use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};
use crate::provider::{ImageInput, Message, ProviderKind};
use crate::usage::Usage;
use crate::validate::{Validate, ValidationError};

/// Result of a successful extraction containing the typed value and usage info.
#[derive(Debug, Clone)]
pub struct ExtractResult<T> {
    /// The extracted, deserialized value.
    pub value: T,
    /// Token usage and cost information.
    pub usage: Usage,
}

/// LLM client for structured data extraction.
///
/// Supports OpenAI (via structured output), Anthropic (via tool use),
/// and any OpenAI-compatible API.
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    pub(crate) provider: ProviderKind,
    pub(crate) default_model: Option<String>,
    pub(crate) default_system: Option<String>,
    pub(crate) default_max_retries: u32,
    pub(crate) default_temperature: Option<f64>,
    pub(crate) default_max_tokens: u32,
}

impl Client {
    /// Create a client for OpenAI models.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::openai("sk-...");
    /// ```
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            provider: ProviderKind::OpenAi {
                api_key: api_key.into(),
                base_url: "https://api.openai.com/v1".into(),
            },
            default_model: None,
            default_system: None,
            default_max_retries: 2,
            default_temperature: Some(0.0),
            default_max_tokens: 4096,
        }
    }

    /// Create a client for Anthropic models.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::anthropic("sk-ant-...");
    /// ```
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            provider: ProviderKind::Anthropic {
                api_key: api_key.into(),
                base_url: "https://api.anthropic.com/v1".into(),
            },
            default_model: None,
            default_system: None,
            default_max_retries: 2,
            default_temperature: None,
            default_max_tokens: 4096,
        }
    }

    /// Create a client for any Anthropic-compatible API.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::anthropic_compatible(
    ///     "sk-ant-...",
    ///     "https://custom-anthropic-proxy.example.com/v1",
    /// );
    /// ```
    pub fn anthropic_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            provider: ProviderKind::Anthropic {
                api_key: api_key.into(),
                base_url: base_url.into(),
            },
            default_model: None,
            default_system: None,
            default_max_retries: 2,
            default_temperature: None,
            default_max_tokens: 4096,
        }
    }

    /// Create a client for any OpenAI-compatible API (e.g. DeepSeek, Together, local).
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::openai_compatible(
    ///     "sk-...",
    ///     "https://api.deepseek.com/v1",
    /// );
    /// ```
    pub fn openai_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            provider: ProviderKind::OpenAi {
                api_key: api_key.into(),
                base_url: base_url.into(),
            },
            default_model: None,
            default_system: None,
            default_max_retries: 2,
            default_temperature: Some(0.0),
            default_max_tokens: 4096,
        }
    }

    /// Set the default model for all extractions.
    pub fn with_model(self, model: impl Into<String>) -> Self {
        Self {
            default_model: Some(model.into()),
            ..self
        }
    }

    /// Set the default system prompt.
    pub fn with_system(self, system: impl Into<String>) -> Self {
        Self {
            default_system: Some(system.into()),
            ..self
        }
    }

    /// Set the default max retries on parse/validation failure.
    pub fn with_max_retries(self, retries: u32) -> Self {
        Self {
            default_max_retries: retries,
            ..self
        }
    }

    /// Set the default temperature.
    pub fn with_temperature(self, temp: f64) -> Self {
        Self {
            default_temperature: Some(temp),
            ..self
        }
    }

    /// Set the default max output tokens.
    pub fn with_max_tokens(self, tokens: u32) -> Self {
        Self {
            default_max_tokens: tokens,
            ..self
        }
    }

    /// Begin a batch extraction over multiple prompts with configurable concurrency.
    ///
    /// Returns a [`BatchBuilder`] that processes prompts concurrently.
    ///
    /// ```rust,no_run
    /// # use serde::Deserialize;
    /// # use schemars::JsonSchema;
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Contact { name: String }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = instructors::Client::openai("sk-...");
    /// let prompts = vec!["John Doe".into(), "Jane Smith".into()];
    /// let results = client.extract_batch::<Contact>(prompts)
    ///     .concurrency(5)
    ///     .run()
    ///     .await;
    ///
    /// for result in results {
    ///     println!("{}", result?.value.name);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_batch<T>(&self, prompts: Vec<String>) -> crate::batch::BatchBuilder<'_, T>
    where
        T: DeserializeOwned + JsonSchema + Send + 'static,
    {
        crate::batch::BatchBuilder::new(self, prompts)
    }

    /// Extract a list of items from the prompt.
    ///
    /// Internally wraps the target type in a `Vec<T>` for the LLM to populate.
    ///
    /// ```rust,no_run
    /// # use serde::Deserialize;
    /// # use schemars::JsonSchema;
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Entity { name: String, entity_type: String }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = instructors::Client::openai("sk-...");
    /// let entities: Vec<Entity> = client
    ///     .extract_many("Apple CEO Tim Cook met with Google CEO Sundar Pichai")
    ///     .await?.value;
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_many<T>(&self, prompt: impl Into<String>) -> ExtractBuilder<'_, Vec<T>>
    where
        T: DeserializeOwned + JsonSchema,
    {
        ExtractBuilder {
            client: self,
            prompt: prompt.into(),
            model: self.default_model.clone(),
            system: self.default_system.clone(),
            temperature: self.default_temperature,
            max_tokens: self.default_max_tokens,
            max_retries: self.default_max_retries,
            context: None,
            images: Vec::new(),
            history: None,
            validator: None,
            on_request: None,
            on_response: None,
            _phantom: PhantomData,
        }
    }

    /// Begin an extraction request. The return type `T` must implement
    /// [`serde::Deserialize`] and [`schemars::JsonSchema`].
    ///
    /// ```rust,no_run
    /// # use serde::Deserialize;
    /// # use schemars::JsonSchema;
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Contact { name: String }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = instructors::Client::openai("sk-...");
    /// let contact: Contact = client.extract("John Doe, john@example.com").await?.value;
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract<T>(&self, prompt: impl Into<String>) -> ExtractBuilder<'_, T>
    where
        T: DeserializeOwned + JsonSchema,
    {
        ExtractBuilder {
            client: self,
            prompt: prompt.into(),
            model: self.default_model.clone(),
            system: self.default_system.clone(),
            temperature: self.default_temperature,
            max_tokens: self.default_max_tokens,
            max_retries: self.default_max_retries,
            context: None,
            images: Vec::new(),
            history: None,
            validator: None,
            on_request: None,
            on_response: None,
            _phantom: PhantomData,
        }
    }
}

type ValidatorFn<T> = Box<dyn Fn(&T) -> std::result::Result<(), ValidationError> + Send + Sync>;

/// Builder for configuring an extraction request.
///
/// Created by [`Client::extract`]. Call `.await` to execute the request.
pub struct ExtractBuilder<'a, T> {
    client: &'a Client,
    prompt: String,
    model: Option<String>,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: u32,
    max_retries: u32,
    context: Option<String>,
    images: Vec<ImageInput>,
    history: Option<Vec<Message>>,
    validator: Option<ValidatorFn<T>>,
    on_request: Option<RequestHook>,
    on_response: Option<ResponseHook>,
    _phantom: PhantomData<T>,
}

type RequestHook = Box<dyn Fn(&str, &str) + Send + Sync>;
type ResponseHook = Box<dyn Fn(&Usage) + Send + Sync>;

impl<T> ExtractBuilder<'_, T>
where
    T: DeserializeOwned + JsonSchema,
{
    /// Override the model for this request.
    pub fn model(self, model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..self
        }
    }

    /// Override the system prompt for this request.
    pub fn system(self, system: impl Into<String>) -> Self {
        Self {
            system: Some(system.into()),
            ..self
        }
    }

    /// Set the temperature (0.0 = deterministic, 1.0 = creative).
    pub fn temperature(self, temp: f64) -> Self {
        Self {
            temperature: Some(temp),
            ..self
        }
    }

    /// Set the maximum output tokens.
    pub fn max_tokens(self, tokens: u32) -> Self {
        Self {
            max_tokens: tokens,
            ..self
        }
    }

    /// Set the maximum retry attempts on parse/validation failure.
    pub fn max_retries(self, retries: u32) -> Self {
        Self {
            max_retries: retries,
            ..self
        }
    }

    /// Add extra context to the prompt.
    pub fn context(self, ctx: impl Into<String>) -> Self {
        Self {
            context: Some(ctx.into()),
            ..self
        }
    }

    /// Add an image to the extraction prompt (for vision-capable models).
    ///
    /// Can be called multiple times to add multiple images.
    pub fn image(mut self, img: ImageInput) -> Self {
        self.images.push(img);
        self
    }

    /// Add multiple images to the extraction prompt.
    pub fn images(mut self, imgs: Vec<ImageInput>) -> Self {
        self.images.extend(imgs);
        self
    }

    /// Set prior message history for multi-turn conversations.
    ///
    /// Messages are prepended before the extraction prompt.
    ///
    /// ```rust,no_run
    /// # use serde::Deserialize;
    /// # use schemars::JsonSchema;
    /// use instructors::Message;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Summary { text: String }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = instructors::Client::openai("sk-...");
    /// let result = client.extract::<Summary>("summarize the above")
    ///     .messages(vec![
    ///         Message::user("Here is a long document..."),
    ///         Message::assistant("I see the document."),
    ///     ])
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn messages(self, msgs: Vec<Message>) -> Self {
        Self {
            history: Some(msgs),
            ..self
        }
    }

    /// Register a hook called before each API request.
    ///
    /// Receives `(model, prompt)`.
    pub fn on_request<F>(self, f: F) -> Self
    where
        F: Fn(&str, &str) + Send + Sync + 'static,
    {
        Self {
            on_request: Some(Box::new(f)),
            ..self
        }
    }

    /// Register a hook called after a successful extraction.
    ///
    /// Receives the final `Usage`.
    pub fn on_response<F>(self, f: F) -> Self
    where
        F: Fn(&Usage) + Send + Sync + 'static,
    {
        Self {
            on_response: Some(Box::new(f)),
            ..self
        }
    }

    /// Add a custom validation function. If validation fails, the error message
    /// is fed back to the LLM and the request is retried.
    ///
    /// ```rust,no_run
    /// # use serde::Deserialize;
    /// # use schemars::JsonSchema;
    /// #[derive(Deserialize, JsonSchema)]
    /// struct User { name: String, age: u32 }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = instructors::Client::openai("sk-...");
    /// let user: User = client.extract("...")
    ///     .validate(|u: &User| {
    ///         if u.age > 150 { Err("age must be <= 150".into()) } else { Ok(()) }
    ///     })
    ///     .await?.value;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate<F>(self, f: F) -> Self
    where
        F: Fn(&T) -> std::result::Result<(), ValidationError> + Send + Sync + 'static,
    {
        Self {
            validator: Some(Box::new(f)),
            ..self
        }
    }

    async fn execute(self) -> Result<ExtractResult<T>>
    where
        T: 'static,
    {
        let (root_schema, schema_name) = crate::schema::cached_schema_for::<T>();

        let mut user_content = self.prompt.clone();
        if let Some(ctx) = &self.context {
            user_content = format!("{user_content}\n\nAdditional context:\n{ctx}");
        }

        let model = self
            .model
            .as_deref()
            .unwrap_or(self.client.provider.default_model());
        let system = self.system.as_deref();

        let mut usage = Usage::default();
        let mut last_error = String::new();

        // pre-clone history once outside the retry loop
        let history: Vec<Message> = self
            .history
            .as_ref()
            .map(|h| h.to_vec())
            .unwrap_or_default();

        for attempt in 0..=self.max_retries {
            let prompt_with_retry = if attempt == 0 {
                user_content.clone()
            } else {
                format!(
                    "{user_content}\n\n\
                     IMPORTANT: A previous extraction attempt produced invalid output.\n\
                     Error: {last_error}\n\
                     Please ensure your response is valid JSON matching the schema exactly."
                )
            };

            let mut messages = Vec::with_capacity(history.len() + 1);
            messages.extend(history.iter().cloned());
            messages.push(Message {
                role: "user".into(),
                content: prompt_with_retry.clone(),
                images: self.images.clone(),
            });

            if let Some(ref hook) = self.on_request {
                hook(model, &prompt_with_retry);
            }

            let raw = self
                .client
                .provider
                .send(
                    &self.client.http,
                    model,
                    system,
                    &messages,
                    &root_schema,
                    &schema_name,
                    self.temperature,
                    self.max_tokens,
                )
                .await?;

            usage.accumulate(raw.input_tokens, raw.output_tokens);

            // parse JSON
            let value: T = match serde_json::from_str(&raw.content) {
                Ok(v) => v,
                Err(e) => {
                    last_error = format!("{e} (raw: {})", truncate(&raw.content, 200));
                    if attempt < self.max_retries {
                        usage.retries += 1;
                    }
                    continue;
                }
            };

            // run validator if present
            if let Some(ref validator) = self.validator
                && let Err(e) = validator(&value)
            {
                last_error = format!("validation: {}", e.message);
                if attempt < self.max_retries {
                    usage.retries += 1;
                }
                continue;
            }

            // success
            #[cfg(feature = "cost-tracking")]
            {
                usage.cost = tiktoken::pricing::estimate_cost(
                    model,
                    usage.input_tokens as u64,
                    usage.output_tokens as u64,
                );
            }
            if let Some(ref hook) = self.on_response {
                hook(&usage);
            }
            return Ok(ExtractResult { value, usage });
        }

        // determine error type
        let err = if last_error.starts_with("validation:") {
            Error::ValidationFailed {
                retries: self.max_retries,
                message: last_error,
            }
        } else {
            Error::ExtractionFailed {
                retries: self.max_retries,
                message: last_error,
            }
        };
        Err(err)
    }
}

impl<T> ExtractBuilder<'_, T>
where
    T: DeserializeOwned + JsonSchema + Validate,
{
    /// Enable trait-based validation. Calls `T::validate()` after deserialization.
    /// If validation fails, the error is fed back to the LLM for retry.
    ///
    /// Requires `T: Validate`.
    ///
    /// ```rust,no_run
    /// # use serde::Deserialize;
    /// # use schemars::JsonSchema;
    /// # use instructors::{Validate, ValidationError};
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Email { address: String }
    ///
    /// impl Validate for Email {
    ///     fn validate(&self) -> Result<(), ValidationError> {
    ///         if self.address.contains('@') { Ok(()) }
    ///         else { Err("invalid email".into()) }
    ///     }
    /// }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = instructors::Client::openai("sk-...");
    /// let email: Email = client.extract("...").validated().await?.value;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validated(self) -> Self {
        Self {
            validator: Some(Box::new(|v: &T| v.validate())),
            ..self
        }
    }
}

impl<'a, T> IntoFuture for ExtractBuilder<'a, T>
where
    T: DeserializeOwned + JsonSchema + Send + 'static,
{
    type Output = Result<ExtractResult<T>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..s.floor_char_boundary(max)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_long() {
        let s = "a".repeat(300);
        assert_eq!(truncate(&s, 200).len(), 200);
    }

    #[test]
    fn truncate_unicode() {
        let s = "你好世界你好世界你好世界";
        let t = truncate(s, 6);
        assert!(t.len() <= 6);
        assert!(t.is_char_boundary(t.len()));
    }

    #[test]
    fn truncate_empty() {
        assert_eq!(truncate("", 10), "");
    }

    #[test]
    fn truncate_zero() {
        assert_eq!(truncate("hello", 0), "");
    }

    #[test]
    fn client_builder_openai() {
        let client = Client::openai("test-key")
            .with_model("gpt-4o-mini")
            .with_max_retries(5)
            .with_temperature(0.5)
            .with_max_tokens(2048)
            .with_system("custom system");

        assert_eq!(client.default_model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(client.default_max_retries, 5);
        assert_eq!(client.default_temperature, Some(0.5));
        assert_eq!(client.default_max_tokens, 2048);
        assert_eq!(client.default_system.as_deref(), Some("custom system"));
    }

    #[test]
    fn client_builder_anthropic() {
        let client = Client::anthropic("test-key");
        assert_eq!(client.default_temperature, None);
        assert_eq!(client.default_max_retries, 2);
        assert_eq!(client.default_max_tokens, 4096);
        match &client.provider {
            ProviderKind::Anthropic { base_url, .. } => {
                assert_eq!(base_url, "https://api.anthropic.com/v1");
            }
            _ => panic!("expected Anthropic provider"),
        }
    }

    #[test]
    fn client_openai_compatible() {
        let client = Client::openai_compatible("key", "https://api.deepseek.com/v1");
        match &client.provider {
            ProviderKind::OpenAi { base_url, .. } => {
                assert_eq!(base_url, "https://api.deepseek.com/v1");
            }
            _ => panic!("expected OpenAi provider"),
        }
    }

    #[test]
    fn client_default_models() {
        let openai = Client::openai("key");
        match &openai.provider {
            ProviderKind::OpenAi { .. } => {}
            _ => panic!("wrong provider"),
        }

        let anthropic = Client::anthropic("key");
        match &anthropic.provider {
            ProviderKind::Anthropic { .. } => {}
            _ => panic!("wrong provider"),
        }
    }

    #[test]
    fn openai_defaults() {
        let client = Client::openai("key");
        assert_eq!(client.default_temperature, Some(0.0));
        assert_eq!(client.default_max_retries, 2);
        match &client.provider {
            ProviderKind::OpenAi { base_url, .. } => {
                assert_eq!(base_url, "https://api.openai.com/v1");
            }
            _ => panic!("expected OpenAi"),
        }
    }

    #[test]
    fn extract_builder_defaults() {
        let client = Client::openai("key").with_model("gpt-4o-mini");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Dummy {
            x: i32,
        }

        let builder = client.extract::<Dummy>("test prompt");
        assert_eq!(builder.prompt, "test prompt");
        assert_eq!(builder.model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(builder.max_retries, 2);
        assert_eq!(builder.temperature, Some(0.0));
        assert_eq!(builder.max_tokens, 4096);
        assert!(builder.context.is_none());
        assert!(builder.validator.is_none());
    }

    #[test]
    fn extract_builder_overrides() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Dummy {
            x: i32,
        }

        let builder = client
            .extract::<Dummy>("prompt")
            .model("gpt-3.5-turbo")
            .system("be precise")
            .temperature(0.7)
            .max_tokens(1024)
            .max_retries(5)
            .context("extra info");

        assert_eq!(builder.model.as_deref(), Some("gpt-3.5-turbo"));
        assert_eq!(builder.system.as_deref(), Some("be precise"));
        assert_eq!(builder.temperature, Some(0.7));
        assert_eq!(builder.max_tokens, 1024);
        assert_eq!(builder.max_retries, 5);
        assert_eq!(builder.context.as_deref(), Some("extra info"));
    }

    #[test]
    fn client_anthropic_compatible() {
        let client = Client::anthropic_compatible("key", "https://proxy.example.com/v1");
        match &client.provider {
            ProviderKind::Anthropic { base_url, .. } => {
                assert_eq!(base_url, "https://proxy.example.com/v1");
            }
            _ => panic!("expected Anthropic provider"),
        }
    }

    #[test]
    fn extract_many_inherits_defaults() {
        let client = Client::openai("key")
            .with_model("gpt-4o-mini")
            .with_system("custom")
            .with_temperature(0.7);

        #[derive(serde::Deserialize, JsonSchema)]
        struct Item {
            x: i32,
        }

        let builder = client.extract_many::<Item>("test");
        assert_eq!(builder.model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(builder.system.as_deref(), Some("custom"));
        assert_eq!(builder.temperature, Some(0.7));
    }

    #[test]
    fn extract_builder_messages() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Dummy {
            x: i32,
        }

        let builder = client
            .extract::<Dummy>("test")
            .messages(vec![Message::user("hello"), Message::assistant("hi")]);
        assert!(builder.history.is_some());
        assert_eq!(builder.history.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn extract_builder_hooks() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Dummy {
            x: i32,
        }

        let builder = client
            .extract::<Dummy>("test")
            .on_request(|_, _| {})
            .on_response(|_| {});
        assert!(builder.on_request.is_some());
        assert!(builder.on_response.is_some());
    }

    #[test]
    fn extract_builder_image() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Dummy {
            x: i32,
        }

        let builder = client
            .extract::<Dummy>("describe")
            .image(ImageInput::Url("https://example.com/img.png".into()))
            .image(ImageInput::Base64 {
                media_type: "image/jpeg".into(),
                data: "abc".into(),
            });
        assert_eq!(builder.images.len(), 2);
    }

    #[test]
    fn extract_builder_images_batch() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Dummy {
            x: i32,
        }

        let builder = client.extract::<Dummy>("describe").images(vec![
            ImageInput::Url("https://a.com/1.png".into()),
            ImageInput::Url("https://a.com/2.png".into()),
        ]);
        assert_eq!(builder.images.len(), 2);
    }

    #[test]
    fn extract_builder_with_validator() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Dummy {
            x: i32,
        }

        let builder = client.extract::<Dummy>("prompt").validate(|d| {
            if d.x > 0 {
                Ok(())
            } else {
                Err("x must be positive".into())
            }
        });

        assert!(builder.validator.is_some());
        // test the validator function directly
        let validator = builder.validator.as_ref().unwrap();
        let good = Dummy { x: 1 };
        assert!(validator(&good).is_ok());
        let bad = Dummy { x: -1 };
        let err = validator(&bad).unwrap_err();
        assert_eq!(err.message, "x must be positive");
    }

    #[test]
    fn extract_builder_validated_trait() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct Strict {
            value: i32,
        }

        impl Validate for Strict {
            fn validate(&self) -> std::result::Result<(), ValidationError> {
                if self.value >= 0 {
                    Ok(())
                } else {
                    Err("must be non-negative".into())
                }
            }
        }

        let builder = client.extract::<Strict>("prompt").validated();
        assert!(builder.validator.is_some());
        let validator = builder.validator.as_ref().unwrap();
        assert!(validator(&Strict { value: 5 }).is_ok());
        assert!(validator(&Strict { value: -1 }).is_err());
    }
}
