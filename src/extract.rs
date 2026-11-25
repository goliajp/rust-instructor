use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::time::Duration;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::backoff::BackoffConfig;
use crate::client::{Client, ExtractResult};
use crate::error::Result;
use crate::partial::PartialValue;
use crate::provider::{ImageInput, Message};
use crate::reasoning::Reasoning;
use crate::usage::Usage;
use crate::validate::{Validate, ValidationError};

pub(crate) type ValidatorFn<T> =
    Box<dyn Fn(&T) -> std::result::Result<(), ValidationError> + Send + Sync>;
type RequestHook = Box<dyn Fn(&str, &str) + Send + Sync>;
type ResponseHook = Box<dyn Fn(&Usage) + Send + Sync>;
type StreamHook = Box<dyn Fn(&str) + Send + Sync>;
type PartialHook = Box<dyn Fn(PartialValue<'_>) + Send + Sync>;

/// Builder for configuring an extraction request.
///
/// Created by [`Client::extract`]. Call `.await` to execute the request.
pub struct ExtractBuilder<'a, T> {
    pub(crate) client: &'a Client,
    pub(crate) prompt: String,
    pub(crate) model: Option<String>,
    pub(crate) system: Option<String>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) max_retries: u32,
    pub(crate) backoff: Option<BackoffConfig>,
    pub(crate) timeout: Duration,
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) context: Option<String>,
    pub(crate) images: Vec<ImageInput>,
    pub(crate) history: Option<Vec<Message>>,
    pub(crate) validator: Option<ValidatorFn<T>>,
    pub(crate) on_request: Option<RequestHook>,
    pub(crate) on_response: Option<ResponseHook>,
    pub(crate) on_stream: Option<StreamHook>,
    pub(crate) on_partial: Option<PartialHook>,
    // `extract_many`: the schema root is `{"items": T}`
    pub(crate) list: bool,
    _phantom: PhantomData<T>,
}

impl<'a, T> ExtractBuilder<'a, T> {
    pub(crate) fn new(client: &'a Client, prompt: String, list: bool) -> Self {
        Self {
            client,
            prompt,
            model: client.default_model.clone(),
            system: client.default_system.clone(),
            temperature: client.default_temperature,
            max_tokens: client.default_max_tokens,
            max_retries: client.default_max_retries,
            backoff: client.default_backoff.clone(),
            timeout: client.default_timeout,
            reasoning: client.default_reasoning,
            context: None,
            images: Vec::new(),
            history: None,
            validator: None,
            on_request: None,
            on_response: None,
            on_stream: None,
            on_partial: None,
            list,
            _phantom: PhantomData,
        }
    }
}

impl<T> ExtractBuilder<'_, T>
where
    T: DeserializeOwned + JsonSchema,
{
    /// Override the model for this request. Fallback clients keep their own.
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
    ///
    /// Not sent to Anthropic: the current Claude generations reject a
    /// non-default `temperature` with a 400.
    pub fn temperature(self, temp: f64) -> Self {
        Self {
            temperature: Some(temp),
            ..self
        }
    }

    /// Set the maximum output tokens. See [`Client::with_max_tokens`] for
    /// the field each provider receives.
    pub fn max_tokens(self, tokens: u32) -> Self {
        Self {
            max_tokens: Some(tokens),
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

    /// Set the reasoning level for this request. See [`Reasoning`].
    pub fn reasoning(self, reasoning: Reasoning) -> Self {
        Self {
            reasoning: Some(reasoning),
            ..self
        }
    }

    /// Enable exponential backoff for retryable HTTP errors.
    pub fn retry_backoff(self, config: BackoffConfig) -> Self {
        Self {
            backoff: Some(config),
            ..self
        }
    }

    /// Set the overall request timeout for this extraction.
    ///
    /// Covers the entire extraction including all retries, backoff, and fallback.
    pub fn timeout(self, timeout: Duration) -> Self {
        Self { timeout, ..self }
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

    /// Stream the response and call `f` with each raw text delta.
    ///
    /// The final result is still a complete `ExtractResult<T>` parsed from
    /// the whole response.
    pub fn on_stream<F>(self, f: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        Self {
            on_stream: Some(Box::new(f)),
            ..self
        }
    }

    /// Stream the response and call `f` after each delta with the JSON
    /// parsed so far.
    ///
    /// The value is a borrowed view of an incremental parser that reads each
    /// byte once, so the cost of the whole stream grows linearly with its
    /// length. For [`Client::extract_many`] the view is the list itself.
    /// A retry starts a new value.
    ///
    /// ```rust,no_run
    /// # use serde::Deserialize;
    /// # use schemars::JsonSchema;
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Article { title: String, body: String }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = instructors::Client::openai("sk-...");
    /// let article: Article = client
    ///     .extract("...")
    ///     .on_partial(|v| {
    ///         if let Some(title) = v.get("title").and_then(|t| t.as_str()) {
    ///             println!("title so far: {title}");
    ///         }
    ///     })
    ///     .await?
    ///     .value;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_partial<F>(self, f: F) -> Self
    where
        F: Fn(PartialValue<'_>) + Send + Sync + 'static,
    {
        Self {
            on_partial: Some(Box::new(f)),
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
}

impl<T> ExtractBuilder<'_, T>
where
    T: DeserializeOwned + JsonSchema + Validate,
{
    /// Enable trait-based validation. Calls `T::validate()` after deserialization.
    /// If validation fails, the error is fed back to the LLM for retry.
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
    T: DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    type Output = Result<ExtractResult<T>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize, JsonSchema)]
    struct Dummy {
        x: i32,
    }

    #[test]
    fn inherits_client_defaults() {
        let client = Client::openai("key")
            .with_model("gpt-5.6-luna")
            .with_system("custom")
            .with_reasoning(Reasoning::Budget(10));
        let b = client.extract::<Dummy>("p");
        assert_eq!(b.prompt, "p");
        assert_eq!(b.model.as_deref(), Some("gpt-5.6-luna"));
        assert_eq!(b.system.as_deref(), Some("custom"));
        assert_eq!(b.max_retries, 2);
        assert_eq!(b.temperature, Some(0.0));
        assert_eq!(b.max_tokens, None);
        assert_eq!(b.reasoning, Some(Reasoning::Budget(10)));
        assert!(!b.list);
        assert!(client.extract_many::<Dummy>("p").list);
    }

    #[test]
    fn overrides() {
        let client = Client::openai("key");
        let b = client
            .extract::<Dummy>("p")
            .model("m")
            .system("s")
            .temperature(0.7)
            .max_tokens(1024)
            .max_retries(5)
            .reasoning(Reasoning::Off)
            .context("c")
            .timeout(Duration::from_secs(9))
            .image(ImageInput::Url("u".into()))
            .images(vec![ImageInput::Url("v".into())])
            .messages(vec![Message::user("hi")])
            .on_request(|_, _| {})
            .on_response(|_| {})
            .on_stream(|_| {})
            .on_partial(|_| {});
        assert_eq!(b.model.as_deref(), Some("m"));
        assert_eq!(b.max_tokens, Some(1024));
        assert_eq!(b.reasoning, Some(Reasoning::Off));
        assert_eq!(b.images.len(), 2);
        assert_eq!(b.timeout, Duration::from_secs(9));
        assert!(b.on_partial.is_some());
    }

    #[test]
    fn validators() {
        let client = Client::openai("key");
        let b = client
            .extract::<Dummy>("p")
            .validate(|d| if d.x > 0 { Ok(()) } else { Err("neg".into()) });
        let v = b.validator.as_ref().unwrap();
        assert!(v(&Dummy { x: 1 }).is_ok());
        assert_eq!(v(&Dummy { x: -1 }).unwrap_err().message, "neg");
    }
}
