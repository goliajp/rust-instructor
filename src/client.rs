use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};
use crate::provider::{Message, ProviderKind};
use crate::usage::Usage;

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
/// Supports OpenAI (via structured output) and Anthropic (via tool use).
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    pub(crate) provider: ProviderKind,
    default_model: Option<String>,
    default_system: Option<String>,
    default_max_retries: u32,
    default_temperature: Option<f64>,
    default_max_tokens: u32,
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
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Set the default system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.default_system = Some(system.into());
        self
    }

    /// Set the default max retries on parse/validation failure.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.default_max_retries = retries;
        self
    }

    /// Set the default temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.default_temperature = Some(temp);
        self
    }

    /// Set the default max output tokens.
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.default_max_tokens = tokens;
        self
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
            _phantom: PhantomData,
        }
    }
}

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
    _phantom: PhantomData<T>,
}

impl<T> ExtractBuilder<'_, T>
where
    T: DeserializeOwned + JsonSchema,
{
    /// Override the model for this request.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Override the system prompt for this request.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the temperature (0.0 = deterministic, 1.0 = creative).
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum output tokens.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set the maximum retry attempts on parse failure.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Add extra context to the prompt.
    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    async fn execute(self) -> Result<ExtractResult<T>> {
        let schema_name = T::schema_name().to_string();
        let root_schema = schemars::schema_for!(T);

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

            let messages = vec![Message {
                role: "user".into(),
                content: prompt_with_retry,
            }];

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

            match serde_json::from_str::<T>(&raw.content) {
                Ok(value) => {
                    #[cfg(feature = "cost-tracking")]
                    {
                        usage.cost = tiktoken::pricing::estimate_cost(
                            model,
                            usage.input_tokens as u64,
                            usage.output_tokens as u64,
                        )
                    }
                    return Ok(ExtractResult { value, usage });
                }
                Err(e) => {
                    last_error = format!("{e} (raw output: {})", truncate(&raw.content, 200));
                    if attempt < self.max_retries {
                        usage.retries += 1;
                    }
                }
            }
        }

        Err(Error::ExtractionFailed {
            retries: self.max_retries,
            message: last_error,
        })
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
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let s = "a".repeat(300);
        assert_eq!(truncate(&s, 200).len(), 200);
    }

    #[test]
    fn test_truncate_unicode() {
        let s = "你好世界你好世界你好世界";
        let t = truncate(s, 6);
        assert!(t.len() <= 6);
        assert!(t.is_char_boundary(t.len()));
    }

    #[test]
    fn test_client_builder() {
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
    fn test_openai_compatible() {
        let client = Client::openai_compatible("key", "https://api.deepseek.com/v1");
        match &client.provider {
            ProviderKind::OpenAi { base_url, .. } => {
                assert_eq!(base_url, "https://api.deepseek.com/v1");
            }
            _ => panic!("expected OpenAi provider"),
        }
    }

    #[test]
    fn test_anthropic_defaults() {
        let client = Client::anthropic("test-key");
        assert_eq!(client.default_temperature, None);
        match &client.provider {
            ProviderKind::Anthropic { base_url, .. } => {
                assert_eq!(base_url, "https://api.anthropic.com/v1");
            }
            _ => panic!("expected Anthropic provider"),
        }
    }
}
