use std::time::Duration;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::backoff::BackoffConfig;
use crate::extract::ExtractBuilder;
use crate::provider::{OaiDialect, ProviderKind};
use crate::reasoning::Reasoning;
use crate::usage::Usage;

/// Result of a successful extraction containing the typed value and usage info.
#[derive(Debug, Clone)]
pub struct ExtractResult<T> {
    /// The extracted, deserialized value.
    pub value: T,
    /// Token usage and cost information.
    pub usage: Usage,
}

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// LLM client for structured data extraction.
///
/// Speaks three wire protocols: OpenAI Chat Completions (`response_format`
/// with a strict JSON schema), Anthropic Messages (`output_config.format`, or
/// a tool), and Gemini `generateContent` (`responseSchema`). The `_compatible`
/// constructors point any of them at another base URL.
///
/// All clients in a process share one HTTP connection pool.
#[derive(Clone)]
pub struct Client {
    pub(crate) http: reqwest::Client,
    pub(crate) provider: ProviderKind,
    pub(crate) default_model: Option<String>,
    pub(crate) default_system: Option<String>,
    pub(crate) default_max_retries: u32,
    pub(crate) default_temperature: Option<f64>,
    pub(crate) default_max_tokens: Option<u32>,
    pub(crate) default_backoff: Option<BackoffConfig>,
    pub(crate) default_timeout: Duration,
    pub(crate) default_reasoning: Option<Reasoning>,
    pub(crate) fallbacks: Vec<Client>,
}

impl Client {
    fn with_provider(provider: ProviderKind, temperature: Option<f64>) -> Self {
        Self {
            http: crate::provider::shared_client(),
            provider,
            default_model: None,
            default_system: None,
            default_max_retries: 2,
            default_temperature: temperature,
            default_max_tokens: None,
            default_backoff: None,
            default_timeout: DEFAULT_TIMEOUT,
            default_reasoning: None,
            fallbacks: Vec::new(),
        }
    }

    /// Create a client for OpenAI models.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::openai("sk-...");
    /// ```
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::openai_compatible(api_key, "https://api.openai.com/v1")
    }

    /// Create a client for any OpenAI-compatible Chat Completions API.
    ///
    /// The base URL's host selects request details for APIs this crate knows:
    /// `api.openai.com` gets `max_completion_tokens`; DeepSeek
    /// (`deepseek.com`), Qwen (`aliyuncs.com`), Kimi (`moonshot.cn`,
    /// `moonshot.ai`) and GLM (`bigmodel.cn`, `z.ai`) get their own
    /// [`Reasoning`] parameters. Any other host is treated as generic
    /// OpenAI-compatible.
    ///
    /// The request always asks for a strict `json_schema` response format, so
    /// the server must support it.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::openai_compatible(
    ///     "sk-...",
    ///     "https://api.together.xyz/v1",
    /// );
    /// ```
    pub fn openai_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let dialect = OaiDialect::from_base_url(&base_url);
        let provider = ProviderKind::OpenAi {
            api_key: api_key.into(),
            base_url,
            dialect,
        };
        Self::with_provider(provider, Some(0.0))
    }

    /// Create a client for Anthropic models.
    ///
    /// Uses structured output (`output_config.format`), which every current
    /// Claude model accepts. A type that refers to itself cannot be expressed
    /// there, so it is sent as a tool instead.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::anthropic("sk-ant-...");
    /// ```
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        let provider = ProviderKind::Anthropic {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".into(),
            structured_output: true,
        };
        Self::with_provider(provider, None)
    }

    /// Create a client for any Anthropic-compatible API.
    ///
    /// Sends the schema as a forced tool call, which compatible servers
    /// support more widely than `output_config`. Call
    /// [`with_anthropic_structured_output`](Self::with_anthropic_structured_output)
    /// when the server is Anthropic's own API behind a proxy, since the
    /// newest Claude models reject forced tool calls.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::anthropic_compatible(
    ///     "sk-ant-...",
    ///     "https://custom-anthropic-proxy.example.com/v1",
    /// );
    /// ```
    pub fn anthropic_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let provider = ProviderKind::Anthropic {
            api_key: api_key.into(),
            base_url: base_url.into(),
            structured_output: false,
        };
        Self::with_provider(provider, None)
    }

    /// Create a client for Google Gemini models.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::gemini("AIza...");
    /// ```
    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::gemini_compatible(api_key, "https://generativelanguage.googleapis.com/v1beta")
    }

    /// Create a client for any Gemini-compatible API.
    ///
    /// ```rust,no_run
    /// let client = instructors::Client::gemini_compatible(
    ///     "AIza...",
    ///     "https://custom-gemini-proxy.example.com/v1beta",
    /// );
    /// ```
    pub fn gemini_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let provider = ProviderKind::Gemini {
            api_key: api_key.into(),
            base_url: base_url.into(),
        };
        Self::with_provider(provider, None)
    }

    /// Set the default model for all extractions.
    ///
    /// A fallback client uses its own default model, not the primary's.
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
    ///
    /// Not sent to Anthropic: the current Claude generations reject a
    /// non-default `temperature` with a 400. Sent to the OpenAI and Gemini
    /// families.
    pub fn with_temperature(self, temp: f64) -> Self {
        Self {
            default_temperature: Some(temp),
            ..self
        }
    }

    /// Set the default reasoning level. See [`Reasoning`] for what each
    /// provider receives. Not set by default, and then nothing is sent.
    pub fn with_reasoning(self, reasoning: Reasoning) -> Self {
        Self {
            default_reasoning: Some(reasoning),
            ..self
        }
    }

    /// Add a fallback client. If the primary provider fails after exhausting
    /// retries, the request is sent to each fallback in order, with the
    /// fallback's own default model. When all fail, the error lists every
    /// provider's error.
    pub fn with_fallback(mut self, fallback: Client) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    /// Set the default max output tokens.
    ///
    /// Sent as `max_completion_tokens` to OpenAI, `max_tokens` to
    /// OpenAI-compatible APIs and Anthropic, and `maxOutputTokens` to Gemini.
    /// When unset, OpenAI-family requests carry no limit, and Anthropic and
    /// Gemini requests carry 16384.
    pub fn with_max_tokens(self, tokens: u32) -> Self {
        Self {
            default_max_tokens: Some(tokens),
            ..self
        }
    }

    /// Send the schema as Anthropic structured output
    /// (`output_config.format`). This is the default for
    /// [`Client::anthropic`]; use it on an
    /// [`anthropic_compatible`](Self::anthropic_compatible) client whose
    /// server supports it.
    ///
    /// No effect on OpenAI or Gemini clients.
    pub fn with_anthropic_structured_output(mut self) -> Self {
        if let ProviderKind::Anthropic {
            structured_output, ..
        } = &mut self.provider
        {
            *structured_output = true;
        }
        self
    }

    /// Send the schema as a tool (`tools` + forced `tool_choice`) instead of
    /// structured output, as 1.4 did by default. Models that reject forced
    /// tool calls (Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and later) need
    /// structured output.
    ///
    /// No effect on OpenAI or Gemini clients.
    pub fn with_anthropic_tool_use(mut self) -> Self {
        if let ProviderKind::Anthropic {
            structured_output, ..
        } = &mut self.provider
        {
            *structured_output = false;
        }
        self
    }

    /// Enable exponential backoff for retryable HTTP errors (429, 500, 502,
    /// 503, 504, 529).
    pub fn with_retry_backoff(self, config: BackoffConfig) -> Self {
        Self {
            default_backoff: Some(config),
            ..self
        }
    }

    /// Set the overall request timeout. Default: 60 seconds.
    ///
    /// This timeout covers the entire extraction including all retries,
    /// backoff, and fallback attempts.
    pub fn with_timeout(self, timeout: Duration) -> Self {
        Self {
            default_timeout: timeout,
            ..self
        }
    }

    /// Begin a batch extraction over multiple prompts with configurable concurrency.
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
        T: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    {
        crate::batch::BatchBuilder::new(self, prompts)
    }

    /// Extract a list of items from the prompt.
    ///
    /// The schema sent is an object `{"items": [T, ...]}`, because strict
    /// structured output needs an object at the root; the result is the list.
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
        ExtractBuilder::new(self, prompt.into(), true)
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
        ExtractBuilder::new(self, prompt.into(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_pick_provider_and_defaults() {
        let c = Client::openai("k");
        assert_eq!(c.default_temperature, Some(0.0));
        assert_eq!(c.default_max_tokens, None);
        assert!(matches!(
            c.provider,
            ProviderKind::OpenAi { ref base_url, dialect: OaiDialect::OpenAi, .. }
                if base_url == "https://api.openai.com/v1"
        ));

        let c = Client::openai_compatible("k", "https://api.deepseek.com/v1");
        assert!(matches!(
            c.provider,
            ProviderKind::OpenAi {
                dialect: OaiDialect::DeepSeek,
                ..
            }
        ));

        let c = Client::anthropic("k");
        assert_eq!(c.default_temperature, None);
        assert!(matches!(
            c.provider,
            ProviderKind::Anthropic {
                structured_output: true,
                ..
            }
        ));
        let c = c.with_anthropic_tool_use();
        assert!(matches!(
            c.provider,
            ProviderKind::Anthropic {
                structured_output: false,
                ..
            }
        ));

        let c = Client::anthropic_compatible("k", "https://proxy.example.com/v1");
        assert!(matches!(
            c.provider,
            ProviderKind::Anthropic {
                structured_output: false,
                ..
            }
        ));

        let c = Client::gemini("k");
        assert!(matches!(
            c.provider,
            ProviderKind::Gemini { ref base_url, .. }
                if base_url == "https://generativelanguage.googleapis.com/v1beta"
        ));
        assert_eq!(c.default_timeout, Duration::from_secs(60));
    }

    #[test]
    fn setters() {
        let c = Client::openai("k")
            .with_model("gpt-5.6-luna")
            .with_max_retries(5)
            .with_temperature(0.5)
            .with_max_tokens(2048)
            .with_system("s")
            .with_reasoning(Reasoning::Off)
            .with_timeout(Duration::from_secs(5))
            .with_retry_backoff(BackoffConfig::default())
            .with_fallback(Client::anthropic("a"));
        assert_eq!(c.default_model.as_deref(), Some("gpt-5.6-luna"));
        assert_eq!(c.default_max_retries, 5);
        assert_eq!(c.default_temperature, Some(0.5));
        assert_eq!(c.default_max_tokens, Some(2048));
        assert_eq!(c.default_system.as_deref(), Some("s"));
        assert_eq!(c.default_reasoning, Some(Reasoning::Off));
        assert_eq!(c.default_timeout, Duration::from_secs(5));
        assert!(c.default_backoff.is_some());
        assert_eq!(c.fallbacks.len(), 1);
    }
}
