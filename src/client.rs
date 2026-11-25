use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;

use crate::batch::Batch;
use crate::exchange::RetryPolicy;
use crate::extract::Extract;
use crate::reasoning::Reasoning;
use crate::schema::{Schema, entry_for};
use crate::transport::Transport;
use crate::wire::{Host, OpenAiApi, OutputMode, Provider, Wire};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// LLM client for structured extraction. Cloning is cheap.
///
/// Speaks OpenAI Chat Completions (and, opt-in, the Responses API),
/// Anthropic Messages and Gemini `generateContent`. The `_compatible`
/// constructors point any of them at another base URL.
#[derive(Clone)]
pub struct Client {
    pub(crate) inner: Arc<Config>,
}

#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) provider: Provider,
    pub(crate) transport: Option<Arc<dyn Transport>>,
    pub(crate) model: Option<String>,
    pub(crate) system: Option<String>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) mode: OutputMode,
    pub(crate) retry: RetryPolicy,
    pub(crate) max_attempts: u32,
    pub(crate) timeout: Duration,
    pub(crate) fallbacks: Vec<Client>,
}

#[cfg(feature = "reqwest")]
fn default_transport() -> Option<Arc<dyn Transport>> {
    Some(crate::transport::shared())
}

#[cfg(not(feature = "reqwest"))]
fn default_transport() -> Option<Arc<dyn Transport>> {
    None
}

impl Client {
    fn new(wire: Wire, key: String, base_url: String, mode: OutputMode) -> Self {
        let temperature = matches!(wire, Wire::OpenAiChat(_)).then_some(0.0);
        Self {
            inner: Arc::new(Config {
                provider: Provider {
                    wire,
                    key,
                    base_url,
                },
                transport: default_transport(),
                model: None,
                system: None,
                temperature,
                max_tokens: None,
                reasoning: None,
                mode,
                retry: RetryPolicy::default(),
                max_attempts: 3,
                timeout: DEFAULT_TIMEOUT,
                fallbacks: Vec::new(),
            }),
        }
    }

    fn with(mut self, f: impl FnOnce(&mut Config)) -> Self {
        f(Arc::make_mut(&mut self.inner));
        self
    }

    /// A client for OpenAI (Chat Completions; see [`with_api`](Self::with_api)).
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::openai_compatible(api_key, "https://api.openai.com/v1")
    }

    /// A client for any OpenAI-compatible Chat Completions API.
    ///
    /// The base URL's host selects details for APIs this crate knows:
    /// `api.openai.com` gets `max_completion_tokens`; DeepSeek
    /// (`deepseek.com`), Qwen (`aliyuncs.com`), Kimi (`moonshot.cn`,
    /// `moonshot.ai`, `kimi.ai`, `kimi.com`) and GLM (`bigmodel.cn`, `z.ai`)
    /// get their own [`Reasoning`] parameters. DeepSeek, Qwen and GLM default
    /// to [`OutputMode::JsonObject`], the others to [`OutputMode::Schema`];
    /// change it with [`with_output_mode`](Self::with_output_mode).
    pub fn openai_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let host = Host::from_base_url(&base_url);
        Self::new(
            Wire::OpenAiChat(host),
            api_key.into(),
            base_url,
            host.default_mode(),
        )
    }

    /// A client for Anthropic. Uses structured output
    /// (`output_config.format`); a type that refers to itself goes as a tool.
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::new(
            Wire::Anthropic,
            api_key.into(),
            "https://api.anthropic.com/v1".into(),
            OutputMode::Schema,
        )
    }

    /// A client for any Anthropic-compatible API. Sends the schema as a
    /// forced tool call ([`OutputMode::Tool`]), which compatible servers
    /// support more widely than `output_config`.
    pub fn anthropic_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new(
            Wire::Anthropic,
            api_key.into(),
            base_url.into(),
            OutputMode::Tool,
        )
    }

    /// A client for Google Gemini.
    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::gemini_compatible(api_key, "https://generativelanguage.googleapis.com/v1beta")
    }

    /// A client for any Gemini-compatible API.
    pub fn gemini_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new(
            Wire::Gemini,
            api_key.into(),
            base_url.into(),
            OutputMode::Schema,
        )
    }

    /// The default model. A fallback client uses its own.
    pub fn with_model(self, model: impl Into<String>) -> Self {
        let model = model.into();
        self.with(|c| c.model = Some(model))
    }

    /// The default system prompt.
    pub fn with_system(self, system: impl Into<String>) -> Self {
        let system = system.into();
        self.with(|c| c.system = Some(system))
    }

    /// The default temperature. OpenAI-family clients send 0.0 unless set;
    /// never sent to Anthropic, whose current models reject a non-default
    /// value.
    pub fn with_temperature(self, t: f64) -> Self {
        self.with(|c| c.temperature = Some(t))
    }

    /// The default output limit: `max_completion_tokens` on OpenAI,
    /// `max_tokens` on compatible APIs and Anthropic, `max_output_tokens` on
    /// the Responses API, `maxOutputTokens` on Gemini. Unset, OpenAI-family
    /// requests carry none and Anthropic and Gemini requests carry 16384.
    pub fn with_max_tokens(self, n: u32) -> Self {
        self.with(|c| c.max_tokens = Some(n))
    }

    /// The default reasoning level; see [`Reasoning`].
    pub fn with_reasoning(self, r: Reasoning) -> Self {
        self.with(|c| c.reasoning = Some(r))
    }

    /// How the schema reaches the model; overrides the default chosen from
    /// the provider and host.
    pub fn with_output_mode(self, mode: OutputMode) -> Self {
        self.with(|c| c.mode = mode)
    }

    /// Use the Responses API instead of Chat Completions. Only for OpenAI
    /// clients; no effect on others.
    pub fn with_api(self, api: OpenAiApi) -> Self {
        self.with(|c| {
            c.provider.wire = match (&c.provider.wire, api) {
                (Wire::OpenAiChat(_) | Wire::OpenAiResponses, OpenAiApi::Responses) => {
                    Wire::OpenAiResponses
                }
                (Wire::OpenAiResponses, OpenAiApi::ChatCompletions) => {
                    Wire::OpenAiChat(Host::from_base_url(&c.provider.base_url))
                }
                (w, _) => w.clone(),
            }
        })
    }

    /// HTTP-level retries (429, 5xx, 529, connection errors). On by default:
    /// [`RetryPolicy::default`]. [`RetryPolicy::none`] turns them off.
    pub fn with_retry(self, p: RetryPolicy) -> Self {
        self.with(|c| c.retry = p)
    }

    /// How many answers to ask for before giving up on parse or validation
    /// errors. Default 3. Each retry shows the model its previous answer and
    /// the error.
    pub fn with_max_attempts(self, n: u32) -> Self {
        self.with(|c| c.max_attempts = n.max(1))
    }

    /// The overall timeout of one extraction: every attempt, retry wait and
    /// fallback. Default 60 s.
    pub fn with_timeout(self, d: Duration) -> Self {
        self.with(|c| c.timeout = d)
    }

    /// A provider to try when this one fails (after its own retries). Tried
    /// in order, each with its own model, output mode and transport.
    pub fn with_fallback(self, fallback: Client) -> Self {
        self.with(|c| c.fallbacks.push(fallback))
    }

    /// Send requests through `t` instead of the default reqwest transport.
    pub fn with_transport(self, t: impl Transport + 'static) -> Self {
        let t: Arc<dyn Transport> = Arc::new(t);
        self.with(|c| c.transport = Some(t))
    }

    /// Extract a `T` from the prompt.
    ///
    /// ```rust,no_run
    /// use instructors::prelude::*;
    ///
    /// #[derive(Deserialize, Schema)]
    /// struct Contact { name: String }
    ///
    /// # async fn run() -> instructors::Result<()> {
    /// let client = Client::openai("sk-...");
    /// let contact: Contact = client.extract("John Doe, john@example.com").await?.value;
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract<T: DeserializeOwned + Schema>(
        &self,
        prompt: impl Into<String>,
    ) -> Extract<'_, T> {
        Extract::new(self, prompt.into(), entry_for::<T>(false), false)
    }

    /// Extract a `T` that derives `schemars::JsonSchema` (feature
    /// `schemars`), for types that cannot switch to [`Schema`] yet. Field
    /// order follows schemars: alphabetical unless `schemars/preserve_order`
    /// is enabled.
    #[cfg(feature = "schemars")]
    pub fn extract_schemars<T>(&self, prompt: impl Into<String>) -> Extract<'_, T>
    where
        T: DeserializeOwned + schemars::JsonSchema + 'static,
    {
        Extract::new(
            self,
            prompt.into(),
            crate::schema::schemars::entry_for::<T>(false),
            false,
        )
    }

    /// Extract a list of `T`. The schema sent is `{"items": [T, ...]}`,
    /// because strict structured output needs an object at the root.
    pub fn extract_many<T: DeserializeOwned + Schema>(
        &self,
        prompt: impl Into<String>,
    ) -> Extract<'_, Vec<T>> {
        Extract::new(self, prompt.into(), entry_for::<T>(true), true)
    }

    /// Extract a `T` from each prompt, a few at a time.
    pub fn extract_batch<T: DeserializeOwned + Schema>(
        &self,
        prompts: impl IntoIterator<Item = String>,
    ) -> Batch<'_, T> {
        Batch::new(self, prompts.into_iter().collect())
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // the api key stays out of logs
        f.debug_struct("Client")
            .field("provider", &self.inner.provider.name())
            .field("base_url", &self.inner.provider.base_url)
            .field("model", &self.inner.model)
            .field("mode", &self.inner.mode)
            .field("fallbacks", &self.inner.fallbacks.len())
            .finish_non_exhaustive()
    }
}
