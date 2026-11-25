mod anthropic;
mod gemini;
mod http;
mod openai;

pub(crate) use http::shared_client;
pub(crate) use openai::OaiDialect;

use crate::error::Result;
use crate::reasoning::Reasoning;
use crate::schema::SchemaEntry;

/// Called with each content delta while a response streams in.
pub(crate) type DeltaSink<'a> = Option<&'a mut (dyn FnMut(&str) + Send)>;

/// Everything one provider request needs.
pub(crate) struct RequestSpec<'a> {
    pub model: &'a str,
    pub system: Option<&'a str>,
    pub messages: &'a [Message],
    pub schema: &'a SchemaEntry,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub reasoning: Option<Reasoning>,
}

/// Why the model stopped, when it matters to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Stop {
    Done,
    /// Cut off by the output token limit; carries the provider's reason.
    Truncated(String),
}

pub(crate) struct RawResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub stop: Stop,
}

#[derive(Clone)]
pub(crate) enum ProviderKind {
    OpenAi {
        api_key: String,
        base_url: String,
        dialect: OaiDialect,
    },
    Anthropic {
        api_key: String,
        base_url: String,
        /// Send the schema as `output_config.format` instead of as a tool.
        structured_output: bool,
    },
    Gemini {
        api_key: String,
        base_url: String,
    },
}

/// Image input for vision-capable models.
#[derive(Clone, Debug)]
pub enum ImageInput {
    /// Image from a URL (OpenAI and Anthropic both support this).
    Url(String),
    /// Base64-encoded image data with its MIME type (e.g. `"image/png"`).
    Base64 { media_type: String, data: String },
}

/// One message of conversation history.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Optional images attached to this message.
    pub images: Vec<ImageInput>,
}

impl Message {
    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    /// Create a user message with images.
    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageInput>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            images,
        }
    }
}

pub(crate) const DEFAULT_SYSTEM: &str = "Extract the requested information from the given text. \
     Return valid JSON matching the provided schema.";

impl ProviderKind {
    /// The model used when the caller sets none.
    ///
    /// Each is its provider's current balanced tier. Model generations turn
    /// over roughly twice a year and retired ids stop resolving, so these are
    /// expected to move with minor releases.
    pub(crate) fn default_model(&self) -> &str {
        match self {
            Self::OpenAi { dialect, .. } => dialect.default_model(),
            Self::Anthropic { .. } => "claude-sonnet-5",
            Self::Gemini { .. } => "gemini-3.6-flash",
        }
    }

    /// Provider name for error messages and tracing.
    pub(crate) fn kind_name(&self) -> &'static str {
        match self {
            Self::OpenAi { dialect, .. } => dialect.name(),
            Self::Anthropic { .. } => "anthropic",
            Self::Gemini { .. } => "gemini",
        }
    }

    pub(crate) async fn send(
        &self,
        http: &reqwest::Client,
        spec: &RequestSpec<'_>,
        on_delta: DeltaSink<'_>,
    ) -> Result<RawResponse> {
        match self {
            Self::OpenAi {
                api_key,
                base_url,
                dialect,
            } => openai::send(http, base_url, api_key, *dialect, spec, on_delta).await,
            Self::Anthropic {
                api_key,
                base_url,
                structured_output,
            } => anthropic::send(http, base_url, api_key, *structured_output, spec, on_delta).await,
            Self::Gemini { api_key, base_url } => {
                gemini::send(http, base_url, api_key, spec, on_delta).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "hello");
        assert!(msg.images.is_empty());

        let msg = Message::assistant("hi there");
        assert_eq!(msg.role, "assistant");

        let msg = Message::user_with_images(
            "describe this",
            vec![ImageInput::Url("https://example.com/img.png".into())],
        );
        assert_eq!(msg.images.len(), 1);
    }

    #[test]
    fn default_models() {
        let openai = ProviderKind::OpenAi {
            api_key: "k".into(),
            base_url: "u".into(),
            dialect: OaiDialect::OpenAi,
        };
        assert_eq!(openai.default_model(), "gpt-5.6-terra");
        assert_eq!(openai.kind_name(), "openai");

        let anthropic = ProviderKind::Anthropic {
            api_key: "k".into(),
            base_url: "u".into(),
            structured_output: true,
        };
        assert_eq!(anthropic.default_model(), "claude-sonnet-5");
        assert_eq!(anthropic.kind_name(), "anthropic");

        let gemini = ProviderKind::Gemini {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(gemini.default_model(), "gemini-3.6-flash");
        assert_eq!(gemini.kind_name(), "gemini");
    }
}
