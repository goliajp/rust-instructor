mod anthropic;
mod gemini;
mod openai;

pub(crate) use anthropic::send_anthropic;
pub(crate) use gemini::send_gemini;
pub(crate) use openai::send_openai;

pub(crate) type StreamCallback<'a> = Option<&'a (dyn Fn(&str) + Send + Sync)>;

use crate::error::Result;
use schemars::Schema;

pub(crate) struct RawResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Clone)]
pub(crate) enum ProviderKind {
    OpenAi { api_key: String, base_url: String },
    Anthropic { api_key: String, base_url: String },
    Gemini { api_key: String, base_url: String },
}

/// Image input for vision-capable models.
#[derive(Clone, Debug)]
pub enum ImageInput {
    /// Image from a URL (OpenAI and Anthropic both support this).
    Url(String),
    /// Base64-encoded image data with its MIME type (e.g. `"image/png"`).
    Base64 { media_type: String, data: String },
}

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

impl ProviderKind {
    pub(crate) fn default_model(&self) -> &str {
        match self {
            Self::OpenAi { .. } => "gpt-4o",
            Self::Anthropic { .. } => "claude-sonnet-4-20250514",
            Self::Gemini { .. } => "gemini-2.5-flash",
        }
    }

    /// Returns the provider name for diagnostics and tracing.
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) fn kind_name(&self) -> &'static str {
        match self {
            Self::OpenAi { .. } => "openai",
            Self::Anthropic { .. } => "anthropic",
            Self::Gemini { .. } => "gemini",
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn send(
        &self,
        http: &reqwest::Client,
        model: &str,
        system: Option<&str>,
        messages: &[Message],
        schema: &Schema,
        schema_name: &str,
        temperature: Option<f64>,
        max_tokens: u32,
        on_stream: StreamCallback<'_>,
    ) -> Result<RawResponse> {
        match self {
            Self::OpenAi { api_key, base_url } => {
                send_openai(
                    http,
                    base_url,
                    api_key,
                    model,
                    system,
                    messages,
                    schema,
                    schema_name,
                    temperature,
                    on_stream,
                )
                .await
            }
            Self::Anthropic { api_key, base_url } => {
                send_anthropic(
                    http, base_url, api_key, model, system, messages, schema, max_tokens, on_stream,
                )
                .await
            }
            Self::Gemini { api_key, base_url } => {
                send_gemini(
                    http,
                    base_url,
                    api_key,
                    model,
                    system,
                    messages,
                    schema,
                    temperature,
                    max_tokens,
                    on_stream,
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_user() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "hello");
        assert!(msg.images.is_empty());
    }

    #[test]
    fn message_assistant() {
        let msg = Message::assistant("hi there");
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "hi there");
        assert!(msg.images.is_empty());
    }

    #[test]
    fn message_clone() {
        let msg = Message::user("test");
        let cloned = msg.clone();
        assert_eq!(cloned.role, msg.role);
        assert_eq!(cloned.content, msg.content);
    }

    #[test]
    fn message_debug() {
        let msg = Message::user("test");
        let debug = format!("{msg:?}");
        assert!(debug.contains("user"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn message_user_with_images() {
        let msg = Message::user_with_images(
            "describe this",
            vec![ImageInput::Url("https://example.com/img.png".into())],
        );
        assert_eq!(msg.role, "user");
        assert_eq!(msg.images.len(), 1);
    }

    #[test]
    fn image_input_clone_and_debug() {
        let url = ImageInput::Url("https://example.com/img.png".into());
        let cloned = url.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("Url"));

        let b64 = ImageInput::Base64 {
            media_type: "image/png".into(),
            data: "abc123".into(),
        };
        let debug = format!("{b64:?}");
        assert!(debug.contains("Base64"));
    }

    #[test]
    fn default_model_openai() {
        let provider = ProviderKind::OpenAi {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "gpt-4o");
    }

    #[test]
    fn default_model_anthropic() {
        let provider = ProviderKind::Anthropic {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn default_model_gemini() {
        let provider = ProviderKind::Gemini {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "gemini-2.5-flash");
    }

    #[test]
    fn kind_name_all() {
        let openai = ProviderKind::OpenAi {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(openai.kind_name(), "openai");

        let anthropic = ProviderKind::Anthropic {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(anthropic.kind_name(), "anthropic");

        let gemini = ProviderKind::Gemini {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(gemini.kind_name(), "gemini");
    }
}
