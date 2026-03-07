mod anthropic;
mod openai;

pub(crate) use anthropic::send_anthropic;
pub(crate) use openai::send_openai;

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
}

#[derive(Clone, Debug)]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
}

impl ProviderKind {
    pub(crate) fn default_model(&self) -> &str {
        match self {
            Self::OpenAi { .. } => "gpt-4o",
            Self::Anthropic { .. } => "claude-sonnet-4-20250514",
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
                )
                .await
            }
            Self::Anthropic { api_key, base_url } => {
                send_anthropic(
                    http, base_url, api_key, model, system, messages, schema, max_tokens,
                )
                .await
            }
        }
    }
}
