use crate::error::{Error, Result};
use crate::schema;
use schemars::Schema;
use serde::{Deserialize, Serialize};

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

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    response_format: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[allow(clippy::too_many_arguments)]
async fn send_openai(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    system: Option<&str>,
    messages: &[Message],
    schema: &Schema,
    schema_name: &str,
    temperature: Option<f64>,
) -> Result<RawResponse> {
    let mut oai_messages = Vec::with_capacity(messages.len() + 1);

    let sys = system.unwrap_or(
        "Extract the requested information from the given text. \
         Return valid JSON matching the provided schema.",
    );
    oai_messages.push(OpenAiMessage {
        role: "system".into(),
        content: sys.into(),
    });

    for msg in messages {
        oai_messages.push(OpenAiMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
        });
    }

    let response_format = schema::wrap_for_openai(schema, schema_name);

    let body = OpenAiRequest {
        model: model.into(),
        messages: oai_messages,
        response_format,
        temperature,
    };

    let resp = http
        .post(format!("{base_url}/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(Error::Api {
            status: status.as_u16(),
            message: text,
        });
    }

    let data: OpenAiResponse = resp.json().await?;
    let choice = data
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Other("no choices in response".into()))?;
    let usage = data.usage.unwrap_or(OpenAiUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
    });

    Ok(RawResponse {
        content: choice.message.content,
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
    })
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    tools: Vec<AnthropicTool>,
    tool_choice: AnthropicToolChoice,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Serialize)]
struct AnthropicToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
    name: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[allow(clippy::too_many_arguments)]
async fn send_anthropic(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    system: Option<&str>,
    messages: &[Message],
    schema: &Schema,
    max_tokens: u32,
) -> Result<RawResponse> {
    let ant_messages: Vec<AnthropicMessage> = messages
        .iter()
        .map(|m| AnthropicMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    let sys_text = system.unwrap_or("Extract the requested information from the given text.");

    let input_schema = schema::clean_for_anthropic(schema);

    let body = AnthropicRequest {
        model: model.into(),
        max_tokens,
        messages: ant_messages,
        system: Some(sys_text.into()),
        tools: vec![AnthropicTool {
            name: "extract".into(),
            description: "Extract structured data from the input".into(),
            input_schema,
        }],
        tool_choice: AnthropicToolChoice {
            choice_type: "tool".into(),
            name: "extract".into(),
        },
    };

    let resp = http
        .post(format!("{base_url}/messages"))
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(Error::Api {
            status: status.as_u16(),
            message: text,
        });
    }

    let data: AnthropicResponse = resp.json().await?;
    let usage = data.usage.unwrap_or(AnthropicUsage {
        input_tokens: 0,
        output_tokens: 0,
    });

    let tool_block = data
        .content
        .into_iter()
        .find(|b| b.block_type == "tool_use")
        .ok_or_else(|| Error::Other("no tool_use block in response".into()))?;

    let input = tool_block.input.unwrap_or(serde_json::Value::Null);
    let content = serde_json::to_string(&input)?;

    Ok(RawResponse {
        content,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
    })
}
