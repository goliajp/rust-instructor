use schemars::Schema;
use serde::{Deserialize, Serialize};

use super::{ImageInput, Message, RawResponse, StreamCallback};
use crate::error::{Error, Result};
use crate::schema;

#[derive(Serialize)]
struct Request {
    model: String,
    max_tokens: u32,
    messages: Vec<AntMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    tools: Vec<Tool>,
    tool_choice: ToolChoice,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct AntMessage {
    role: String,
    content: AntContent,
}

// anthropic accepts either a plain string or an array of content blocks
#[derive(Serialize)]
#[serde(untagged)]
enum AntContent {
    Text(String),
    Blocks(Vec<AntContentBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AntContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AntImageSource },
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AntImageSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
    #[serde(rename = "url")]
    Url { url: String },
}

#[derive(Serialize)]
struct Tool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Serialize)]
struct ToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
    name: String,
}

#[derive(Deserialize)]
struct Response {
    content: Vec<ContentBlock>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct UsageInfo {
    input_tokens: u32,
    output_tokens: u32,
}

// streaming event types
#[derive(Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    delta: Option<StreamDelta>,
    #[serde(default)]
    usage: Option<UsageInfo>,
    #[serde(default)]
    message: Option<StreamMessage>,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(rename = "type")]
    #[serde(default)]
    delta_type: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
}

#[derive(Deserialize)]
struct StreamMessage {
    #[serde(default)]
    usage: Option<UsageInfo>,
}

fn build_content(msg: &Message) -> AntContent {
    if msg.images.is_empty() {
        return AntContent::Text(msg.content.clone());
    }

    let mut blocks = Vec::with_capacity(msg.images.len() + 1);
    // images first, then text (anthropic convention)
    for img in &msg.images {
        let source = match img {
            ImageInput::Url(u) => AntImageSource::Url { url: u.clone() },
            ImageInput::Base64 { media_type, data } => AntImageSource::Base64 {
                media_type: media_type.clone(),
                data: data.clone(),
            },
        };
        blocks.push(AntContentBlock::Image { source });
    }
    blocks.push(AntContentBlock::Text {
        text: msg.content.clone(),
    });
    AntContent::Blocks(blocks)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_anthropic(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    system: Option<&str>,
    messages: &[Message],
    schema: &Schema,
    max_tokens: u32,
    on_stream: StreamCallback<'_>,
) -> Result<RawResponse> {
    let streaming = on_stream.is_some();

    let ant_messages: Vec<AntMessage> = messages
        .iter()
        .map(|m| AntMessage {
            role: m.role.clone(),
            content: build_content(m),
        })
        .collect();

    let sys_text = system.unwrap_or("Extract the requested information from the given text.");
    let input_schema = schema::clean_for_anthropic(schema);

    let body = Request {
        model: model.into(),
        max_tokens,
        messages: ant_messages,
        system: Some(sys_text.into()),
        tools: vec![Tool {
            name: "extract".into(),
            description: "Extract structured data from the input".into(),
            input_schema,
        }],
        tool_choice: ToolChoice {
            choice_type: "tool".into(),
            name: "extract".into(),
        },
        stream: streaming,
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

    if streaming {
        read_stream(resp, on_stream.unwrap()).await
    } else {
        read_response(resp).await
    }
}

async fn read_response(resp: reqwest::Response) -> Result<RawResponse> {
    let data: Response = resp.json().await?;
    let usage = data.usage.unwrap_or(UsageInfo {
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

async fn read_stream(
    resp: reqwest::Response,
    callback: &(dyn Fn(&str) + Send + Sync),
) -> Result<RawResponse> {
    use futures::StreamExt;

    let mut accumulated = String::new();
    let mut input_tokens = 0u32;
    let mut output_tokens = 0u32;
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        // process complete SSE lines
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim_end_matches('\r').to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                    match event.event_type.as_str() {
                        "message_start" => {
                            if let Some(msg) = event.message
                                && let Some(usage) = msg.usage
                            {
                                input_tokens = usage.input_tokens;
                            }
                        }
                        "content_block_delta" => {
                            if let Some(delta) = event.delta
                                && delta.delta_type.as_deref() == Some("input_json_delta")
                                && let Some(partial) = delta.partial_json
                            {
                                callback(&partial);
                                accumulated.push_str(&partial);
                            }
                        }
                        "message_delta" => {
                            if let Some(usage) = event.usage {
                                output_tokens = usage.output_tokens;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(RawResponse {
        content: accumulated,
        input_tokens,
        output_tokens,
    })
}
