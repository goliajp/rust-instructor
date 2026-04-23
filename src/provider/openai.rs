use schemars::Schema;
use serde::{Deserialize, Serialize};

use super::{ImageInput, Message, RawResponse, StreamCallback};
use crate::error::{Error, Result};
use crate::schema;

#[derive(Serialize)]
struct Request {
    model: String,
    messages: Vec<OaiMessage>,
    response_format: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OaiMessage {
    role: String,
    content: OaiContent,
}

// openai accepts either a plain string or an array of content parts
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum OaiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OaiImageUrl },
}

#[derive(Serialize, Deserialize)]
struct OaiImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct Choice {
    message: OaiMessage,
}

#[derive(Deserialize)]
struct UsageInfo {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// streaming response types
#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: Option<StreamDelta>,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

fn build_content(msg: &Message) -> OaiContent {
    if msg.images.is_empty() {
        return OaiContent::Text(msg.content.clone());
    }

    let mut parts = Vec::with_capacity(1 + msg.images.len());
    parts.push(OaiContentPart::Text {
        text: msg.content.clone(),
    });
    for img in &msg.images {
        let url = match img {
            ImageInput::Url(u) => u.clone(),
            ImageInput::Base64 { media_type, data } => {
                format!("data:{media_type};base64,{data}")
            }
        };
        parts.push(OaiContentPart::ImageUrl {
            image_url: OaiImageUrl { url },
        });
    }
    OaiContent::Parts(parts)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_openai(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    system: Option<&str>,
    messages: &[Message],
    schema: &Schema,
    schema_name: &str,
    temperature: Option<f64>,
    on_stream: StreamCallback<'_>,
) -> Result<RawResponse> {
    let streaming = on_stream.is_some();

    let mut oai_messages = Vec::with_capacity(messages.len() + 1);

    let sys = system.unwrap_or(
        "Extract the requested information from the given text. \
         Return valid JSON matching the provided schema.",
    );
    oai_messages.push(OaiMessage {
        role: "system".into(),
        content: OaiContent::Text(sys.into()),
    });

    for msg in messages {
        oai_messages.push(OaiMessage {
            role: msg.role.clone(),
            content: build_content(msg),
        });
    }

    let response_format = schema::wrap_for_openai(schema, schema_name);

    let body = Request {
        model: model.into(),
        messages: oai_messages,
        response_format,
        temperature,
        stream: streaming,
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

    if streaming {
        read_stream(resp, on_stream.unwrap()).await
    } else {
        read_response(resp).await
    }
}

async fn read_response(resp: reqwest::Response) -> Result<RawResponse> {
    let data: Response = resp.json().await?;
    let choice = data
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Other("no choices in response".into()))?;
    let usage = data.usage.unwrap_or(UsageInfo {
        prompt_tokens: 0,
        completion_tokens: 0,
    });

    let content_text = match choice.message.content {
        OaiContent::Text(t) => t,
        OaiContent::Parts(parts) => parts
            .into_iter()
            .find_map(|p| match p {
                OaiContentPart::Text { text } => Some(text),
                _ => None,
            })
            .unwrap_or_default(),
    };

    Ok(RawResponse {
        content: content_text,
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
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
            let line: String = buffer[..pos].trim_end_matches('\r').into();
            buffer.drain(..pos + 1);

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    if let Some(usage) = chunk.usage {
                        input_tokens = usage.prompt_tokens;
                        output_tokens = usage.completion_tokens;
                    }
                    for choice in chunk.choices {
                        if let Some(delta) = choice.delta
                            && let Some(content) = delta.content
                        {
                            callback(&content);
                            accumulated.push_str(&content);
                        }
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
