use schemars::Schema;
use serde::{Deserialize, Serialize};

use super::{ImageInput, Message, RawResponse, StreamCallback};
use crate::error::{Error, Result};
use crate::schema;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    contents: Vec<GemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GemContent>,
    generation_config: GenerationConfig,
}

#[derive(Serialize)]
struct GemContent {
    role: String,
    parts: Vec<GemPart>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum GemPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileData {
    file_uri: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    response_mime_type: String,
    response_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct Response {
    candidates: Option<Vec<Candidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<CandidateContent>,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Option<Vec<ResponsePart>>,
}

#[derive(Deserialize)]
struct ResponsePart {
    text: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
}

fn build_parts(msg: &Message) -> Vec<GemPart> {
    let mut parts = Vec::with_capacity(1 + msg.images.len());

    for img in &msg.images {
        match img {
            ImageInput::Base64 { media_type, data } => {
                parts.push(GemPart::InlineData {
                    inline_data: InlineData {
                        mime_type: media_type.clone(),
                        data: data.clone(),
                    },
                });
            }
            ImageInput::Url(url) => {
                parts.push(GemPart::FileData {
                    file_data: FileData {
                        file_uri: url.clone(),
                    },
                });
            }
        }
    }

    parts.push(GemPart::Text {
        text: msg.content.clone(),
    });

    parts
}

fn gemini_role(role: &str) -> &str {
    match role {
        "assistant" => "model",
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_gemini(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    system: Option<&str>,
    messages: &[Message],
    schema: &Schema,
    temperature: Option<f64>,
    max_tokens: u32,
    on_stream: StreamCallback<'_>,
) -> Result<RawResponse> {
    let streaming = on_stream.is_some();

    let contents: Vec<GemContent> = messages
        .iter()
        .map(|m| GemContent {
            role: gemini_role(&m.role).into(),
            parts: build_parts(m),
        })
        .collect();

    let sys = system
        .unwrap_or("Extract the requested information from the given text. Return valid JSON matching the provided schema.");

    let system_instruction = GemContent {
        role: "user".into(),
        parts: vec![GemPart::Text { text: sys.into() }],
    };

    let response_schema = schema::clean_for_gemini(schema);

    let body = Request {
        contents,
        system_instruction: Some(system_instruction),
        generation_config: GenerationConfig {
            temperature,
            max_output_tokens: Some(max_tokens),
            response_mime_type: "application/json".into(),
            response_schema,
        },
    };

    let method = if streaming {
        "streamGenerateContent"
    } else {
        "generateContent"
    };

    let url = if streaming {
        format!("{base_url}/models/{model}:{method}?alt=sse&key={api_key}")
    } else {
        format!("{base_url}/models/{model}:{method}?key={api_key}")
    };

    let resp = http
        .post(&url)
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
    let usage = data.usage_metadata.unwrap_or(UsageMetadata {
        prompt_token_count: 0,
        candidates_token_count: 0,
    });

    let text = data
        .candidates
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.content)
        .and_then(|c| c.parts)
        .and_then(|p| p.into_iter().next())
        .and_then(|p| p.text)
        .ok_or_else(|| Error::Other("no content in gemini response".into()))?;

    Ok(RawResponse {
        content: text,
        input_tokens: usage.prompt_token_count,
        output_tokens: usage.candidates_token_count,
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

        while let Some(pos) = buffer.find('\n') {
            let line: String = buffer[..pos].trim_end_matches('\r').into();
            buffer.drain(..pos + 1);

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ")
                && let Ok(event) = serde_json::from_str::<Response>(data)
            {
                if let Some(usage) = event.usage_metadata {
                    input_tokens = usage.prompt_token_count;
                    output_tokens = usage.candidates_token_count;
                }

                if let Some(text) = event
                    .candidates
                    .and_then(|c| c.into_iter().next())
                    .and_then(|c| c.content)
                    .and_then(|c| c.parts)
                    .and_then(|p| p.into_iter().next())
                    .and_then(|p| p.text)
                {
                    callback(&text);
                    accumulated.push_str(&text);
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
