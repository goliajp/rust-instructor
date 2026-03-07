use schemars::Schema;
use serde::{Deserialize, Serialize};

use super::{Message, RawResponse};
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
}

#[derive(Serialize)]
struct AntMessage {
    role: String,
    content: String,
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
) -> Result<RawResponse> {
    let ant_messages: Vec<AntMessage> = messages
        .iter()
        .map(|m| AntMessage {
            role: m.role.clone(),
            content: m.content.clone(),
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
