use schemars::Schema;
use serde::{Deserialize, Serialize};

use super::{Message, RawResponse};
use crate::error::{Error, Result};
use crate::schema;

#[derive(Serialize)]
struct Request {
    model: String,
    messages: Vec<OaiMessage>,
    response_format: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct OaiMessage {
    role: String,
    content: String,
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
) -> Result<RawResponse> {
    let mut oai_messages = Vec::with_capacity(messages.len() + 1);

    let sys = system.unwrap_or(
        "Extract the requested information from the given text. \
         Return valid JSON matching the provided schema.",
    );
    oai_messages.push(OaiMessage {
        role: "system".into(),
        content: sys.into(),
    });

    for msg in messages {
        oai_messages.push(OaiMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
        });
    }

    let response_format = schema::wrap_for_openai(schema, schema_name);

    let body = Request {
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

    Ok(RawResponse {
        content: choice.message.content,
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
    })
}
