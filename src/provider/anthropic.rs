use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::{DeltaSink, ImageInput, Message, RawResponse, RequestSpec, Stop, http};
use crate::error::{Error, Result};
use crate::reasoning::{Effort, Reasoning};
use crate::schema::Dialect;

const DEFAULT_MAX_TOKENS: u32 = 16384;
const SYSTEM: &str = "Extract the requested information from the given text.";
const TOOL_HINT: &str = " Answer by calling the `extract` tool.";

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<AntMessage<'a>>,
    system: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Tool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct OutputConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<OutputFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<&'static str>,
}

#[derive(Serialize)]
struct OutputFormat<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    schema: &'a RawValue,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Thinking {
    Adaptive,
    Enabled { budget_tokens: u32 },
    Disabled,
}

#[derive(Serialize)]
struct AntMessage<'a> {
    role: &'a str,
    content: AntContent<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AntContent<'a> {
    Text(&'a str),
    Blocks(Vec<AntBlock<'a>>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AntBlock<'a> {
    Text { text: &'a str },
    Image { source: ImageSource<'a> },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ImageSource<'a> {
    Base64 { media_type: &'a str, data: &'a str },
    Url { url: &'a str },
}

#[derive(Serialize)]
struct Tool<'a> {
    name: &'static str,
    description: &'static str,
    input_schema: &'a RawValue,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ToolChoice {
    Tool { name: &'static str },
    Auto,
}

#[derive(Deserialize)]
struct Response<'a> {
    #[serde(borrow, default)]
    content: Vec<Block<'a>>,
    stop_reason: Option<String>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct Block<'a> {
    #[serde(rename = "type")]
    kind: String,
    #[serde(borrow, default)]
    input: Option<&'a RawValue>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize, Default)]
struct UsageInfo {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    kind: String,
    delta: Option<StreamDelta>,
    usage: Option<UsageInfo>,
    message: Option<StreamMessage>,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(rename = "type")]
    kind: Option<String>,
    partial_json: Option<String>,
    text: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamMessage {
    usage: Option<UsageInfo>,
}

fn build_content(msg: &Message) -> AntContent<'_> {
    if msg.images.is_empty() {
        return AntContent::Text(&msg.content);
    }
    // images first, then text (anthropic convention)
    let mut blocks: Vec<AntBlock<'_>> = msg
        .images
        .iter()
        .map(|img| AntBlock::Image {
            source: match img {
                ImageInput::Url(url) => ImageSource::Url { url },
                ImageInput::Base64 { media_type, data } => ImageSource::Base64 { media_type, data },
            },
        })
        .collect();
    blocks.push(AntBlock::Text { text: &msg.content });
    AntContent::Blocks(blocks)
}

fn thinking_for(r: Option<Reasoning>) -> (Option<Thinking>, Option<&'static str>) {
    match r {
        None => (None, None),
        Some(Reasoning::Effort(e)) => {
            let effort = match e {
                Effort::Minimal => "low",
                e => e.as_str(),
            };
            (Some(Thinking::Adaptive), Some(effort))
        }
        Some(Reasoning::Budget(n)) => (Some(Thinking::Enabled { budget_tokens: n }), None),
        Some(Reasoning::Off) => (Some(Thinking::Disabled), None),
    }
}

pub(crate) async fn send(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    structured_output: bool,
    spec: &RequestSpec<'_>,
    on_delta: DeltaSink<'_>,
) -> Result<RawResponse> {
    // structured output cannot express recursive schemas; a tool can.
    // forced tool choice is rejected by the newest models and whenever
    // thinking is on, so the tool path asks for the call instead
    let use_output = structured_output && !spec.schema.recursive();
    let forced = !structured_output && spec.reasoning.is_none();
    let (thinking, effort) = thinking_for(spec.reasoning);

    let mut system = spec.system.unwrap_or(SYSTEM).to_string();
    let (tools, tool_choice, format) = if use_output {
        let format = OutputFormat {
            kind: "json_schema",
            schema: spec.schema.get(Dialect::AnthropicOutput),
        };
        (Vec::new(), None, Some(format))
    } else {
        let tool = Tool {
            name: "extract",
            description: "Extract structured data from the input",
            input_schema: spec.schema.get(Dialect::AnthropicTool),
        };
        let choice = if forced {
            ToolChoice::Tool { name: "extract" }
        } else {
            system.push_str(TOOL_HINT);
            ToolChoice::Auto
        };
        (vec![tool], Some(choice), None)
    };
    let output_config =
        (format.is_some() || effort.is_some()).then_some(OutputConfig { format, effort });

    let body = Request {
        model: spec.model,
        max_tokens: spec.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages: spec
            .messages
            .iter()
            .map(|m| AntMessage {
                role: &m.role,
                content: build_content(m),
            })
            .collect(),
        system: &system,
        tools,
        tool_choice,
        output_config,
        thinking,
        stream: on_delta.is_some(),
    };

    let req = http
        .post(format!("{base_url}/messages"))
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01");
    let resp = http::post(req, &body).await?;

    let mode = match (use_output, forced) {
        (true, _) => Mode::Output,
        (false, true) => Mode::ForcedTool,
        (false, false) => Mode::AutoTool,
    };
    match on_delta {
        Some(sink) => read_stream(resp, use_output, sink).await,
        None => read_response(resp, mode).await,
    }
}

fn finish(content: String, usage: UsageInfo, stop_reason: Option<&str>) -> Result<RawResponse> {
    let stop = match stop_reason {
        Some("refusal") => {
            let text = if content.is_empty() {
                "stop_reason: refusal"
            } else {
                &content
            };
            return Err(Error::refused("anthropic", text));
        }
        Some(r @ ("max_tokens" | "model_context_window_exceeded")) => {
            Stop::Truncated(format!("stop_reason: {r}"))
        }
        _ => Stop::Done,
    };
    Ok(RawResponse {
        content,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        stop,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Output,
    ForcedTool,
    AutoTool,
}

async fn read_response(resp: reqwest::Response, mode: Mode) -> Result<RawResponse> {
    let bytes = resp.bytes().await?;
    let data: Response<'_> = serde_json::from_slice(&bytes)?;
    let mut texts = data
        .content
        .iter()
        .filter(|b| b.kind == "text")
        .filter_map(|b| b.text.as_deref())
        .peekable();
    let has_text = texts.peek().is_some();
    let text: String = texts.collect();
    let tool_input = data
        .content
        .iter()
        .find_map(|b| b.input.filter(|_| b.kind == "tool_use"));
    let truncated = matches!(
        data.stop_reason.as_deref(),
        Some("max_tokens" | "model_context_window_exceeded")
    );
    let content = match (mode, tool_input) {
        _ if data.stop_reason.as_deref() == Some("refusal") => text,
        (Mode::Output, _) if !has_text && !truncated => {
            return Err(Error::Other("no text block in response".into()));
        }
        (Mode::Output, _) => text,
        (_, Some(input)) => input.get().to_string(),
        (Mode::ForcedTool, None) if !truncated => {
            return Err(Error::Other("no tool_use block in response".into()));
        }
        // the model answered in prose instead of calling the tool; kept so
        // the retry message can quote it
        (_, None) => text,
    };
    finish(
        content,
        data.usage.unwrap_or_default(),
        data.stop_reason.as_deref(),
    )
}

async fn read_stream(
    resp: reqwest::Response,
    use_output: bool,
    sink: &mut (dyn FnMut(&str) + Send),
) -> Result<RawResponse> {
    let mut content = String::new();
    let mut usage = UsageInfo::default();
    let mut stop_reason = None;
    http::read_sse(resp, |data| {
        let Ok(event) = serde_json::from_str::<StreamEvent>(data) else {
            return;
        };
        match event.kind.as_str() {
            "message_start" => {
                if let Some(u) = event.message.and_then(|m| m.usage) {
                    usage.input_tokens = u.input_tokens;
                }
            }
            // structured output streams `text_delta`, a tool `input_json_delta`
            "content_block_delta" => {
                let Some(delta) = event.delta else { return };
                let piece = match (use_output, delta.kind.as_deref()) {
                    (true, Some("text_delta")) => delta.text,
                    (false, Some("input_json_delta")) => delta.partial_json,
                    _ => None,
                };
                if let Some(p) = piece {
                    sink(&p);
                    content.push_str(&p);
                }
            }
            "message_delta" => {
                if let Some(u) = event.usage {
                    usage.output_tokens = u.output_tokens;
                }
                if let Some(r) = event.delta.and_then(|d| d.stop_reason) {
                    stop_reason = Some(r);
                }
            }
            _ => {}
        }
    })
    .await?;
    finish(content, usage, stop_reason.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_mapping() {
        assert_eq!(thinking_for(None), (None, None));
        assert_eq!(
            thinking_for(Some(Reasoning::Effort(Effort::Minimal))),
            (Some(Thinking::Adaptive), Some("low"))
        );
        assert_eq!(
            thinking_for(Some(Reasoning::Effort(Effort::Max))),
            (Some(Thinking::Adaptive), Some("max"))
        );
        assert_eq!(
            thinking_for(Some(Reasoning::Budget(2048))),
            (
                Some(Thinking::Enabled {
                    budget_tokens: 2048
                }),
                None
            )
        );
        assert_eq!(
            thinking_for(Some(Reasoning::Off)),
            (Some(Thinking::Disabled), None)
        );
        let v = serde_json::to_value(Thinking::Enabled { budget_tokens: 9 }).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"type": "enabled", "budget_tokens": 9})
        );
    }
}
