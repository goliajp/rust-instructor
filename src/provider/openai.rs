use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::{DEFAULT_SYSTEM, DeltaSink, ImageInput, Message, RawResponse, RequestSpec, Stop, http};
use crate::error::{Error, Result};
use crate::reasoning::Reasoning;
use crate::schema::Dialect;

/// Which OpenAI-compatible API is on the other end. It decides the name of
/// the output-limit field and how [`Reasoning`] is spelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OaiDialect {
    OpenAi,
    Generic,
    DeepSeek,
    Qwen,
    Kimi,
    Glm,
}

impl OaiDialect {
    /// Recognize the hosted APIs by their base URL host.
    pub(crate) fn from_base_url(url: &str) -> Self {
        let host = url
            .split("://")
            .nth(1)
            .unwrap_or(url)
            .split(['/', ':'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let is = |domain: &str| host == domain || host.ends_with(&format!(".{domain}"));
        if is("api.openai.com") {
            Self::OpenAi
        } else if is("deepseek.com") {
            Self::DeepSeek
        } else if is("aliyuncs.com") {
            Self::Qwen
        } else if is("moonshot.cn") || is("moonshot.ai") || is("kimi.ai") || is("kimi.com") {
            Self::Kimi
        } else if is("bigmodel.cn") || is("z.ai") {
            Self::Glm
        } else {
            Self::Generic
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Generic => "openai-compatible",
            Self::DeepSeek => "deepseek",
            Self::Qwen => "qwen",
            Self::Kimi => "kimi",
            Self::Glm => "glm",
        }
    }

    pub(crate) fn default_model(self) -> &'static str {
        "gpt-5.6-terra"
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: Vec<OaiMessage<'a>>,
    response_format: &'a RawValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(flatten)]
    reasoning: ReasoningFields,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize, Default, Debug, PartialEq)]
struct ReasoningFields {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingSwitch>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

#[derive(Serialize, Debug, PartialEq)]
struct ThinkingSwitch {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct OaiMessage<'a> {
    role: &'a str,
    content: OaiContent<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum OaiContent<'a> {
    Text(&'a str),
    Parts(Vec<OaiPart<'a>>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OaiPart<'a> {
    Text { text: &'a str },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    choices: Vec<Choice>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct Choice {
    message: RespMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct RespMessage {
    content: Option<RespContent>,
    refusal: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RespContent {
    Text(String),
    Parts(Vec<RespPart>),
}

#[derive(Deserialize)]
struct RespPart {
    text: Option<String>,
}

#[derive(Deserialize, Default)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: Option<StreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
    refusal: Option<String>,
}

fn build_content(msg: &Message) -> OaiContent<'_> {
    if msg.images.is_empty() {
        return OaiContent::Text(&msg.content);
    }
    let mut parts = Vec::with_capacity(1 + msg.images.len());
    parts.push(OaiPart::Text { text: &msg.content });
    for img in &msg.images {
        let url = match img {
            ImageInput::Url(u) => u.clone(),
            ImageInput::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
        };
        parts.push(OaiPart::ImageUrl {
            image_url: ImageUrl { url },
        });
    }
    OaiContent::Parts(parts)
}

fn reasoning_fields(dialect: OaiDialect, r: Option<Reasoning>) -> Result<ReasoningFields> {
    let mut f = ReasoningFields::default();
    let Some(r) = r else {
        return Ok(f);
    };
    let switch = |on: bool| ThinkingSwitch {
        kind: if on { "enabled" } else { "disabled" },
    };
    let no_budget = || {
        Err(Error::Other(format!(
            "{} has no thinking budget parameter; use Reasoning::Effort",
            dialect.name()
        )))
    };
    match (dialect, r) {
        (OaiDialect::OpenAi | OaiDialect::Generic, Reasoning::Effort(e)) => {
            f.reasoning_effort = Some(e.as_str())
        }
        (OaiDialect::OpenAi | OaiDialect::Generic, Reasoning::Off) => {
            f.reasoning_effort = Some("none")
        }
        (OaiDialect::Qwen, Reasoning::Effort(_)) => f.enable_thinking = Some(true),
        (OaiDialect::Qwen, Reasoning::Budget(n)) => {
            f.enable_thinking = Some(true);
            f.thinking_budget = Some(n);
        }
        (OaiDialect::Qwen, Reasoning::Off) => f.enable_thinking = Some(false),
        (_, Reasoning::Effort(e)) => {
            f.thinking = Some(switch(true));
            f.reasoning_effort = Some(e.coarse());
        }
        (_, Reasoning::Off) => f.thinking = Some(switch(false)),
        (_, Reasoning::Budget(_)) => return no_budget(),
    }
    Ok(f)
}

pub(crate) async fn send(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    dialect: OaiDialect,
    spec: &RequestSpec<'_>,
    on_delta: DeltaSink<'_>,
) -> Result<RawResponse> {
    let streaming = on_delta.is_some();
    let mut messages = Vec::with_capacity(spec.messages.len() + 1);
    messages.push(OaiMessage {
        role: "system",
        content: OaiContent::Text(spec.system.unwrap_or(DEFAULT_SYSTEM)),
    });
    messages.extend(spec.messages.iter().map(|m| OaiMessage {
        role: &m.role,
        content: build_content(m),
    }));

    // openai deprecated `max_tokens` in favour of `max_completion_tokens`;
    // the compatible APIs still read `max_tokens`
    let (max_tokens, max_completion_tokens) = match dialect {
        OaiDialect::OpenAi => (None, spec.max_tokens),
        _ => (spec.max_tokens, None),
    };

    let body = Request {
        model: spec.model,
        messages,
        response_format: spec.schema.get(Dialect::OpenAi),
        temperature: spec.temperature,
        max_tokens,
        max_completion_tokens,
        reasoning: reasoning_fields(dialect, spec.reasoning)?,
        stream: streaming,
        stream_options: streaming.then_some(StreamOptions {
            include_usage: true,
        }),
    };

    let req = http
        .post(format!("{base_url}/chat/completions"))
        .bearer_auth(api_key);
    let resp = http::post(req, &body).await?;

    match on_delta {
        Some(sink) => read_stream(resp, dialect, sink).await,
        None => read_response(resp, dialect).await,
    }
}

fn finish(
    dialect: OaiDialect,
    content: String,
    usage: UsageInfo,
    refusal: Option<String>,
    finish_reason: Option<&str>,
) -> Result<RawResponse> {
    if let Some(text) = refusal.filter(|t| !t.is_empty()) {
        return Err(Error::refused(dialect.name(), &text));
    }
    let stop = match finish_reason {
        Some("length") => Stop::Truncated("finish_reason: length".into()),
        Some("content_filter") if content.is_empty() => {
            return Err(Error::refused(dialect.name(), "content_filter"));
        }
        _ => Stop::Done,
    };
    Ok(RawResponse {
        content,
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        stop,
    })
}

async fn read_response(resp: reqwest::Response, dialect: OaiDialect) -> Result<RawResponse> {
    let data: Response = http::json(resp).await?;
    let choice = data
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Other("no choices in response".into()))?;
    let content = match choice.message.content {
        Some(RespContent::Text(t)) => t,
        Some(RespContent::Parts(parts)) => parts.into_iter().filter_map(|p| p.text).collect(),
        None => String::new(),
    };
    finish(
        dialect,
        content,
        data.usage.unwrap_or_default(),
        choice.message.refusal,
        choice.finish_reason.as_deref(),
    )
}

async fn read_stream(
    resp: reqwest::Response,
    dialect: OaiDialect,
    sink: &mut (dyn FnMut(&str) + Send),
) -> Result<RawResponse> {
    let mut content = String::new();
    let mut refusal = String::new();
    let mut usage = UsageInfo::default();
    let mut finish_reason = None;
    http::read_sse(resp, |data| {
        let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) else {
            return;
        };
        if let Some(u) = chunk.usage {
            usage = u;
        }
        for choice in chunk.choices {
            if let Some(r) = choice.finish_reason {
                finish_reason = Some(r);
            }
            let Some(delta) = choice.delta else { continue };
            if let Some(r) = delta.refusal {
                refusal.push_str(&r);
            }
            if let Some(text) = delta.content {
                sink(&text);
                content.push_str(&text);
            }
        }
    })
    .await?;
    finish(
        dialect,
        content,
        usage,
        Some(refusal),
        finish_reason.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning::Effort;

    #[test]
    fn dialect_from_url() {
        let d = OaiDialect::from_base_url;
        assert_eq!(d("https://api.openai.com/v1"), OaiDialect::OpenAi);
        assert_eq!(d("https://api.deepseek.com/v1"), OaiDialect::DeepSeek);
        assert_eq!(
            d("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            OaiDialect::Qwen
        );
        assert_eq!(d("https://api.moonshot.ai/v1"), OaiDialect::Kimi);
        assert_eq!(d("https://open.bigmodel.cn/api/paas/v4"), OaiDialect::Glm);
        assert_eq!(d("https://api.z.ai/api/paas/v4"), OaiDialect::Glm);
        assert_eq!(d("http://127.0.0.1:8080/v1"), OaiDialect::Generic);
        assert_eq!(d("https://notdeepseek.com/v1"), OaiDialect::Generic);
        assert_eq!(d("https://api.together.xyz/v1"), OaiDialect::Generic);
    }

    #[test]
    fn reasoning_mapping() {
        let f = |d, r| serde_json::to_value(reasoning_fields(d, Some(r)).unwrap()).unwrap();
        let hi = Reasoning::Effort(Effort::High);
        assert_eq!(
            f(OaiDialect::OpenAi, hi),
            serde_json::json!({"reasoning_effort": "high"})
        );
        assert_eq!(
            f(OaiDialect::OpenAi, Reasoning::Off),
            serde_json::json!({"reasoning_effort": "none"})
        );
        assert_eq!(
            f(OaiDialect::DeepSeek, Reasoning::Effort(Effort::Medium)),
            serde_json::json!({"thinking": {"type": "enabled"}, "reasoning_effort": "high"})
        );
        assert_eq!(
            f(OaiDialect::Glm, Reasoning::Off),
            serde_json::json!({"thinking": {"type": "disabled"}})
        );
        assert_eq!(
            f(OaiDialect::Qwen, Reasoning::Budget(512)),
            serde_json::json!({"enable_thinking": true, "thinking_budget": 512})
        );
        assert_eq!(
            f(OaiDialect::Qwen, Reasoning::Off),
            serde_json::json!({"enable_thinking": false})
        );
        assert!(reasoning_fields(OaiDialect::OpenAi, Some(Reasoning::Budget(1))).is_err());
        assert!(reasoning_fields(OaiDialect::Kimi, Some(Reasoning::Budget(1))).is_err());
        assert_eq!(
            reasoning_fields(OaiDialect::OpenAi, None).unwrap(),
            ReasoningFields::default()
        );
    }
}
