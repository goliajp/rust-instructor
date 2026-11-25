//! Anthropic Messages.

use std::borrow::Cow;

use serde::Deserialize;

use super::{
    Answer, Arr, Fail, HttpRequest, Obj, OutputMode, Provider, Req, Stop, envelope, json_headers,
    system_with_schema, type_obj,
};
use crate::json::Raw;
use crate::message::{Part, Role};
use crate::reasoning::{Effort, Reasoning};
use crate::schema::Dialect;
use crate::usage::Usage;

const DEFAULT_MAX_TOKENS: u32 = 16384;
const SYSTEM: &str = "Extract the requested information from the given text.";
const TOOL_HINT: &str = " Answer by calling the `extract` tool.";

/// Where the answer comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Plan {
    /// `output_config.format`: text blocks.
    Output,
    /// A tool; forced unless thinking is on.
    Tool { forced: bool },
    /// The schema in the system prompt: text blocks.
    Text,
}

fn plan(r: &Req<'_>) -> Plan {
    match r.mode {
        // structured output cannot express recursive schemas; a tool can.
        // forced tool choice is rejected whenever thinking is on and by the
        // newest models, so this path asks for the call instead
        OutputMode::Schema if r.schema.recursive() => Plan::Tool { forced: false },
        OutputMode::Schema => Plan::Output,
        OutputMode::Tool => Plan::Tool {
            forced: r.reasoning.is_none(),
        },
        OutputMode::JsonObject | OutputMode::Prompt => Plan::Text,
    }
}

fn thinking(r: Option<Reasoning>) -> (Option<Reasoning>, Option<&'static str>) {
    match r {
        Some(Reasoning::Effort(e)) => {
            let effort = match e {
                Effort::Minimal => "low",
                e => e.as_str(),
            };
            (r, Some(effort))
        }
        other => (other, None),
    }
}

fn content(out: &mut Vec<u8>, parts: &[Part]) {
    if let [Part::Text(t)] = parts {
        crate::json::write_str(out, t);
        return;
    }
    // images and documents first, then text (anthropic convention)
    let mut a = Arr::new(out);
    let media = parts.iter().filter(|p| !matches!(p, Part::Text(_)));
    let texts = parts.iter().filter(|p| matches!(p, Part::Text(_)));
    for p in media.chain(texts) {
        let mut o = Obj::new(a.item());
        match p {
            Part::Text(t) => {
                o.str("type", "text");
                o.str("text", t);
            }
            Part::ImageUrl(u) => {
                o.str("type", "image");
                let mut s = Obj::new(o.key("source"));
                s.str("type", "url");
                s.str("url", u);
                s.end();
            }
            Part::Image { media_type, data } | Part::Document { media_type, data } => {
                let kind = if matches!(p, Part::Image { .. }) {
                    "image"
                } else {
                    "document"
                };
                o.str("type", kind);
                let mut s = Obj::new(o.key("source"));
                s.str("type", "base64");
                s.str("media_type", media_type);
                s.str("data", data);
                s.end();
            }
        }
        o.end();
    }
    a.end();
}

pub(crate) fn request(p: &Provider, r: &Req<'_>) -> Result<HttpRequest, Fail> {
    let plan = plan(r);
    let (thinking, effort) = thinking(r.reasoning);
    let mut body = Vec::with_capacity(1024 + r.schema.get(Dialect::AnthropicTool).len());
    let mut o = Obj::new(&mut body);
    o.str("model", r.model);
    o.u64(
        "max_tokens",
        r.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS).into(),
    );
    let mut msgs = Arr::new(o.key("messages"));
    for msg in r.messages {
        let mut m = Obj::new(msgs.item());
        m.str(
            "role",
            match msg.role() {
                Role::User => "user",
                Role::Assistant => "assistant",
            },
        );
        content(m.key("content"), msg.parts());
        m.end();
    }
    msgs.end();
    let base = r.system.unwrap_or(SYSTEM);
    let system = match plan {
        Plan::Tool { forced: false } => Cow::Owned(format!("{base}{TOOL_HINT}")),
        Plan::Text => Cow::Owned(system_with_schema(base, r)),
        _ => Cow::Borrowed(base),
    };
    o.str("system", &system);
    if let Plan::Tool { forced } = plan {
        let mut tools = Arr::new(o.key("tools"));
        let mut t = Obj::new(tools.item());
        t.str("name", "extract");
        t.str("description", "Extract structured data from the input");
        t.raw("input_schema", r.schema.get(Dialect::AnthropicTool));
        t.end();
        tools.end();
        let mut tc = Obj::new(o.key("tool_choice"));
        if forced {
            tc.str("type", "tool");
            tc.str("name", "extract");
        } else {
            tc.str("type", "auto");
        }
        tc.end();
    }
    if plan == Plan::Output || effort.is_some() {
        let mut oc = Obj::new(o.key("output_config"));
        if plan == Plan::Output {
            let mut f = Obj::new(oc.key("format"));
            f.str("type", "json_schema");
            f.raw("schema", r.schema.get(Dialect::AnthropicOutput));
            f.end();
        }
        if let Some(e) = effort {
            oc.str("effort", e);
        }
        oc.end();
    }
    match thinking {
        None => {}
        Some(Reasoning::Effort(_)) => type_obj(o.key("thinking"), "adaptive"),
        Some(Reasoning::Budget(n)) => {
            let mut t = Obj::new(o.key("thinking"));
            t.str("type", "enabled");
            t.u64("budget_tokens", n.into());
            t.end();
        }
        Some(Reasoning::Off) => type_obj(o.key("thinking"), "disabled"),
    }
    if r.stream {
        o.bool("stream", true);
    }
    o.end();
    Ok(HttpRequest {
        method: "POST",
        url: format!("{}/messages", p.base_url),
        headers: json_headers(&[("x-api-key", &p.key), ("anthropic-version", "2023-06-01")]),
        body,
        stream: r.stream,
    })
}

#[derive(Deserialize, Default)]
struct UsageIn {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

impl UsageIn {
    fn input(&self, u: &mut Usage) {
        u.input_tokens =
            self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens;
        u.cached_input_tokens = self.cache_read_input_tokens;
        u.cache_write_tokens = self.cache_creation_input_tokens;
    }
}

#[derive(Deserialize)]
struct Response<'a> {
    #[serde(default, borrow)]
    content: Vec<Block<'a>>,
    #[serde(default, borrow)]
    stop_reason: Option<Cow<'a, str>>,
    #[serde(default)]
    usage: Option<UsageIn>,
}

#[derive(Deserialize)]
struct Block<'a> {
    #[serde(rename = "type", borrow)]
    kind: Cow<'a, str>,
    #[serde(default, borrow)]
    input: Option<Raw<'a>>,
    #[serde(default, borrow)]
    text: Option<Cow<'a, str>>,
}

fn finish(content: String, usage: Usage, stop_reason: Option<&str>) -> Result<Answer, Fail> {
    let stop = match stop_reason {
        Some("refusal") => {
            let text = if content.is_empty() {
                "stop_reason: refusal".to_owned()
            } else {
                content
            };
            return Err(Fail::Refusal(text));
        }
        Some(r @ ("max_tokens" | "model_context_window_exceeded")) => {
            Stop::Truncated(format!("stop_reason: {r}"))
        }
        _ => Stop::Done,
    };
    Ok(Answer {
        content,
        usage,
        stop,
    })
}

pub(crate) fn response(r: &Req<'_>, body: &[u8]) -> Result<Answer, Fail> {
    let data: Response<'_> = envelope(body)?;
    let mut usage = Usage::default();
    if let Some(u) = &data.usage {
        u.input(&mut usage);
        usage.output_tokens = u.output_tokens;
    }
    let text = || -> String {
        data.content
            .iter()
            .filter(|b| b.kind == "text")
            .filter_map(|b| b.text.as_deref())
            .collect()
    };
    let tool_input = data
        .content
        .iter()
        .find_map(|b| b.input.filter(|_| b.kind == "tool_use"));
    let content = match (plan(r), tool_input) {
        (Plan::Tool { .. }, Some(input)) if data.stop_reason.as_deref() != Some("refusal") => {
            input.0.to_owned()
        }
        // the model answered in prose instead of calling the tool; kept so
        // the re-ask can quote it
        _ => text(),
    };
    finish(content, usage, data.stop_reason.as_deref())
}

#[derive(Deserialize)]
struct Event<'a> {
    #[serde(rename = "type", borrow)]
    kind: Cow<'a, str>,
    #[serde(default, borrow)]
    delta: Option<EventDelta<'a>>,
    #[serde(default)]
    usage: Option<UsageIn>,
    #[serde(default)]
    message: Option<EventMessage>,
}

#[derive(Deserialize)]
struct EventDelta<'a> {
    #[serde(rename = "type", default, borrow)]
    kind: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    partial_json: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    text: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    stop_reason: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct EventMessage {
    #[serde(default)]
    usage: Option<UsageIn>,
}

pub(crate) struct StreamAcc {
    tool: bool,
    content: String,
    usage: Usage,
    stop_reason: Option<String>,
}

impl StreamAcc {
    pub(crate) fn new(r: &Req<'_>) -> Self {
        Self {
            tool: matches!(plan(r), Plan::Tool { .. }),
            content: String::new(),
            usage: Usage::default(),
            stop_reason: None,
        }
    }

    pub(crate) fn event(&mut self, data: &str, sink: &mut dyn FnMut(&str)) {
        let Ok(event) = crate::json::from_str::<Event<'_>>(data) else {
            return;
        };
        match event.kind.as_ref() {
            "message_start" => {
                if let Some(u) = event.message.and_then(|m| m.usage) {
                    u.input(&mut self.usage);
                }
            }
            // text modes stream `text_delta`, a tool `input_json_delta`
            "content_block_delta" => {
                let Some(delta) = event.delta else { return };
                let piece = match (self.tool, delta.kind.as_deref()) {
                    (false, Some("text_delta")) => delta.text,
                    (true, Some("input_json_delta")) => delta.partial_json,
                    _ => None,
                };
                if let Some(p) = piece {
                    sink(&p);
                    self.content.push_str(&p);
                }
            }
            "message_delta" => {
                if let Some(u) = event.usage {
                    self.usage.output_tokens = u.output_tokens;
                }
                if let Some(r) = event.delta.and_then(|d| d.stop_reason) {
                    self.stop_reason = Some(r.into_owned());
                }
            }
            _ => {}
        }
    }

    pub(crate) fn finish(self) -> Result<Answer, Fail> {
        finish(self.content, self.usage, self.stop_reason.as_deref())
    }
}
