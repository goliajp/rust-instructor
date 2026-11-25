//! OpenAI Responses API.

use std::borrow::Cow;

use serde::Deserialize;

use super::{
    Answer, Arr, DEFAULT_SYSTEM, Fail, HttpRequest, Obj, OutputMode, Provider, Req, Stop, envelope,
    json_headers, system_with_schema, type_obj,
};
use crate::message::{Part, Role};
use crate::reasoning::Reasoning;
use crate::schema::Dialect;
use crate::usage::Usage;

fn content(out: &mut Vec<u8>, role: Role, parts: &[Part]) {
    let mut a = Arr::new(out);
    for p in parts {
        let mut o = Obj::new(a.item());
        match (role, p) {
            (Role::Assistant, Part::Text(t)) => {
                o.str("type", "output_text");
                o.str("text", t);
            }
            (_, Part::Text(t)) => {
                o.str("type", "input_text");
                o.str("text", t);
            }
            (_, Part::ImageUrl(u)) => {
                o.str("type", "input_image");
                o.str("image_url", u);
            }
            (_, Part::Image { media_type, data }) => {
                o.str("type", "input_image");
                o.str("image_url", &format!("data:{media_type};base64,{data}"));
            }
            (_, Part::Document { media_type, data }) => {
                o.str("type", "input_file");
                o.str("filename", "document.pdf");
                o.str("file_data", &format!("data:{media_type};base64,{data}"));
            }
        }
        o.end();
    }
    a.end();
}

pub(crate) fn request(p: &Provider, r: &Req<'_>) -> Result<HttpRequest, Fail> {
    if r.mode == OutputMode::Tool {
        return Err(Fail::Unsupported(
            "OutputMode::Tool on the Responses API; use Schema".into(),
        ));
    }
    if let Some(Reasoning::Budget(_)) = r.reasoning {
        return Err(Fail::Unsupported(
            "openai has no thinking budget parameter; use Reasoning::Effort".into(),
        ));
    }
    let mut body = Vec::with_capacity(1024 + r.schema.get(Dialect::Strict).len());
    let mut o = Obj::new(&mut body);
    o.str("model", r.model);
    let base = r.system.unwrap_or(DEFAULT_SYSTEM);
    match r.mode {
        OutputMode::JsonObject | OutputMode::Prompt => {
            o.str("instructions", &system_with_schema(base, r))
        }
        _ => o.str("instructions", base),
    }
    let mut input = Arr::new(o.key("input"));
    for msg in r.messages {
        let mut m = Obj::new(input.item());
        let role = msg.role();
        m.str(
            "role",
            match role {
                Role::User => "user",
                Role::Assistant => "assistant",
            },
        );
        content(m.key("content"), role, msg.parts());
        m.end();
    }
    input.end();
    if r.mode != OutputMode::Prompt {
        let mut text = Obj::new(o.key("text"));
        if r.mode == OutputMode::Schema {
            let mut f = Obj::new(text.key("format"));
            f.str("type", "json_schema");
            f.str("name", r.schema.name());
            f.bool("strict", true);
            f.raw("schema", r.schema.get(Dialect::Strict));
            f.end();
        } else {
            type_obj(text.key("format"), "json_object");
        }
        text.end();
    }
    if let Some(t) = r.temperature {
        o.f64("temperature", t);
    }
    if let Some(n) = r.max_tokens {
        o.u64("max_output_tokens", n.into());
    }
    match r.reasoning {
        Some(Reasoning::Effort(e)) => {
            let mut re = Obj::new(o.key("reasoning"));
            re.str("effort", e.as_str());
            re.end();
        }
        Some(Reasoning::Off) => {
            let mut re = Obj::new(o.key("reasoning"));
            re.str("effort", "none");
            re.end();
        }
        _ => {}
    }
    if r.stream {
        o.bool("stream", true);
    }
    o.end();
    Ok(HttpRequest {
        method: "POST",
        url: format!("{}/responses", p.base_url),
        headers: json_headers(&[("authorization", &format!("Bearer {}", p.key))]),
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
    input_tokens_details: Option<Details>,
    #[serde(default)]
    output_tokens_details: Option<Details>,
}

#[derive(Deserialize, Default)]
struct Details {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    reasoning_tokens: u64,
}

#[derive(Deserialize)]
struct Response<'a> {
    #[serde(default, borrow)]
    status: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    incomplete_details: Option<Incomplete<'a>>,
    #[serde(default, borrow)]
    output: Vec<Item<'a>>,
    #[serde(default)]
    usage: Option<UsageIn>,
}

#[derive(Deserialize)]
struct Incomplete<'a> {
    #[serde(default, borrow)]
    reason: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct Item<'a> {
    #[serde(rename = "type", borrow)]
    kind: Cow<'a, str>,
    #[serde(default, borrow)]
    content: Vec<ItemPart<'a>>,
}

#[derive(Deserialize)]
struct ItemPart<'a> {
    #[serde(rename = "type", borrow)]
    kind: Cow<'a, str>,
    #[serde(default, borrow)]
    text: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    refusal: Option<Cow<'a, str>>,
}

fn to_usage(u: &UsageIn) -> Usage {
    Usage {
        input_tokens: u.input_tokens,
        cached_input_tokens: u
            .input_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens),
        output_tokens: u.output_tokens,
        reasoning_tokens: u
            .output_tokens_details
            .as_ref()
            .map_or(0, |d| d.reasoning_tokens),
        ..Usage::default()
    }
}

fn finish(r: &Response<'_>, content: String, refusal: String) -> Result<Answer, Fail> {
    let usage = r.usage.as_ref().map(to_usage).unwrap_or_default();
    if !refusal.is_empty() {
        return Err(Fail::Refusal(refusal, usage));
    }
    let stop = match (r.status.as_deref(), &r.incomplete_details) {
        (Some("incomplete"), Some(d)) if d.reason.as_deref() == Some("content_filter") => {
            return Err(Fail::Refusal("incomplete: content_filter".into(), usage));
        }
        (Some("incomplete"), d) => Stop::Truncated(format!(
            "incomplete: {}",
            d.as_ref()
                .and_then(|d| d.reason.as_deref())
                .unwrap_or("unknown")
        )),
        _ => Stop::Done,
    };
    Ok(Answer {
        content,
        usage,
        stop,
    })
}

fn collect(r: &Response<'_>) -> (String, String) {
    let mut text = String::new();
    let mut refusal = String::new();
    for item in r.output.iter().filter(|i| i.kind == "message") {
        for p in &item.content {
            match p.kind.as_ref() {
                "output_text" => text.push_str(p.text.as_deref().unwrap_or("")),
                "refusal" => refusal.push_str(p.refusal.as_deref().unwrap_or("")),
                _ => {}
            }
        }
    }
    (text, refusal)
}

pub(crate) fn response(body: &[u8]) -> Result<Answer, Fail> {
    let data: Response<'_> = envelope(body)?;
    let (text, refusal) = collect(&data);
    finish(&data, text, refusal)
}

#[derive(Deserialize)]
struct Event<'a> {
    #[serde(rename = "type", borrow)]
    kind: Cow<'a, str>,
    #[serde(default, borrow)]
    delta: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    response: Option<Response<'a>>,
}

#[derive(Default)]
pub(crate) struct StreamAcc {
    content: String,
    refusal: String,
    end: Option<Result<Answer, Fail>>,
}

impl StreamAcc {
    pub(crate) fn event(&mut self, data: &str, sink: &mut dyn FnMut(&str)) {
        let Ok(e) = crate::json::from_str::<Event<'_>>(data) else {
            return;
        };
        match (e.kind.as_ref(), e.delta, e.response) {
            ("response.output_text.delta", Some(d), _) => {
                sink(&d);
                self.content.push_str(&d);
            }
            ("response.refusal.delta", Some(d), _) => self.refusal.push_str(&d),
            ("response.completed" | "response.incomplete" | "response.failed", _, Some(r)) => {
                let content = std::mem::take(&mut self.content);
                let refusal = std::mem::take(&mut self.refusal);
                self.end = Some(finish(&r, content, refusal));
            }
            _ => {}
        }
    }

    pub(crate) fn finish(self) -> Result<Answer, Fail> {
        match self.end {
            Some(r) => r,
            None => {
                if !self.refusal.is_empty() {
                    return Err(Fail::Refusal(self.refusal, Usage::default()));
                }
                Ok(Answer {
                    content: self.content,
                    usage: Usage::default(),
                    stop: Stop::Done,
                })
            }
        }
    }
}
