//! OpenAI Chat Completions and the compatible APIs.

use std::borrow::Cow;

use serde::Deserialize;

mod content;

use super::{
    Answer, Arr, DEFAULT_SYSTEM, Fail, HttpRequest, Obj, OutputMode, Provider, Req, Stop, envelope,
    json_headers, system_with_schema, type_obj,
};
use crate::json::Raw;
use crate::message::Role;
use crate::reasoning::Reasoning;
use crate::schema::Dialect;
use crate::usage::Usage;
use content::content;

/// Which OpenAI-compatible API is on the other end, from the base URL host.
/// It decides the output-limit field, the reasoning parameters and the
/// default [`OutputMode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Host {
    OpenAi,
    Generic,
    DeepSeek,
    Qwen,
    Kimi,
    Glm,
}

impl Host {
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

    /// DeepSeek and GLM document only `json_object`; Qwen supports strict
    /// schemas on some models only.
    pub(crate) fn default_mode(self) -> OutputMode {
        match self {
            Self::DeepSeek | Self::Glm | Self::Qwen => OutputMode::JsonObject,
            Self::OpenAi | Self::Generic | Self::Kimi => OutputMode::Schema,
        }
    }
}

fn reasoning(o: &mut Obj<'_>, host: Host, r: Option<Reasoning>) -> Result<(), Fail> {
    let Some(r) = r else { return Ok(()) };
    let switch = |o: &mut Obj<'_>, on: bool| {
        type_obj(o.key("thinking"), if on { "enabled" } else { "disabled" });
    };
    match (host, r) {
        (Host::OpenAi | Host::Generic, Reasoning::Effort(e)) => {
            o.str("reasoning_effort", e.as_str())
        }
        (Host::OpenAi | Host::Generic, Reasoning::Off) => o.str("reasoning_effort", "none"),
        (Host::Qwen, Reasoning::Effort(_)) => o.bool("enable_thinking", true),
        (Host::Qwen, Reasoning::Budget(n)) => {
            o.bool("enable_thinking", true);
            o.u64("thinking_budget", n.into());
        }
        (Host::Qwen, Reasoning::Off) => o.bool("enable_thinking", false),
        (_, Reasoning::Effort(e)) => {
            switch(o, true);
            o.str("reasoning_effort", e.coarse());
        }
        (_, Reasoning::Off) => switch(o, false),
        (_, Reasoning::Budget(_)) => {
            return Err(Fail::Unsupported(format!(
                "{} has no thinking budget parameter; use Reasoning::Effort",
                host.name()
            )));
        }
    }
    Ok(())
}

pub(crate) fn request(p: &Provider, host: Host, r: &Req<'_>) -> Result<HttpRequest, Fail> {
    let mut body = Vec::with_capacity(1024 + r.schema.get(Dialect::Strict).len());
    let mut o = Obj::new(&mut body);
    o.str("model", r.model);
    let base = r.system.unwrap_or(DEFAULT_SYSTEM);
    let system = match r.mode {
        OutputMode::JsonObject | OutputMode::Prompt => Cow::Owned(system_with_schema(base, r)),
        OutputMode::Schema | OutputMode::Tool => Cow::Borrowed(base),
    };
    let mut msgs = Arr::new(o.key("messages"));
    {
        let mut m = Obj::new(msgs.item());
        m.str("role", "system");
        m.str("content", &system);
        m.end();
    }
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
    match r.mode {
        OutputMode::Schema => {
            let mut rf = Obj::new(o.key("response_format"));
            rf.str("type", "json_schema");
            let mut js = Obj::new(rf.key("json_schema"));
            js.str("name", r.schema.name());
            js.bool("strict", true);
            js.raw("schema", r.schema.get(Dialect::Strict));
            js.end();
            rf.end();
        }
        OutputMode::Tool => {
            let mut tools = Arr::new(o.key("tools"));
            let mut t = Obj::new(tools.item());
            t.str("type", "function");
            let mut f = Obj::new(t.key("function"));
            f.str("name", "extract");
            f.str("description", "Extract structured data from the input");
            f.raw("parameters", r.schema.get(Dialect::Strict));
            f.bool("strict", true);
            f.end();
            t.end();
            tools.end();
            let mut tc = Obj::new(o.key("tool_choice"));
            tc.str("type", "function");
            let mut f = Obj::new(tc.key("function"));
            f.str("name", "extract");
            f.end();
            tc.end();
        }
        OutputMode::JsonObject => type_obj(o.key("response_format"), "json_object"),
        OutputMode::Prompt => {}
    }
    if let Some(t) = r.temperature {
        o.f64("temperature", t);
    }
    // openai deprecated `max_tokens` in favour of `max_completion_tokens`;
    // the compatible APIs still read `max_tokens`
    if let Some(n) = r.max_tokens {
        let key = match host {
            Host::OpenAi => "max_completion_tokens",
            _ => "max_tokens",
        };
        o.u64(key, n.into());
    }
    reasoning(&mut o, host, r.reasoning)?;
    if r.stream {
        o.bool("stream", true);
        let mut so = Obj::new(o.key("stream_options"));
        so.bool("include_usage", true);
        so.end();
    }
    o.end();
    Ok(HttpRequest {
        method: "POST",
        url: format!("{}/chat/completions", p.base_url),
        headers: json_headers(&[("authorization", &format!("Bearer {}", p.key))]),
        body,
        stream: r.stream,
    })
}

#[derive(Deserialize, Default)]
struct UsageIn {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<Details>,
    #[serde(default)]
    completion_tokens_details: Option<Details>,
}

#[derive(Deserialize, Default)]
struct Details {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    cache_write_tokens: u64,
    #[serde(default)]
    reasoning_tokens: u64,
}

impl UsageIn {
    fn to_usage(&self) -> Usage {
        Usage {
            input_tokens: self.prompt_tokens,
            cached_input_tokens: self
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |d| d.cached_tokens),
            cache_write_tokens: self
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |d| d.cache_write_tokens),
            output_tokens: self.completion_tokens,
            reasoning_tokens: self
                .completion_tokens_details
                .as_ref()
                .map_or(0, |d| d.reasoning_tokens),
            ..Usage::default()
        }
    }
}

#[derive(Deserialize)]
struct Response<'a> {
    #[serde(default, borrow)]
    choices: Vec<Choice<'a>>,
    #[serde(default)]
    usage: Option<UsageIn>,
}

#[derive(Deserialize)]
struct Choice<'a> {
    #[serde(borrow)]
    message: RespMessage<'a>,
    #[serde(default, borrow)]
    finish_reason: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct RespMessage<'a> {
    #[serde(default, borrow, deserialize_with = "text_or_parts")]
    content: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    refusal: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    tool_calls: Vec<ToolCall<'a>>,
}

#[derive(Deserialize)]
struct ToolCall<'a> {
    #[serde(borrow)]
    function: Function<'a>,
}

#[derive(Deserialize)]
struct Function<'a> {
    #[serde(default, borrow)]
    arguments: Option<Cow<'a, str>>,
}

/// `content` is a string, an array of `{"text": ..}` parts, or null.
fn text_or_parts<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<Cow<'de, str>>, D::Error> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Option<Cow<'de, str>>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a string, an array of parts or null")
        }
        fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E> {
            Ok(Some(Cow::Borrowed(v)))
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(Cow::Owned(v.to_owned())))
        }
        fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
            Ok(Some(Cow::Owned(v)))
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(self)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            #[derive(Deserialize)]
            struct PartIn {
                #[serde(default)]
                text: Option<String>,
            }
            let mut out = String::new();
            while let Some(p) = a.next_element::<PartIn>()? {
                out.push_str(p.text.as_deref().unwrap_or(""));
            }
            Ok(Some(Cow::Owned(out)))
        }
    }
    d.deserialize_any(V)
}

fn finish(
    content: String,
    usage: Usage,
    refusal: Option<&str>,
    finish_reason: Option<&str>,
) -> Result<Answer, Fail> {
    if let Some(text) = refusal.filter(|t| !t.is_empty()) {
        return Err(Fail::Refusal(text.to_owned(), usage));
    }
    let stop = match finish_reason {
        Some("length") => Stop::Truncated("finish_reason: length".into()),
        Some("content_filter") if content.is_empty() => {
            return Err(Fail::Refusal("finish_reason: content_filter".into(), usage));
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
    let usage = data
        .usage
        .as_ref()
        .map(UsageIn::to_usage)
        .unwrap_or_default();
    let Some(choice) = data.choices.into_iter().next() else {
        return Err(Fail::Malformed("no choices in response".into()));
    };
    let m = choice.message;
    let content = match (r.mode, m.tool_calls.first()) {
        (OutputMode::Tool, Some(call)) => call.function.arguments.as_deref().unwrap_or(""),
        _ => m.content.as_deref().unwrap_or(""),
    };
    finish(
        content.to_owned(),
        usage,
        m.refusal.as_deref(),
        choice.finish_reason.as_deref(),
    )
}

#[derive(Deserialize)]
struct Chunk<'a> {
    #[serde(default, borrow)]
    choices: Vec<ChunkChoice<'a>>,
    #[serde(default)]
    usage: Option<UsageIn>,
    #[serde(default, borrow)]
    error: Option<Raw<'a>>,
}

#[derive(Deserialize)]
struct ChunkChoice<'a> {
    #[serde(default, borrow)]
    delta: Option<Delta<'a>>,
    #[serde(default, borrow)]
    finish_reason: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct Delta<'a> {
    #[serde(default, borrow)]
    content: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    refusal: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    tool_calls: Vec<ToolCall<'a>>,
}

#[derive(Default)]
pub(crate) struct StreamAcc {
    content: String,
    refusal: String,
    usage: Usage,
    finish_reason: Option<String>,
    error: Option<String>,
}

impl StreamAcc {
    pub(crate) fn event(&mut self, data: &str, sink: &mut dyn FnMut(&str)) {
        let Ok(chunk) = crate::json::from_str::<Chunk<'_>>(data) else {
            return;
        };
        // an error after the stream has started arrives as a chunk
        if let Some(e) = chunk.error {
            self.error = Some(e.0.to_owned());
            return;
        }
        if let Some(u) = &chunk.usage {
            self.usage = u.to_usage();
        }
        for choice in chunk.choices {
            if let Some(r) = choice.finish_reason {
                self.finish_reason = Some(r.into_owned());
            }
            let Some(delta) = choice.delta else { continue };
            if let Some(r) = delta.refusal {
                self.refusal.push_str(&r);
            }
            let piece = match delta.tool_calls.first() {
                Some(call) => call.function.arguments.as_deref(),
                None => delta.content.as_deref(),
            };
            if let Some(text) = piece.filter(|t| !t.is_empty()) {
                sink(text);
                self.content.push_str(text);
            }
        }
    }

    pub(crate) fn finish(self) -> Result<Answer, Fail> {
        if let Some(e) = self.error {
            return Err(Fail::Api(e));
        }
        finish(
            self.content,
            self.usage,
            Some(&self.refusal),
            self.finish_reason.as_deref(),
        )
    }
}
