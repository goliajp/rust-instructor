//! Request bytes out, response bytes in, one module per wire protocol. No
//! I/O and no async here.

mod answer;
mod anthropic;
mod gemini;
mod openai_chat;
mod openai_resp;
#[cfg(test)]
mod tests;
mod write;

pub(crate) use answer::openai_schema;
pub(crate) use openai_chat::Host;
pub(crate) use write::{Arr, Obj, type_obj};

use crate::message::Message;
use crate::reasoning::Reasoning;
use crate::schema::SchemaEntry;
use crate::usage::Usage;

/// How the schema reaches the model.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// The provider's native structured output: OpenAI `json_schema`
    /// (strict), Anthropic `output_config.format`, Gemini
    /// `responseFormat.text.schema`.
    Schema,
    /// A tool whose input is the schema, called by the model (forced where
    /// the model allows it). Not available on Gemini or the OpenAI
    /// Responses API.
    Tool,
    /// JSON mode (`response_format: {"type": "json_object"}`) with the schema
    /// in the system prompt. Anthropic has no JSON mode and gets
    /// [`Prompt`](Self::Prompt).
    JsonObject,
    /// No format parameter; the schema goes in the system prompt and the
    /// JSON is taken from the answer text.
    Prompt,
}

/// Which OpenAI API an OpenAI client calls.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenAiApi {
    /// `POST /chat/completions`, the default; every compatible server has it.
    ChatCompletions,
    /// `POST /responses`.
    Responses,
}

/// HTTP headers in order. Names compare case-insensitively.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Headers(Vec<(String, String)>);

impl Headers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.0.push((name.into(), value.into()));
    }

    /// The first value for `name`.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// One HTTP request, ready to send.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpRequest {
    /// Always `"POST"` today.
    pub method: &'static str,
    pub url: String,
    pub headers: Headers,
    /// JSON bytes.
    pub body: Vec<u8>,
    /// The response is a server-sent event stream.
    pub stream: bool,
}

/// The wire protocol and its host-specific details.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Wire {
    OpenAiChat(Host),
    OpenAiResponses,
    Anthropic,
    Gemini,
}

/// One provider endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Provider {
    pub(crate) wire: Wire,
    pub(crate) key: String,
    pub(crate) base_url: String,
}

/// Everything one request needs.
pub(crate) struct Req<'a> {
    pub(crate) model: &'a str,
    pub(crate) system: Option<&'a str>,
    pub(crate) messages: &'a [Message],
    pub(crate) schema: &'static SchemaEntry,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) mode: OutputMode,
    pub(crate) stream: bool,
}

/// Why the model stopped, when it matters to the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    Done,
    /// Cut off by the output token limit; the provider's reason.
    Truncated(String),
}

/// What one response carried.
#[derive(Debug, PartialEq)]
pub(crate) struct Answer {
    pub(crate) content: String,
    pub(crate) usage: Usage,
    pub(crate) stop: Stop,
}

/// A response that is not an answer.
#[derive(Debug, PartialEq)]
pub(crate) enum Fail {
    /// The model declined; the usage of the response that said so.
    Refusal(String, Usage),
    /// The body is not the provider's response format.
    Malformed(String),
    /// The provider reported an error inside a successful HTTP response: an
    /// error event in a stream, or a failed Responses API response.
    Api(String),
    Unsupported(String),
}

pub(crate) const DEFAULT_SYSTEM: &str = "Extract the requested information from the given text. \
     Return valid JSON matching the provided schema.";

/// Appended to the system prompt when the tool is offered, not forced.
pub(crate) const TOOL_HINT: &str = " Answer by calling the `extract` tool.";

pub(crate) const SCHEMA_IN_PROMPT: &str =
    "\n\nAnswer with one JSON value and no other text. It must match this JSON Schema:\n";

impl Provider {
    pub(crate) fn name(&self) -> &'static str {
        match &self.wire {
            Wire::OpenAiChat(h) => h.name(),
            Wire::OpenAiResponses => "openai",
            Wire::Anthropic => "anthropic",
            Wire::Gemini => "gemini",
        }
    }

    /// The model used when the caller sets none. Each is its provider's
    /// current balanced tier; they move with minor releases.
    pub(crate) fn default_model(&self) -> &'static str {
        match &self.wire {
            Wire::OpenAiChat(Host::DeepSeek) => "deepseek-flash",
            Wire::OpenAiChat(Host::Qwen) => "qwen3.7-plus",
            Wire::OpenAiChat(_) | Wire::OpenAiResponses => "gpt-5.6-terra",
            Wire::Anthropic => "claude-sonnet-5",
            Wire::Gemini => "gemini-3.6-flash",
        }
    }

    pub(crate) fn request(&self, r: &Req<'_>) -> Result<HttpRequest, Fail> {
        match &self.wire {
            Wire::OpenAiChat(h) => openai_chat::request(self, *h, r),
            Wire::OpenAiResponses => openai_resp::request(self, r),
            Wire::Anthropic => anthropic::request(self, r),
            Wire::Gemini => gemini::request(self, r),
        }
    }

    pub(crate) fn response(&self, r: &Req<'_>, body: &[u8]) -> Result<Answer, Fail> {
        match &self.wire {
            Wire::OpenAiChat(_) => openai_chat::response(r, body),
            Wire::OpenAiResponses => openai_resp::response(body),
            Wire::Anthropic => anthropic::response(r, body),
            Wire::Gemini => gemini::response(body),
        }
    }

    pub(crate) fn stream(&self, r: &Req<'_>) -> Stream {
        match &self.wire {
            Wire::OpenAiChat(_) => Stream::OpenAiChat(Default::default()),
            Wire::OpenAiResponses => Stream::OpenAiResponses(Default::default()),
            Wire::Anthropic => Stream::Anthropic(anthropic::StreamAcc::new(r)),
            Wire::Gemini => Stream::Gemini(Default::default()),
        }
    }
}

/// Accumulates one streamed response, event by event.
pub(crate) enum Stream {
    OpenAiChat(openai_chat::StreamAcc),
    OpenAiResponses(openai_resp::StreamAcc),
    Anthropic(anthropic::StreamAcc),
    Gemini(gemini::StreamAcc),
}

impl Stream {
    /// One SSE `data:` payload; `sink` receives each answer text delta.
    pub(crate) fn event(&mut self, data: &str, sink: &mut dyn FnMut(&str)) {
        match self {
            Self::OpenAiChat(a) => a.event(data, sink),
            Self::OpenAiResponses(a) => a.event(data, sink),
            Self::Anthropic(a) => a.event(data, sink),
            Self::Gemini(a) => a.event(data, sink),
        }
    }

    pub(crate) fn finish(self) -> Result<Answer, Fail> {
        match self {
            Self::OpenAiChat(a) => a.finish(),
            Self::OpenAiResponses(a) => a.finish(),
            Self::Anthropic(a) => a.finish(),
            Self::Gemini(a) => a.finish(),
        }
    }
}

/// System prompt with the schema appended, for the modes that carry the
/// schema in text.
pub(crate) fn system_with_schema(base: &str, r: &Req<'_>) -> String {
    use crate::schema::Dialect;
    let schema = r.schema.get(Dialect::AnthropicTool);
    let mut s = String::with_capacity(base.len() + SCHEMA_IN_PROMPT.len() + schema.len());
    s.push_str(base);
    s.push_str(SCHEMA_IN_PROMPT);
    s.push_str(schema);
    s
}

pub(crate) fn json_headers(extra: &[(&str, &str)]) -> Headers {
    let mut h = Headers::new();
    h.push("content-type", "application/json");
    for (k, v) in extra {
        h.push(*k, *v);
    }
    h
}

/// Parse a response envelope, mapping errors to [`Fail::Malformed`].
pub(crate) fn envelope<'de, T: serde::Deserialize<'de>>(body: &'de [u8]) -> Result<T, Fail> {
    crate::json::from_slice(body).map_err(|e| {
        let text = String::from_utf8_lossy(body);
        Fail::Malformed(format!("{e} (body: {})", crate::text::clip(&text, 200)))
    })
}
