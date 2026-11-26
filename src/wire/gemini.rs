//! Gemini `generateContent`.

use std::borrow::Cow;

use serde::Deserialize;

use super::{
    Answer, Arr, DEFAULT_SYSTEM, Fail, HttpRequest, Obj, OutputMode, Provider, Req, Stop, envelope,
    json_headers, system_with_schema,
};
use crate::message::{Part, Role};
use crate::reasoning::{Effort, Reasoning};
use crate::schema::Dialect;
use crate::usage::Usage;

const DEFAULT_MAX_TOKENS: u32 = 16384;

fn parts(out: &mut Vec<u8>, parts: &[Part]) {
    let mut a = Arr::new(out);
    // media first, then text, as in the API's examples
    let media = parts.iter().filter(|p| !matches!(p, Part::Text(_)));
    let texts = parts.iter().filter(|p| matches!(p, Part::Text(_)));
    for p in media.chain(texts) {
        let mut o = Obj::new(a.item());
        match p {
            Part::Text(t) => o.str("text", t),
            Part::ImageUrl(u) => {
                let mut f = Obj::new(o.key("fileData"));
                f.str("fileUri", u);
                f.end();
            }
            Part::Image { media_type, data } | Part::Document { media_type, data } => {
                let mut i = Obj::new(o.key("inlineData"));
                i.str("mimeType", media_type);
                i.str("data", data);
                i.end();
            }
        }
        o.end();
    }
    a.end();
}

fn thinking(o: &mut Obj<'_>, r: Option<Reasoning>) {
    let Some(r) = r else { return };
    let mut t = Obj::new(o.key("thinkingConfig"));
    match r {
        Reasoning::Effort(e) => {
            let level = match e {
                Effort::Minimal => "MINIMAL",
                Effort::Low => "LOW",
                Effort::Medium => "MEDIUM",
                Effort::High => "HIGH",
                _ => "XHIGH",
            };
            t.str("thinkingLevel", level);
        }
        Reasoning::Budget(n) => t.u64("thinkingBudget", n.into()),
        Reasoning::Off => t.u64("thinkingBudget", 0),
    }
    t.end();
}

pub(crate) fn request(p: &Provider, r: &Req<'_>) -> Result<HttpRequest, Fail> {
    if r.mode == OutputMode::Tool {
        return Err(Fail::Unsupported(
            "OutputMode::Tool; use Schema, JsonObject or Prompt".into(),
        ));
    }
    let mut body = Vec::with_capacity(1024 + r.schema.get(Dialect::GeminiJson).len());
    let mut o = Obj::new(&mut body);
    let mut contents = Arr::new(o.key("contents"));
    for msg in r.messages {
        let mut m = Obj::new(contents.item());
        m.str(
            "role",
            match msg.role() {
                Role::User => "user",
                Role::Assistant => "model",
            },
        );
        parts(m.key("parts"), msg.parts());
        m.end();
    }
    contents.end();
    let base = r.system.unwrap_or(DEFAULT_SYSTEM);
    let system = match r.mode {
        OutputMode::JsonObject | OutputMode::Prompt => Cow::Owned(system_with_schema(base, r)),
        _ => Cow::Borrowed(base),
    };
    {
        let mut si = Obj::new(o.key("systemInstruction"));
        si.str("role", "user");
        let mut ps = Arr::new(si.key("parts"));
        let mut t = Obj::new(ps.item());
        t.str("text", &system);
        t.end();
        ps.end();
        si.end();
    }
    let mut g = Obj::new(o.key("generationConfig"));
    if let Some(t) = r.temperature {
        g.f64("temperature", t);
    }
    g.u64(
        "maxOutputTokens",
        r.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS).into(),
    );
    // `responseFormat.text` is what the REST reference documents;
    // `responseMimeType` and `responseJsonSchema` are marked deprecated
    if r.mode != OutputMode::Prompt {
        let mut rf = Obj::new(g.key("responseFormat"));
        let mut t = Obj::new(rf.key("text"));
        t.str("mimeType", "APPLICATION_JSON");
        if r.mode == OutputMode::Schema {
            t.raw("schema", r.schema.get(Dialect::GeminiJson));
        }
        t.end();
        rf.end();
    }
    thinking(&mut g, r.reasoning);
    g.end();
    o.end();
    // the key goes in a header so it never appears in a url or an error
    let url = if r.stream {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            p.base_url, r.model
        )
    } else {
        format!("{}/models/{}:generateContent", p.base_url, r.model)
    };
    Ok(HttpRequest {
        method: "POST",
        url,
        headers: json_headers(&[("x-goog-api-key", &p.key)]),
        body,
        stream: r.stream,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Response<'a> {
    #[serde(default, borrow)]
    candidates: Vec<Candidate<'a>>,
    #[serde(default)]
    usage_metadata: Option<UsageIn>,
    #[serde(default, borrow)]
    prompt_feedback: Option<Feedback<'a>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate<'a> {
    #[serde(default, borrow)]
    content: Option<Content<'a>>,
    #[serde(default, borrow)]
    finish_reason: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct Content<'a> {
    #[serde(default, borrow)]
    parts: Vec<RespPart<'a>>,
}

#[derive(Deserialize)]
struct RespPart<'a> {
    #[serde(default, borrow)]
    text: Option<Cow<'a, str>>,
    #[serde(default)]
    thought: bool,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct UsageIn {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    thoughts_token_count: u64,
    #[serde(default)]
    cached_content_token_count: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Feedback<'a> {
    #[serde(default, borrow)]
    block_reason: Option<Cow<'a, str>>,
}

#[derive(Default)]
pub(crate) struct StreamAcc {
    content: String,
    usage: Usage,
    finish_reason: Option<String>,
    block_reason: Option<String>,
}

impl StreamAcc {
    fn add(&mut self, r: Response<'_>, sink: &mut dyn FnMut(&str)) {
        if let Some(u) = r.usage_metadata {
            // thinking tokens are billed as output
            self.usage = Usage {
                input_tokens: u.prompt_token_count,
                cached_input_tokens: u.cached_content_token_count,
                output_tokens: u.candidates_token_count + u.thoughts_token_count,
                reasoning_tokens: u.thoughts_token_count,
                ..Usage::default()
            };
        }
        if let Some(b) = r.prompt_feedback.and_then(|f| f.block_reason) {
            self.block_reason = Some(b.into_owned());
        }
        let Some(c) = r.candidates.into_iter().next() else {
            return;
        };
        if let Some(f) = c.finish_reason {
            self.finish_reason = Some(f.into_owned());
        }
        for p in c.content.map(|c| c.parts).unwrap_or_default() {
            if p.thought {
                continue;
            }
            if let Some(t) = p.text {
                sink(&t);
                self.content.push_str(&t);
            }
        }
    }

    pub(crate) fn event(&mut self, data: &str, sink: &mut dyn FnMut(&str)) {
        if let Ok(r) = crate::json::from_str::<Response<'_>>(data) {
            self.add(r, sink);
        }
    }

    pub(crate) fn finish(self) -> Result<Answer, Fail> {
        if let Some(b) = self.block_reason {
            return Err(Fail::Refusal(format!("prompt blocked: {b}"), self.usage));
        }
        let stop = match self.finish_reason.as_deref() {
            // CONTINUATION: the per-request limit was reached and the answer
            // is not complete
            Some(r @ ("MAX_TOKENS" | "CONTINUATION")) => {
                Stop::Truncated(format!("finishReason: {r}"))
            }
            Some(
                r @ ("SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
                | "IMAGE_SAFETY"),
            ) => {
                return Err(Fail::Refusal(format!("finishReason: {r}"), self.usage));
            }
            _ => Stop::Done,
        };
        Ok(Answer {
            content: self.content,
            usage: self.usage,
            stop,
        })
    }
}

pub(crate) fn response(body: &[u8]) -> Result<Answer, Fail> {
    let data: Response<'_> = envelope(body)?;
    let mut acc = StreamAcc::default();
    acc.add(data, &mut |_| {});
    acc.finish()
}
