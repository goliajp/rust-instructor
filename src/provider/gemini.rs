use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::{DEFAULT_SYSTEM, DeltaSink, ImageInput, Message, RawResponse, RequestSpec, Stop, http};
use crate::error::{Error, Result};
use crate::reasoning::{Effort, Reasoning};
use crate::schema::Dialect;

const DEFAULT_MAX_TOKENS: u32 = 16384;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Request<'a> {
    contents: Vec<GemContent<'a>>,
    system_instruction: GemContent<'a>,
    generation_config: GenerationConfig<'a>,
}

#[derive(Serialize)]
struct GemContent<'a> {
    role: &'a str,
    parts: Vec<GemPart<'a>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum GemPart<'a> {
    Text {
        text: &'a str,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData<'a>,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData<'a>,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineData<'a> {
    mime_type: &'a str,
    data: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileData<'a> {
    file_uri: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    max_output_tokens: u32,
    response_mime_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_json_schema: Option<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
struct ThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Response {
    candidates: Option<Vec<Candidate>>,
    usage_metadata: Option<UsageMetadata>,
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: Option<CandidateContent>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Option<Vec<ResponsePart>>,
}

#[derive(Deserialize)]
struct ResponsePart {
    text: Option<String>,
    #[serde(default)]
    thought: bool,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    // billed as output
    #[serde(default)]
    thoughts_token_count: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptFeedback {
    block_reason: Option<String>,
}

fn build_parts(msg: &Message) -> Vec<GemPart<'_>> {
    let mut parts: Vec<GemPart<'_>> = msg
        .images
        .iter()
        .map(|img| match img {
            ImageInput::Base64 { media_type, data } => GemPart::InlineData {
                inline_data: InlineData {
                    mime_type: media_type,
                    data,
                },
            },
            ImageInput::Url(url) => GemPart::FileData {
                file_data: FileData { file_uri: url },
            },
        })
        .collect();
    parts.push(GemPart::Text { text: &msg.content });
    parts
}

fn thinking_config(r: Option<Reasoning>) -> Option<ThinkingConfig> {
    let (level, budget) = match r? {
        Reasoning::Effort(e) => {
            let level = match e {
                Effort::Minimal => "MINIMAL",
                Effort::Low => "LOW",
                Effort::Medium => "MEDIUM",
                Effort::High => "HIGH",
                _ => "XHIGH",
            };
            (Some(level), None)
        }
        Reasoning::Budget(n) => (None, Some(n)),
        Reasoning::Off => (None, Some(0)),
    };
    Some(ThinkingConfig {
        thinking_level: level,
        thinking_budget: budget,
    })
}

pub(crate) async fn send(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    spec: &RequestSpec<'_>,
    on_delta: DeltaSink<'_>,
) -> Result<RawResponse> {
    // `responseSchema` has no `$ref`, so a recursive type goes as JSON Schema
    let (response_schema, response_json_schema) = if spec.schema.recursive() {
        (None, Some(spec.schema.get(Dialect::GeminiJson)))
    } else {
        (Some(spec.schema.get(Dialect::Gemini)), None)
    };
    let body = Request {
        contents: spec
            .messages
            .iter()
            .map(|m| GemContent {
                role: if m.role == "assistant" {
                    "model"
                } else {
                    &m.role
                },
                parts: build_parts(m),
            })
            .collect(),
        system_instruction: GemContent {
            role: "user",
            parts: vec![GemPart::Text {
                text: spec.system.unwrap_or(DEFAULT_SYSTEM),
            }],
        },
        generation_config: GenerationConfig {
            temperature: spec.temperature,
            max_output_tokens: spec.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            response_mime_type: "application/json",
            response_schema,
            response_json_schema,
            thinking_config: thinking_config(spec.reasoning),
        },
    };

    // the key goes in a header so it never appears in a url or an error
    let url = match on_delta {
        Some(_) => format!(
            "{base_url}/models/{}:streamGenerateContent?alt=sse",
            spec.model
        ),
        None => format!("{base_url}/models/{}:generateContent", spec.model),
    };
    let req = http.post(url).header("x-goog-api-key", api_key);
    let resp = http::post(req, &body).await?;

    match on_delta {
        Some(sink) => read_stream(resp, sink).await,
        None => {
            let data: Response = http::json(resp).await?;
            let mut acc = Acc::default();
            acc.add(data, &mut |_| {});
            acc.finish()
        }
    }
}

#[derive(Default)]
struct Acc {
    content: String,
    usage: UsageMetadata,
    finish_reason: Option<String>,
    block_reason: Option<String>,
}

impl Acc {
    fn add(&mut self, r: Response, sink: &mut dyn FnMut(&str)) {
        if let Some(u) = r.usage_metadata {
            self.usage = u;
        }
        if let Some(b) = r.prompt_feedback.and_then(|f| f.block_reason) {
            self.block_reason = Some(b);
        }
        let Some(c) = r.candidates.and_then(|c| c.into_iter().next()) else {
            return;
        };
        if let Some(f) = c.finish_reason {
            self.finish_reason = Some(f);
        }
        let parts = c.content.and_then(|c| c.parts).unwrap_or_default();
        for p in parts.into_iter().filter(|p| !p.thought) {
            if let Some(t) = p.text {
                sink(&t);
                self.content.push_str(&t);
            }
        }
    }

    fn finish(self) -> Result<RawResponse> {
        if let Some(b) = self.block_reason {
            return Err(Error::refused("gemini", &format!("prompt blocked: {b}")));
        }
        let stop = match self.finish_reason.as_deref() {
            Some("MAX_TOKENS") => Stop::Truncated("finishReason: MAX_TOKENS".into()),
            Some(
                r @ ("SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
                | "IMAGE_SAFETY"),
            ) => return Err(Error::refused("gemini", &format!("finishReason: {r}"))),
            _ => Stop::Done,
        };
        Ok(RawResponse {
            content: self.content,
            input_tokens: self.usage.prompt_token_count,
            output_tokens: self.usage.candidates_token_count + self.usage.thoughts_token_count,
            stop,
        })
    }
}

async fn read_stream(
    resp: reqwest::Response,
    sink: &mut (dyn FnMut(&str) + Send),
) -> Result<RawResponse> {
    let mut acc = Acc::default();
    http::read_sse(resp, |data| {
        if let Ok(event) = serde_json::from_str::<Response>(data) {
            acc.add(event, &mut *sink);
        }
    })
    .await?;
    acc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_mapping() {
        assert_eq!(thinking_config(None), None);
        let level = |e| {
            thinking_config(Some(Reasoning::Effort(e)))
                .unwrap()
                .thinking_level
        };
        assert_eq!(level(Effort::Minimal), Some("MINIMAL"));
        assert_eq!(level(Effort::XHigh), Some("XHIGH"));
        assert_eq!(level(Effort::Max), Some("XHIGH"));
        let off = thinking_config(Some(Reasoning::Off)).unwrap();
        assert_eq!(off.thinking_budget, Some(0));
        assert_eq!(off.thinking_level, None);
        let v = serde_json::to_value(thinking_config(Some(Reasoning::Budget(1024)))).unwrap();
        assert_eq!(v, serde_json::json!({"thinkingBudget": 1024}));
    }
}
