use serde_json::{Value, json};

use super::*;
use crate::Schema;
use crate::message::Part;
use crate::reasoning::Effort;
use crate::schema::entry_for;

#[derive(serde::Deserialize, Schema)]
#[expect(dead_code, reason = "only the schema is used")]
struct C {
    name: String,
}

fn provider(wire: Wire) -> Provider {
    Provider {
        wire,
        key: "k".into(),
        base_url: "http://h".into(),
    }
}

fn req<'a>(msgs: &'a [Message], mode: OutputMode, reasoning: Option<Reasoning>) -> Req<'a> {
    Req {
        model: "m",
        system: None,
        messages: msgs,
        schema: entry_for::<C>(false),
        temperature: None,
        max_tokens: Some(9),
        reasoning,
        mode,
        stream: false,
    }
}

fn body(p: &Provider, r: &Req<'_>) -> Value {
    serde_json::from_slice(&p.request(r).unwrap().body).unwrap()
}

fn chat(host: Host) -> Provider {
    provider(Wire::OpenAiChat(host))
}

#[test]
fn host_detection_and_default_modes() {
    let d = Host::from_base_url;
    assert_eq!(d("https://api.openai.com/v1"), Host::OpenAi);
    assert_eq!(d("https://api.deepseek.com/v1"), Host::DeepSeek);
    assert_eq!(
        d("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        Host::Qwen
    );
    assert_eq!(d("https://api.moonshot.ai/v1"), Host::Kimi);
    assert_eq!(d("https://open.bigmodel.cn/api/paas/v4"), Host::Glm);
    assert_eq!(d("https://api.z.ai/api/paas/v4"), Host::Glm);
    assert_eq!(d("http://127.0.0.1:8080/v1"), Host::Generic);
    assert_eq!(d("https://notdeepseek.com/v1"), Host::Generic);
    for h in [Host::DeepSeek, Host::Glm, Host::Qwen] {
        assert_eq!(h.default_mode(), OutputMode::JsonObject, "{h:?}");
    }
    for h in [Host::OpenAi, Host::Generic, Host::Kimi] {
        assert_eq!(h.default_mode(), OutputMode::Schema, "{h:?}");
    }
}

#[test]
fn reasoning_parameters_per_host() {
    let m = [Message::user("x")];
    let f = |h, r| {
        let mut b = body(&chat(h), &req(&m, OutputMode::Schema, Some(r)));
        let keep = [
            "reasoning_effort",
            "thinking",
            "enable_thinking",
            "thinking_budget",
        ];
        b.as_object_mut()
            .unwrap()
            .retain(|k, _| keep.contains(&k.as_str()));
        b
    };
    let hi = Reasoning::Effort(Effort::High);
    assert_eq!(f(Host::OpenAi, hi), json!({"reasoning_effort": "high"}));
    assert_eq!(
        f(Host::OpenAi, Reasoning::Off),
        json!({"reasoning_effort": "none"})
    );
    assert_eq!(
        f(Host::DeepSeek, Reasoning::Effort(Effort::Medium)),
        json!({"thinking": {"type": "enabled"}, "reasoning_effort": "high"})
    );
    assert_eq!(
        f(Host::Glm, Reasoning::Off),
        json!({"thinking": {"type": "disabled"}})
    );
    assert_eq!(
        f(Host::Kimi, Reasoning::Effort(Effort::Max)),
        json!({"thinking": {"type": "enabled"}, "reasoning_effort": "max"})
    );
    assert_eq!(
        f(Host::Qwen, Reasoning::Budget(512)),
        json!({"enable_thinking": true, "thinking_budget": 512})
    );
    assert_eq!(
        f(Host::Qwen, Reasoning::Off),
        json!({"enable_thinking": false})
    );
    for h in [Host::OpenAi, Host::Kimi, Host::DeepSeek] {
        let r = req(&m, OutputMode::Schema, Some(Reasoning::Budget(1)));
        assert!(
            matches!(chat(h).request(&r), Err(Fail::Unsupported(_))),
            "{h:?}"
        );
    }
}

#[test]
fn openai_host_sends_max_completion_tokens() {
    let m = [Message::user("x")];
    let b = body(&chat(Host::OpenAi), &req(&m, OutputMode::Schema, None));
    assert_eq!(b["max_completion_tokens"], 9);
    assert!(b.get("max_tokens").is_none());
    let b = body(&chat(Host::DeepSeek), &req(&m, OutputMode::Schema, None));
    assert_eq!(b["max_tokens"], 9);
}

#[test]
fn json_object_mode_puts_the_schema_in_the_prompt() {
    let m = [Message::user("x")];
    let b = body(
        &chat(Host::DeepSeek),
        &req(&m, OutputMode::JsonObject, None),
    );
    assert_eq!(b["response_format"], json!({"type": "json_object"}));
    let system = b["messages"][0]["content"].as_str().unwrap();
    // deepseek rejects json_object requests whose prompt lacks the word
    assert!(system.contains("JSON"), "{system}");
    assert!(system.contains(r#""properties":{"name""#), "{system}");
    let b = body(&chat(Host::Generic), &req(&m, OutputMode::Prompt, None));
    assert!(b.get("response_format").is_none());
    assert!(
        b["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("JSON Schema")
    );
}

#[test]
fn openai_tool_mode() {
    let m = [Message::user("x")];
    let b = body(&chat(Host::OpenAi), &req(&m, OutputMode::Tool, None));
    assert_eq!(b["tools"][0]["function"]["name"], "extract");
    assert_eq!(b["tools"][0]["function"]["strict"], true);
    assert_eq!(b["tools"][0]["function"]["parameters"]["type"], "object");
    assert_eq!(b["tool_choice"]["function"]["name"], "extract");
    assert!(b.get("response_format").is_none());
    let r = req(&m, OutputMode::Tool, None);
    let resp = json!({"choices": [{"message": {"content": null, "tool_calls": [
        {"function": {"name": "extract", "arguments": "{\"name\":\"a\"}"}}]},
        "finish_reason": "tool_calls"}]});
    let a = chat(Host::OpenAi)
        .response(&r, resp.to_string().as_bytes())
        .unwrap();
    assert_eq!(a.content, r#"{"name":"a"}"#);
}

#[test]
fn openai_usage_details_and_stops() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Schema, None);
    let p = chat(Host::OpenAi);
    let resp = json!({"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                  "prompt_tokens_details": {"cached_tokens": 40},
                  "completion_tokens_details": {"reasoning_tokens": 30}}});
    let a = p.response(&r, resp.to_string().as_bytes()).unwrap();
    assert_eq!(a.stop, Stop::Truncated("finish_reason: length".into()));
    assert_eq!(
        (a.usage.input_tokens, a.usage.cached_input_tokens),
        (100, 40)
    );
    assert_eq!((a.usage.output_tokens, a.usage.reasoning_tokens), (50, 30));
    let resp = json!({"choices": [{"message": {"content": null, "refusal": "no"}}]});
    assert_eq!(
        p.response(&r, resp.to_string().as_bytes()),
        Err(Fail::Refusal("no".into(), Usage::default()))
    );
    let resp = json!({"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"text": "b"}]}}]});
    assert_eq!(
        p.response(&r, resp.to_string().as_bytes()).unwrap().content,
        "ab"
    );
    assert!(matches!(
        p.response(&r, b"not json"),
        Err(Fail::Malformed(_))
    ));
    assert!(matches!(
        p.response(&r, br#"{"choices":[]}"#),
        Err(Fail::Malformed(_))
    ));
}

#[test]
fn openai_stream_with_escapes_and_usage() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Schema, None);
    let mut s = chat(Host::OpenAi).stream(&r);
    let mut seen = String::new();
    for e in [
        r#"{"choices":[{"delta":{"content":"{\"n\":\"a\\\"b"}}]}"#,
        r#"{"choices":[{"delta":{"content":"\"}"},"finish_reason":"stop"}]}"#,
        r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}"#,
        "[DONE]",
    ] {
        s.event(e, &mut |d| seen.push_str(d));
    }
    let a = s.finish().unwrap();
    assert_eq!(a.content, r#"{"n":"a\"b"}"#);
    assert_eq!(seen, a.content);
    assert_eq!(a.usage.total_tokens(), 7);
}

#[test]
fn anthropic_cache_counters_are_summed_into_input() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Schema, None);
    let p = provider(Wire::Anthropic);
    let resp = json!({"content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}});
    let a = p.response(&r, resp.to_string().as_bytes()).unwrap();
    assert_eq!(a.usage.input_tokens, 130);
    assert_eq!(a.usage.cached_input_tokens, 100);
    assert_eq!(a.usage.cache_write_tokens, 20);
}

#[test]
fn anthropic_tool_answer_and_prose_fallback() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Tool, None);
    let p = provider(Wire::Anthropic);
    let b = body(&p, &r);
    assert_eq!(b["tool_choice"], json!({"type": "tool", "name": "extract"}));
    let resp = json!({"content": [{"type": "tool_use", "id": "t", "name": "extract",
        "input": {"name": "a\u{1F600}"}}], "stop_reason": "tool_use"});
    let a = p.response(&r, resp.to_string().as_bytes()).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&a.content).unwrap(),
        json!({"name": "a\u{1F600}"})
    );
    let resp =
        json!({"content": [{"type": "text", "text": "I think..."}], "stop_reason": "end_turn"});
    assert_eq!(
        p.response(&r, resp.to_string().as_bytes()).unwrap().content,
        "I think..."
    );
    // thinking on: the tool is offered, not forced
    let r = req(&m, OutputMode::Tool, Some(Reasoning::Budget(1024)));
    let b = body(&p, &r);
    assert_eq!(b["tool_choice"], json!({"type": "auto"}));
    assert!(b["system"].as_str().unwrap().ends_with("`extract` tool."));
}

#[test]
fn anthropic_prompt_mode_and_documents() {
    let m = [Message::user("x").with(Part::document("application/pdf", "JVBE"))];
    let r = req(&m, OutputMode::JsonObject, None);
    let b = body(&provider(Wire::Anthropic), &r);
    assert!(b.get("tools").is_none() && b.get("output_config").is_none());
    assert!(b["system"].as_str().unwrap().contains("JSON Schema"));
    assert_eq!(b["messages"][0]["content"][0]["type"], "document");
    assert_eq!(b["messages"][0]["content"][1]["text"], "x");
}

#[test]
fn gemini_always_sends_json_schema() {
    let m = [Message::user("x")];
    let b = body(&provider(Wire::Gemini), &req(&m, OutputMode::Schema, None));
    let g = &b["generationConfig"];
    assert!(g.get("responseSchema").is_none());
    assert_eq!(g["responseJsonSchema"]["type"], "object");
    assert_eq!(g["responseMimeType"], "application/json");
    let r = req(&m, OutputMode::Tool, None);
    assert!(matches!(
        provider(Wire::Gemini).request(&r),
        Err(Fail::Unsupported(_))
    ));
}

#[test]
fn gemini_usage_and_safety() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Schema, None);
    let p = provider(Wire::Gemini);
    let resp = json!({"candidates": [{"content": {"parts": [
            {"text": "thinking", "thought": true}, {"text": "{}"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                          "thoughtsTokenCount": 7, "cachedContentTokenCount": 4}});
    let a = p.response(&r, resp.to_string().as_bytes()).unwrap();
    assert_eq!(a.content, "{}");
    assert_eq!((a.usage.output_tokens, a.usage.reasoning_tokens), (12, 7));
    assert_eq!(a.usage.cached_input_tokens, 4);
    let resp = json!({"candidates": [{"finishReason": "SAFETY"}]});
    assert!(matches!(
        p.response(&r, resp.to_string().as_bytes()),
        Err(Fail::Refusal(..))
    ));
    let resp = json!({"promptFeedback": {"blockReason": "OTHER"}});
    assert!(matches!(
        p.response(&r, resp.to_string().as_bytes()),
        Err(Fail::Refusal(..))
    ));
}

#[test]
fn responses_request_shape() {
    let m = [
        Message::user("earlier"),
        Message::assistant("noted"),
        Message::user("x").with(Part::image_url("https://x/i.png")),
    ];
    let mut r = req(&m, OutputMode::Schema, Some(Reasoning::Effort(Effort::Low)));
    r.system = Some("sys");
    r.stream = true;
    let p = provider(Wire::OpenAiResponses);
    let hr = p.request(&r).unwrap();
    assert_eq!(hr.url, "http://h/responses");
    assert_eq!(hr.headers.get("Authorization"), Some("Bearer k"));
    let b: Value = serde_json::from_slice(&hr.body).unwrap();
    assert_eq!(b["instructions"], "sys");
    assert_eq!(b["input"][1]["content"][0]["type"], "output_text");
    assert_eq!(b["input"][2]["content"][1]["type"], "input_image");
    assert_eq!(b["text"]["format"]["type"], "json_schema");
    assert_eq!(b["text"]["format"]["name"], "C");
    assert_eq!(b["text"]["format"]["strict"], true);
    assert_eq!(b["text"]["format"]["schema"]["additionalProperties"], false);
    assert_eq!(b["reasoning"], json!({"effort": "low"}));
    assert_eq!(b["max_output_tokens"], 9);
    assert_eq!(b["stream"], true);
    let r = req(&m, OutputMode::Schema, Some(Reasoning::Budget(9)));
    assert!(matches!(p.request(&r), Err(Fail::Unsupported(_))));
}

#[test]
fn responses_answers() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Schema, None);
    let p = provider(Wire::OpenAiResponses);
    let resp = json!({"status": "completed", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "{\"name\":\"a\"}"}]}],
        "usage": {"input_tokens": 9, "output_tokens": 8,
                  "input_tokens_details": {"cached_tokens": 2},
                  "output_tokens_details": {"reasoning_tokens": 3}}});
    let a = p.response(&r, resp.to_string().as_bytes()).unwrap();
    assert_eq!(a.content, r#"{"name":"a"}"#);
    assert_eq!(a.usage.reasoning_tokens, 3);
    assert_eq!(a.usage.cached_input_tokens, 2);
    let resp = json!({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"na"}]}]});
    let a = p.response(&r, resp.to_string().as_bytes()).unwrap();
    assert_eq!(
        a.stop,
        Stop::Truncated("incomplete: max_output_tokens".into())
    );
    let resp =
        json!({"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]});
    assert_eq!(
        p.response(&r, resp.to_string().as_bytes()),
        Err(Fail::Refusal("no".into(), Usage::default()))
    );
    let mut s = p.stream(&r);
    let mut seen = String::new();
    for e in [
        r#"{"type":"response.output_text.delta","delta":"{\"name\":"}"#,
        r#"{"type":"response.output_text.delta","delta":"\"a\"}"}"#,
        r#"{"type":"response.completed","response":{"status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":2}}}"#,
    ] {
        s.event(e, &mut |d| seen.push_str(d));
    }
    let a = s.finish().unwrap();
    assert_eq!(a.content, r#"{"name":"a"}"#);
    assert_eq!(seen, a.content);
    assert_eq!(a.usage.total_tokens(), 3);
}
