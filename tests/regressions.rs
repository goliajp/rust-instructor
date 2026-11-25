// Request and response handling defects found in 1.4.1, one test each.

mod support;

use instructors::Client;
use instructors::{OutputMode, RetryPolicy, Schema};
use serde::Deserialize;
use serde_json::json;
use support::{Reply, anthropic_text, gemini_ok, openai_ok, start};

#[derive(Debug, Deserialize, Schema)]
struct Contact {
    name: String,
}

#[derive(Debug, Deserialize, Schema)]
struct Node {
    name: String,
    children: Vec<Node>,
}

#[derive(Debug, Deserialize, Schema)]
struct Doc {
    title: String,
    root: Node,
}

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, Schema)]
struct Article {
    title: String,
    format: String,
}

const DOC: &str = r#"{"title":"t","root":{"name":"a","children":[{"name":"b","children":[]}]}}"#;

#[tokio::test]
async fn recursive_type_inside_a_struct_does_not_overflow() {
    let (base, log) = start(vec![Reply::json("/chat", 200, openai_ok(DOC))]).await;
    let r = Client::openai_compatible("k", base)
        .extract::<Doc>("x")
        .await
        .unwrap();
    assert_eq!(r.value.root.children[0].name, "b");
    let schema = &log.body(0)["response_format"]["json_schema"]["schema"];
    assert_eq!(
        schema["$defs"]["Node"]["properties"]["children"]["items"]["$ref"],
        "#/$defs/Node"
    );
}

#[tokio::test]
async fn recursive_type_on_anthropic_structured_output_uses_a_tool() {
    let tool = json!({
        "content": [{"type": "tool_use", "id": "t", "name": "extract",
                     "input": serde_json::from_str::<serde_json::Value>(DOC).unwrap()}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let (base, log) = start(vec![Reply::json("/messages", 200, tool)]).await;
    let r = Client::anthropic_compatible("k", base)
        .with_output_mode(OutputMode::Schema)
        .extract::<Doc>("x")
        .await
        .unwrap();
    assert_eq!(r.value.title, "t");
    let body = log.body(0);
    assert!(body.get("output_config").is_none());
    assert_eq!(body["tool_choice"], json!({"type": "auto"}));
}

#[tokio::test]
async fn recursive_type_on_gemini_uses_json_schema() {
    let (base, log) = start(vec![Reply::json(":generate", 200, gemini_ok(DOC))]).await;
    Client::gemini_compatible("k", base)
        .extract::<Doc>("x")
        .await
        .unwrap();
    let cfg = &log.body(0)["generationConfig"];
    assert!(cfg.get("responseSchema").is_none());
    assert!(cfg["responseJsonSchema"]["$defs"]["Node"].is_object());
}

#[tokio::test]
async fn fields_named_title_and_format_are_sent() {
    let (base, log) = start(vec![Reply::json(
        "/chat",
        200,
        openai_ok(r#"{"title":"a","format":"b"}"#),
    )])
    .await;
    Client::openai_compatible("k", base)
        .extract::<Article>("x")
        .await
        .unwrap();
    let schema = &log.body(0)["response_format"]["json_schema"]["schema"];
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("title")) && required.contains(&json!("format")));
    assert!(schema["properties"]["title"].is_object());
}

#[tokio::test]
async fn max_tokens_reaches_openai_compatible_apis() {
    let (base, log) = start(vec![Reply::json(
        "/chat",
        200,
        openai_ok(r#"{"name":"a"}"#),
    )])
    .await;
    let c = Client::openai_compatible("k", base);
    c.extract::<Contact>("x").max_tokens(77).await.unwrap();
    c.extract::<Contact>("x").await.unwrap();
    assert_eq!(log.body(0)["max_tokens"], 77);
    assert!(log.body(0).get("max_completion_tokens").is_none());
    // unset stays unset, as in 1.4
    assert!(log.body(1).get("max_tokens").is_none());
}

#[tokio::test]
async fn extract_many_sends_an_object_root() {
    let (base, log) = start(vec![Reply::json(
        "/chat",
        200,
        openai_ok(r#"{"items":[{"name":"a"},{"name":"b"}]}"#),
    )])
    .await;
    let r = Client::openai_compatible("k", base)
        .extract_many::<Contact>("x")
        .await
        .unwrap();
    assert_eq!(r.value.len(), 2);
    let js = &log.body(0)["response_format"]["json_schema"];
    assert_eq!(js["strict"], true);
    assert_eq!(js["schema"]["type"], "object");
    assert_eq!(js["schema"]["properties"]["items"]["type"], "array");
}

fn split_utf8_stream() -> Vec<Vec<u8>> {
    let mut first = b"data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"name\\\":\\\"".to_vec();
    first.push(0xe6); // first byte of U+6771
    let mut second = vec![0x9d, 0xb1];
    second.extend_from_slice(b"\\\"}\"}}]}\n\n");
    let usage = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\ndata: [DONE]\n\n".to_vec();
    vec![first, second, usage]
}

#[tokio::test]
async fn stream_keeps_a_character_split_across_reads() {
    let (base, _) = start(vec![Reply::sse("/chat", split_utf8_stream())]).await;
    let r = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .on_delta(|_| {})
        .await
        .unwrap();
    assert_eq!(r.value.name, "東");
}

#[tokio::test]
async fn openai_stream_asks_for_and_counts_usage() {
    let (base, log) = start(vec![Reply::sse("/chat", split_utf8_stream())]).await;
    let r = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .on_delta(|_| {})
        .await
        .unwrap();
    assert_eq!(
        log.body(0)["stream_options"],
        json!({"include_usage": true})
    );
    assert_eq!(r.usage.input_tokens, 11);
    assert_eq!(r.usage.output_tokens, 7);
}

#[tokio::test]
async fn fallback_sends_its_own_model_and_reports_every_error() {
    let (base, log) = start(vec![
        Reply::json("/chat", 500, json!({"error": "primary down"})),
        Reply::json("/messages", 400, json!({"error": "fallback rejected"})),
    ])
    .await;
    let err = Client::openai_compatible("k", base.clone())
        .with_model("gpt-5.6-terra")
        .with_retry(RetryPolicy::none())
        .with_fallback(Client::anthropic_compatible("k", base))
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    let reqs = log.requests();
    assert_eq!(reqs[0].1["model"], "gpt-5.6-terra");
    assert_eq!(reqs[1].1["model"], "claude-sonnet-5");
    // each provider's error is one link of the chain, the last tried first
    let texts: Vec<String> = err.chain().map(ToString::to_string).collect();
    assert_eq!(texts.len(), 2, "{texts:?}");
    assert!(texts[0].contains("fallback rejected"), "{texts:?}");
    assert!(texts[1].contains("primary down"), "{texts:?}");
    assert!(matches!(err, instructors::Error::Fallback { .. }));
    let links: Vec<&instructors::Error> = err.chain().collect();
    assert!(matches!(
        links[1],
        instructors::Error::Api { status: 500, .. }
    ));
}

#[tokio::test]
async fn gemini_counts_thinking_tokens_as_output() {
    let body = json!({
        "candidates": [{"content": {"parts": [
            {"text": "planning...", "thought": true},
            {"text": "{\"name\":\"a\"}"}
        ]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 900}
    });
    let (base, _) = start(vec![Reply::json(":generate", 200, body)]).await;
    let r = Client::gemini_compatible("k", base)
        .extract::<Contact>("x")
        .await
        .unwrap();
    assert_eq!(r.value.name, "a");
    assert_eq!(r.usage.output_tokens, 905);
}

#[tokio::test]
async fn gemini_key_stays_out_of_urls_and_errors() {
    let (base, log) = start(vec![Reply {
        path_contains: ":generate",
        status: 200,
        content_type: "application/json",
        chunks: vec![b"not json".to_vec()],
    }])
    .await;
    let err = Client::gemini_compatible("AIzaSECRETKEY", base)
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    assert!(!err.to_string().contains("SECRET"), "{err}");
    assert!(!log.requests()[0].0.contains("SECRET"));
}

#[tokio::test]
async fn openai_refusal_is_reported_and_not_retried() {
    let body = json!({
        "choices": [{"message": {"role": "assistant", "content": null,
                                 "refusal": "I can't help with that."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    let (base, log) = start(vec![Reply::json("/chat", 200, body)]).await;
    let err = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    assert_eq!(err.refusal(), Some("I can't help with that."));
    assert_eq!(log.requests().len(), 1);
}

#[tokio::test]
async fn gemini_max_tokens_is_reported_as_truncation() {
    let body = json!({
        "candidates": [{"finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"promptTokenCount": 10, "thoughtsTokenCount": 16384}
    });
    let (base, log) = start(vec![Reply::json(":generate", 200, body)]).await;
    let err = Client::gemini_compatible("k", base)
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    assert!(err.is_truncated(), "{err}");
    assert_eq!(log.requests().len(), 1);
}

#[tokio::test]
async fn openai_length_cut_is_truncation_not_a_repaired_value() {
    let body = json!({
        "choices": [{"message": {"role": "assistant", "content": "{\"name\":\"Ada Lov"},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    let (base, _) = start(vec![Reply::json("/chat", 200, body)]).await;
    let err = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    assert!(err.is_truncated(), "{err}");
}

#[tokio::test]
async fn anthropic_refusal_stop_reason() {
    let body = json!({
        "content": [{"type": "text", "text": "I won't do that."}],
        "stop_reason": "refusal",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let (base, _) = start(vec![Reply::json("/messages", 200, body)]).await;
    let err = Client::anthropic_compatible("k", base)
        .with_output_mode(OutputMode::Schema)
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    assert_eq!(err.refusal(), Some("I won't do that."));
}

#[tokio::test]
async fn anthropic_structured_output_strips_unsupported_keywords() {
    #[derive(Deserialize, Schema)]
    #[expect(dead_code, reason = "fields are only filled by deserialization")]
    struct Item {
        qty: u32,
        price: f64,
    }
    let (base, log) = start(vec![Reply::json(
        "/messages",
        200,
        anthropic_text(r#"{"qty":1,"price":2.5}"#),
    )])
    .await;
    Client::anthropic_compatible("k", base)
        .with_output_mode(OutputMode::Schema)
        .extract::<Item>("x")
        .await
        .unwrap();
    let schema = &log.body(0)["output_config"]["format"]["schema"];
    assert_eq!(schema["additionalProperties"], false);
    let qty = &schema["properties"]["qty"];
    assert!(
        qty.get("minimum").is_none() && qty.get("format").is_none(),
        "{qty}"
    );
}

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, Schema)]
struct Reasoned {
    reasoning: String,
    answer: String,
    confidence: f64,
}

#[tokio::test]
async fn schema_properties_keep_declaration_order() {
    const R: &str = r#"{"reasoning":"r","answer":"a","confidence":1}"#;
    let (base, log) = start(vec![
        Reply::json("/chat", 200, openai_ok(R)),
        Reply::json("/messages", 200, anthropic_text(R)),
        Reply::json(":generate", 200, gemini_ok(R)),
    ])
    .await;
    Client::openai_compatible("k", base.clone())
        .extract::<Reasoned>("x")
        .await
        .unwrap();
    Client::anthropic_compatible("k", base.clone())
        .with_output_mode(OutputMode::Schema)
        .extract::<Reasoned>("x")
        .await
        .unwrap();
    Client::gemini_compatible("k", base)
        .extract::<Reasoned>("x")
        .await
        .unwrap();
    // the raw bodies, since serde_json's map sorts keys
    for i in 0..3 {
        let raw = log.raw(i);
        let at = |k: &str| raw.find(&format!("\"{k}\":")).unwrap();
        assert!(
            at("reasoning") < at("answer") && at("answer") < at("confidence"),
            "{raw}"
        );
    }
    assert!(log.body(2)["generationConfig"]["responseJsonSchema"].is_object());
}
