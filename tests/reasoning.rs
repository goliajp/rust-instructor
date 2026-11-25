// What each provider receives for a `Reasoning` setting, and streaming
// partial values.

mod support;

use std::sync::{Arc, Mutex};

use instructors::{Client, Effort, Reasoning};
use instructors::{OutputMode, Schema};
use serde::Deserialize;
use serde_json::json;
use support::{Reply, anthropic_text, gemini_ok, openai_ok, start};

#[derive(Debug, Deserialize, Schema)]
struct Contact {
    name: String,
}

const OK: &str = r#"{"name":"a"}"#;

#[tokio::test]
async fn nothing_is_sent_by_default() {
    let (base, log) = start(vec![
        Reply::json("/chat", 200, openai_ok(OK)),
        Reply::json("/messages", 200, anthropic_text(OK)),
        Reply::json(":generate", 200, gemini_ok(OK)),
    ])
    .await;
    Client::openai_compatible("k", base.clone())
        .extract::<Contact>("x")
        .await
        .unwrap();
    Client::anthropic_compatible("k", base.clone())
        .with_output_mode(OutputMode::Schema)
        .extract::<Contact>("x")
        .await
        .unwrap();
    Client::gemini_compatible("k", base)
        .extract::<Contact>("x")
        .await
        .unwrap();
    let oai = log.body(0);
    assert!(oai.get("reasoning_effort").is_none());
    let ant = log.body(1);
    assert!(ant.get("thinking").is_none());
    assert!(ant["output_config"].get("effort").is_none());
    assert!(
        log.body(2)["generationConfig"]
            .get("thinkingConfig")
            .is_none()
    );
}

#[tokio::test]
async fn openai_effort() {
    let (base, log) = start(vec![Reply::json("/chat", 200, openai_ok(OK))]).await;
    let c = Client::openai_compatible("k", base).with_reasoning(Reasoning::Effort(Effort::Low));
    c.extract::<Contact>("x").await.unwrap();
    c.extract::<Contact>("x")
        .reasoning(Reasoning::Off)
        .await
        .unwrap();
    assert_eq!(log.body(0)["reasoning_effort"], "low");
    assert_eq!(log.body(1)["reasoning_effort"], "none");
}

#[tokio::test]
async fn openai_budget_is_an_error_before_sending() {
    let (base, log) = start(vec![Reply::json("/chat", 200, openai_ok(OK))]).await;
    let err = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .reasoning(Reasoning::Budget(1000))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Reasoning::Effort"), "{err}");
    assert!(log.requests().is_empty());
}

#[tokio::test]
async fn anthropic_effort_budget_and_off() {
    let (base, log) = start(vec![Reply::json("/messages", 200, anthropic_text(OK))]).await;
    let c = Client::anthropic_compatible("k", base).with_output_mode(OutputMode::Schema);
    for r in [
        Reasoning::Effort(Effort::XHigh),
        Reasoning::Budget(4096),
        Reasoning::Off,
    ] {
        c.extract::<Contact>("x").reasoning(r).await.unwrap();
    }
    let b = log.body(0);
    assert_eq!(b["thinking"], json!({"type": "adaptive"}));
    assert_eq!(b["output_config"]["effort"], "xhigh");
    assert!(b["output_config"]["format"].is_object());
    assert_eq!(
        log.body(1)["thinking"],
        json!({"type": "enabled", "budget_tokens": 4096})
    );
    assert_eq!(log.body(2)["thinking"], json!({"type": "disabled"}));
}

#[tokio::test]
async fn anthropic_tool_mode_with_thinking_does_not_force_the_tool() {
    let tool = json!({
        "content": [{"type": "thinking", "thinking": ""},
                    {"type": "tool_use", "id": "t", "name": "extract", "input": {"name": "a"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let (base, log) = start(vec![Reply::json("/messages", 200, tool)]).await;
    let r = Client::anthropic_compatible("k", base)
        .extract::<Contact>("x")
        .reasoning(Reasoning::Effort(Effort::Low))
        .await
        .unwrap();
    assert_eq!(r.value.name, "a");
    let b = log.body(0);
    assert_eq!(b["tool_choice"], json!({"type": "auto"}));
    assert_eq!(b["output_config"], json!({"effort": "low"}));
}

#[tokio::test]
async fn gemini_level_and_budget() {
    let (base, log) = start(vec![Reply::json(":generate", 200, gemini_ok(OK))]).await;
    let c = Client::gemini_compatible("k", base);
    c.extract::<Contact>("x")
        .reasoning(Reasoning::Effort(Effort::Medium))
        .await
        .unwrap();
    c.extract::<Contact>("x")
        .reasoning(Reasoning::Budget(512))
        .await
        .unwrap();
    let cfg = |i| log.body(i)["generationConfig"]["thinkingConfig"].clone();
    assert_eq!(cfg(0), json!({"thinkingLevel": "MEDIUM"}));
    assert_eq!(cfg(1), json!({"thinkingBudget": 512}));
}

#[tokio::test]
async fn batch_passes_reasoning_through() {
    let (base, log) = start(vec![Reply::json("/chat", 200, openai_ok(OK))]).await;
    let c = Client::openai_compatible("k", base);
    let results = c
        .extract_batch::<Contact>(vec!["a".into(), "b".into()])
        .reasoning(Reasoning::Effort(Effort::High))
        .run()
        .await;
    assert!(results.iter().all(|r| r.is_ok()));
    for (_, body) in log.requests() {
        assert_eq!(body["reasoning_effort"], "high");
    }
}

fn delta_events(content: &str, size: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let chars: Vec<char> = content.chars().collect();
    for piece in chars.chunks(size) {
        let s: String = piece.iter().collect();
        let ev = json!({"choices": [{"delta": {"content": s}}]});
        out.push(format!("data: {ev}\n\n").into_bytes());
    }
    out.push(b"data: [DONE]\n\n".to_vec());
    out
}

#[tokio::test]
async fn on_partial_sees_the_value_grow() {
    #[expect(dead_code, reason = "fields are only filled by deserialization")]
    #[derive(Debug, Deserialize, Schema)]
    struct Article {
        title: String,
        tags: Vec<String>,
    }
    let content = r#"{"title":"Streaming JSON","tags":["rust","llm"]}"#;
    let (base, _) = start(vec![Reply::sse("/chat", delta_events(content, 6))]).await;
    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::default();
    let seen2 = seen.clone();
    let r = Client::openai_compatible("k", base)
        .extract::<Article>("x")
        .on_partial(move |v| {
            seen2
                .lock()
                .unwrap()
                .push(serde_json::from_str::<serde_json::Value>(&v.to_json()).unwrap())
        })
        .await
        .unwrap();
    assert_eq!(r.value.tags, ["rust", "llm"]);
    let seen = seen.lock().unwrap();
    assert!(seen.len() >= 5);
    assert!(
        seen.iter()
            .any(|v| v["title"].as_str().is_some_and(|t| t.len() < 14))
    );
    assert_eq!(
        seen.last().unwrap(),
        &json!({"title": "Streaming JSON", "tags": ["rust", "llm"]})
    );
}

#[tokio::test]
async fn on_partial_for_extract_many_shows_the_list() {
    let content = r#"{"items":[{"name":"a"},{"name":"b"}]}"#;
    let (base, _) = start(vec![Reply::sse("/chat", delta_events(content, 5))]).await;
    let last: Arc<Mutex<Option<serde_json::Value>>> = Arc::default();
    let last2 = last.clone();
    let r = Client::openai_compatible("k", base)
        .extract_many::<Contact>("x")
        .on_partial(move |v| {
            *last2.lock().unwrap() =
                Some(serde_json::from_str::<serde_json::Value>(&v.to_json()).unwrap())
        })
        .await
        .unwrap();
    assert_eq!(r.value.len(), 2);
    assert_eq!(
        last.lock().unwrap().clone().unwrap(),
        json!([{"name": "a"}, {"name": "b"}])
    );
}
