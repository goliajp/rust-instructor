// Per-provider and per-model request details: default models, when the
// tool is forced, what each host refuses before sending.

use instructors::{
    Client, Effort, Error, Headers, Message, OutputMode, Part, Reasoning, Schema, Step,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, Schema)]
#[expect(dead_code, reason = "only the schema is used")]
struct Contact {
    name: String,
}

fn first(c: &Client, mode: OutputMode, r: Option<Reasoning>) -> Value {
    let mut b = c.extract::<Contact>("x").output_mode(mode);
    if let Some(r) = r {
        b = b.reasoning(r);
    }
    match b.exchange().next() {
        Step::Send(req) => serde_json::from_slice(&req.body).unwrap(),
        Step::Done(Err(e)) => panic!("{e}"),
        _ => panic!("no request"),
    }
}

#[test]
fn deepseek_and_qwen_have_their_own_defaults() {
    let d = Client::deepseek("k");
    let Step::Send(req) = d.extract::<Contact>("x").exchange().next() else {
        panic!()
    };
    assert_eq!(req.url, "https://api.deepseek.com/chat/completions");
    let b: Value = serde_json::from_slice(&req.body).unwrap();
    assert_eq!(b["model"], "deepseek-flash");
    assert_eq!(b["response_format"], json!({"type": "json_object"}));
    assert!(b.get("temperature").is_none() && b.get("thinking").is_none());
    let q = Client::qwen("k");
    let Step::Send(req) = q.extract::<Contact>("x").exchange().next() else {
        panic!()
    };
    assert!(
        req.url.starts_with("https://dashscope-intl.aliyuncs.com/"),
        "{}",
        req.url
    );
    let b: Value = serde_json::from_slice(&req.body).unwrap();
    assert_eq!(b["model"], "qwen3.7-plus");
    assert_eq!(b["response_format"], json!({"type": "json_object"}));
    assert!(b.get("enable_thinking").is_none());
    // any aliyuncs.com host, such as a workspace domain, gets the same model
    let w = Client::openai_compatible(
        "k",
        "https://ws1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    );
    assert_eq!(
        first(&w, OutputMode::JsonObject, None)["model"],
        "qwen3.7-plus"
    );
    assert_eq!(
        first(&Client::openai("k"), OutputMode::Schema, None)["model"],
        "gpt-5.6-terra"
    );
}

#[test]
fn deepseek_and_qwen_force_the_tool_only_with_thinking_off() {
    for c in [Client::deepseek("k"), Client::qwen("k")] {
        for r in [None, Some(Reasoning::Effort(Effort::High))] {
            let b = first(&c, OutputMode::Tool, r);
            assert_eq!(b["tool_choice"], "auto", "{r:?}");
            let system = b["messages"][0]["content"].as_str().unwrap();
            assert!(
                system.ends_with("Answer by calling the `extract` tool."),
                "{system}"
            );
        }
        let b = first(&c, OutputMode::Tool, Some(Reasoning::Off));
        assert_eq!(b["tool_choice"]["function"]["name"], "extract");
    }
    let b = first(&Client::openai("k"), OutputMode::Tool, None);
    assert_eq!(b["tool_choice"]["function"]["name"], "extract");
}

#[test]
fn qwen_refuses_documents_before_sending() {
    let mut ex = Client::qwen("k")
        .extract::<Contact>("x")
        .messages([Message::user("see").with(Part::document("application/pdf", "JVBE"))])
        .exchange();
    match ex.next() {
        Step::Done(Err(Error::Unsupported { provider, what })) => {
            assert_eq!(provider, "qwen");
            assert!(what.contains("documents"), "{what}");
        }
        _ => panic!("expected Unsupported"),
    }
}

#[test]
fn anthropic_model_table() {
    let c = Client::anthropic("k");
    let with = |model: &str, mode, r| first(&c.clone().with_model(model), mode, r);
    // structured output for current and unknown models
    for m in ["claude-sonnet-5", "claude-opus-5-5", "claude-next-9"] {
        let b = with(m, OutputMode::Schema, None);
        assert!(b["output_config"]["format"].is_object(), "{m}");
        assert!(b.get("tools").is_none(), "{m}");
    }
    // models without it get the tool
    for m in [
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-1",
        "claude-sonnet-4-20250514",
    ] {
        let b = with(m, OutputMode::Schema, None);
        assert!(b.get("output_config").is_none(), "{m}");
        assert_eq!(
            b["tool_choice"],
            json!({"type": "tool", "name": "extract"}),
            "{m}"
        );
    }
    // models that reject a forced tool, and thinking, get it offered
    for (m, r) in [
        ("claude-sonnet-5-5", None),
        ("claude-fable-5-2-20261001", None),
        ("claude-sonnet-5", Some(Reasoning::Effort(Effort::Low))),
        ("claude-sonnet-5", Some(Reasoning::Budget(2048))),
    ] {
        let b = with(m, OutputMode::Tool, r);
        assert_eq!(b["tool_choice"], json!({"type": "auto"}), "{m} {r:?}");
        assert!(
            b["system"].as_str().unwrap().ends_with("`extract` tool."),
            "{m}"
        );
    }
    let b = with("claude-sonnet-5", OutputMode::Tool, Some(Reasoning::Off));
    assert_eq!(b["tool_choice"], json!({"type": "tool", "name": "extract"}));
    // a longer id is not the same model
    let b = with("claude-sonnet-5-50", OutputMode::Tool, None);
    assert_eq!(b["tool_choice"]["type"], "tool");
}

#[test]
fn no_temperature_unless_set() {
    for c in [
        Client::openai("k"),
        Client::deepseek("k"),
        Client::gemini("k"),
    ] {
        let mode = OutputMode::JsonObject;
        let b = first(&c, mode, None);
        assert!(b.get("temperature").is_none(), "{b}");
        assert!(b["generationConfig"].get("temperature").is_none(), "{b}");
        let b = first(&c.with_temperature(0.2), mode, None);
        let t = b
            .get("temperature")
            .or(b["generationConfig"].get("temperature"));
        assert_eq!(t, Some(&json!(0.2)));
    }
}

#[test]
fn gemini_continuation_is_a_truncation() {
    let mut ex = Client::gemini("k").extract::<Contact>("x").exchange();
    let Step::Send(_) = ex.next() else { panic!() };
    let body = json!({"candidates": [{"content": {"parts": [{"text": "{\"name\": \"A"}]},
                                      "finishReason": "CONTINUATION"}]});
    ex.receive(200, &Headers::new(), body.to_string().as_bytes());
    match ex.next() {
        Step::Done(Err(e)) => assert!(e.is_truncated(), "{e}"),
        _ => panic!("expected a truncation error"),
    }
}
