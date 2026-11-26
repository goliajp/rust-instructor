// Streamed answers: errors inside the stream and the Anthropic tool plan.

use super::*;

#[test]
fn errors_inside_a_stream() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Schema, None);
    let cases = [
        (
            chat(Host::OpenAi),
            r#"{"error":{"message":"overloaded","type":"server_error"}}"#,
        ),
        (
            provider(Wire::Anthropic),
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ),
        (
            provider(Wire::OpenAiResponses),
            r#"{"type":"error","code":"server_error","message":"boom","param":null,"sequence_number":3}"#,
        ),
        (
            provider(Wire::OpenAiResponses),
            r#"{"type":"response.failed","response":{"status":"failed","output":[],"error":{"code":"server_error","message":"boom"}}}"#,
        ),
    ];
    for (p, error) in cases {
        let mut s = p.stream(&r);
        s.event(
            r#"{"type":"response.output_text.delta","delta":"{"}"#,
            &mut |_| {},
        );
        s.event(error, &mut |_| {});
        match s.finish() {
            Err(Fail::Api(m)) => assert!(
                m.contains("server_error") || m.contains("overloaded_error"),
                "{m}"
            ),
            other => panic!("{}: {other:?}", p.name()),
        }
    }
    let resp = json!({"status": "failed", "output": [], "error": {"code": "server_error", "message": "boom"}});
    assert!(matches!(
        provider(Wire::OpenAiResponses).response(&r, resp.to_string().as_bytes()),
        Err(Fail::Api(m)) if m.contains("boom")
    ));
}

#[test]
fn anthropic_streamed_tool_plan_keeps_the_prose_fallback() {
    let m = [Message::user("x")];
    let r = req(&m, OutputMode::Tool, Some(Reasoning::Off));
    let p = provider(Wire::Anthropic);
    let run = |events: &[&str]| {
        let mut s = p.stream(&r);
        let mut seen = String::new();
        events
            .iter()
            .for_each(|e| s.event(e, &mut |d| seen.push_str(d)));
        (s.finish().unwrap().content, seen)
    };
    let text = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I think..."}}"#;
    assert_eq!(run(&[text]), ("I think...".into(), String::new()));
    let start = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t","name":"extract","input":{}}}"#;
    let json = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}"#;
    assert_eq!(run(&[text, start, json]), ("{}".into(), "{}".into()));
}
