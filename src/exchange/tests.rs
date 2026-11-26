//! The state machine, scripted: no HTTP, no runtime.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::{Client, Schema};

#[derive(Debug, serde::Deserialize, Schema, PartialEq)]
struct C {
    name: String,
    email: Option<String>,
}

fn openai(content: &str) -> Vec<u8> {
    json!({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    .to_string()
    .into_bytes()
}

fn client() -> Client {
    Client::openai_compatible("k", "http://a").with_retry(RetryPolicy::none())
}

fn send<T>(ex: &mut Exchange<T>) -> HttpRequest
where
    T: serde::de::DeserializeOwned + std::fmt::Debug,
{
    match ex.next() {
        Step::Send(r) => r,
        other => panic!("expected a request, got {other:?}"),
    }
}

fn done<T>(ex: &mut Exchange<T>) -> Result<Extracted<T>>
where
    T: serde::de::DeserializeOwned + std::fmt::Debug,
{
    match ex.next() {
        Step::Done(r) => r,
        other => panic!("expected the end, got {other:?}"),
    }
}

fn body(r: &HttpRequest) -> Value {
    serde_json::from_slice(&r.body).unwrap()
}

const OK: &str = r#"{"name":"Ada","email":null}"#;

#[test]
fn one_request() {
    let mut ex = client().extract::<C>("x").exchange();
    let r = send(&mut ex);
    assert_eq!(r.url, "http://a/chat/completions");
    ex.receive(200, &Headers::new(), &openai(OK));
    let out = done(&mut ex).unwrap();
    assert_eq!(out.value.name, "Ada");
    assert_eq!(out.usage.requests, 1);
    assert_eq!(out.usage.total_tokens(), 15);
    assert!(!out.truncated);
}

#[test]
fn retry_after_is_honoured() {
    let mut ex = Client::openai_compatible("k", "http://a")
        .extract::<C>("x")
        .exchange();
    send(&mut ex);
    let mut h = Headers::new();
    h.push("retry-after", "2");
    ex.receive(429, &h, b"slow down");
    assert!(matches!(ex.next(), Step::Wait(d) if d == Duration::from_secs(2)));
    send(&mut ex);
    ex.receive(200, &Headers::new(), &openai(OK));
    assert_eq!(done(&mut ex).unwrap().usage.requests, 1);
}

#[test]
fn http_errors_end_after_the_retries() {
    let p = RetryPolicy::new().max_retries(1).jitter(false);
    let mut ex = client().with_retry(p).extract::<C>("x").exchange();
    send(&mut ex);
    ex.receive(503, &Headers::new(), b"busy");
    assert!(matches!(ex.next(), Step::Wait(d) if d == Duration::from_millis(500)));
    send(&mut ex);
    ex.receive(503, &Headers::new(), b"busy");
    match done(&mut ex) {
        Err(Error::Api {
            status: 503,
            message,
            ..
        }) => assert_eq!(message, "busy"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn bad_answer_is_shown_back_to_the_model() {
    let mut ex = client().extract::<C>("x").exchange();
    send(&mut ex);
    ex.receive(200, &Headers::new(), &openai("{\"nom\": 1}"));
    let second = body(&send(&mut ex));
    let msgs = second["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 4, "{msgs:?}");
    assert_eq!(msgs[1]["content"], "x");
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["content"], "{\"nom\": 1}");
    assert_eq!(msgs[3]["role"], "user");
    let turn = msgs[3]["content"].as_str().unwrap();
    assert!(turn.contains("missing field `name`"), "{turn}");
    ex.receive(200, &Headers::new(), &openai(OK));
    let out = done(&mut ex).unwrap();
    assert_eq!(out.usage.requests, 2);
    assert_eq!(out.usage.retries(), 1);
}

#[test]
fn parse_errors_end_after_max_attempts_with_usage() {
    let mut ex = client().with_max_attempts(2).extract::<C>("x").exchange();
    for _ in 0..2 {
        send(&mut ex);
        ex.receive(200, &Headers::new(), &openai("nope"));
    }
    let e = done(&mut ex).unwrap_err();
    match &e {
        Error::Parse { attempts, raw, .. } => {
            assert_eq!(*attempts, 2);
            assert_eq!(raw, "nope");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(e.usage().unwrap().requests, 2);
    assert_eq!(e.usage().unwrap().total_tokens(), 30);
}

#[test]
fn validation_failure() {
    let mut ex = client()
        .with_max_attempts(1)
        .extract::<C>("x")
        .validate(|c| Err(format!("{} is not allowed", c.name).into()))
        .exchange();
    send(&mut ex);
    ex.receive(200, &Headers::new(), &openai(OK));
    match done(&mut ex) {
        Err(Error::Validation { message, .. }) => assert_eq!(message, "Ada is not allowed"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn refusal_is_not_retried() {
    let mut ex = client().extract::<C>("x").exchange();
    send(&mut ex);
    let b = json!({"choices": [{"message": {"content": null, "refusal": "no"}}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}});
    ex.receive(200, &Headers::new(), b.to_string().as_bytes());
    let e = done(&mut ex).unwrap_err();
    assert_eq!(e.refusal(), Some("no"));
    assert_eq!(e.usage().unwrap().requests, 1);
    assert_eq!(e.usage().unwrap().total_tokens(), 2);
}

fn cut(content: &str) -> Vec<u8> {
    json!({"choices": [{"message": {"content": content}, "finish_reason": "length"}]})
        .to_string()
        .into_bytes()
}

#[test]
fn truncation_fails_unless_accepted() {
    let mut ex = client().extract::<C>("x").exchange();
    send(&mut ex);
    ex.receive(200, &Headers::new(), &cut(r#"{"name":"Ada","email":"ad"#));
    let e = done(&mut ex).unwrap_err();
    assert!(e.is_truncated());

    let mut ex = client().extract::<C>("x").accept_truncated().exchange();
    send(&mut ex);
    ex.receive(200, &Headers::new(), &cut(r#"{"name":"Ada","email":"ad"#));
    let out = done(&mut ex).unwrap();
    assert!(out.truncated);
    assert_eq!(out.value.name, "Ada");
}

#[test]
fn fallback_chain() {
    let primary = client();
    let second = Client::anthropic_compatible("k", "http://b").with_retry(RetryPolicy::none());
    let mut ex = primary.with_fallback(second).extract::<C>("x").exchange();
    send(&mut ex);
    ex.receive(500, &Headers::new(), b"boom");
    assert_eq!(ex.provider_index(), 1);
    let r = send(&mut ex);
    assert_eq!(r.url, "http://b/messages");
    let b = json!({"content": [], "stop_reason": "refusal",
                   "usage": {"input_tokens": 3, "output_tokens": 1}});
    ex.receive(200, &Headers::new(), b.to_string().as_bytes());
    let e = done(&mut ex).unwrap_err();
    let chain: Vec<&str> = e.chain().map(|x| x.provider().unwrap()).collect();
    assert_eq!(chain, ["anthropic", "openai-compatible"]);
    assert_eq!(e.refusal(), Some("stop_reason: refusal"));
    let src = std::error::Error::source(&e).unwrap().to_string();
    assert!(src.contains("boom"), "{src}");
}

#[test]
fn fallback_success_counts_every_provider() {
    let second = Client::gemini_compatible("k", "http://g");
    let mut ex = client()
        .with_max_attempts(1)
        .with_fallback(second)
        .extract::<C>("x")
        .exchange();
    send(&mut ex);
    ex.receive(200, &Headers::new(), &openai("nope"));
    let r = send(&mut ex);
    assert!(r.url.starts_with("http://g/models/"), "{}", r.url);
    let g = json!({"candidates": [{"content": {"parts": [{"text": OK}]}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}});
    ex.receive(200, &Headers::new(), g.to_string().as_bytes());
    let out = done(&mut ex).unwrap();
    assert_eq!(out.usage.requests, 2);
    assert_eq!(out.usage.total_tokens(), 17);
}

#[test]
fn unsupported_setting_sends_nothing() {
    let mut ex = client()
        .extract::<C>("x")
        .reasoning(crate::Reasoning::Budget(10))
        .exchange();
    assert!(matches!(
        done(&mut ex),
        Err(Error::Unsupported {
            provider: "openai-compatible",
            ..
        })
    ));
}

#[test]
fn retryable_transport_error_waits() {
    let mut ex = Client::openai_compatible("k", "http://a")
        .with_retry(RetryPolicy::new().jitter(false))
        .extract::<C>("x")
        .exchange();
    send(&mut ex);
    ex.receive_error(TransportError::new("refused").retryable());
    assert!(matches!(ex.next(), Step::Wait(_)));
    send(&mut ex);
    ex.receive_error(TransportError::new("bad"));
    assert!(matches!(done(&mut ex), Err(Error::Transport { .. })));
}

#[test]
fn streamed_in_pieces() {
    let deltas = Arc::new(Mutex::new(String::new()));
    let d2 = deltas.clone();
    let mut ex = client()
        .extract::<C>("x")
        .on_delta(move |d| d2.lock().unwrap().push_str(d))
        .exchange();
    let r = send(&mut ex);
    assert!(r.stream);
    ex.receive_head(200, &Headers::new());
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"name\\\":\"}}]}\n\n\
               data: {\"choices\":[{\"delta\":{\"content\":\"\\\"Ada\\\"}\"},\"finish_reason\":\"stop\"}]}\n\n\
               data: [DONE]\n\n";
    for piece in sse.as_bytes().chunks(7) {
        ex.receive_chunk(piece);
    }
    ex.receive_end();
    let out = done(&mut ex).unwrap();
    assert_eq!(out.value.name, "Ada");
    assert_eq!(*deltas.lock().unwrap(), r#"{"name":"Ada"}"#);
}

#[test]
fn items_arrive_as_they_complete() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    let mut ex = client()
        .extract_many::<C>("x")
        .on_item(move |c: C| s2.lock().unwrap().push(c.name))
        .exchange();
    send(&mut ex);
    ex.receive_head(200, &Headers::new());
    let content = r#"{"items":[{"name":"a","email":null},{"name":"b","email":null}]}"#;
    for (i, piece) in content.as_bytes().chunks(5).enumerate() {
        let piece = std::str::from_utf8(piece).unwrap();
        let ev = json!({"choices": [{"delta": {"content": piece}}]});
        ex.receive_chunk(format!("data: {ev}\n\n").as_bytes());
        if i == 6 {
            assert_eq!(*seen.lock().unwrap(), ["a"]);
        }
    }
    ex.receive_end();
    assert_eq!(done(&mut ex).unwrap().value.len(), 2);
    assert_eq!(*seen.lock().unwrap(), ["a", "b"]);
}

#[test]
fn misuse_is_an_error_not_a_panic() {
    let mut ex = client().extract::<C>("x").exchange();
    send(&mut ex);
    assert!(matches!(
        ex.next(),
        Step::Done(Err(Error::Transport { .. }))
    ));
    ex.receive(200, &Headers::new(), &openai(OK));
    assert!(done(&mut ex).is_ok());
    assert!(done(&mut ex).is_err());
}
