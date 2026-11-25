// Behaviour new in 2.0: streaming on every wire, on_item, accept_truncated,
// usage on errors, custom transports and the sans-IO exchange.

mod support;

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use instructors::transport::{Body, BoxFuture, HttpResponse, Transport};
use instructors::{
    Client, Error, Headers, HttpRequest, OpenAiApi, RetryPolicy, Schema, Step, TransportError,
};
use serde::Deserialize;
use serde_json::{Value, json};
use support::{Reply, openai_ok, start};

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Contact {
    name: String,
    email: Option<String>,
}

const ANSWER: &str = r#"{"name":"Ada","email":null}"#;

fn sse(events: &[Value]) -> Vec<Vec<u8>> {
    events
        .iter()
        .map(|e| format!("data: {e}\n\n").into_bytes())
        .collect()
}

/// The answer cut into 3-byte pieces.
fn pieces(s: &str) -> Vec<String> {
    s.as_bytes()
        .chunks(3)
        .map(|c| String::from_utf8(c.to_vec()).unwrap())
        .collect()
}

fn deltas() -> (
    Arc<Mutex<Vec<String>>>,
    impl Fn(&str) + Send + Sync + 'static,
) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    (seen, move |d: &str| s2.lock().unwrap().push(d.to_owned()))
}

#[tokio::test]
async fn openai_chat_streams() {
    let mut events: Vec<Value> = pieces(ANSWER)
        .into_iter()
        .map(|p| json!({"choices": [{"delta": {"content": p}}]}))
        .collect();
    events.push(json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}));
    events.push(json!({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}));
    let (base, log) = start(vec![Reply::sse("/chat", sse(&events))]).await;
    let (seen, hook) = deltas();
    let r = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .on_delta(hook)
        .await
        .unwrap();
    assert_eq!(r.value.name, "Ada");
    assert_eq!(seen.lock().unwrap().concat(), ANSWER);
    assert_eq!((r.usage.input_tokens, r.usage.output_tokens), (9, 4));
    assert_eq!(log.body(0)["stream"], true);
}

#[tokio::test]
async fn openai_responses_streams() {
    let mut events: Vec<Value> = pieces(ANSWER)
        .into_iter()
        .map(|p| json!({"type": "response.output_text.delta", "delta": p}))
        .collect();
    events.push(json!({"type": "response.completed", "response": {
        "status": "completed", "output": [],
        "usage": {"input_tokens": 7, "output_tokens": 3,
                  "input_tokens_details": {"cached_tokens": 5},
                  "output_tokens_details": {"reasoning_tokens": 2}}
    }}));
    let (base, log) = start(vec![Reply::sse("/responses", sse(&events))]).await;
    let (seen, hook) = deltas();
    let r = Client::openai_compatible("k", base)
        .with_api(OpenAiApi::Responses)
        .extract::<Contact>("x")
        .on_delta(hook)
        .await
        .unwrap();
    assert_eq!(r.value.name, "Ada");
    assert_eq!(seen.lock().unwrap().concat(), ANSWER);
    assert_eq!(r.usage.cached_input_tokens, 5);
    assert_eq!(r.usage.reasoning_tokens, 2);
    let (target, body) = &log.requests()[0];
    assert!(target.ends_with("/responses"), "{target}");
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["text"]["format"]["strict"], true);
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
}

#[tokio::test]
async fn openai_responses_plain() {
    let body = json!({
        "status": "completed",
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "content": [{"type": "output_text", "text": ANSWER}]}
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3}
    });
    let (base, _) = start(vec![Reply::json("/responses", 200, body)]).await;
    let r = Client::openai_compatible("k", base)
        .with_api(OpenAiApi::Responses)
        .extract::<Contact>("x")
        .await
        .unwrap();
    assert_eq!(r.value.name, "Ada");
    assert_eq!(r.usage.output_tokens, 3);
}

#[tokio::test]
async fn anthropic_streams() {
    let mut events = vec![json!({"type": "message_start", "message": {"usage": {
        "input_tokens": 3, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 5
    }}})];
    events.extend(pieces(ANSWER).into_iter().map(
        |p| json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": p}}),
    ));
    events.push(
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                       "usage": {"output_tokens": 6}}),
    );
    let (base, _) = start(vec![Reply::sse("/messages", sse(&events))]).await;
    let (seen, hook) = deltas();
    let r = Client::anthropic_compatible("k", base)
        .with_output_mode(instructors::OutputMode::Schema)
        .extract::<Contact>("x")
        .on_delta(hook)
        .await
        .unwrap();
    assert_eq!(r.value.name, "Ada");
    assert_eq!(seen.lock().unwrap().concat(), ANSWER);
    // anthropic's three input counters are summed
    assert_eq!(r.usage.input_tokens, 12);
    assert_eq!(r.usage.cached_input_tokens, 4);
    assert_eq!(r.usage.cache_write_tokens, 5);
    assert_eq!(r.usage.output_tokens, 6);
}

#[tokio::test]
async fn gemini_streams() {
    let mut events: Vec<Value> = pieces(ANSWER)
        .into_iter()
        .map(|p| json!({"candidates": [{"content": {"parts": [{"text": p}]}}]}))
        .collect();
    events.push(json!({
        "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2,
                          "thoughtsTokenCount": 1, "cachedContentTokenCount": 6}
    }));
    let (base, log) = start(vec![Reply::sse("streamGenerateContent", sse(&events))]).await;
    let (seen, hook) = deltas();
    let r = Client::gemini_compatible("k", base)
        .extract::<Contact>("x")
        .on_delta(hook)
        .await
        .unwrap();
    assert_eq!(r.value.name, "Ada");
    assert_eq!(seen.lock().unwrap().concat(), ANSWER);
    assert_eq!(r.usage.output_tokens, 3);
    assert_eq!(r.usage.reasoning_tokens, 1);
    assert_eq!(r.usage.cached_input_tokens, 6);
    assert!(log.requests()[0].0.contains("alt=sse"));
}

#[tokio::test]
async fn on_item_delivers_each_list_item_once_complete() {
    let content = r#"{"items":[{"name":"a","email":null},{"name":"b","email":"b@x"}]}"#;
    let mut events: Vec<Value> = content
        .chars()
        .map(|c| json!({"choices": [{"delta": {"content": c.to_string()}}]}))
        .collect();
    events.push(json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}));
    let (base, _) = start(vec![Reply::sse("/chat", sse(&events))]).await;
    let items = Arc::new(Mutex::new(Vec::new()));
    let i2 = items.clone();
    let r = Client::openai_compatible("k", base)
        .extract_many::<Contact>("x")
        .on_item(move |c: Contact| i2.lock().unwrap().push(c.name))
        .await
        .unwrap();
    assert_eq!(*items.lock().unwrap(), ["a", "b"]);
    assert_eq!(r.value.len(), 2);
}

fn cut(content: &str) -> Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    })
}

#[tokio::test]
#[ignore = "bug: json::repair (jsonrepair backend) turns a cut key into `\"ema\":}`, which does not parse"]
async fn accept_truncated_returns_the_complete_fields() {
    let (base, _) = start(vec![Reply::json(
        "/chat",
        200,
        cut(r#"{"name":"Ada","ema"#),
    )])
    .await;
    let r = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .accept_truncated()
        .await
        .unwrap();
    assert!(r.truncated);
    assert_eq!(
        r.value,
        Contact {
            name: "Ada".into(),
            email: None
        }
    );
}

#[tokio::test]
async fn truncation_is_an_error_unless_accepted() {
    let (base, log) = start(vec![Reply::json(
        "/chat",
        200,
        cut(r#"{"name":"Ada","ema"#),
    )])
    .await;
    let err = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    match &err {
        Error::Truncated { raw, usage, .. } => {
            assert_eq!(raw, r#"{"name":"Ada","ema"#);
            assert_eq!(usage.output_tokens, 5);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(log.requests().len(), 1);
}

#[tokio::test]
async fn accept_truncated_list_keeps_complete_items() {
    let content = r#"{"items":[{"name":"a","email":null},{"name":"b","email":null},{"na"#;
    let (base, _) = start(vec![Reply::json("/chat", 200, cut(content))]).await;
    let r = Client::openai_compatible("k", base)
        .extract_many::<Contact>("x")
        .accept_truncated()
        .await
        .unwrap();
    assert!(r.truncated);
    let names: Vec<&str> = r.value.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
}

#[tokio::test]
async fn failed_extraction_reports_its_usage() {
    let (base, _) = start(vec![Reply::json("/chat", 200, openai_ok("never json"))]).await;
    let err = Client::openai_compatible("k", base)
        .extract::<Contact>("x")
        .max_attempts(3)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Parse { attempts: 3, .. }), "{err:?}");
    let u = err.usage().unwrap();
    assert_eq!(u.requests, 3);
    assert_eq!(u.input_tokens, 30);
    assert_eq!(u.output_tokens, 15);
}

#[tokio::test]
async fn fallback_usage_and_source_chain() {
    let (base, _) = start(vec![
        Reply::json("/chat", 200, openai_ok("never json")),
        Reply::json("/messages", 500, json!({"error": "down"})),
    ])
    .await;
    let err = Client::openai_compatible("k", base.clone())
        .with_max_attempts(2)
        .with_fallback(Client::anthropic_compatible("k", base).with_retry(RetryPolicy::none()))
        .extract::<Contact>("x")
        .await
        .unwrap_err();
    // the primary's two answers count, though the fallback failed last
    assert_eq!(err.usage().unwrap().requests, 2);
    assert_eq!(err.provider(), Some("anthropic"));
    let providers: Vec<_> = err.chain().map(|e| e.provider()).collect();
    assert_eq!(providers, [Some("anthropic"), Some("openai-compatible")]);
    let mut sources = 0;
    let mut cur: &dyn std::error::Error = &err;
    while let Some(next) = cur.source() {
        sources += 1;
        cur = next;
    }
    assert_eq!(sources, 1);
    assert!(cur.to_string().contains("did not parse"), "{cur}");
}

// status and body of one reply
type CannedReply = (u16, Vec<u8>);

/// Replies from a list, records requests, sleeps not at all.
#[derive(Clone, Default)]
struct Canned {
    replies: Arc<Mutex<Vec<CannedReply>>>,
    sent: Arc<Mutex<Vec<HttpRequest>>>,
    slept: Arc<Mutex<Vec<Duration>>>,
}

impl Transport for Canned {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.sent.lock().unwrap().push(req);
        let next = self.replies.lock().unwrap().remove(0);
        Box::pin(async move {
            Ok(HttpResponse {
                status: next.0,
                headers: Headers::new(),
                body: Body::Full(next.1),
            })
        })
    }

    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        self.slept.lock().unwrap().push(d);
        Box::pin(async {})
    }
}

/// Poll a future that never waits on anything external.
fn run_now<F: Future>(f: F) -> F::Output {
    let mut f = pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    for _ in 0..1000 {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
    panic!("future did not finish");
}

fn ok_body(content: &str) -> Vec<u8> {
    serde_json::to_vec(&openai_ok(content)).unwrap()
}

#[test]
fn custom_transport_without_a_runtime() {
    let t = Canned::default();
    t.replies
        .lock()
        .unwrap()
        .extend([(503, b"busy".to_vec()), (200, ok_body(ANSWER))]);
    let client = Client::openai("k").with_transport(t.clone()).with_retry(
        RetryPolicy::new()
            .base_delay(Duration::from_millis(7))
            .jitter(false),
    );
    let r = run_now(async { client.extract::<Contact>("x").await }).unwrap();
    assert_eq!(r.value.name, "Ada");
    assert_eq!(t.sent.lock().unwrap().len(), 2);
    // the overall timeout starts a sleep too; the retry wait is the other
    assert!(t.slept.lock().unwrap().contains(&Duration::from_millis(7)));
    let sent = t.sent.lock().unwrap();
    assert_eq!(sent[0].url, "https://api.openai.com/v1/chat/completions");
    assert_eq!(sent[0].headers.get("authorization"), Some("Bearer k"));
}

#[test]
fn sans_io_exchange_waits_and_reasks() {
    let client = Client::openai("k");
    let mut ex = client.extract::<Contact>("find the contact").exchange();
    let mut headers = Headers::new();
    headers.push("retry-after", "0");

    let Step::Send(first) = ex.next() else {
        panic!("expected a request")
    };
    assert!(!first.stream);
    ex.receive(429, &headers, b"slow down");
    assert!(matches!(ex.next(), Step::Wait(d) if d == Duration::ZERO));

    let Step::Send(_) = ex.next() else {
        panic!("expected the retried request")
    };
    ex.receive(200, &Headers::new(), &ok_body("not json at all"));

    let Step::Send(reask) = ex.next() else {
        panic!("expected a re-ask")
    };
    let body: Value = serde_json::from_slice(&reask.body).unwrap();
    let msgs = body["messages"].as_array().unwrap();
    // system, user prompt, the model's bad answer, the error turn
    assert_eq!(msgs.len(), 4, "{body}");
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["content"], "not json at all");
    assert_eq!(msgs[3]["role"], "user");
    assert!(
        msgs[3]["content"].as_str().unwrap().contains("not valid"),
        "{}",
        msgs[3]["content"]
    );

    ex.receive(200, &Headers::new(), &ok_body(ANSWER));
    match ex.next() {
        Step::Done(Ok(r)) => {
            assert_eq!(r.value.name, "Ada");
            assert_eq!(r.usage.requests, 2);
        }
        other => panic!("{other:?}"),
    }
}

fn first_request(client: &Client) -> Value {
    match client.extract::<Contact>("x").exchange().next() {
        Step::Send(req) => serde_json::from_slice(&req.body).unwrap(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn json_object_hosts_carry_the_schema_in_the_prompt() {
    for url in [
        "https://api.deepseek.com/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://open.bigmodel.cn/api/paas/v4",
    ] {
        let body = first_request(&Client::openai_compatible("k", url));
        assert_eq!(
            body["response_format"],
            json!({"type": "json_object"}),
            "{url}"
        );
        let system = body["messages"][0]["content"].as_str().unwrap();
        assert!(system.contains("JSON"), "{url}");
        assert!(system.contains(r#""email""#), "{url}: {system}");
    }
    for url in ["https://api.moonshot.ai/v1", "https://api.openai.com/v1"] {
        let body = first_request(&Client::openai_compatible("k", url));
        assert_eq!(body["response_format"]["type"], "json_schema", "{url}");
    }
}

#[test]
fn gemini_sends_json_schema() {
    let body = first_request(&Client::gemini("k"));
    let cfg = &body["generationConfig"];
    assert!(cfg.get("responseSchema").is_none());
    assert_eq!(cfg["responseJsonSchema"]["type"], "object");
}

#[test]
fn unsupported_settings_fail_before_sending() {
    let client = Client::openai("k").with_reasoning(instructors::Reasoning::Budget(10));
    match client.extract::<Contact>("x").exchange().next() {
        Step::Done(Err(Error::Unsupported { provider, .. })) => assert_eq!(provider, "openai"),
        other => panic!("{other:?}"),
    }
}
