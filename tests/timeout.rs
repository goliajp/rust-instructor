use std::time::Duration;

use instructors::{Client, Error};
use instructors::{RetryPolicy, Schema};
use serde::Deserialize;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, Schema)]
struct Contact {
    name: String,
    email: Option<String>,
}

fn openai_response(json_content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    })
}

#[tokio::test]
async fn timeout_triggers() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Slow", "email": null}"#))
                .set_delay(Duration::from_secs(2)),
        )
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", server.uri()).with_timeout(Duration::from_millis(100));
    let err = client.extract::<Contact>("test").await.unwrap_err();
    assert!(matches!(err, Error::Timeout { .. }));
}

#[tokio::test]
async fn timeout_not_triggered() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Fast", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", server.uri()).with_timeout(Duration::from_secs(5));
    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Fast");
}

#[tokio::test]
async fn timeout_covers_retries() {
    let server = MockServer::start().await;

    // always returns bad json, plus a delay
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response("invalid"))
                .set_delay(Duration::from_millis(50)),
        )
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", server.uri()).with_timeout(Duration::from_millis(100));
    let err = client
        .extract::<Contact>("test")
        .max_attempts(11)
        .await
        .unwrap_err();

    // should time out before exhausting 10 retries
    assert!(matches!(err, Error::Timeout { .. }));
}

#[tokio::test]
async fn timeout_covers_backoff() {
    let server = MockServer::start().await;

    // always returns 429
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", server.uri())
        .with_retry(
        RetryPolicy::new()
            .base_delay(Duration::from_millis(200))
            .max_retries(10)
            .jitter(false),
    )
        .with_timeout(Duration::from_millis(100));

    let err = client.extract::<Contact>("test").await.unwrap_err();
    assert!(matches!(err, Error::Timeout { .. }));
}

#[tokio::test]
async fn timeout_error_display() {
    let err = Error::Timeout {
        after: Duration::from_secs(60),
        usage: instructors::Usage::default(),
    };
    let s = err.to_string();
    assert!(s.contains("timed out"));
    assert!(s.contains("60"));
}

#[tokio::test]
async fn per_request_timeout_override() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Test", "email": null}"#))
                .set_delay(Duration::from_millis(200)),
        )
        .mount(&server)
        .await;

    // client timeout is generous, but per-request timeout is tight
    let client =
        Client::openai_compatible("key", server.uri()).with_timeout(Duration::from_secs(10));
    let err = client
        .extract::<Contact>("test")
        .timeout(Duration::from_millis(50))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Timeout { .. }));
}
