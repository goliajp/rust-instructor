use instructors::{BackoffConfig, Client, Error};
use schemars::JsonSchema;
use serde::Deserialize;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Deserialize, JsonSchema)]
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
async fn retry_429_with_backoff() {
    let server = MockServer::start().await;

    // first call returns 429, second succeeds
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "John", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", &server.uri()).with_retry_backoff(BackoffConfig {
            base_delay: std::time::Duration::from_millis(10),
            max_http_retries: 3,
            jitter: false,
            ..Default::default()
        });

    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "John");
}

#[tokio::test]
async fn retry_503_with_backoff() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Jane", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", &server.uri()).with_retry_backoff(BackoffConfig {
            base_delay: std::time::Duration::from_millis(10),
            max_http_retries: 3,
            jitter: false,
            ..Default::default()
        });

    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Jane");
}

#[tokio::test]
async fn no_retry_400() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", &server.uri()).with_retry_backoff(BackoffConfig {
            base_delay: std::time::Duration::from_millis(10),
            max_http_retries: 3,
            jitter: false,
            ..Default::default()
        });

    let err = client.extract::<Contact>("test").await.unwrap_err();
    match err {
        Error::Api { status, .. } => assert_eq!(status, 400),
        _ => panic!("expected Api error, got: {err:?}"),
    }
}

#[tokio::test]
async fn no_retry_401() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .expect(1)
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", &server.uri()).with_retry_backoff(BackoffConfig {
            base_delay: std::time::Duration::from_millis(10),
            max_http_retries: 3,
            jitter: false,
            ..Default::default()
        });

    let err = client.extract::<Contact>("test").await.unwrap_err();
    match err {
        Error::Api { status, .. } => assert_eq!(status, 401),
        _ => panic!("expected Api error, got: {err:?}"),
    }
}

#[tokio::test]
async fn exhaust_http_retries_then_fallback() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;

    // primary always 429
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&primary)
        .await;

    // fallback succeeds
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Fallback", "email": null}"#)),
        )
        .expect(1)
        .mount(&fallback)
        .await;

    let client = Client::openai_compatible("key", &primary.uri())
        .with_retry_backoff(BackoffConfig {
            base_delay: std::time::Duration::from_millis(10),
            max_http_retries: 2,
            jitter: false,
            ..Default::default()
        })
        .with_fallback(Client::openai_compatible("key2", &fallback.uri()));

    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Fallback");
}

#[tokio::test]
async fn backoff_with_parse_retry() {
    let server = MockServer::start().await;

    // first: 429
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // second: bad json
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("not json")))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // third: good json
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "OK", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client =
        Client::openai_compatible("key", &server.uri()).with_retry_backoff(BackoffConfig {
            base_delay: std::time::Duration::from_millis(10),
            max_http_retries: 2,
            jitter: false,
            ..Default::default()
        });

    let result = client
        .extract::<Contact>("test")
        .max_retries(2)
        .await
        .unwrap();
    assert_eq!(result.value.name, "OK");
    assert_eq!(result.usage.retries, 1); // one parse retry
}

#[tokio::test]
async fn no_backoff_by_default() {
    let server = MockServer::start().await;

    // 429 should fail immediately without backoff configured
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();
    match err {
        Error::Api { status, .. } => assert_eq!(status, 429),
        _ => panic!("expected Api error"),
    }
}
