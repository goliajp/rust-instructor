use std::sync::{Arc, Mutex};

use instructors::{Client, Error, ImageInput, Validate, ValidationError};
use schemars::JsonSchema;
use serde::Deserialize;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct StrictContact {
    name: String,
    email: String,
}

impl Validate for StrictContact {
    fn validate(&self) -> Result<(), ValidationError> {
        if !self.email.contains('@') {
            return Err(format!("'{}' is not a valid email", self.email).into());
        }
        Ok(())
    }
}

fn anthropic_response(json_value: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_test",
            "name": "extract",
            "input": json_value
        }],
        "usage": {
            "input_tokens": 40,
            "output_tokens": 15
        }
    })
}

#[tokio::test]
async fn extract_contact() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "ant-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response(
            serde_json::json!({"name": "Alice", "email": "alice@test.com"}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("ant-key", &server.uri());
    let result = client.extract::<Contact>("extract contact").await.unwrap();

    assert_eq!(result.value.name, "Alice");
    assert_eq!(result.value.email, Some("alice@test.com".into()));
    assert_eq!(result.usage.input_tokens, 40);
    assert_eq!(result.usage.output_tokens, 15);
    assert_eq!(result.usage.total_tokens, 55);
}

#[tokio::test]
async fn extract_optional_null() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response(
            serde_json::json!({"name": "Bob", "email": null}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let result = client.extract::<Contact>("Bob").await.unwrap();

    assert_eq!(result.value.name, "Bob");
    assert_eq!(result.value.email, None);
}

#[tokio::test]
async fn api_error_401() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string(r#"{"error":"invalid api key"}"#))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("bad-key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    match err {
        Error::Api { status, message } => {
            assert_eq!(status, 401);
            assert!(message.contains("invalid api key"));
        }
        _ => panic!("expected Api error, got: {err:?}"),
    }
}

#[tokio::test]
async fn api_error_429() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(429).set_body_string(r#"{"error":"rate_limit_exceeded"}"#),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    match err {
        Error::Api { status, .. } => assert_eq!(status, 429),
        _ => panic!("expected Api error"),
    }
}

#[tokio::test]
async fn no_tool_use_block() {
    let server = MockServer::start().await;

    let response_text_only = serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "I cannot extract that"
        }],
        "usage": {
            "input_tokens": 40,
            "output_tokens": 15
        }
    });

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_text_only))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    assert!(matches!(err, Error::Other(_)));
}

#[tokio::test]
async fn no_usage_in_response() {
    let server = MockServer::start().await;

    let response_no_usage = serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_test",
            "name": "extract",
            "input": {"name": "Test", "email": null}
        }]
    });

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_no_usage))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let result = client.extract::<Contact>("test").await.unwrap();

    assert_eq!(result.usage.input_tokens, 0);
    assert_eq!(result.usage.output_tokens, 0);
}

#[tokio::test]
async fn retry_on_bad_json_from_tool() {
    let server = MockServer::start().await;

    // first: tool_use with input that doesn't match schema
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(anthropic_response(serde_json::json!({"wrong_field": 123}))),
        )
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // second: correct
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response(
            serde_json::json!({"name": "Fixed", "email": null}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .max_retries(2)
        .await
        .unwrap();

    assert_eq!(result.value.name, "Fixed");
    assert_eq!(result.usage.retries, 1);
}

#[tokio::test]
async fn trait_validation_with_anthropic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response(
            serde_json::json!({"name": "Alice", "email": "alice@example.com"}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let result = client
        .extract::<StrictContact>("Alice")
        .validated()
        .await
        .unwrap();

    assert_eq!(result.value.name, "Alice");
    assert!(result.value.email.contains('@'));
}

#[tokio::test]
async fn trait_validation_fails_anthropic() {
    let server = MockServer::start().await;

    // always returns invalid email
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response(
            serde_json::json!({"name": "Bob", "email": "not-an-email"}),
        )))
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let err = client
        .extract::<StrictContact>("Bob")
        .validated()
        .max_retries(1)
        .await
        .unwrap_err();

    assert!(matches!(err, Error::ValidationFailed { retries: 1, .. }));
}

#[tokio::test]
async fn custom_model_anthropic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response(
            serde_json::json!({"name": "Test", "email": null}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .model("claude-opus-4-20250514")
        .await
        .unwrap();

    assert_eq!(result.value.name, "Test");
}

#[tokio::test]
async fn extract_with_image_anthropic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response(
            serde_json::json!({"name": "Cat", "email": "cat@test.com"}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::anthropic_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("what animal is this?")
        .image(ImageInput::Base64 {
            media_type: "image/png".into(),
            data: "dGVzdA==".into(),
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Cat");
}

fn anthropic_stream_events(json_content: &str) -> String {
    let mut sse = String::new();

    // message_start with usage
    let msg_start = serde_json::json!({
        "type": "message_start",
        "message": {
            "usage": { "input_tokens": 25, "output_tokens": 0 }
        }
    });
    sse.push_str(&format!("event: message_start\ndata: {msg_start}\n\n"));

    // content_block_start
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "tool_use", "id": "toolu_test", "name": "extract", "input": {} }
    });
    sse.push_str(&format!("event: content_block_start\ndata: {block_start}\n\n"));

    // stream JSON char-by-char as input_json_delta
    for ch in json_content.chars() {
        let delta = serde_json::json!({
            "type": "content_block_delta",
            "delta": {
                "type": "input_json_delta",
                "partial_json": ch.to_string()
            }
        });
        sse.push_str(&format!("event: content_block_delta\ndata: {delta}\n\n"));
    }

    // message_delta with output usage
    let msg_delta = serde_json::json!({
        "type": "message_delta",
        "usage": { "input_tokens": 0, "output_tokens": 12 }
    });
    sse.push_str(&format!("event: message_delta\ndata: {msg_delta}\n\n"));

    // message_stop
    let msg_stop = serde_json::json!({ "type": "message_stop" });
    sse.push_str(&format!("event: message_stop\ndata: {msg_stop}\n\n"));

    sse
}

#[tokio::test]
async fn extract_with_streaming_anthropic() {
    let server = MockServer::start().await;

    let json_content = r#"{"name":"Stream","email":"s@t.com"}"#;
    let sse_body = anthropic_stream_events(json_content);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .expect(1)
        .mount(&server)
        .await;

    let chunks: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let chunks_clone = chunks.clone();

    let client = Client::anthropic_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .on_stream(move |chunk| {
            chunks_clone.lock().unwrap().push(chunk.to_string());
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Stream");
    assert_eq!(result.value.email, Some("s@t.com".into()));
    assert_eq!(result.usage.input_tokens, 25);
    assert_eq!(result.usage.output_tokens, 12);

    let collected = chunks.lock().unwrap();
    assert!(!collected.is_empty());
    let reassembled: String = collected.iter().cloned().collect();
    assert_eq!(reassembled, json_content);
}
