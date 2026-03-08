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
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct UserProfile {
    name: String,
    age: u32,
}

impl Validate for UserProfile {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.name.is_empty() {
            return Err("name must not be empty".into());
        }
        if self.age > 150 {
            return Err(format!("age {} is unrealistic", self.age).into());
        }
        Ok(())
    }
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
async fn extract_contact() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response(
            r#"{"name": "John Doe", "email": "john@example.com"}"#,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("test-key", &server.uri());
    let result = client
        .extract::<Contact>("extract contact from: John Doe john@example.com")
        .await
        .unwrap();

    assert_eq!(result.value.name, "John Doe");
    assert_eq!(result.value.email, Some("john@example.com".into()));
    assert_eq!(result.usage.input_tokens, 50);
    assert_eq!(result.usage.output_tokens, 20);
    assert_eq!(result.usage.total_tokens, 70);
    assert_eq!(result.usage.retries, 0);
}

#[tokio::test]
async fn extract_with_optional_null() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Jane", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client.extract::<Contact>("Jane").await.unwrap();

    assert_eq!(result.value.name, "Jane");
    assert_eq!(result.value.email, None);
}

#[tokio::test]
async fn extract_enum() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response(r#""Positive""#)))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client.extract::<Sentiment>("I love it!").await.unwrap();

    assert!(matches!(result.value, Sentiment::Positive));
}

#[tokio::test]
async fn retry_on_invalid_json() {
    let server = MockServer::start().await;

    // first call returns bad JSON, second returns good
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("not valid json")))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Fixed", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .max_retries(2)
        .await
        .unwrap();

    assert_eq!(result.value.name, "Fixed");
    assert_eq!(result.usage.retries, 1);
    assert_eq!(result.usage.input_tokens, 100); // 50 * 2 attempts
}

#[tokio::test]
async fn exhaust_retries() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(openai_response("bad json every time")),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let err = client
        .extract::<Contact>("test")
        .max_retries(1)
        .await
        .unwrap_err();

    assert!(matches!(err, Error::ExtractionFailed { retries: 1, .. }));
}

#[tokio::test]
async fn api_error_status() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    match err {
        Error::Api { status, message } => {
            assert_eq!(status, 429);
            assert!(message.contains("rate limited"));
        }
        _ => panic!("expected Api error, got: {err:?}"),
    }
}

#[tokio::test]
async fn api_error_500() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500).set_body_string(r#"{"error":"internal server error"}"#),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    match err {
        Error::Api { status, .. } => assert_eq!(status, 500),
        _ => panic!("expected Api error"),
    }
}

#[tokio::test]
async fn closure_validation_passes() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Alice", "age": 30}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<UserProfile>("Alice, 30")
        .validate(|u: &UserProfile| {
            if u.age > 150 {
                Err("too old".into())
            } else {
                Ok(())
            }
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Alice");
    assert_eq!(result.value.age, 30);
}

#[tokio::test]
async fn closure_validation_retries() {
    let server = MockServer::start().await;

    // first: valid JSON but fails validation
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Bob", "age": 999}"#)),
        )
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // second: passes validation
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Bob", "age": 25}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<UserProfile>("Bob")
        .validate(|u: &UserProfile| {
            if u.age > 150 {
                Err(format!("age {} unrealistic", u.age).into())
            } else {
                Ok(())
            }
        })
        .max_retries(2)
        .await
        .unwrap();

    assert_eq!(result.value.age, 25);
    assert_eq!(result.usage.retries, 1);
}

#[tokio::test]
async fn trait_validation_passes() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Alice", "age": 30}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<UserProfile>("Alice")
        .validated()
        .await
        .unwrap();

    assert_eq!(result.value.name, "Alice");
}

#[tokio::test]
async fn trait_validation_fails_exhausted() {
    let server = MockServer::start().await;

    // always returns invalid data (age > 150)
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Old", "age": 999}"#)),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let err = client
        .extract::<UserProfile>("someone old")
        .validated()
        .max_retries(1)
        .await
        .unwrap_err();

    assert!(matches!(err, Error::ValidationFailed { retries: 1, .. }));
}

#[tokio::test]
async fn custom_model() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Test", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .model("gpt-4o-mini")
        .await
        .unwrap();

    assert_eq!(result.value.name, "Test");
}

#[tokio::test]
async fn custom_system_prompt() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Test", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .system("be very precise")
        .await
        .unwrap();

    assert_eq!(result.value.name, "Test");
}

#[tokio::test]
async fn context_appended() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Test", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .context("from a business card")
        .await
        .unwrap();

    assert_eq!(result.value.name, "Test");
}

#[tokio::test]
async fn no_usage_in_response() {
    let server = MockServer::start().await;

    let response_no_usage = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": r#"{"name": "Test", "email": null}"#
            },
            "finish_reason": "stop"
        }]
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_no_usage))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client.extract::<Contact>("test").await.unwrap();

    assert_eq!(result.usage.input_tokens, 0);
    assert_eq!(result.usage.output_tokens, 0);
}

#[tokio::test]
async fn empty_choices_error() {
    let server = MockServer::start().await;

    let response_empty = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [],
        "usage": { "prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10 }
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_empty))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    assert!(matches!(err, Error::Other(_)));
}

#[tokio::test]
async fn zero_retries_no_retry() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("invalid json")))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let err = client
        .extract::<Contact>("test")
        .max_retries(0)
        .await
        .unwrap_err();

    assert!(matches!(err, Error::ExtractionFailed { retries: 0, .. }));
}

#[tokio::test]
async fn client_with_defaults() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Default", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri())
        .with_model("gpt-4o-mini")
        .with_system("extract data")
        .with_temperature(0.5)
        .with_max_retries(0)
        .with_max_tokens(1024);

    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Default");
}

#[tokio::test]
async fn extract_with_image_url() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Cat", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("what animal is this?")
        .image(ImageInput::Url("https://example.com/cat.jpg".into()))
        .await
        .unwrap();

    assert_eq!(result.value.name, "Cat");
}

#[tokio::test]
async fn extract_with_image_base64() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Dog", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("what animal is this?")
        .image(ImageInput::Base64 {
            media_type: "image/jpeg".into(),
            data: "dGVzdA==".into(),
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Dog");
}

#[tokio::test]
async fn extract_with_multiple_images() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Comparison", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("compare these images")
        .images(vec![
            ImageInput::Url("https://example.com/a.jpg".into()),
            ImageInput::Url("https://example.com/b.jpg".into()),
        ])
        .await
        .unwrap();

    assert_eq!(result.value.name, "Comparison");
}

#[tokio::test]
async fn fallback_to_second_provider() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;

    // primary returns 500
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
        .expect(1)
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
        .with_fallback(Client::openai_compatible("key2", &fallback.uri()));

    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Fallback");
}

#[tokio::test]
async fn fallback_not_used_on_success() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Primary", "email": null}"#)),
        )
        .expect(1)
        .mount(&primary)
        .await;

    // fallback should NOT be called
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Fallback", "email": null}"#)),
        )
        .expect(0)
        .mount(&fallback)
        .await;

    let client = Client::openai_compatible("key", &primary.uri())
        .with_fallback(Client::openai_compatible("key2", &fallback.uri()));

    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Primary");
}

#[tokio::test]
async fn fallback_chain_all_fail() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("error"))
        .mount(&primary)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
        .mount(&fallback)
        .await;

    let client = Client::openai_compatible("key", &primary.uri())
        .with_fallback(Client::openai_compatible("key2", &fallback.uri()));

    let err = client
        .extract::<Contact>("test")
        .max_retries(0)
        .await
        .unwrap_err();

    // should return the primary error
    assert!(matches!(err, Error::Api { status: 500, .. }));
}

fn openai_stream_chunks(json_content: &str) -> String {
    // simulate SSE streaming by splitting JSON content char-by-char
    let mut sse = String::new();
    for ch in json_content.chars() {
        let chunk = serde_json::json!({
            "choices": [{
                "delta": { "content": ch.to_string() }
            }]
        });
        sse.push_str(&format!("data: {chunk}\n\n"));
    }
    // final chunk with usage
    let usage_chunk = serde_json::json!({
        "choices": [],
        "usage": { "prompt_tokens": 30, "completion_tokens": 10 }
    });
    sse.push_str(&format!("data: {usage_chunk}\n\n"));
    sse.push_str("data: [DONE]\n\n");
    sse
}

#[tokio::test]
async fn streaming_with_empty_data_lines() {
    let server = MockServer::start().await;

    // SSE body with empty data lines and blank lines mixed in
    let mut sse = String::new();
    sse.push_str("data: \n\n"); // empty data line
    let chunk1 = serde_json::json!({"choices": [{"delta": {"content": r#"{"name":"#}}]});
    sse.push_str(&format!("data: {chunk1}\n\n"));
    sse.push_str("\n"); // blank line
    let chunk2 =
        serde_json::json!({"choices": [{"delta": {"content": r#" "Empty", "email": null}"#}}]});
    sse.push_str(&format!("data: {chunk2}\n\n"));
    let usage =
        serde_json::json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}});
    sse.push_str(&format!("data: {usage}\n\n"));
    sse.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .on_stream(|_| {})
        .await
        .unwrap();

    assert_eq!(result.value.name, "Empty");
}

#[tokio::test]
async fn streaming_multibyte_utf8_split() {
    let server = MockServer::start().await;

    // stream a JSON with multi-byte chars split across chunks
    let mut sse = String::new();
    let chunk1 = serde_json::json!({"choices": [{"delta": {"content": r#"{"name": "日"#}}]});
    sse.push_str(&format!("data: {chunk1}\n\n"));
    let chunk2 =
        serde_json::json!({"choices": [{"delta": {"content": r#"本語", "email": null}"#}}]});
    sse.push_str(&format!("data: {chunk2}\n\n"));
    let usage =
        serde_json::json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}});
    sse.push_str(&format!("data: {usage}\n\n"));
    sse.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .on_stream(|_| {})
        .await
        .unwrap();

    assert_eq!(result.value.name, "日本語");
}

#[tokio::test]
async fn fallback_retry_count_accumulates() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;

    // primary: always returns bad JSON (exhausts retries)
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("bad json")))
        .mount(&primary)
        .await;

    // fallback: first attempt bad, second good
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("also bad")))
        .up_to_n_times(1)
        .expect(1)
        .mount(&fallback)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "OK", "email": null}"#)),
        )
        .expect(1)
        .mount(&fallback)
        .await;

    let client = Client::openai_compatible("key", &primary.uri())
        .with_fallback(Client::openai_compatible("key2", &fallback.uri()));

    let result = client
        .extract::<Contact>("test")
        .max_retries(1)
        .await
        .unwrap();

    assert_eq!(result.value.name, "OK");
    // fallback had 1 retry
    assert_eq!(result.usage.retries, 1);
}

#[tokio::test]
async fn extract_with_streaming() {
    let server = MockServer::start().await;

    let json_content = r#"{"name": "Streamed", "email": null}"#;
    let sse_body = openai_stream_chunks(json_content);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .on_stream(move |chunk| {
            chunks_clone.lock().unwrap().push(chunk.to_string());
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Streamed");
    assert_eq!(result.usage.input_tokens, 30);
    assert_eq!(result.usage.output_tokens, 10);

    let collected = chunks.lock().unwrap();
    assert!(
        !collected.is_empty(),
        "stream callback should have been called"
    );
    let reassembled: String = collected.iter().cloned().collect();
    assert_eq!(reassembled, json_content);
}
