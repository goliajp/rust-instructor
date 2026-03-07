use instructors::{Client, Error, Validate, ValidationError};
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
