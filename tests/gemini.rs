use instructors::{Client, Error, ImageInput};
use schemars::JsonSchema;
use serde::Deserialize;
use wiremock::matchers::{method, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
}

fn gemini_response(json_content: &str) -> serde_json::Value {
    serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{ "text": json_content }],
                "role": "model"
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 40,
            "candidatesTokenCount": 15,
            "totalTokenCount": 55
        }
    })
}

#[tokio::test]
async fn extract_contact() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gemini_response(
            r#"{"name": "John Doe", "email": "john@example.com"}"#,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("test-key", &server.uri());
    let result = client
        .extract::<Contact>("extract contact from: John Doe john@example.com")
        .await
        .unwrap();

    assert_eq!(result.value.name, "John Doe");
    assert_eq!(result.value.email, Some("john@example.com".into()));
    assert_eq!(result.usage.input_tokens, 40);
    assert_eq!(result.usage.output_tokens, 15);
}

#[tokio::test]
async fn extract_with_optional_null() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Jane", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", &server.uri());
    let result = client.extract::<Contact>("Jane").await.unwrap();

    assert_eq!(result.value.name, "Jane");
    assert_eq!(result.value.email, None);
}

#[tokio::test]
async fn retry_on_invalid_json() {
    let server = MockServer::start().await;

    // first: bad JSON
    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gemini_response("not valid json")))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // second: good JSON
    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Fixed", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test")
        .max_retries(2)
        .await
        .unwrap();

    assert_eq!(result.value.name, "Fixed");
    assert_eq!(result.usage.retries, 1);
}

#[tokio::test]
async fn api_error_status() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", &server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    match err {
        Error::Api { status, message } => {
            assert_eq!(status, 400);
            assert!(message.contains("bad request"));
        }
        _ => panic!("expected Api error, got: {err:?}"),
    }
}

#[tokio::test]
async fn extract_with_streaming() {
    let server = MockServer::start().await;

    let mut sse = String::new();
    let chunk1 = gemini_response(r#"{"name": "#);
    sse.push_str(&format!("data: {chunk1}\n\n"));
    let chunk2 = serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{ "text": r#""Streamed", "email": null}"# }],
                "role": "model"
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 40,
            "candidatesTokenCount": 15,
            "totalTokenCount": 55
        }
    });
    sse.push_str(&format!("data: {chunk2}\n\n"));

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", &server.uri());
    let chunks: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let chunks_clone = chunks.clone();

    let result = client
        .extract::<Contact>("test")
        .on_stream(move |chunk| {
            chunks_clone.lock().unwrap().push(chunk.to_string());
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Streamed");
    let collected = chunks.lock().unwrap();
    assert!(!collected.is_empty());
}

#[tokio::test]
async fn extract_with_image() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Cat", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("what is this?")
        .image(ImageInput::Base64 {
            media_type: "image/jpeg".into(),
            data: "dGVzdA==".into(),
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Cat");
}

#[tokio::test]
async fn gemini_auth_in_query_param() {
    let server = MockServer::start().await;

    // only respond if key query param is present
    Mock::given(method("POST"))
        .and(query_param("key", "secret-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Auth", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("secret-api-key", &server.uri());
    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Auth");
}
