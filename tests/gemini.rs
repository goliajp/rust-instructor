use instructors::{Client, Error, GeminiThinking, ImageInput, ThinkingLevel};
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

    let client = Client::gemini_compatible("test-key", server.uri());
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

    let client = Client::gemini_compatible("key", server.uri());
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

    let client = Client::gemini_compatible("key", server.uri());
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

    let client = Client::gemini_compatible("key", server.uri());
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

    let client = Client::gemini_compatible("key", server.uri());
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

    let client = Client::gemini_compatible("key", server.uri());
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

    let client = Client::gemini_compatible("secret-api-key", server.uri());
    let result = client.extract::<Contact>("test").await.unwrap();
    assert_eq!(result.value.name, "Auth");
}

async fn sent_body(server: &MockServer) -> serde_json::Value {
    let requests = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests[0].body).unwrap()
}

#[tokio::test]
async fn thinking_level_sent_in_generation_config() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Thought", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri())
        .with_gemini_thinking(GeminiThinking::Level(ThinkingLevel::High));
    let result = client.extract::<Contact>("test").await.unwrap();

    assert_eq!(result.value.name, "Thought");

    let config = &sent_body(&server).await["generationConfig"]["thinkingConfig"];
    assert_eq!(config["thinkingLevel"], "high");
    // the two parameters are mutually exclusive — a request carrying both is a
    // 400, so only ever one key may reach the wire
    assert!(
        config.get("thinkingBudget").is_none(),
        "thinkingBudget must not accompany thinkingLevel, got: {config}"
    );
}

#[tokio::test]
async fn thinking_budget_sent_in_generation_config() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Budgeted", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri())
        .with_model("gemini-2.5-flash")
        .with_gemini_thinking(GeminiThinking::Budget(1024));
    let result = client.extract::<Contact>("test").await.unwrap();

    assert_eq!(result.value.name, "Budgeted");

    let config = &sent_body(&server).await["generationConfig"]["thinkingConfig"];
    assert_eq!(config["thinkingBudget"], 1024);
    assert!(
        config.get("thinkingLevel").is_none(),
        "thinkingLevel must not accompany thinkingBudget, got: {config}"
    );
}

#[tokio::test]
async fn dynamic_thinking_budget_sent_as_negative_one() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Dynamic", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri())
        .with_gemini_thinking(GeminiThinking::Budget(-1));
    client.extract::<Contact>("test").await.unwrap();

    // -1 asks the model to size the budget itself; it must survive as a
    // signed number rather than being clamped or dropped
    let body = sent_body(&server).await;
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
        -1
    );
}

#[tokio::test]
async fn last_thinking_setting_wins() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Last", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    // one slot, so a second call replaces the first rather than adding a
    // second key
    let client = Client::gemini_compatible("key", server.uri())
        .with_gemini_thinking(GeminiThinking::Level(ThinkingLevel::High))
        .with_gemini_thinking(GeminiThinking::Budget(512));
    client.extract::<Contact>("test").await.unwrap();

    let config = &sent_body(&server).await["generationConfig"]["thinkingConfig"];
    assert_eq!(config["thinkingBudget"], 512);
    assert!(config.get("thinkingLevel").is_none());
    assert_eq!(
        config.as_object().unwrap().len(),
        1,
        "thinkingConfig must carry exactly one key, got: {config}"
    );
}

#[tokio::test]
async fn thinking_level_serializes_lowercase() {
    for (level, expected) in [
        (ThinkingLevel::Minimal, "minimal"),
        (ThinkingLevel::Low, "low"),
        (ThinkingLevel::Medium, "medium"),
        (ThinkingLevel::High, "high"),
    ] {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(query_param("key", "key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(gemini_response(r#"{"name": "L", "email": null}"#)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = Client::gemini_compatible("key", server.uri())
            .with_gemini_thinking(GeminiThinking::Level(level));
        client.extract::<Contact>("test").await.unwrap();

        let body = sent_body(&server).await;
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingLevel"], expected,
            "{level:?} must serialize as {expected}"
        );
    }
}

#[tokio::test]
async fn no_thinking_config_when_unset() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Default", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri());
    client.extract::<Contact>("test").await.unwrap();

    let body = sent_body(&server).await;
    assert!(
        body["generationConfig"].get("thinkingConfig").is_none(),
        "thinkingConfig must be absent, got: {}",
        body["generationConfig"]
    );
}

#[tokio::test]
async fn thinking_level_survives_client_clone() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(gemini_response(r#"{"name": "Cloned", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri())
        .with_gemini_thinking(GeminiThinking::Level(ThinkingLevel::Low))
        .with_max_retries(0);
    let cloned = client.clone();
    cloned.extract::<Contact>("test").await.unwrap();

    let body = sent_body(&server).await;
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        "low"
    );
}

#[tokio::test]
async fn thoughts_tokens_counted_as_output() {
    let server = MockServer::start().await;

    let with_thoughts = serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{ "text": r#"{"name": "Deep", "email": null}"# }],
                "role": "model"
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 40,
            "candidatesTokenCount": 15,
            "thoughtsTokenCount": 900,
            "totalTokenCount": 955
        }
    });

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(with_thoughts))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri())
        .with_gemini_thinking(GeminiThinking::Level(ThinkingLevel::High));
    let result = client.extract::<Contact>("test").await.unwrap();

    // thoughts are billed as output tokens, so they must not fall out of the
    // count — otherwise raising the thinking level deflates the cost estimate
    assert_eq!(result.usage.input_tokens, 40);
    assert_eq!(result.usage.output_tokens, 915);
    assert_eq!(result.usage.total_tokens, 955);
}

#[tokio::test]
async fn thoughts_tokens_counted_while_streaming() {
    let server = MockServer::start().await;

    let chunk = serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{ "text": r#"{"name": "Streamed", "email": null}"# }],
                "role": "model"
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 40,
            "candidatesTokenCount": 15,
            "thoughtsTokenCount": 100,
            "totalTokenCount": 155
        }
    });

    Mock::given(method("POST"))
        .and(query_param("key", "key"))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!("data: {chunk}\n\n")),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri())
        .with_gemini_thinking(GeminiThinking::Level(ThinkingLevel::Medium));
    let result = client
        .extract::<Contact>("test")
        .on_stream(|_| {})
        .await
        .unwrap();

    assert_eq!(result.value.name, "Streamed");
    assert_eq!(result.usage.output_tokens, 115);
}
