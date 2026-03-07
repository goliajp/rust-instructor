use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use instructors::{Client, Message, Validate, ValidationError};
use schemars::JsonSchema;
use serde::Deserialize;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Entity {
    name: String,
    entity_type: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Summary {
    text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Classification {
    category: String,
    confidence: f64,
}

impl Validate for Classification {
    fn validate(&self) -> Result<(), ValidationError> {
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(format!("confidence {} must be between 0 and 1", self.confidence).into());
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
            "message": { "role": "assistant", "content": json_content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70 }
    })
}

// -- extract_many tests --

#[tokio::test]
async fn extract_many_entities() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response(
            r#"[{"name": "Apple", "entity_type": "Company"}, {"name": "Tim Cook", "entity_type": "Person"}]"#,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract_many::<Entity>("Apple CEO Tim Cook")
        .await
        .unwrap();

    assert_eq!(result.value.len(), 2);
    assert_eq!(result.value[0].name, "Apple");
    assert_eq!(result.value[1].name, "Tim Cook");
}

#[tokio::test]
async fn extract_many_empty() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("[]")))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract_many::<Entity>("no entities here")
        .await
        .unwrap();

    assert!(result.value.is_empty());
}

// -- message history tests --

#[tokio::test]
async fn message_history() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"text": "Summary of the document"}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Summary>("summarize the above")
        .messages(vec![
            Message::user("Here is a long document about AI..."),
            Message::assistant("I see the document about AI."),
        ])
        .await
        .unwrap();

    assert_eq!(result.value.text, "Summary of the document");
}

#[tokio::test]
async fn message_constructors() {
    let user = Message::user("hello");
    assert_eq!(user.role, "user");
    assert_eq!(user.content, "hello");

    let assistant = Message::assistant("hi there");
    assert_eq!(assistant.role, "assistant");
    assert_eq!(assistant.content, "hi there");
}

// -- hooks tests --

#[tokio::test]
async fn on_request_hook() {
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

    let call_count = Arc::new(AtomicU32::new(0));
    let count = call_count.clone();

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("test prompt")
        .on_request(move |_model, _prompt| {
            count.fetch_add(1, Ordering::SeqCst);
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Test");
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn on_response_hook() {
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

    let total_tokens = Arc::new(AtomicU32::new(0));
    let tokens = total_tokens.clone();

    let client = Client::openai_compatible("key", &server.uri());
    let _ = client
        .extract::<Contact>("test")
        .on_response(move |usage| {
            tokens.store(usage.total_tokens, Ordering::SeqCst);
        })
        .await
        .unwrap();

    assert_eq!(total_tokens.load(Ordering::SeqCst), 70);
}

#[tokio::test]
async fn hooks_called_on_retry() {
    let server = MockServer::start().await;

    // first bad, second good
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("bad")))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "OK", "email": null}"#)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let request_count = Arc::new(AtomicU32::new(0));
    let count = request_count.clone();

    let client = Client::openai_compatible("key", &server.uri());
    let _ = client
        .extract::<Contact>("test")
        .max_retries(1)
        .on_request(move |_, _| {
            count.fetch_add(1, Ordering::SeqCst);
        })
        .await
        .unwrap();

    // on_request called for each attempt
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
}

// -- classification with validation --

#[tokio::test]
async fn classification_validated() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response(
            r#"{"category": "Technology", "confidence": 0.95}"#,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Classification>("AI and machine learning")
        .validated()
        .await
        .unwrap();

    assert_eq!(result.value.category, "Technology");
    assert!((result.value.confidence - 0.95).abs() < f64::EPSILON);
}

// -- combined features --

#[tokio::test]
async fn extract_many_with_validation() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response(
            r#"[{"name": "Alice", "email": "alice@test.com"}, {"name": "Bob", "email": null}]"#,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract_many::<Contact>("some text with contacts")
        .validate(|contacts: &Vec<Contact>| {
            if contacts.is_empty() {
                Err("expected at least one contact".into())
            } else {
                Ok(())
            }
        })
        .await
        .unwrap();

    assert_eq!(result.value.len(), 2);
}

#[tokio::test]
async fn extract_with_context_and_history() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response(
            r#"{"name": "Jane", "email": "jane@co.com"}"#,
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let result = client
        .extract::<Contact>("extract the contact")
        .messages(vec![Message::user("Here is a business card image...")])
        .context("The card is from Acme Corp")
        .await
        .unwrap();

    assert_eq!(result.value.name, "Jane");
}
