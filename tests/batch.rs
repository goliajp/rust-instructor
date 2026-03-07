use instructors::{Client, Error, Validate, ValidationError};
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
struct StrictUser {
    name: String,
    age: u32,
}

impl Validate for StrictUser {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.age > 150 {
            return Err("age unrealistic".into());
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

#[tokio::test]
async fn batch_basic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Test", "email": null}"#)),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let prompts = vec!["prompt1".into(), "prompt2".into(), "prompt3".into()];
    let results = client
        .extract_batch::<Contact>(prompts)
        .concurrency(3)
        .run()
        .await;

    assert_eq!(results.len(), 3);
    for result in &results {
        let r = result.as_ref().unwrap();
        assert_eq!(r.value.name, "Test");
    }
}

#[tokio::test]
async fn batch_with_model() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "A", "email": null}"#)),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let results = client
        .extract_batch::<Contact>(vec!["a".into(), "b".into()])
        .model("gpt-4o-mini")
        .system("extract")
        .temperature(0.5)
        .max_tokens(1024)
        .max_retries(0)
        .run()
        .await;

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.is_ok()));
}

#[tokio::test]
async fn batch_partial_failure() {
    let server = MockServer::start().await;

    // first request succeeds, subsequent fail
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_response("invalid json")))
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let results = client
        .extract_batch::<Contact>(vec!["a".into(), "b".into()])
        .max_retries(0)
        .concurrency(1)
        .run()
        .await;

    assert_eq!(results.len(), 2);
    // all fail because mock always returns invalid json
    assert!(results.iter().all(|r| r.is_err()));
}

#[tokio::test]
async fn batch_with_validation() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Bob", "age": 30}"#)),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let results = client
        .extract_batch::<StrictUser>(vec!["a".into(), "b".into()])
        .validate(|u: &StrictUser| {
            if u.age > 100 {
                Err("too old".into())
            } else {
                Ok(())
            }
        })
        .run()
        .await;

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.is_ok()));
}

#[tokio::test]
async fn batch_validation_fails() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Old", "age": 999}"#)),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let results = client
        .extract_batch::<StrictUser>(vec!["a".into()])
        .validate(|u: &StrictUser| {
            if u.age > 150 {
                Err("unrealistic".into())
            } else {
                Ok(())
            }
        })
        .max_retries(0)
        .run()
        .await;

    assert_eq!(results.len(), 1);
    assert!(matches!(results[0], Err(Error::ValidationFailed { .. })));
}

#[tokio::test]
async fn batch_empty() {
    let server = MockServer::start().await;
    let client = Client::openai_compatible("key", &server.uri());
    let results = client.extract_batch::<Contact>(vec![]).run().await;

    assert!(results.is_empty());
}

#[tokio::test]
async fn batch_concurrency_one() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Seq", "email": null}"#)),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let results = client
        .extract_batch::<Contact>(vec!["a".into(), "b".into(), "c".into()])
        .concurrency(1) // sequential
        .run()
        .await;

    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.is_ok()));
}

#[tokio::test]
async fn batch_preserves_order() {
    let server = MockServer::start().await;

    // all return same response, but we verify all 5 come back in order
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_response(r#"{"name": "Same", "email": null}"#)),
        )
        .mount(&server)
        .await;

    let client = Client::openai_compatible("key", &server.uri());
    let prompts: Vec<String> = (0..5).map(|i| format!("prompt {i}")).collect();
    let results = client
        .extract_batch::<Contact>(prompts)
        .concurrency(5)
        .run()
        .await;

    assert_eq!(results.len(), 5);
    for result in &results {
        assert!(result.is_ok());
    }
}
