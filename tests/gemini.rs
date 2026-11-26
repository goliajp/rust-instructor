use instructors::Schema;
use instructors::{Client, Error, Part};
use serde::Deserialize;
use wiremock::matchers::{header, method, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Deserialize, Schema)]
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
        .and(header("x-goog-api-key", "test-key"))
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
        .and(header("x-goog-api-key", "key"))
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
        .and(header("x-goog-api-key", "key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gemini_response("not valid json")))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // second: good JSON
    Mock::given(method("POST"))
        .and(header("x-goog-api-key", "key"))
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
        .max_attempts(3)
        .await
        .unwrap();

    assert_eq!(result.value.name, "Fixed");
    assert_eq!(result.usage.retries(), 1);
}

#[tokio::test]
async fn api_error_status() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(header("x-goog-api-key", "key"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let client = Client::gemini_compatible("key", server.uri());
    let err = client.extract::<Contact>("test").await.unwrap_err();

    match err {
        Error::Api {
            status, message, ..
        } => {
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
        .and(header("x-goog-api-key", "key"))
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
        .on_delta(move |chunk| {
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
        .and(header("x-goog-api-key", "key"))
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
        .part(Part::Image {
            media_type: "image/jpeg".into(),
            data: "dGVzdA==".into(),
        })
        .await
        .unwrap();

    assert_eq!(result.value.name, "Cat");
}

#[tokio::test]
async fn gemini_key_in_header_not_url() {
    let server = MockServer::start().await;

    // only respond if the key header is present
    Mock::given(method("POST"))
        .and(header("x-goog-api-key", "secret-api-key"))
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

    let requests = server.received_requests().await.unwrap();
    assert!(!requests[0].url.as_str().contains("secret-api-key"));
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Node {
    name: String,
    children: Vec<Node>,
    by_tag: std::collections::BTreeMap<String, Node>,
    pair: Option<Box<Node>>,
}

/// The schema a request carries and the value read back from `answer`, or
/// the re-ask that followed.
fn exchange<T: serde::de::DeserializeOwned + Schema>(
    client: &Client,
    answer: &str,
) -> (serde_json::Value, Result<T, String>) {
    use instructors::{Headers, Step};
    let mut ex = client.extract::<T>("x").exchange();
    let Step::Send(req) = ex.next() else { panic!() };
    let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
    let reply = gemini_response(answer).to_string();
    ex.receive(200, &Headers::new(), reply.as_bytes());
    let got = match ex.next() {
        Step::Done(Ok(r)) => Ok(r.value),
        Step::Send(again) => Err(String::from_utf8_lossy(&again.body).into_owned()),
        Step::Done(Err(e)) => Err(e.to_string()),
        Step::Wait(_) => panic!("wait"),
    };
    (body, got)
}

#[test]
fn recursive_lists_and_maps_are_optional_and_read_as_empty() {
    let gemini = Client::gemini("k");
    let answer = r#"{"name": "root", "children": [{"name": "leaf", "pair": null}], "by_tag": {"t": {"name": "x"}}}"#;
    let (body, got) = exchange::<Node>(&gemini, answer);
    let schema = &body["generationConfig"]["responseFormat"]["text"]["schema"];
    let def = &schema["$defs"]["Node"];
    let required = if def.is_null() {
        &schema["required"]
    } else {
        &def["required"]
    };
    assert_eq!(*required, serde_json::json!(["name"]), "{schema}");
    let root = got.unwrap();
    assert_eq!(root.children[0].name, "leaf");
    assert!(root.children[0].children.is_empty());
    assert!(root.children[0].by_tag.is_empty());
    assert!(root.by_tag["t"].children.is_empty());

    // a field that is not a recursive list or map stays required
    let (_, got) = exchange::<Node>(&gemini, r#"{"children": []}"#);
    assert!(got.unwrap_err().contains("name"));

    // other providers still require the recursive fields
    let mut ex = Client::openai("k").extract::<Node>("x").exchange();
    let instructors::Step::Send(req) = ex.next() else {
        panic!()
    };
    let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
    let s = body["response_format"]["json_schema"]["schema"].to_string();
    assert!(
        s.contains(r#""required":["name","children","by_tag","pair"]"#),
        "{s}"
    );
}
