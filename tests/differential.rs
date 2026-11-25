// Every request 2.0 builds, compared with what 1.5.1 sent for the same call.
// 1.5.1's requests are recorded by a loopback server; 2.0's come from the
// sans-io exchange without any i/o. Bodies are compared as JSON values;
// each intended difference is a named exception below.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use instructors::{Client, Effort, Message, OutputMode, Part, Reasoning, Schema, Step};
use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, schemars::JsonSchema, Schema)]
struct Contact {
    /// Full name as written.
    name: String,
    email: Option<String>,
}

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, schemars::JsonSchema, Schema)]
struct Node {
    name: String,
    children: Vec<Node>,
}

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, schemars::JsonSchema, Schema)]
enum Currency {
    Usd,
    Jpy,
}

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, schemars::JsonSchema, Schema)]
struct LineItem {
    description: String,
    quantity: u32,
    unit_price: f64,
    sku: Option<String>,
}

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, schemars::JsonSchema, Schema)]
struct Invoice {
    invoice_number: String,
    vendor: String,
    currency: Currency,
    items: Vec<LineItem>,
    notes: Option<String>,
    total: f64,
}

#[derive(Debug, PartialEq)]
struct Captured {
    path: String,
    headers: Vec<(String, String)>,
    body: Value,
}

type Recorded = Arc<Mutex<Vec<Captured>>>;

const KEPT_HEADERS: [&str; 5] = [
    "content-type",
    "authorization",
    "x-api-key",
    "anthropic-version",
    "x-goog-api-key",
];

/// A server that records each request and answers 400, so 1.5.1 stops
/// after one request.
async fn recorder() -> (String, Recorded) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let rec: Recorded = Arc::default();
    let rec2 = rec.clone();
    tokio::spawn(async move {
        loop {
            let (sock, _) = listener.accept().await.unwrap();
            let rec = rec2.clone();
            tokio::spawn(async move {
                let (r, mut w) = sock.into_split();
                let mut r = BufReader::new(r);
                let mut line = String::new();
                loop {
                    line.clear();
                    if r.read_line(&mut line).await.unwrap_or(0) == 0 {
                        return;
                    }
                    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                    let mut headers = Vec::new();
                    let mut len = 0usize;
                    loop {
                        line.clear();
                        r.read_line(&mut line).await.unwrap();
                        if line == "\r\n" {
                            break;
                        }
                        let (k, v) = line.trim_end().split_once(':').unwrap();
                        let (k, v) = (k.to_ascii_lowercase(), v.trim().to_string());
                        if k == "content-length" {
                            len = v.parse().unwrap();
                        }
                        if KEPT_HEADERS.contains(&k.as_str()) {
                            headers.push((k, v));
                        }
                    }
                    let mut buf = vec![0u8; len];
                    r.read_exact(&mut buf).await.unwrap();
                    headers.sort();
                    rec.lock().unwrap().push(Captured {
                        path,
                        headers,
                        body: serde_json::from_slice(&buf).unwrap(),
                    });
                    let msg = b"recorded";
                    let head = format!("HTTP/1.1 400 X\r\ncontent-length: {}\r\n\r\n", msg.len());
                    w.write_all(head.as_bytes()).await.unwrap();
                    w.write_all(msg).await.unwrap();
                }
            });
        }
    });
    (base, rec)
}

type V1Call = Box<dyn FnOnce(String) -> Pin<Box<dyn Future<Output = ()> + Send>>>;
type V2Call = Box<dyn FnOnce(String) -> Option<instructors::HttpRequest>>;

/// 1.5.1's first request, or `None` when it sent nothing.
async fn v1(call: V1Call) -> Option<Captured> {
    let (base, rec) = recorder().await;
    call(base).await;
    rec.lock().unwrap().drain(..).next()
}

fn v2(call: V2Call) -> Option<Captured> {
    let base = "http://h".to_string();
    let req = call(base.clone())?;
    let mut headers: Vec<(String, String)> = req
        .headers
        .iter()
        .filter(|(k, _)| KEPT_HEADERS.contains(&k.to_ascii_lowercase().as_str()))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.to_string()))
        .collect();
    headers.sort();
    Some(Captured {
        path: req.url[base.len()..].to_string(),
        headers,
        body: serde_json::from_slice(&req.body).unwrap(),
    })
}

fn first<T>(ex: &mut instructors::Exchange<T>) -> Option<instructors::HttpRequest>
where
    T: serde::de::DeserializeOwned,
{
    match ex.next() {
        Step::Send(r) => Some(r),
        _ => None,
    }
}

/// `required` lists are compared as sets: 1.5 sorted them, 2.0 keeps
/// declaration order (checked in the schema tests).
fn normalize(v: &mut Value) {
    match v {
        Value::Object(m) => {
            for (k, x) in m.iter_mut() {
                if k == "required"
                    && let Value::Array(a) = x
                {
                    a.sort_by_key(|e| e.to_string());
                }
                normalize(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(normalize),
        _ => {}
    }
}

/// Gemini: 1.5 sent the OpenAPI-subset `responseSchema` unless the type was
/// recursive; 2.0 always sends `responseJsonSchema`.
fn gemini_schema_key(old: &mut Captured, new: &mut Captured) {
    let (Some(o), Some(n)) = (
        old.body.get_mut("generationConfig"),
        new.body.get_mut("generationConfig"),
    ) else {
        return;
    };
    if let Some(Value::Object(o)) = Some(o)
        && o.remove("responseSchema").is_some()
        && let Value::Object(n) = n
    {
        assert!(n.remove("responseJsonSchema").is_some());
    }
}

static COMPARED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SKIPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn compare(case: &str, old: Option<Captured>, new: Option<Captured>) {
    use std::sync::atomic::Ordering::Relaxed;
    let (mut old, mut new) = match (old, new) {
        (None, None) => {
            SKIPPED.fetch_add(1, Relaxed);
            return;
        }
        (Some(o), Some(n)) => (o, n),
        (o, n) => panic!("{case}: 1.5 sent {o:?}, 2.0 sent {n:?}"),
    };
    gemini_schema_key(&mut old, &mut new);
    normalize(&mut old.body);
    normalize(&mut new.body);
    assert_eq!(old.path, new.path, "{case}");
    assert_eq!(old.headers, new.headers, "{case}");
    assert_eq!(old.body, new.body, "{case}");
    COMPARED.fetch_add(1, Relaxed);
}

#[derive(Clone, Copy, Debug)]
enum P {
    OpenAiCompatible,
    AnthropicOutput,
    AnthropicTool,
    Gemini,
}

const PROVIDERS: [P; 4] = [
    P::OpenAiCompatible,
    P::AnthropicOutput,
    P::AnthropicTool,
    P::Gemini,
];

fn c1(p: P, base: String) -> instructors_1::Client {
    use instructors_1::Client as C;
    match p {
        P::OpenAiCompatible => C::openai_compatible("k", base),
        P::AnthropicOutput => C::anthropic_compatible("k", base).with_anthropic_structured_output(),
        P::AnthropicTool => C::anthropic_compatible("k", base),
        P::Gemini => C::gemini_compatible("k", base),
    }
}

fn c2(p: P, base: String) -> Client {
    match p {
        P::OpenAiCompatible => Client::openai_compatible("k", base),
        P::AnthropicOutput => {
            Client::anthropic_compatible("k", base).with_output_mode(OutputMode::Schema)
        }
        P::AnthropicTool => Client::anthropic_compatible("k", base),
        P::Gemini => Client::gemini_compatible("k", base),
    }
}

/// One case: the same settings applied through each version's builder.
macro_rules! case {
    ($name:expr, $p:expr, $ty:ty, |$b1:ident| $e1:expr, |$b2:ident| $e2:expr) => {{
        let p = $p;
        let old = v1(Box::new(move |base| {
            Box::pin(async move {
                let client = c1(p, base);
                let $b1 = client
                    .extract::<$ty>("Contact John at john@x.com")
                    .max_retries(0);
                let _ = $e1.await;
            })
        }))
        .await;
        let new = v2(Box::new(move |base| {
            let client = c2(p, base);
            let $b2 = client.extract::<$ty>("Contact John at john@x.com");
            first(&mut $e2.exchange())
        }));
        compare(&format!("{} {:?}", $name, p), old, new);
    }};
}

#[tokio::test]
async fn single_requests_match_1_5() {
    for p in PROVIDERS {
        case!("plain", p, Contact, |b| b, |b| b);
        case!("invoice", p, Invoice, |b| b, |b| b);
        case!("recursive", p, Node, |b| b, |b| b);
        case!("system", p, Contact, |b| b.system("s"), |b| b.system("s"));
        case!("model", p, Contact, |b| b.model("m-1"), |b| b.model("m-1"));
        case!("temperature", p, Contact, |b| b.temperature(0.3), |b| b
            .temperature(0.3));
        case!("max_tokens", p, Contact, |b| b.max_tokens(1000), |b| b
            .max_tokens(1000));
        case!(
            "effort",
            p,
            Contact,
            |b| b.reasoning(instructors_1::Reasoning::Effort(
                instructors_1::Effort::High
            )),
            |b| b.reasoning(Reasoning::Effort(Effort::High))
        );
        case!(
            "effort minimal",
            p,
            Contact,
            |b| b.reasoning(instructors_1::Reasoning::Effort(
                instructors_1::Effort::Minimal
            )),
            |b| b.reasoning(Reasoning::Effort(Effort::Minimal))
        );
        case!(
            "budget",
            p,
            Contact,
            |b| b.reasoning(instructors_1::Reasoning::Budget(1024)),
            |b| b.reasoning(Reasoning::Budget(1024))
        );
        case!(
            "off",
            p,
            Contact,
            |b| b.reasoning(instructors_1::Reasoning::Off),
            |b| b.reasoning(Reasoning::Off)
        );
        case!(
            "history",
            p,
            Contact,
            |b| b.messages(vec![
                instructors_1::Message::user("earlier"),
                instructors_1::Message::assistant("noted"),
            ]),
            |b| b.messages([Message::user("earlier"), Message::assistant("noted")])
        );
        case!(
            "image url",
            p,
            Contact,
            |b| b.image(instructors_1::ImageInput::Url("https://x/i.png".into())),
            |b| b.part(Part::image_url("https://x/i.png"))
        );
        case!(
            "image data",
            p,
            Contact,
            |b| b.image(instructors_1::ImageInput::Base64 {
                media_type: "image/png".into(),
                data: "AAAA".into()
            }),
            |b| b.part(Part::image("image/png", "AAAA"))
        );
        case!("context", p, Contact, |b| b.context("more"), |b| b
            .context("more"));
        case!("streaming", p, Contact, |b| b.on_stream(|_| {}), |b| b
            .on_delta(|_| {}));
        case!(
            "streaming partial",
            p,
            Contact,
            |b| b.on_partial(|_| {}),
            |b| b.on_partial(|_| {})
        );
    }
    // only openai-compatible + budget sends nothing in both versions
    use std::sync::atomic::Ordering::Relaxed;
    assert!(COMPARED.load(Relaxed) >= 67, "{}", COMPARED.load(Relaxed));
    assert_eq!(SKIPPED.load(Relaxed), 1);
}

#[tokio::test]
async fn list_requests_match_1_5() {
    for p in PROVIDERS {
        let old = v1(Box::new(move |base| {
            Box::pin(async move {
                let client = c1(p, base);
                let _ = client.extract_many::<Contact>("x").max_retries(0).await;
            })
        }))
        .await;
        let new = v2(Box::new(move |base| {
            first(&mut c2(p, base).extract_many::<Contact>("x").exchange())
        }));
        compare(&format!("extract_many {p:?}"), old, new);
        let old = v1(Box::new(move |base| {
            Box::pin(async move {
                let client = c1(p, base);
                let _ = client.extract_many::<Node>("x").max_retries(0).await;
            })
        }))
        .await;
        let new = v2(Box::new(move |base| {
            first(&mut c2(p, base).extract_many::<Node>("x").exchange())
        }));
        compare(&format!("extract_many recursive {p:?}"), old, new);
    }
}

#[tokio::test]
async fn client_defaults_match_1_5() {
    for p in PROVIDERS {
        let old = v1(Box::new(move |base| {
            Box::pin(async move {
                let client = c1(p, base)
                    .with_model("cm")
                    .with_system("cs")
                    .with_max_tokens(77)
                    .with_temperature(0.5)
                    .with_reasoning(instructors_1::Reasoning::Effort(instructors_1::Effort::Low));
                let _ = client.extract::<Contact>("x").max_retries(0).await;
            })
        }))
        .await;
        let new = v2(Box::new(move |base| {
            let client = c2(p, base)
                .with_model("cm")
                .with_system("cs")
                .with_max_tokens(77)
                .with_temperature(0.5)
                .with_reasoning(Reasoning::Effort(Effort::Low));
            first(&mut client.extract::<Contact>("x").exchange())
        }));
        compare(&format!("client defaults {p:?}"), old, new);
    }
}
