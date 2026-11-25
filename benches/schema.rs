// Per-request client work without I/O, through the public sans-IO exchange:
// building the request (cached schema bytes spliced into the body) and a
// full round of request + canned response + parse.

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use instructors::{Client, Headers, Schema, Step};
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[expect(
    dead_code,
    reason = "the benches only build and parse; fields are never read"
)]
#[derive(Deserialize, Schema)]
struct Simple {
    name: String,
    age: u32,
}

#[expect(
    dead_code,
    reason = "the benches only build and parse; fields are never read"
)]
#[derive(Deserialize, Schema)]
struct Medium {
    name: String,
    email: Option<String>,
    phone: Option<String>,
    age: u32,
    active: bool,
    tags: Vec<String>,
}

#[expect(
    dead_code,
    reason = "the benches only build and parse; fields are never read"
)]
#[derive(Deserialize, Schema)]
struct Address {
    street: String,
    city: String,
    state: String,
    zip: String,
    country: String,
}

#[expect(
    dead_code,
    reason = "the benches only build and parse; fields are never read"
)]
#[derive(Deserialize, Schema)]
struct Complex {
    first_name: String,
    last_name: String,
    email: String,
    phone: Option<String>,
    age: u32,
    address: Address,
    tags: Vec<String>,
    metadata: HashMap<String, String>,
}

const SIMPLE: &str = r#"{"name": "John Doe", "age": 30}"#;
const MEDIUM: &str = r#"{"name": "John", "email": "john@test.com", "phone": "+1234", "age": 30, "active": true, "tags": ["dev", "rust"]}"#;
const COMPLEX: &str = r#"{"first_name": "John", "last_name": "Doe", "email": "john@test.com", "phone": "+1234", "age": 30, "address": {"street": "123 Main", "city": "NYC", "state": "NY", "zip": "10001", "country": "US"}, "tags": ["dev"], "metadata": {"role": "eng"}}"#;

fn request<T: DeserializeOwned + Schema>(client: &Client) -> usize {
    match client.extract::<T>("John Doe, 30").exchange().next() {
        Step::Send(req) => req.body.len(),
        _ => unreachable!("the first step sends"),
    }
}

fn openai_body(content: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20}
    }))
    .unwrap()
}

fn round<T: DeserializeOwned + Schema>(client: &Client, body: &[u8]) -> T {
    let mut ex = client.extract::<T>("John Doe, 30").exchange();
    let Step::Send(_) = ex.next() else {
        unreachable!("the first step sends")
    };
    ex.receive(200, &Headers::new(), body);
    match ex.next() {
        Step::Done(Ok(r)) => r.value,
        _ => unreachable!("the canned answer parses"),
    }
}

fn bench_request(c: &mut Criterion) {
    let mut group = c.benchmark_group("request");
    for (name, client) in [
        ("openai", Client::openai("k")),
        ("anthropic", Client::anthropic("k")),
        ("gemini", Client::gemini("k")),
    ] {
        group.bench_function(format!("{name}/simple"), |b| {
            b.iter(|| black_box(request::<Simple>(&client)))
        });
        group.bench_function(format!("{name}/complex"), |b| {
            b.iter(|| black_box(request::<Complex>(&client)))
        });
    }
    group.finish();
}

fn bench_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("round");
    let client = Client::openai("k");
    let (s, m, x) = (
        openai_body(SIMPLE),
        openai_body(MEDIUM),
        openai_body(COMPLEX),
    );
    group.bench_function("simple", |b| {
        b.iter(|| black_box(round::<Simple>(&client, &s)))
    });
    group.bench_function("medium", |b| {
        b.iter(|| black_box(round::<Medium>(&client, &m)))
    });
    group.bench_function("complex", |b| {
        b.iter(|| black_box(round::<Complex>(&client, &x)))
    });
    group.finish();
}

criterion_group!(benches, bench_request, bench_round);
criterion_main!(benches);
