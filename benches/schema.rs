use criterion::{Criterion, black_box, criterion_group, criterion_main};
use schemars::JsonSchema;
use serde::Deserialize;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// test structs of varying complexity

#[derive(Deserialize, JsonSchema)]
struct Simple {
    name: String,
    age: u32,
}

#[derive(Deserialize, JsonSchema)]
struct Medium {
    name: String,
    email: Option<String>,
    phone: Option<String>,
    age: u32,
    active: bool,
    tags: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct Address {
    street: String,
    city: String,
    state: String,
    zip: String,
    country: String,
}

#[derive(Deserialize, JsonSchema)]
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

// approach 1: no cache (baseline)
fn schema_no_cache<T: JsonSchema>() -> serde_json::Value {
    let schema = schemars::schema_for!(T);
    serde_json::to_value(&schema).unwrap()
}

// approach 2: global OnceLock + Mutex<HashMap<TypeId>>
fn schema_oncelock_map<T: JsonSchema + 'static>() -> serde_json::Value {
    static CACHE: OnceLock<Mutex<HashMap<TypeId, serde_json::Value>>> = OnceLock::new();
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard
        .entry(TypeId::of::<T>())
        .or_insert_with(|| {
            let schema = schemars::schema_for!(T);
            serde_json::to_value(&schema).unwrap()
        })
        .clone()
}

// approach 3: thread_local + RefCell<HashMap<TypeId>>
fn schema_thread_local<T: JsonSchema + 'static>() -> serde_json::Value {
    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<TypeId, serde_json::Value>> =
            std::cell::RefCell::new(HashMap::new());
    }
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache
            .entry(TypeId::of::<T>())
            .or_insert_with(|| {
                let schema = schemars::schema_for!(T);
                serde_json::to_value(&schema).unwrap()
            })
            .clone()
    })
}

fn bench_schema_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_generation");

    // simple struct
    group.bench_function("simple/no_cache", |b| {
        b.iter(|| black_box(schema_no_cache::<Simple>()))
    });
    group.bench_function("simple/oncelock_map", |b| {
        b.iter(|| black_box(schema_oncelock_map::<Simple>()))
    });
    group.bench_function("simple/thread_local", |b| {
        b.iter(|| black_box(schema_thread_local::<Simple>()))
    });

    // medium struct
    group.bench_function("medium/no_cache", |b| {
        b.iter(|| black_box(schema_no_cache::<Medium>()))
    });
    group.bench_function("medium/oncelock_map", |b| {
        b.iter(|| black_box(schema_oncelock_map::<Medium>()))
    });
    group.bench_function("medium/thread_local", |b| {
        b.iter(|| black_box(schema_thread_local::<Medium>()))
    });

    // complex struct
    group.bench_function("complex/no_cache", |b| {
        b.iter(|| black_box(schema_no_cache::<Complex>()))
    });
    group.bench_function("complex/oncelock_map", |b| {
        b.iter(|| black_box(schema_oncelock_map::<Complex>()))
    });
    group.bench_function("complex/thread_local", |b| {
        b.iter(|| black_box(schema_thread_local::<Complex>()))
    });

    group.finish();
}

fn bench_json_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_parse");

    let simple_json = r#"{"name": "John Doe", "age": 30}"#;
    let medium_json = r#"{"name": "John", "email": "john@test.com", "phone": "+1234", "age": 30, "active": true, "tags": ["dev", "rust"]}"#;
    let complex_json = r#"{"first_name": "John", "last_name": "Doe", "email": "john@test.com", "phone": "+1234", "age": 30, "address": {"street": "123 Main", "city": "NYC", "state": "NY", "zip": "10001", "country": "US"}, "tags": ["dev"], "metadata": {"role": "eng"}}"#;

    group.bench_function("simple", |b| {
        b.iter(|| black_box(serde_json::from_str::<Simple>(simple_json).unwrap()))
    });
    group.bench_function("medium", |b| {
        b.iter(|| black_box(serde_json::from_str::<Medium>(medium_json).unwrap()))
    });
    group.bench_function("complex", |b| {
        b.iter(|| black_box(serde_json::from_str::<Complex>(complex_json).unwrap()))
    });

    group.finish();
}

criterion_group!(benches, bench_schema_generation, bench_json_parse);
criterion_main!(benches);
