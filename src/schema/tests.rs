//! The derive against schemars 1 on a type corpus: same schema per dialect
//! (compared as JSON values, so key order aside), and properties in
//! declaration order.

#![expect(dead_code, reason = "corpus types are only inspected, never built")]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use super::cache::{Dialect, entry_for};
use super::oracle_15 as oracle;
use crate::Schema;
use crate::partial::PartialParser;

#[derive(Deserialize, JsonSchema, Schema)]
/// A doc on the struct.
/// Second line.
struct Prims {
    a: bool,
    b: u8,
    c: u16,
    d: u32,
    e: u64,
    f: usize,
    g: i8,
    h: i32,
    i: i64,
    j: f32,
    k: f64,
    l: String,
    m: char,
    n: (),
    o: i128,
    p: u128,
    q: isize,
}

/// Inner doc
#[derive(Deserialize, JsonSchema, Schema)]
struct Inner {
    /// field doc
    x: i32,
}

#[derive(Deserialize, JsonSchema, Schema)]
enum E1 {
    A,
    B,
    /// c doc
    C,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Opts {
    a: Option<u32>,
    b: Option<String>,
    c: Option<Inner>,
    d: Option<Option<bool>>,
    e: Option<Vec<u8>>,
    f: Option<E1>,
}

#[derive(Deserialize, JsonSchema, Schema)]
enum E2 {
    A,
    B(i32),
    C { x: u8 },
    D(i32, String),
}

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(tag = "kind")]
enum E3 {
    A,
    C { x: u8 },
}

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(tag = "kind")]
enum E3Newtype {
    I(Inner),
}

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(tag = "t", content = "c")]
enum E4 {
    A,
    B(i32),
    C { x: u8 },
}

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(untagged)]
enum E5 {
    B(i32),
    C { x: u8 },
    S(String),
}

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(rename_all = "camelCase")]
struct Ren {
    first_name: String,
    #[serde(rename = "LAST")]
    last_name: String,
    #[serde(skip)]
    skipped: u32,
    #[serde(skip_deserializing)]
    sd: u8,
    #[serde(alias = "other")]
    some_field: bool,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct WithDefault {
    #[serde(default)]
    dflt: u32,
    req: u32,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Flat {
    a: u8,
    #[serde(flatten)]
    fl: Inner,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Colls {
    v: Vec<String>,
    vd: VecDeque<u8>,
    a: [u8; 3],
    hm: HashMap<String, u32>,
    bm: BTreeMap<String, Inner>,
    hs: HashSet<String>,
    bs: BTreeSet<u8>,
    t: (u8, String),
    b: Box<Inner>,
    bx: Box<str>,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Tup(u32, String);

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(transparent)]
struct Tr {
    x: Inner,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Gen<T> {
    item: T,
    items: Vec<T>,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct UsesGen {
    a: Gen<u32>,
    b: Gen<String>,
}

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(deny_unknown_fields)]
struct Deny {
    a: u8,
}

#[derive(Deserialize, JsonSchema, Schema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum E6 {
    FirstOne,
    SecondOne,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Node {
    name: String,
    children: Vec<Node>,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Doc {
    title: String,
    root: Node,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Twice {
    a: Inner,
    b: Inner,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Article {
    title: String,
    format: String,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Invoice {
    invoice_number: String,
    issued_on: String,
    vendor: String,
    vendor_address: Address,
    bill_to: Address,
    currency: Currency,
    items: Vec<LineItem>,
    notes: Option<String>,
    total: f64,
}

#[derive(Deserialize, JsonSchema, Schema)]
enum Currency {
    Usd,
    Eur,
    Jpy,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct Address {
    street: String,
    city: String,
    postal_code: String,
    country: String,
}

#[derive(Deserialize, JsonSchema, Schema)]
struct LineItem {
    description: String,
    quantity: u32,
    unit_price: f64,
    sku: Option<String>,
}

const PAIRS: [(Dialect, oracle::Dialect); 3] = [
    (Dialect::AnthropicOutput, oracle::Dialect::AnthropicOutput),
    (Dialect::AnthropicTool, oracle::Dialect::AnthropicTool),
    (Dialect::GeminiJson, oracle::Dialect::GeminiJson),
];

fn ours<T: Schema>(list: bool, d: Dialect) -> Value {
    serde_json::from_str(entry_for::<T>(list).get(d)).unwrap()
}

// 1.5 built list schemas from `Vec<T>`
fn theirs<T: JsonSchema + 'static>(list: bool, d: oracle::Dialect) -> Value {
    let raw = if list {
        oracle::entry_for::<Vec<T>>(true).get(d).get().to_owned()
    } else {
        oracle::entry_for::<T>(false).get(d).get().to_owned()
    };
    serde_json::from_str(&raw).unwrap()
}

/// The first place two values differ, for assertion messages.
fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                let p = format!("{path}/{k}");
                match (x.get(k), y.get(k)) {
                    // 1.5 listed `required` alphabetically; order is checked
                    // separately
                    (Some(Value::Array(u)), Some(Value::Array(v))) if k == "required" => {
                        let mut u: Vec<String> = u.iter().map(|x| x.to_string()).collect();
                        let mut v: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                        u.sort();
                        v.sort();
                        if u != v {
                            return Some(format!("{p}: {u:?} vs {v:?}"));
                        }
                    }
                    (Some(u), Some(v)) => {
                        if let Some(d) = diff(u, v, &p) {
                            return Some(d);
                        }
                    }
                    (u, v) => return Some(format!("{p}: {u:?} vs {v:?}")),
                }
            }
            None
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .enumerate()
            .find_map(|(i, (u, v))| diff(u, v, &format!("{path}/{i}"))),
        _ => (a != b).then(|| format!("{path}: {a} vs {b}")),
    }
}

fn same<T: Schema + JsonSchema + 'static>() {
    for list in [false, true] {
        let name = std::any::type_name::<T>();
        let mut want = theirs::<T>(list, oracle::Dialect::OpenAi);
        let want_name = want["json_schema"]["name"].take();
        let want_strict = want["json_schema"]["schema"].take();
        let got = ours::<T>(list, Dialect::Strict);
        assert_eq!(
            diff(&got, &want_strict, ""),
            None,
            "{name} list={list} strict"
        );
        assert_eq!(
            entry_for::<T>(list).name(),
            want_name.as_str().unwrap(),
            "{name} list={list}"
        );
        for (d, o) in PAIRS {
            let (a, b) = (ours::<T>(list, d), theirs::<T>(list, o));
            assert_eq!(diff(&a, &b, ""), None, "{name} list={list} {d:?}");
        }
    }
}

#[test]
fn matches_schemars() {
    same::<Prims>();
    same::<Inner>();
    same::<E1>();
    same::<Opts>();
    same::<E2>();
    same::<E3>();
    same::<E4>();
    same::<E5>();
    same::<Ren>();
    same::<Colls>();
    same::<UsesGen>();
    same::<Deny>();
    same::<E6>();
    same::<Node>();
    same::<Doc>();
    same::<Twice>();
    same::<Article>();
    same::<Invoice>();
}

// schemars writes `"default": 0` for a `#[serde(default)]` field; the
// derive does not (it would need `Serialize`), and it is only an annotation
fn without_default(mut v: Value) -> Value {
    fn walk(v: &mut Value) {
        match v {
            Value::Object(m) => {
                m.remove("default");
                m.values_mut().for_each(walk);
            }
            Value::Array(a) => a.iter_mut().for_each(walk),
            _ => {}
        }
    }
    walk(&mut v);
    v
}

#[test]
fn tuple_struct_matches_but_is_named_by_its_items() {
    // the derive keeps no name for tuple structs; only `json_schema.name`
    // and the Gemini title differ
    let mut want = theirs::<Tup>(false, oracle::Dialect::OpenAi);
    assert_eq!(
        ours::<Tup>(false, Dialect::Strict),
        want["json_schema"]["schema"].take()
    );
    assert_eq!(entry_for::<Tup>(false).name(), "Tuple_of_uint32_and_string");
}

#[test]
fn default_fields_are_optional_like_schemars() {
    for (d, o) in PAIRS {
        assert_eq!(
            ours::<WithDefault>(false, d),
            without_default(theirs::<WithDefault>(false, o))
        );
    }
}

#[test]
fn flatten_merges_fields() {
    // schemars also copies the flattened type's description onto the
    // parent; the derive keeps the parent's own (none here)
    let v = ours::<Flat>(false, Dialect::AnthropicTool);
    assert_eq!(keys(&v["properties"]), ["a", "x"]);
    assert_eq!(v["required"], serde_json::json!(["a", "x"]));
    assert!(v.get("description").is_none());
}

#[test]
fn internally_tagged_newtype_keeps_its_fields() {
    // 1.5 lost `x` here: the reference's inlining let the variant's
    // `properties` replace the struct's
    let v = ours::<E3Newtype>(false, Dialect::AnthropicTool);
    let variant = &v["oneOf"][0];
    assert_eq!(keys(&variant["properties"]), ["kind", "x"]);
    assert_eq!(variant["required"], serde_json::json!(["kind", "x"]));
}

#[test]
fn transparent_and_newtype_are_the_inner_schema() {
    assert!(std::ptr::eq(Tr::schema(), Inner::schema()));
}

#[test]
fn generic_instances_are_distinct_and_stable() {
    assert!(!std::ptr::eq(Gen::<u32>::schema(), Gen::<String>::schema()));
    assert!(std::ptr::eq(Gen::<u32>::schema(), Gen::<u32>::schema()));
}

/// Keys of an object in the order the bytes list them.
fn ordered_keys(raw: &str, path: &[&str]) -> Vec<String> {
    let mut p = PartialParser::new();
    p.feed(raw);
    let mut v = p.root().unwrap();
    for k in path {
        v = v.get(k).unwrap();
    }
    v.entries().map(|(k, _)| k.to_owned()).collect()
}

fn keys(v: &Value) -> Vec<String> {
    let raw = serde_json::to_string(v).unwrap();
    ordered_keys(&raw, &[])
}

#[test]
fn properties_follow_declaration_order() {
    let want = [
        "invoice_number",
        "issued_on",
        "vendor",
        "vendor_address",
        "bill_to",
        "currency",
        "items",
        "notes",
        "total",
    ];
    for d in [
        Dialect::Strict,
        Dialect::AnthropicOutput,
        Dialect::AnthropicTool,
        Dialect::GeminiJson,
    ] {
        let raw = entry_for::<Invoice>(false).get(d);
        assert_eq!(ordered_keys(raw, &["properties"]), want, "{d:?}");
        let raw = entry_for::<Invoice>(true).get(d);
        let line = ordered_keys(
            raw,
            &[
                "properties",
                "items",
                "items",
                "properties",
                "items",
                "items",
                "properties",
            ],
        );
        assert_eq!(
            line,
            ["description", "quantity", "unit_price", "sku"],
            "{d:?}"
        );
    }
    let raw = entry_for::<Invoice>(false).get(Dialect::Strict);
    let mut p = PartialParser::new();
    p.feed(raw);
    let req: Vec<String> = p
        .root()
        .unwrap()
        .get("required")
        .unwrap()
        .items()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(req, want);
}

#[test]
fn schema_attributes() {
    #[derive(Deserialize, Schema)]
    struct S {
        #[schema(description = "when", format = "date-time")]
        at: String,
        #[schema(min = 1, max = 5)]
        stars: u8,
        #[schema(min = -1.5)]
        temp: f64,
        #[serde(deserialize_with = "de_flag")]
        #[schema(like = bool)]
        flag: bool,
    }
    fn de_flag<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
        bool::deserialize(d)
    }
    let v = ours::<S>(false, Dialect::GeminiJson);
    assert_eq!(v["properties"]["at"]["description"], "when");
    assert_eq!(v["properties"]["at"]["format"], "date-time");
    assert_eq!(v["properties"]["stars"]["minimum"], 1);
    assert_eq!(v["properties"]["stars"]["maximum"], 5);
    assert_eq!(v["properties"]["temp"]["minimum"], -1.5);
    assert_eq!(v["properties"]["flag"]["type"], "boolean");
}
