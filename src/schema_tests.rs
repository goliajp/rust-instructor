use super::*;
use schemars::JsonSchema;

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Simple {
    name: String,
    age: u32,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct WithOption {
    name: String,
    email: Option<String>,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Nested {
    contact: Simple,
    tags: Vec<String>,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Node {
    name: String,
    children: Vec<Node>,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Doc {
    title: String,
    root: Node,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Article {
    title: String,
    format: String,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
enum Shape {
    Circle { r: f64 },
    Square { side: f64 },
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Drawing {
    shape: Shape,
}

fn json<T: JsonSchema + 'static>(list: bool, d: Dialect) -> Value {
    serde_json::from_str(entry_for::<T>(list).get(d).get()).unwrap()
}

fn has_ref(v: &Value) -> bool {
    v.to_string().contains("$ref")
}

#[test]
fn removes_meta_and_inlines_refs() {
    let v = json::<Nested>(false, Dialect::AnthropicTool);
    assert!(v.get("$schema").is_none());
    assert!(v.get("$defs").is_none());
    assert!(!has_ref(&v));
    assert_eq!(v["properties"]["contact"]["type"], "object");
}

#[test]
fn openai_strict_shape() {
    let rf = json::<WithOption>(false, Dialect::OpenAi);
    assert_eq!(rf["type"], "json_schema");
    let js = &rf["json_schema"];
    assert_eq!(js["name"], "WithOption");
    assert_eq!(js["strict"], true);
    let s = &js["schema"];
    assert!(s.get("title").is_none());
    assert_eq!(s["additionalProperties"], false);
    let required = s["required"].as_array().unwrap();
    assert!(required.contains(&"name".into()));
    assert!(required.contains(&"email".into()));
}

#[test]
fn nested_objects_are_closed() {
    let s = json::<Nested>(false, Dialect::OpenAi)["json_schema"]["schema"].clone();
    assert_eq!(s["properties"]["contact"]["additionalProperties"], false);
}

#[test]
fn fields_named_like_keywords_survive() {
    // a property map is not a schema: `title` and `format` here are fields
    for d in [
        Dialect::OpenAi,
        Dialect::AnthropicOutput,
        Dialect::AnthropicTool,
        Dialect::Gemini,
    ] {
        let mut v = json::<Article>(false, d);
        if d == Dialect::OpenAi {
            v = v["json_schema"]["schema"].clone();
        }
        assert!(v["properties"]["title"].is_object(), "{d:?}: {v}");
        assert!(v["properties"]["format"].is_object(), "{d:?}: {v}");
    }
}

#[test]
fn recursive_type_keeps_root_ref() {
    let e = entry_for::<Node>(false);
    assert!(e.recursive());
    let v = json::<Node>(false, Dialect::OpenAi);
    assert!(has_ref(&v));
}

#[test]
fn nested_recursive_type_does_not_overflow() {
    // used to recurse forever while inlining `$defs`
    let e = entry_for::<Doc>(false);
    assert!(e.recursive());
    let v = json::<Doc>(false, Dialect::OpenAi);
    let s = &v["json_schema"]["schema"];
    assert_eq!(s["properties"]["root"]["type"], "object");
    let node = &s["$defs"]["Node"];
    assert_eq!(node["additionalProperties"], false);
    assert_eq!(
        node["properties"]["children"]["items"]["$ref"],
        "#/$defs/Node"
    );
}

#[test]
fn list_is_wrapped_in_object_root() {
    let v = json::<Vec<Simple>>(true, Dialect::OpenAi);
    let s = &v["json_schema"]["schema"];
    assert_eq!(s["type"], "object");
    assert_eq!(s["required"], serde_json::json!(["items"]));
    assert_eq!(s["properties"]["items"]["type"], "array");
    assert_eq!(s["properties"]["items"]["items"]["type"], "object");
    assert!(!has_ref(s));
}

#[test]
fn list_of_recursive_type_moves_defs_to_wrapper() {
    let v = json::<Vec<Node>>(true, Dialect::GeminiJson);
    assert_eq!(v["type"], "object");
    assert!(v["$defs"]["Node"].is_object());
}

#[test]
fn anthropic_output_drops_unsupported_keywords() {
    let v = json::<Simple>(false, Dialect::AnthropicOutput);
    let age = &v["properties"]["age"];
    assert!(age.get("minimum").is_none(), "{age}");
    assert!(age.get("format").is_none(), "{age}");
    assert_eq!(v["additionalProperties"], false);
}

#[test]
fn one_of_becomes_any_of_for_strict_dialects() {
    for d in [Dialect::OpenAi, Dialect::AnthropicOutput] {
        let v = json::<Drawing>(false, d);
        let s = v.to_string();
        assert!(!s.contains("oneOf"), "{d:?}: {s}");
        assert!(s.contains("anyOf"), "{d:?}: {s}");
    }
}

#[test]
fn gemini_basic() {
    let v = json::<Simple>(false, Dialect::Gemini);
    assert!(v.get("title").is_none());
    assert!(v.get("additionalProperties").is_none());
    assert!(v["properties"]["age"].get("format").is_none());
    assert_eq!(v["type"], "object");
}

#[test]
fn gemini_nullable() {
    let v = json::<WithOption>(false, Dialect::Gemini);
    let email = &v["properties"]["email"];
    assert_eq!(email["nullable"], true);
    assert_eq!(email["type"], "string");
    assert!(email.get("anyOf").is_none());
}

#[test]
fn gemini_nested_has_no_additional_properties() {
    let v = json::<Nested>(false, Dialect::Gemini);
    assert!(!v.to_string().contains("additionalProperties"));
    assert_eq!(v["properties"]["contact"]["type"], "object");
}

#[test]
fn cached_bytes_are_shared() {
    let a = entry_for::<Simple>(false);
    let b = entry_for::<Simple>(false);
    assert!(Arc::ptr_eq(&a, &b));
    let p1 = a.get(Dialect::OpenAi) as *const RawValue;
    let p2 = b.get(Dialect::OpenAi) as *const RawValue;
    assert_eq!(p1, p2);
}

#[test]
fn name_is_sanitized() {
    assert_eq!(sanitize_name("Wrapper<Foo>"), "Wrapper_Foo_");
    assert_eq!(sanitize_name(""), "output");
    assert_eq!(sanitize_name(&"x".repeat(80)).len(), 64);
}
