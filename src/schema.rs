use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use schemars::{JsonSchema, Schema};
use serde_json::{Map, Value};

/// Cached schema generation. Uses thread_local to avoid lock contention
/// in concurrent batch scenarios. Returns `Arc<Schema>` to eliminate
/// full schema cloning on every extraction request.
pub(crate) fn cached_schema_for<T: JsonSchema + 'static>() -> (Arc<Schema>, String) {
    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<TypeId, (Arc<Schema>, String)>> =
            std::cell::RefCell::new(HashMap::new());
    }
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let entry = cache.entry(TypeId::of::<T>()).or_insert_with(|| {
            let schema = schemars::schema_for!(T);
            let name = T::schema_name().to_string();
            (Arc::new(schema), name)
        });
        (Arc::clone(&entry.0), entry.1.clone())
    })
}

/// Convert a schemars Schema to a clean JSON Value with inlined refs.
pub(crate) fn from_schema(schema: &Schema) -> Value {
    let mut value = serde_json::to_value(schema).unwrap_or(Value::Null);

    if let Some(obj) = value.as_object_mut() {
        obj.remove("$schema");
    }

    let definitions = extract_definitions(&value);
    inline_refs(&mut value, &definitions);

    if let Some(obj) = value.as_object_mut() {
        obj.remove("$defs");
    }

    value
}

/// Prepare the `response_format` payload for OpenAI structured output (strict mode).
pub(crate) fn wrap_for_openai(schema: &Schema, name: &str) -> Value {
    let mut schema_value = from_schema(schema);
    remove_key_recursive(&mut schema_value, "title");
    remove_key_recursive(&mut schema_value, "format");
    add_additional_properties_false(&mut schema_value);
    make_all_properties_required(&mut schema_value);

    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": true,
            "schema": schema_value
        }
    })
}

/// Clean up the schema for use as Anthropic tool `input_schema`.
pub(crate) fn clean_for_anthropic(schema: &Schema) -> Value {
    let mut schema_value = from_schema(schema);
    remove_key_recursive(&mut schema_value, "title");
    schema_value
}

/// Clean up the schema for Gemini `response_schema`.
///
/// Removes `title`, `format`, and `additionalProperties`, and converts
/// `anyOf: [{type: T}, {type: "null"}]` into `{type: T, nullable: true}`.
pub(crate) fn clean_for_gemini(schema: &Schema) -> Value {
    let mut schema_value = from_schema(schema);
    remove_key_recursive(&mut schema_value, "title");
    remove_key_recursive(&mut schema_value, "format");
    remove_key_recursive(&mut schema_value, "additionalProperties");
    convert_nullable_for_gemini(&mut schema_value);
    schema_value
}

/// Convert nullable types for Gemini compatibility.
///
/// Handles two patterns:
/// 1. schemars v1: `"type": ["string", "null"]` → `"type": "string", "nullable": true`
/// 2. anyOf pattern: `anyOf: [{type: T}, {type: "null"}]` → `{type: T, nullable: true}`
fn convert_nullable_for_gemini(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // handle schemars v1 type-array pattern: "type": ["string", "null"]
            if let Some(type_val) = map.get("type")
                && let Some(type_arr) = type_val.as_array()
                && type_arr.len() == 2
            {
                let null_idx = type_arr.iter().position(|v| v.as_str() == Some("null"));
                if let Some(idx) = null_idx {
                    let non_null_idx = 1 - idx;
                    let real_type = type_arr[non_null_idx].clone();
                    map.insert("type".into(), real_type);
                    map.insert("nullable".into(), Value::Bool(true));
                }
            }

            // handle anyOf pattern: anyOf: [{type: T}, {type: "null"}]
            if let Some(any_of) = map.remove("anyOf")
                && let Some(variants) = any_of.as_array()
                && variants.len() == 2
            {
                let null_idx = variants
                    .iter()
                    .position(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"));
                if let Some(idx) = null_idx {
                    let non_null_idx = 1 - idx;
                    let mut non_null = variants[non_null_idx].clone();
                    if let Some(obj) = non_null.as_object_mut() {
                        obj.insert("nullable".into(), Value::Bool(true));
                    }
                    convert_nullable_for_gemini(&mut non_null);
                    if let Some(inner) = non_null.as_object() {
                        for (k, v) in inner {
                            map.insert(k.clone(), v.clone());
                        }
                    }
                } else {
                    // not a nullable pattern, restore anyOf
                    map.insert("anyOf".into(), any_of);
                }
            }

            for v in map.values_mut() {
                convert_nullable_for_gemini(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                convert_nullable_for_gemini(v);
            }
        }
        _ => {}
    }
}

fn extract_definitions(value: &Value) -> Map<String, Value> {
    value
        .get("$defs")
        .and_then(|d| d.as_object())
        .cloned()
        .unwrap_or_default()
}

fn inline_refs(value: &mut Value, definitions: &Map<String, Value>) {
    match value {
        Value::Object(map) => {
            if let Some(ref_str) = map.get("$ref").and_then(|r| r.as_str()).map(String::from)
                && let Some(name) = ref_str.strip_prefix("#/$defs/")
                && let Some(def) = definitions.get(name)
            {
                let mut resolved = def.clone();
                inline_refs(&mut resolved, definitions);
                *value = resolved;
                return;
            }
            for v in map.values_mut() {
                inline_refs(v, definitions);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                inline_refs(v, definitions);
            }
        }
        _ => {}
    }
}

fn remove_key_recursive(value: &mut Value, key: &str) {
    match value {
        Value::Object(map) => {
            map.remove(key);
            for v in map.values_mut() {
                remove_key_recursive(v, key);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                remove_key_recursive(v, key);
            }
        }
        _ => {}
    }
}

fn add_additional_properties_false(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("object")
                && map.contains_key("properties")
            {
                map.insert("additionalProperties".into(), Value::Bool(false));
            }
            for v in map.values_mut() {
                add_additional_properties_false(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                add_additional_properties_false(v);
            }
        }
        _ => {}
    }
}

fn make_all_properties_required(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("object")
                && let Some(props) = map.get("properties").and_then(|p| p.as_object())
            {
                let all_keys: Vec<Value> = props.keys().map(|k| Value::String(k.clone())).collect();
                map.insert("required".into(), Value::Array(all_keys));
            }
            for v in map.values_mut() {
                make_all_properties_required(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                make_all_properties_required(v);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;

    #[derive(JsonSchema)]
    struct Simple {
        name: String,
        age: u32,
    }

    #[derive(JsonSchema)]
    struct WithOption {
        name: String,
        email: Option<String>,
    }

    #[derive(JsonSchema)]
    struct Nested {
        contact: Simple,
        tags: Vec<String>,
    }

    #[test]
    fn test_from_schema_removes_meta() {
        let schema = schemars::schema_for!(Simple);
        let value = from_schema(&schema);
        assert!(value.get("$schema").is_none());
    }

    #[test]
    fn test_from_schema_inlines_refs() {
        let schema = schemars::schema_for!(Nested);
        let value = from_schema(&schema);
        assert!(value.get("$defs").is_none());
        let json = serde_json::to_string(&value).unwrap();
        assert!(!json.contains("$ref"));
    }

    #[test]
    fn test_openai_strict_mode() {
        let schema = schemars::schema_for!(WithOption);
        let wrapped = wrap_for_openai(&schema, "WithOption");
        let js = &wrapped["json_schema"];
        assert_eq!(js["name"], "WithOption");
        assert_eq!(js["strict"], true);

        let schema_val = &js["schema"];
        assert!(schema_val.get("title").is_none());
        assert_eq!(schema_val["additionalProperties"], false);

        let required = schema_val["required"].as_array().unwrap();
        assert!(required.contains(&Value::String("name".into())));
        assert!(required.contains(&Value::String("email".into())));
    }

    #[test]
    fn test_anthropic_schema() {
        let schema = schemars::schema_for!(Simple);
        let value = clean_for_anthropic(&schema);
        assert!(value.get("title").is_none());
        assert!(value.get("$schema").is_none());
        assert_eq!(value["type"], "object");
    }

    #[test]
    fn test_nested_additional_properties() {
        let schema = schemars::schema_for!(Nested);
        let wrapped = wrap_for_openai(&schema, "Nested");
        let schema_val = &wrapped["json_schema"]["schema"];
        assert_eq!(schema_val["additionalProperties"], false);

        let contact_schema = &schema_val["properties"]["contact"];
        assert_eq!(contact_schema["additionalProperties"], false);
    }

    #[test]
    fn test_gemini_schema_basic() {
        let schema = schemars::schema_for!(Simple);
        let value = clean_for_gemini(&schema);
        assert!(value.get("title").is_none());
        assert!(value.get("format").is_none());
        assert!(value.get("additionalProperties").is_none());
        assert_eq!(value["type"], "object");
    }

    #[test]
    fn test_gemini_schema_nullable() {
        let schema = schemars::schema_for!(WithOption);
        let value = clean_for_gemini(&schema);
        let email = &value["properties"]["email"];
        assert_eq!(email["nullable"], true);
        assert!(email.get("anyOf").is_none());
        // should have a single type string, not an array
        assert!(email["type"].is_string());
        assert_eq!(email["type"], "string");
    }

    #[test]
    fn test_gemini_schema_no_additional_properties() {
        let schema = schemars::schema_for!(Nested);
        let value = clean_for_gemini(&schema);
        let json = serde_json::to_string(&value).unwrap();
        assert!(!json.contains("additionalProperties"));
    }

    #[test]
    fn test_gemini_schema_nested() {
        let schema = schemars::schema_for!(Nested);
        let value = clean_for_gemini(&schema);
        let contact = &value["properties"]["contact"];
        assert_eq!(contact["type"], "object");
        assert!(contact.get("title").is_none());
    }
}
