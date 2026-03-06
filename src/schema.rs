use schemars::schema::RootSchema;
use serde_json::{Map, Value};

/// Convert a schemars RootSchema to a clean JSON Value with inlined refs.
pub(crate) fn from_root_schema(root: &RootSchema) -> Value {
    let mut value = serde_json::to_value(root).unwrap_or(Value::Null);

    if let Some(obj) = value.as_object_mut() {
        obj.remove("$schema");
    }

    let definitions = extract_definitions(&value);
    inline_refs(&mut value, &definitions);

    if let Some(obj) = value.as_object_mut() {
        obj.remove("definitions");
    }

    value
}

/// Prepare the `response_format` payload for OpenAI structured output (strict mode).
pub(crate) fn wrap_for_openai(root: &RootSchema, name: &str) -> Value {
    let mut schema = from_root_schema(root);
    remove_key_recursive(&mut schema, "title");
    remove_key_recursive(&mut schema, "format");
    add_additional_properties_false(&mut schema);
    make_all_properties_required(&mut schema);

    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": true,
            "schema": schema
        }
    })
}

/// Clean up the schema for use as Anthropic tool `input_schema`.
pub(crate) fn clean_for_anthropic(root: &RootSchema) -> Value {
    let mut schema = from_root_schema(root);
    remove_key_recursive(&mut schema, "title");
    schema
}

fn extract_definitions(value: &Value) -> Map<String, Value> {
    value
        .get("definitions")
        .and_then(|d| d.as_object())
        .cloned()
        .unwrap_or_default()
}

fn inline_refs(value: &mut Value, definitions: &Map<String, Value>) {
    match value {
        Value::Object(map) => {
            if let Some(ref_str) = map.get("$ref").and_then(|r| r.as_str()).map(String::from) {
                if let Some(name) = ref_str.strip_prefix("#/definitions/") {
                    if let Some(def) = definitions.get(name) {
                        let mut resolved = def.clone();
                        inline_refs(&mut resolved, definitions);
                        *value = resolved;
                        return;
                    }
                }
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
            if map.get("type").and_then(|t| t.as_str()) == Some("object") {
                if let Some(props) = map.get("properties").and_then(|p| p.as_object()) {
                    let all_keys: Vec<Value> =
                        props.keys().map(|k| Value::String(k.clone())).collect();
                    map.insert("required".into(), Value::Array(all_keys));
                }
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
    fn test_from_root_schema_removes_meta() {
        let root = schemars::schema_for!(Simple);
        let value = from_root_schema(&root);
        assert!(value.get("$schema").is_none());
    }

    #[test]
    fn test_from_root_schema_inlines_refs() {
        let root = schemars::schema_for!(Nested);
        let value = from_root_schema(&root);
        assert!(value.get("definitions").is_none());
        let json = serde_json::to_string(&value).unwrap();
        assert!(!json.contains("$ref"));
    }

    #[test]
    fn test_openai_strict_mode() {
        let root = schemars::schema_for!(WithOption);
        let wrapped = wrap_for_openai(&root, "WithOption");
        let js = &wrapped["json_schema"];
        assert_eq!(js["name"], "WithOption");
        assert_eq!(js["strict"], true);

        let schema = &js["schema"];
        assert!(schema.get("title").is_none());
        assert_eq!(schema["additionalProperties"], false);

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&Value::String("name".into())));
        assert!(required.contains(&Value::String("email".into())));
    }

    #[test]
    fn test_anthropic_schema() {
        let root = schemars::schema_for!(Simple);
        let value = clean_for_anthropic(&root);
        assert!(value.get("title").is_none());
        assert!(value.get("$schema").is_none());
        assert_eq!(value["type"], "object");
    }

    #[test]
    fn test_nested_additional_properties() {
        let root = schemars::schema_for!(Nested);
        let wrapped = wrap_for_openai(&root, "Nested");
        let schema = &wrapped["json_schema"]["schema"];
        assert_eq!(schema["additionalProperties"], false);

        let contact_schema = &schema["properties"]["contact"];
        assert_eq!(contact_schema["additionalProperties"], false);
    }
}
