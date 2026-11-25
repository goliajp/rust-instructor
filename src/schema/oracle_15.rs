#![expect(dead_code, reason = "the 1.5 module is kept whole as a reference")]

//! The 1.5 schema pipeline (schemars, then normalization), kept as the
//! reference the derive is compared against.

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};

use schemars::JsonSchema;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

/// The schema shape each wire format expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Dialect {
    /// OpenAI `response_format` object (`json_schema`, strict).
    OpenAi,
    /// Anthropic `output_config.format.schema`.
    AnthropicOutput,
    /// Anthropic tool `input_schema`.
    AnthropicTool,
    /// Gemini `responseSchema` (OpenAPI subset, no `$ref`).
    Gemini,
    /// Gemini `responseJsonSchema` (JSON Schema, `$ref` allowed).
    GeminiJson,
}

const DIALECTS: usize = 5;

impl Dialect {
    fn index(self) -> usize {
        self as usize
    }
}

/// One type's schema, normalized once, with each dialect's bytes built on
/// first use.
pub(super) struct SchemaEntry {
    name: String,
    recursive: bool,
    base: Value,
    built: [OnceLock<Box<RawValue>>; DIALECTS],
}

impl SchemaEntry {
    pub(super) fn new(root: Value, name: &str, list: bool) -> Self {
        let (base, recursive) = normalize(root, list);
        Self {
            name: sanitize_name(name),
            recursive,
            base,
            built: Default::default(),
        }
    }

    /// True when the schema refers to itself, which some dialects cannot
    /// express.
    pub(super) fn recursive(&self) -> bool {
        self.recursive
    }

    /// Serialized schema for `dialect`, built on first call.
    pub(super) fn get(&self, dialect: Dialect) -> &RawValue {
        self.built[dialect.index()].get_or_init(|| {
            let v = build(&self.base, &self.name, dialect);
            RawValue::from_string(v.to_string()).expect("serde_json output is valid json")
        })
    }
}

/// Cached schema entry for `T`. `list` wraps the schema as `{"items": T}`.
pub(super) fn entry_for<T: JsonSchema + 'static>(list: bool) -> Arc<SchemaEntry> {
    thread_local! {
        static CACHE: RefCell<HashMap<(TypeId, bool), Arc<SchemaEntry>>> =
            RefCell::new(HashMap::new());
    }
    CACHE.with(|cache| {
        cache
            .borrow_mut()
            .entry((TypeId::of::<T>(), list))
            .or_insert_with(|| {
                let root = schemars::schema_for!(T).to_value();
                Arc::new(SchemaEntry::new(root, &T::schema_name(), list))
            })
            .clone()
    })
}

fn sanitize_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(64);
    if out.is_empty() {
        out.push_str("output");
    }
    out
}

/// Strip `$schema`, inline every non-recursive `$ref`, keep `$defs` only for
/// types that refer to themselves, and wrap list schemas in an object root.
fn normalize(mut root: Value, list: bool) -> (Value, bool) {
    let defs = match root.as_object_mut() {
        Some(obj) => {
            obj.remove("$schema");
            match obj.remove("$defs") {
                Some(Value::Object(d)) => d,
                _ => Map::new(),
            }
        }
        None => Map::new(),
    };

    let mut ctx = Inline {
        defs: &defs,
        stack: Vec::new(),
        kept: BTreeMap::new(),
        root_ref: false,
    };
    ctx.schema(&mut root);

    // kept definitions are themselves inlined, which may keep more
    let mut out_defs = Map::new();
    loop {
        let pending: Vec<String> = ctx
            .kept
            .iter()
            .filter(|(_, done)| !**done)
            .map(|(name, _)| name.clone())
            .collect();
        if pending.is_empty() {
            break;
        }
        for name in pending {
            ctx.kept.insert(name.clone(), true);
            let mut def = defs.get(&name).cloned().unwrap_or(Value::Bool(true));
            ctx.stack.push(name.clone());
            ctx.schema(&mut def);
            ctx.stack.pop();
            out_defs.insert(name, def);
        }
    }

    let recursive = ctx.root_ref || !out_defs.is_empty();

    let mut root = if list {
        let mut wrapper = Map::new();
        wrapper.insert("type".into(), "object".into());
        let mut props = Map::new();
        props.insert("items".into(), root);
        wrapper.insert("properties".into(), Value::Object(props));
        wrapper.insert("required".into(), Value::Array(vec!["items".into()]));
        Value::Object(wrapper)
    } else {
        root
    };
    if !out_defs.is_empty()
        && let Some(obj) = root.as_object_mut()
    {
        obj.insert("$defs".into(), Value::Object(out_defs));
    }
    (root, recursive)
}

struct Inline<'a> {
    defs: &'a Map<String, Value>,
    stack: Vec<String>,
    // name -> already emitted
    kept: BTreeMap<String, bool>,
    root_ref: bool,
}

impl Inline<'_> {
    fn schema(&mut self, v: &mut Value) {
        let Some(map) = v.as_object_mut() else {
            return;
        };
        if let Some(r) = map.get("$ref").and_then(Value::as_str) {
            if r == "#" {
                self.root_ref = true;
                return;
            }
            if let Some(name) = r.strip_prefix("#/$defs/") {
                let name = name.to_string();
                if self.stack.contains(&name) {
                    self.kept.entry(name).or_insert(false);
                    return;
                }
                if let Some(def) = self.defs.get(&name) {
                    let mut resolved = def.clone();
                    if let Some(obj) = resolved.as_object_mut() {
                        for (k, val) in map.iter() {
                            if k != "$ref" {
                                obj.insert(k.clone(), val.clone());
                            }
                        }
                    }
                    *v = resolved;
                    self.stack.push(name);
                    self.schema(v);
                    self.stack.pop();
                    return;
                }
            }
            return;
        }
        for_each_subschema(map, |s| self.schema(s));
    }
}

/// Visit every subschema position of one schema object (not property maps
/// themselves, so a field named `title` or `format` is never mistaken for a
/// keyword).
fn for_each_subschema(map: &mut Map<String, Value>, mut f: impl FnMut(&mut Value)) {
    for (key, val) in map.iter_mut() {
        match key.as_str() {
            "properties" | "patternProperties" | "$defs" | "definitions" => {
                if let Some(m) = val.as_object_mut() {
                    m.values_mut().for_each(&mut f);
                }
            }
            "anyOf" | "oneOf" | "allOf" | "prefixItems" => {
                if let Some(a) = val.as_array_mut() {
                    a.iter_mut().for_each(&mut f);
                }
            }
            "items" => match val {
                Value::Array(a) => a.iter_mut().for_each(&mut f),
                other => f(other),
            },
            "additionalProperties"
            | "not"
            | "if"
            | "then"
            | "else"
            | "contains"
            | "propertyNames"
            | "additionalItems"
            | "unevaluatedItems"
            | "unevaluatedProperties" => f(val),
            _ => {}
        }
    }
}

/// Apply `f` to every schema object in `v`, parents before children.
fn walk(v: &mut Value, f: &mut impl FnMut(&mut Map<String, Value>)) {
    if let Some(map) = v.as_object_mut() {
        f(map);
        for_each_subschema(map, |s| walk(s, f));
    }
}

fn build(base: &Value, name: &str, dialect: Dialect) -> Value {
    let mut v = base.clone();
    match dialect {
        Dialect::OpenAi => {
            walk(&mut v, &mut |m| {
                m.remove("title");
                m.remove("format");
                one_of_to_any_of(m);
                close_object(m);
            });
            serde_json::json!({
                "type": "json_schema",
                "json_schema": { "name": name, "strict": true, "schema": v }
            })
        }
        Dialect::AnthropicOutput => {
            walk(&mut v, &mut |m| {
                m.remove("title");
                strip_unsupported_anthropic(m);
                one_of_to_any_of(m);
                close_object(m);
            });
            v
        }
        Dialect::AnthropicTool => {
            walk(&mut v, &mut |m| {
                m.remove("title");
            });
            v
        }
        Dialect::Gemini => {
            walk(&mut v, &mut |m| {
                m.remove("title");
                m.remove("format");
                m.remove("additionalProperties");
                nullable_for_gemini(m);
            });
            v
        }
        Dialect::GeminiJson => v,
    }
}

fn is_object_type(m: &Map<String, Value>) -> bool {
    m.get("type").and_then(Value::as_str) == Some("object")
}

/// Strict modes: no extra keys, every property listed in `required`.
/// Optional fields stay optional through their nullable type.
fn close_object(m: &mut Map<String, Value>) {
    if !is_object_type(m) {
        return;
    }
    let Some(props) = m.get("properties").and_then(Value::as_object) else {
        return;
    };
    let keys: Vec<Value> = props.keys().map(|k| Value::String(k.clone())).collect();
    m.insert("required".into(), Value::Array(keys));
    m.insert("additionalProperties".into(), Value::Bool(false));
}

fn one_of_to_any_of(m: &mut Map<String, Value>) {
    if !m.contains_key("anyOf")
        && let Some(v) = m.remove("oneOf")
    {
        m.insert("anyOf".into(), v);
    }
}

// keywords anthropic structured output rejects with a 400
fn strip_unsupported_anthropic(m: &mut Map<String, Value>) {
    for k in [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    ] {
        m.remove(k);
    }
    if m.get("minItems")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
    {
        m.remove("minItems");
    }
    const FORMATS: [&str; 10] = [
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    ];
    if m.get("format")
        .and_then(Value::as_str)
        .is_some_and(|f| !FORMATS.contains(&f))
    {
        m.remove("format");
    }
}

/// `"type": ["T", "null"]` and `anyOf: [X, {"type": "null"}]` become
/// `nullable: true`, which is how Gemini's OpenAPI subset spells optional.
fn nullable_for_gemini(m: &mut Map<String, Value>) {
    if let Some(Value::Array(types)) = m.get("type")
        && types.len() == 2
        && let Some(idx) = types.iter().position(|t| t.as_str() == Some("null"))
    {
        let real = types[1 - idx].clone();
        m.insert("type".into(), real);
        m.insert("nullable".into(), Value::Bool(true));
    }
    let null_pos = m.get("anyOf").and_then(Value::as_array).and_then(|vs| {
        if vs.len() != 2 {
            return None;
        }
        vs.iter()
            .position(|v| v.get("type").and_then(Value::as_str) == Some("null"))
    });
    if let Some(idx) = null_pos
        && let Some(Value::Array(mut vs)) = m.remove("anyOf")
        && let Value::Object(inner) = vs.swap_remove(1 - idx)
    {
        for (k, v) in inner {
            m.entry(k).or_insert(v);
        }
        m.insert("nullable".into(), Value::Bool(true));
        // the merged schema may itself use the type-array form
        nullable_for_gemini(m);
    }
}
