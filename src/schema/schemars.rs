//! Feature `schemars`: extraction for types that derive
//! `schemars::JsonSchema` instead of [`Schema`](super::Schema).
//!
//! The schemars schema is normalized the way 1.x did it (references inlined,
//! `$defs` kept only for recursive types) and then goes through the same
//! per-provider adjustments as derived schemas. Property order is
//! alphabetical unless `schemars/preserve_order` is enabled.

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

use schemars::JsonSchema;
use serde_json::{Map, Value};

use super::cache::{SchemaEntry, sanitize_name};
use super::lock;
use super::tree::J;

/// Cached schema entry for a schemars type.
pub(crate) fn entry_for<T: JsonSchema + 'static>(list: bool) -> &'static SchemaEntry {
    static MAP: OnceLock<Mutex<HashMap<(TypeId, bool), &'static SchemaEntry>>> = OnceLock::new();
    let mut m = lock(MAP.get_or_init(Default::default));
    m.entry((TypeId::of::<T>(), list)).or_insert_with(|| {
        // 1.x built list schemas from `Vec<T>`
        let root = if list {
            schemars::schema_for!(Vec<T>).to_value()
        } else {
            schemars::schema_for!(T).to_value()
        };
        let name = T::schema_name().into_owned();
        let (base, recursive) = normalize(root, list);
        let name = if list {
            format!("Array_of_{name}")
        } else {
            name
        };
        Box::leak(Box::new(SchemaEntry::from_parts(
            sanitize_name(&name),
            to_j(base),
            recursive,
        )))
    })
}

fn to_j(v: Value) -> J {
    match v {
        Value::Null => J::Num("null".into()),
        Value::Bool(b) => J::Bool(b),
        Value::Number(n) => J::Num(n.to_string()),
        Value::String(s) => J::Str(s),
        Value::Array(a) => J::Arr(a.into_iter().map(to_j).collect()),
        Value::Object(m) => J::Obj(m.into_iter().map(|(k, v)| (k, to_j(v))).collect()),
    }
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
                            match (k.as_str(), obj.get_mut(k), val) {
                                ("$ref", _, _) => {}
                                // an internally tagged variant: tag + fields
                                ("properties", Some(Value::Object(have)), Value::Object(add)) => {
                                    let mut merged = add.clone();
                                    merged.extend(std::mem::take(have));
                                    *have = merged;
                                }
                                ("required", Some(Value::Array(have)), Value::Array(add)) => {
                                    let mut merged = add.clone();
                                    merged.extend(std::mem::take(have));
                                    *have = merged;
                                }
                                _ => {
                                    obj.insert(k.clone(), val.clone());
                                }
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
