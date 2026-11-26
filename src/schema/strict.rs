//! Schemas for providers that take only a strict subset of JSON Schema
//! (OpenAI strict mode, Anthropic structured output), and the way back from
//! their answers.
//!
//! On the wire a map is a list of `{"key": .., "value": ..}` objects, a
//! tuple is an object with the properties `"0"`, `"1"`, …, and a root that
//! is not an object is wrapped as `{"value": ..}`. Keywords a provider does
//! not take are dropped from its schema and checked here once the answer
//! has been turned back into the type's own JSON.

use super::tree::{J, Map, get, obj, remove, s};

pub(super) const WRAP: &str = "value";

pub(crate) fn has_type(m: &Map, t: &str) -> bool {
    match get(m, "type") {
        Some(J::Str(x)) => x == t,
        Some(J::Arr(xs)) => xs.iter().any(|x| matches!(x, J::Str(x) if x == t)),
        _ => false,
    }
}

fn retype(m: &mut Map, from: &str, to: &str) {
    for (k, v) in m.iter_mut().filter(|(k, _)| k == "type") {
        debug_assert_eq!(k, "type");
        match v {
            J::Str(x) if x == from => *x = to.into(),
            J::Arr(xs) => xs
                .iter_mut()
                .filter(|x| matches!(x, J::Str(x) if x == from))
                .for_each(|x| *x = s(to)),
            _ => {}
        }
    }
}

/// A map: an object schema with no properties and a schema for its values.
pub(super) fn is_map(m: &Map) -> bool {
    has_type(m, "object")
        && get(m, "properties").is_none()
        && matches!(
            get(m, "additionalProperties"),
            Some(J::Obj(_) | J::Bool(true))
        )
}

/// Keywords a dialect drops and the answer check enforces.
pub(crate) const OPENAI_DROPS: [&str; 5] = [
    "minLength",
    "maxLength",
    "uniqueItems",
    "minProperties",
    "maxProperties",
];
pub(crate) const ANTHROPIC_DROPS: [&str; 11] = [
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
];

/// Keywords Gemini does not list; it ignores them, so they are checked on
/// the answer instead.
pub(crate) const GEMINI_DROPS: [&str; 3] = ["minLength", "maxLength", "uniqueItems"];

/// True when `f` holds for `v` or any of its subschemas (not property maps
/// themselves, so a field named `minimum` is not a keyword).
fn any_sub(v: &J, f: &mut impl FnMut(&J) -> bool) -> bool {
    if f(v) {
        return true;
    }
    let J::Obj(m) = v else { return false };
    m.iter().any(|(k, x)| match (k.as_str(), x) {
        ("properties" | "$defs", J::Obj(p)) => p.iter().any(|(_, x)| any_sub(x, f)),
        ("anyOf" | "oneOf" | "allOf" | "prefixItems", J::Arr(a)) => a.iter().any(|x| any_sub(x, f)),
        ("items" | "additionalProperties", x) => any_sub(x, f),
        _ => false,
    })
}

/// Whether an answer to a schema built with `drops` needs turning back or
/// checking: the schema has a dropped constraint (`minItems` above 1 counts
/// for Anthropic) or, when `reshape` is on, a map, a tuple or a wrapped
/// root.
pub(crate) fn reshaped(base: &J, drops: &[&str], min_items: bool, reshape: bool) -> bool {
    (reshape && !object_root(base))
        || any_sub(base, &mut |v| {
            let J::Obj(m) = v else { return false };
            (reshape && (is_map(m) || get(m, "prefixItems").is_some()))
                || m.iter().any(|(k, _)| drops.contains(&k.as_str()))
                || (min_items
                    && matches!(get(m, "minItems"), Some(J::Num(n)) if n != "0" && n != "1"))
        })
}

/// The schema accepts any value somewhere (`true` or `{}`), which no strict
/// subset can express.
pub(crate) fn has_any(base: &J) -> bool {
    any_sub(base, &mut |v| match v {
        J::Bool(true) => true,
        J::Obj(m) => m.is_empty(),
        _ => false,
    })
}

/// The canonical root is an object with named properties, so it needs no
/// wrapper.
pub(crate) fn object_root(base: &J) -> bool {
    matches!(base, J::Obj(m) if has_type(m, "object") && get(m, "properties").is_some())
}

fn closed(props: Map, required: Vec<J>) -> J {
    obj([
        ("type", s("object")),
        ("properties", J::Obj(props)),
        ("required", J::Arr(required)),
        ("additionalProperties", J::Bool(false)),
    ])
}

/// Maps and tuples to their strict forms; called on each subschema,
/// parents first, so the new subschemas are visited afterwards.
pub(crate) fn reshape(m: &mut Map) {
    if is_map(m) {
        let value = remove(m, "additionalProperties").expect("checked by is_map");
        let key = obj([("type", s("string"))]);
        let pair = closed(
            vec![("key".into(), key), ("value".into(), value)],
            vec![s("key"), s("value")],
        );
        retype(m, "object", "array");
        m.push(("items".into(), pair));
    } else if let Some(J::Arr(items)) = remove(m, "prefixItems") {
        remove(m, "minItems");
        remove(m, "maxItems");
        remove(m, "items");
        let names: Vec<String> = (0..items.len()).map(|i| i.to_string()).collect();
        let required = names.iter().map(|n| s(n)).collect();
        let props = names.into_iter().zip(items).collect();
        retype(m, "array", "object");
        m.push(("properties".into(), J::Obj(props)));
        m.push(("required".into(), J::Arr(required)));
        m.push(("additionalProperties".into(), J::Bool(false)));
    }
}

fn refs_root(v: &J) -> bool {
    match v {
        J::Obj(m) => m
            .iter()
            .any(|(k, x)| (k == "$ref" && *x == s("#")) || refs_root(x)),
        J::Arr(a) => a.iter().any(refs_root),
        _ => false,
    }
}

fn redirect_root(v: &mut J, to: &str) {
    match v {
        J::Obj(m) => m.iter_mut().for_each(|(k, x)| {
            if k == "$ref" && *x == s("#") {
                *x = s(to);
            } else {
                redirect_root(x, to);
            }
        }),
        J::Arr(a) => a.iter_mut().for_each(|x| redirect_root(x, to)),
        _ => {}
    }
}

/// Wrap a root that is not an object as `{"value": root}`. A root that
/// refers to itself moves into `$defs`.
pub(crate) fn wrap(mut v: J) -> J {
    let mut defs = match &mut v {
        J::Obj(m) => match remove(m, "$defs") {
            Some(J::Obj(d)) => d,
            _ => Map::new(),
        },
        _ => Map::new(),
    };
    let inner = if refs_root(&v) || defs.iter().any(|(_, d)| refs_root(d)) {
        let mut name = "Root".to_owned();
        while defs.iter().any(|(k, _)| *k == name) {
            name.push('_');
        }
        let to = format!("#/$defs/{name}");
        redirect_root(&mut v, &to);
        defs.iter_mut().for_each(|(_, d)| redirect_root(d, &to));
        defs.push((name, v));
        obj([("$ref", J::Str(to))])
    } else {
        v
    };
    let mut out = closed(vec![(WRAP.into(), inner)], vec![s(WRAP)]);
    if !defs.is_empty()
        && let J::Obj(m) = &mut out
    {
        m.push(("$defs".into(), J::Obj(defs)));
    }
    out
}

pub(super) fn text(v: &J) -> String {
    let mut out = Vec::new();
    super::tree::write(v, &mut out);
    let t = String::from_utf8(out).expect("the writer emits utf-8");
    crate::text::clip(&t, 80).to_owned()
}
