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

const WRAP: &str = "value";

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
fn is_map(m: &Map) -> bool {
    has_type(m, "object")
        && get(m, "properties").is_none()
        && matches!(
            get(m, "additionalProperties"),
            Some(J::Obj(_) | J::Bool(true))
        )
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

/// A strict answer back to the type's own JSON, checked against the
/// constraints of the canonical schema `base`.
pub(crate) fn unwire(base: &J, answer: J) -> Result<J, String> {
    // a compatible server that does not enforce the schema may answer
    // with the bare value
    let v = match answer {
        J::Obj(mut m) if !object_root(base) && m.len() == 1 && m[0].0 == WRAP => {
            m.pop().expect("one entry").1
        }
        other => other,
    };
    Back { root: base }.value(base, v, "")
}

/// One item of a list answer (`{"items": [..]}`) back to the item type's
/// JSON.
pub(crate) fn unwire_item(base: &J, item: J) -> Result<J, String> {
    let schema = list_items(base).ok_or("not a list schema")?;
    Back { root: base }.value(schema, item, "/items/-")
}

fn list_items(base: &J) -> Option<&J> {
    let J::Obj(m) = base else { return None };
    let Some(J::Obj(p)) = get(m, "properties") else {
        return None;
    };
    let Some(J::Obj(items)) = get(p, "items") else {
        return None;
    };
    get(items, "items")
}

struct Back<'a> {
    root: &'a J,
}

fn at(path: &str) -> &str {
    if path.is_empty() { "the answer" } else { path }
}

fn kind(v: &J) -> &'static str {
    match v {
        J::Null => "null",
        J::Bool(_) => "boolean",
        J::Num(_) => "number",
        J::Str(_) => "string",
        J::Arr(_) => "array",
        J::Obj(_) => "object",
    }
}

impl<'a> Back<'a> {
    fn resolve(&self, r: &str) -> Result<&'a J, String> {
        if r == "#" {
            return Ok(self.root);
        }
        let name = r.strip_prefix("#/$defs/").ok_or("unknown $ref")?;
        let J::Obj(m) = self.root else {
            return Err("unknown $ref".into());
        };
        match get(m, "$defs") {
            Some(J::Obj(d)) => get(d, name).ok_or_else(|| "unknown $ref".into()),
            _ => Err("unknown $ref".into()),
        }
    }

    fn value(&self, schema: &'a J, v: J, path: &str) -> Result<J, String> {
        let J::Obj(m) = schema else { return Ok(v) };
        if let Some(J::Str(r)) = get(m, "$ref") {
            return self.value(self.resolve(r)?, v, path);
        }
        if let Some(J::Arr(alts)) = get(m, "anyOf").or_else(|| get(m, "oneOf")) {
            let mut first = None;
            for alt in alts {
                match self.value(alt, v.clone(), path) {
                    Ok(x) => return super::check::constraints(m, x, path),
                    Err(e) => {
                        first.get_or_insert(e);
                    }
                }
            }
            return Err(first.unwrap_or_else(|| format!("{}: no variant matches", at(path))));
        }
        let out = self.shape(m, v, path)?;
        super::check::constraints(m, out, path)
    }

    fn shape(&self, m: &'a Map, v: J, path: &str) -> Result<J, String> {
        if let Some(c) = get(m, "const")
            && *c != v
        {
            return Err(format!("{}: expected {}", at(path), text(c)));
        }
        if let Some(J::Arr(e)) = get(m, "enum")
            && !e.contains(&v)
        {
            return Err(format!(
                "{}: {} is not one of the allowed values",
                at(path),
                text(&v)
            ));
        }
        if v == J::Null {
            return match get(m, "type") {
                None => Ok(v),
                Some(_) if has_type(m, "null") => Ok(v),
                Some(_) => Err(format!("{}: expected a value, got null", at(path))),
            };
        }
        if is_map(m) {
            return self.map(m, v, path);
        }
        if let Some(J::Arr(items)) = get(m, "prefixItems") {
            return self.tuple(items, v, path);
        }
        let got = kind(&v);
        let expect = |t: &str, ok: bool| {
            if ok || get(m, "type").is_none() {
                Ok(())
            } else {
                Err(format!("{}: expected {t}, got {got}", at(path)))
            }
        };
        match v {
            J::Obj(entries) => {
                expect("an object", has_type(m, "object"))?;
                self.object(m, entries, path)
            }
            J::Arr(xs) => {
                expect("an array", has_type(m, "array"))?;
                let item = get(m, "items");
                let out = xs
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| match item {
                        Some(sc) => self.value(sc, x, &format!("{path}/{i}")),
                        None => Ok(x),
                    })
                    .collect::<Result<_, _>>()?;
                Ok(J::Arr(out))
            }
            J::Str(_) => expect("a string", has_type(m, "string")).map(|()| v),
            J::Num(_) => {
                expect("a number", has_type(m, "number") || has_type(m, "integer")).map(|()| v)
            }
            J::Bool(_) => expect("a boolean", has_type(m, "boolean")).map(|()| v),
            J::Null => Ok(v),
        }
    }

    fn object(&self, m: &'a Map, entries: Map, path: &str) -> Result<J, String> {
        let props = match get(m, "properties") {
            Some(J::Obj(p)) => Some(p),
            _ => None,
        };
        if let Some(J::Arr(req)) = get(m, "required") {
            for r in req {
                if let J::Str(name) = r
                    && !entries.iter().any(|(k, _)| k == name)
                {
                    return Err(format!("{}: missing \"{name}\"", at(path)));
                }
            }
        }
        let extra = get(m, "additionalProperties");
        let mut out = Map::with_capacity(entries.len());
        for (k, x) in entries {
            let p = format!("{path}/{k}");
            let x = match (props.and_then(|ps| get(ps, &k)), extra) {
                (Some(sc), _) => self.value(sc, x, &p)?,
                (None, Some(J::Bool(false))) => {
                    return Err(format!("{}: unexpected \"{k}\"", at(path)));
                }
                (None, Some(sc @ J::Obj(_))) => self.value(sc, x, &p)?,
                (None, _) => x,
            };
            out.push((k, x));
        }
        Ok(J::Obj(out))
    }

    fn map(&self, m: &'a Map, v: J, path: &str) -> Result<J, String> {
        let value = get(m, "additionalProperties").expect("checked by is_map");
        let J::Arr(pairs) = v else {
            return Err(format!(
                "{}: expected a list of key and value pairs",
                at(path)
            ));
        };
        let mut out = Map::with_capacity(pairs.len());
        for (i, pair) in pairs.into_iter().enumerate() {
            let J::Obj(mut kv) = pair else {
                return Err(format!("{path}/{i}: expected an object with key and value"));
            };
            let (Some(J::Str(k)), Some(x)) = (remove(&mut kv, "key"), remove(&mut kv, "value"))
            else {
                return Err(format!("{path}/{i}: expected a string key and a value"));
            };
            if !kv.is_empty() || out.iter().any(|(o, _)| *o == k) {
                return Err(format!(
                    "{path}/{i}: an extra field or a repeated key \"{k}\""
                ));
            }
            let x = self.value(value, x, &format!("{path}/{k}"))?;
            out.push((k, x));
        }
        Ok(J::Obj(out))
    }

    fn tuple(&self, items: &'a [J], v: J, path: &str) -> Result<J, String> {
        let J::Obj(mut entries) = v else {
            return Err(format!(
                "{}: expected an object with keys \"0\" to \"{}\"",
                at(path),
                items.len().saturating_sub(1)
            ));
        };
        let mut out = Vec::with_capacity(items.len());
        for (i, sc) in items.iter().enumerate() {
            let key = i.to_string();
            let x = remove(&mut entries, &key)
                .ok_or_else(|| format!("{}: missing \"{key}\"", at(path)))?;
            out.push(self.value(sc, x, &format!("{path}/{key}"))?);
        }
        if let Some((k, _)) = entries.first() {
            return Err(format!("{}: unexpected \"{k}\"", at(path)));
        }
        Ok(J::Arr(out))
    }
}

pub(super) fn text(v: &J) -> String {
    let mut out = Vec::new();
    super::tree::write(v, &mut out);
    let t = String::from_utf8(out).expect("the writer emits utf-8");
    crate::text::clip(&t, 80).to_owned()
}
