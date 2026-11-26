//! A JSON value with keys in insertion order, for building schemas and for
//! rewriting answers.

use crate::json;

/// A JSON value with keys in insertion order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum J {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

pub(crate) type Map = Vec<(String, J)>;

pub(crate) fn s(v: &str) -> J {
    J::Str(v.to_owned())
}

pub(crate) fn obj(pairs: impl IntoIterator<Item = (&'static str, J)>) -> J {
    J::Obj(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

pub(crate) fn get<'a>(m: &'a Map, key: &str) -> Option<&'a J> {
    m.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

pub(crate) fn remove(m: &mut Map, key: &str) -> Option<J> {
    let i = m.iter().position(|(k, _)| k == key)?;
    Some(m.remove(i).1)
}

/// Insert or replace, keeping the key's position when it exists.
pub(crate) fn set(m: &mut Map, key: &str, v: J) {
    match m.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = v,
        None => m.push((key.to_owned(), v)),
    }
}

/// Reads a value keeping each number's text as written, so an integer
/// beyond 2^53 or a float's digits pass through unchanged. Only the crate's
/// own JSON deserializer can produce it.
impl<'de> serde::Deserialize<'de> for J {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = json::Raw::deserialize(d)?;
        match raw.0.as_bytes().first() {
            Some(b'-' | b'0'..=b'9') => Ok(J::Num(raw.0.to_owned())),
            _ => json::from_str::<Inner>(raw.0)
                .map(|i| i.0)
                .map_err(serde::de::Error::custom),
        }
    }
}

/// One level of a value; nested values go through `J` again.
struct Inner(J);

impl<'de> serde::Deserialize<'de> for Inner {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(Visit).map(Inner)
    }
}

struct Visit;

impl<'de> serde::de::Visitor<'de> for Visit {
    type Value = J;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_unit<E>(self) -> Result<J, E> {
        Ok(J::Null)
    }
    fn visit_none<E>(self) -> Result<J, E> {
        Ok(J::Null)
    }
    fn visit_bool<E>(self, b: bool) -> Result<J, E> {
        Ok(J::Bool(b))
    }
    fn visit_str<E>(self, v: &str) -> Result<J, E> {
        Ok(J::Str(v.to_owned()))
    }
    fn visit_string<E>(self, v: String) -> Result<J, E> {
        Ok(J::Str(v))
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = a.next_element()? {
            out.push(v);
        }
        Ok(J::Arr(out))
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
        let mut out = Vec::new();
        while let Some((k, v)) = a.next_entry::<String, J>()? {
            out.push((k, v));
        }
        Ok(J::Obj(out))
    }
}

pub(crate) fn write(v: &J, out: &mut Vec<u8>) {
    match v {
        J::Null => out.extend_from_slice(b"null"),
        J::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        J::Num(n) => out.extend_from_slice(n.as_bytes()),
        J::Str(t) => json::write_str(out, t),
        J::Arr(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write(x, out);
            }
            out.push(b']');
        }
        J::Obj(m) => {
            out.push(b'{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                json::write_str(out, k);
                out.push(b':');
                write(x, out);
            }
            out.push(b'}');
        }
    }
}
