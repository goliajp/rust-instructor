//! A JSON value with keys in insertion order, for building schemas.

use crate::json;

/// A JSON value with keys in insertion order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum J {
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

pub(crate) fn write(v: &J, out: &mut Vec<u8>) {
    match v {
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
