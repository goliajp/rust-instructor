use std::fmt;

use super::{Kind, NodeKind, PartialParser};

/// A borrowed view of one value inside a [`PartialParser`].
#[derive(Clone, Copy)]
pub struct PartialValue<'a> {
    pub(super) p: &'a PartialParser,
    pub(super) idx: usize,
}

impl<'a> PartialValue<'a> {
    fn node(&self) -> &'a super::Node {
        &self.p.nodes[self.idx]
    }

    fn at_idx(&self, idx: usize) -> PartialValue<'a> {
        PartialValue { p: self.p, idx }
    }

    /// The JSON type.
    pub fn kind(&self) -> Kind {
        match self.node().kind {
            NodeKind::Null => Kind::Null,
            NodeKind::Bool(_) => Kind::Bool,
            NodeKind::Number(_) => Kind::Number,
            NodeKind::String(_) => Kind::String,
            NodeKind::Array(_) => Kind::Array,
            NodeKind::Object(_) => Kind::Object,
        }
    }

    /// True once the value's closing character has been read.
    pub fn is_complete(&self) -> bool {
        self.node().complete
    }

    pub fn is_null(&self) -> bool {
        matches!(self.node().kind, NodeKind::Null)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self.node().kind {
            NodeKind::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// The string received so far.
    pub fn as_str(&self) -> Option<&'a str> {
        match &self.node().kind {
            NodeKind::String(s) => Some(s),
            _ => None,
        }
    }

    /// The number's text as received, e.g. `"12."` while still streaming.
    pub fn as_number_str(&self) -> Option<&'a str> {
        match &self.node().kind {
            NodeKind::Number(n) => Some(n),
            _ => None,
        }
    }

    /// The number parsed as `f64`, if its text so far is a valid number.
    pub fn as_f64(&self) -> Option<f64> {
        self.as_number_str().and_then(|n| n.parse().ok())
    }

    /// Object member by key (the last one if the key repeats).
    pub fn get(&self, key: &str) -> Option<PartialValue<'a>> {
        match &self.node().kind {
            NodeKind::Object(entries) => entries
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|&(_, idx)| self.at_idx(idx)),
            _ => None,
        }
    }

    /// Array element by position.
    pub fn at(&self, i: usize) -> Option<PartialValue<'a>> {
        match &self.node().kind {
            NodeKind::Array(items) => items.get(i).map(|&idx| self.at_idx(idx)),
            _ => None,
        }
    }

    /// Number of array elements or object members seen so far; 0 otherwise.
    pub fn len(&self) -> usize {
        match &self.node().kind {
            NodeKind::Array(items) => items.len(),
            NodeKind::Object(entries) => entries.len(),
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Array elements; empty for other kinds.
    pub fn items(&self) -> impl Iterator<Item = PartialValue<'a>> + 'a {
        let p = self.p;
        let items: &'a [usize] = match &self.node().kind {
            NodeKind::Array(items) => items,
            _ => &[],
        };
        items.iter().map(move |&idx| PartialValue { p, idx })
    }

    /// Object members in the order received; empty for other kinds.
    pub fn entries(&self) -> impl Iterator<Item = (&'a str, PartialValue<'a>)> + 'a {
        let p = self.p;
        let entries: &'a [(String, usize)] = match &self.node().kind {
            NodeKind::Object(entries) => entries,
            _ => &[],
        };
        entries
            .iter()
            .map(move |(k, idx)| (k.as_str(), PartialValue { p, idx: *idx }))
    }

    /// The value so far as JSON text. A number whose text is not yet a
    /// valid number is written as `null`. This walks the whole subtree.
    pub fn to_json(&self) -> String {
        let mut out = Vec::new();
        self.write_json(&mut out);
        String::from_utf8(out).expect("the writer emits utf-8")
    }

    pub(crate) fn write_json(&self, out: &mut Vec<u8>) {
        match &self.node().kind {
            NodeKind::Null => out.extend_from_slice(b"null"),
            NodeKind::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
            NodeKind::Number(n) => {
                if crate::json::from_str::<f64>(n).is_ok() {
                    out.extend_from_slice(n.as_bytes());
                } else {
                    out.extend_from_slice(b"null");
                }
            }
            NodeKind::String(s) => crate::json::write_str(out, s),
            NodeKind::Array(_) => {
                out.push(b'[');
                for (i, v) in self.items().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    v.write_json(out);
                }
                out.push(b']');
            }
            NodeKind::Object(_) => {
                out.push(b'{');
                for (i, (k, v)) in self.entries().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    crate::json::write_str(out, k);
                    out.push(b':');
                    v.write_json(out);
                }
                out.push(b'}');
            }
        }
    }

    /// Copy into a `serde_json::Value` (feature `serde_json`).
    #[cfg(feature = "serde_json")]
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::from_str(&self.to_json()).unwrap_or(serde_json::Value::Null)
    }
}

impl fmt::Debug for PartialValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_json())
    }
}
