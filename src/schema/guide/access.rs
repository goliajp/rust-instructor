//! What the guided deserializer hands to visitors for objects, lists and
//! externally tagged variants.

use serde::de::{
    self, Deserialize, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor,
};

use super::de::{Cx, Guide};
use super::key::Key;
use super::norm::Norm;
use super::plan::{ANY, Extra, Id, Typed};
use crate::json::de::Deserializer as Json;
use crate::json::{JsonError, Scanner, Str};
use serde::de::value::{MapDeserializer, SeqDeserializer};

pub(super) use super::pairs::{Pairs, Slots};

/// The property `s` of `t`, looked for first where the last one ended,
/// since answers usually keep the schema's order.
#[inline]
fn lookup(t: &Typed, hint: usize, s: &str) -> Option<usize> {
    if let Some((k, _)) = t.props.get(hint)
        && **k == *s
    {
        return Some(hint);
    }
    t.props.iter().position(|(k, _)| **k == *s)
}

/// The key that starts at `at`, read again for a message.
#[cold]
pub(super) fn key_at(de: &Json<'_>, at: usize) -> String {
    let mut sc = Scanner::new(de.sc.b);
    sc.i = at + 1;
    let mut scratch = Vec::new();
    match sc.parse_str(&mut scratch) {
        Ok(Str::Borrowed(s) | Str::Copied(s)) => s.to_owned(),
        Err(_) => "?".to_owned(),
    }
}

/// Where a value sits in its object, for messages.
#[derive(Clone, Copy)]
enum Seg {
    Prop(usize),
    // a key that is not a property, by where it starts
    At(usize),
}

pub(super) struct Obj<'p, 'b, 'de> {
    cx: Cx<'p>,
    de: &'b mut Json<'de>,
    t: &'p Typed,
    first: bool,
    bits: u64,
    hint: usize,
    child: Id,
    seg: Seg,
    seen: Vec<String>,
    // the value to give is an empty list or map for an omitted property
    filling: bool,
}

impl<'p, 'b, 'de> Obj<'p, 'b, 'de> {
    pub(super) fn new(cx: Cx<'p>, de: &'b mut Json<'de>, t: &'p Typed) -> Self {
        Self {
            cx,
            de,
            t,
            first: true,
            bits: 0,
            hint: 0,
            child: ANY,
            seg: Seg::At(0),
            seen: Vec::new(),
            filling: false,
        }
    }

    /// A property left out that reads as empty.
    fn unfilled(&self) -> Option<usize> {
        let m = self.t.fill & !self.bits;
        (self.cx.fill && m != 0).then(|| m.trailing_zeros() as usize)
    }

    fn required(&self) -> Result<(), JsonError> {
        let t = self.t;
        let missing = if t.by_name {
            t.required
                .iter()
                .find(|r| !self.seen.iter().any(|k| *k == ***r))
        } else if self.bits & t.required_mask != t.required_mask {
            t.required.iter().find(|r| {
                let i = t.props.iter().position(|(k, _)| k == *r).unwrap_or(64);
                i < 64 && self.bits & (1 << i) == 0
            })
        } else {
            None
        };
        match missing {
            Some(name) => Err(self.de.err(&format!("missing \"{name}\""))),
            None => Ok(()),
        }
    }

    #[cold]
    fn under(&self, e: JsonError) -> JsonError {
        match self.seg {
            Seg::Prop(i) => e.under(&self.t.props[i].0),
            Seg::At(at) => e.under(&key_at(self.de, at)),
        }
    }
}

impl<'de> MapAccess<'de> for Obj<'_, '_, 'de> {
    type Error = JsonError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, JsonError> {
        let c = match self.de.sc.ws() {
            Some(b'}') => {
                let Some(i) = self.unfilled() else {
                    return self.required().map(|()| None);
                };
                self.bits |= 1 << i;
                self.filling = true;
                self.seg = Seg::Prop(i);
                let name = &self.t.props[i].0;
                if self.t.by_name {
                    self.seen.push(name.to_string());
                }
                return seed.deserialize(Key::copied(name, self.de.sc.i)).map(Some);
            }
            Some(b',') if !self.first => {
                self.de.sc.i += 1;
                self.de.sc.ws()
            }
            Some(c) if self.first => Some(c),
            Some(_) => return Err(self.de.err("expected `,` or `}`")),
            None => return Err(self.de.err("EOF while parsing an object")),
        };
        self.first = false;
        match c {
            Some(b'"') => {}
            Some(b'}') => return Err(self.de.err("trailing comma")),
            Some(_) => return Err(self.de.err("key must be a string")),
            None => return Err(self.de.err("EOF while parsing a value")),
        }
        let at = self.de.sc.i;
        self.de.sc.i += 1;
        let key = self.de.sc.parse_str(&mut self.de.scratch)?;
        let s = match key {
            Str::Borrowed(s) => s,
            Str::Copied(s) => s,
        };
        match lookup(self.t, self.hint, s) {
            Some(i) => {
                if i < 64 {
                    self.bits |= 1 << i;
                }
                self.hint = i + 1;
                self.child = self.t.props[i].1;
                self.seg = Seg::Prop(i);
            }
            None => {
                self.child = match self.t.extra {
                    Extra::Deny => {
                        return Err(JsonError::at(&format!("unexpected \"{s}\""), at));
                    }
                    Extra::Node(n) => n,
                    Extra::Ignore => ANY,
                };
                self.seg = Seg::At(at);
            }
        }
        if self.t.by_name {
            self.seen.push(s.to_owned());
        }
        let key = match key {
            Str::Borrowed(s) => Key::borrowed(s, at),
            Str::Copied(s) => Key::copied(s, at),
        };
        seed.deserialize(key).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, JsonError> {
        if std::mem::take(&mut self.filling) {
            let Seg::Prop(i) = self.seg else {
                unreachable!("a filled property is a property")
            };
            let r = if self.t.fill_map & (1 << i) != 0 {
                seed.deserialize(MapDeserializer::new(std::iter::empty::<((), ())>()))
            } else {
                seed.deserialize(SeqDeserializer::new(std::iter::empty::<()>()))
            };
            return r.map_err(|e| self.under(e));
        }
        match self.de.sc.ws() {
            Some(b':') => self.de.sc.i += 1,
            Some(_) => return Err(self.de.err("expected `:`")),
            None => return Err(self.de.err("EOF while parsing an object")),
        }
        if !self.cx.full && self.cx.plan.is_plain(self.child) {
            return seed.deserialize(&mut *self.de).map_err(|e| self.under(e));
        }
        let g = Guide {
            cx: self.cx,
            de: &mut *self.de,
            node: self.child,
        };
        seed.deserialize(g).map_err(|e| self.under(e))
    }
}

pub(super) struct Arr<'p, 'b, 'de> {
    cx: Cx<'p>,
    de: &'b mut Json<'de>,
    t: &'p Typed,
    n: usize,
    first: bool,
    // each item in a normal form, when items must be unique
    norms: Vec<Vec<u8>>,
}

impl<'p, 'b, 'de> Arr<'p, 'b, 'de> {
    pub(super) fn new(cx: Cx<'p>, de: &'b mut Json<'de>, t: &'p Typed) -> Self {
        Self {
            cx,
            de,
            t,
            n: 0,
            first: true,
            norms: Vec::new(),
        }
    }

    /// The number of items read and their normal forms.
    pub(super) fn done(self) -> (usize, Vec<Vec<u8>>) {
        (self.n, self.norms)
    }
}

impl<'de> SeqAccess<'de> for Arr<'_, '_, 'de> {
    type Error = JsonError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, JsonError> {
        match self.de.sc.ws() {
            Some(b']') => return Ok(None),
            Some(b',') if !self.first => {
                self.de.sc.i += 1;
                if self.de.sc.ws() == Some(b']') {
                    return Err(self.de.err("trailing comma"));
                }
            }
            Some(_) if self.first => {}
            Some(_) => return Err(self.de.err("expected `,` or `]`")),
            None => return Err(self.de.err("EOF while parsing a list")),
        }
        self.first = false;
        let n = self.n;
        let node = match &self.t.prefix {
            Some(p) => p.get(n).copied().unwrap_or(ANY),
            None => self.t.items,
        };
        if self.t.unique {
            let start = self.de.sc.i;
            let g = Guide {
                cx: self.cx,
                de: &mut *self.de,
                node,
            };
            let v = Norm.deserialize(g).map_err(|e| e.under(&n))?;
            let mut out = Vec::new();
            super::super::tree::write(&v, &mut out);
            self.norms.push(out);
            self.de.sc.i = start;
        }
        let g = Guide {
            cx: self.cx,
            de: &mut *self.de,
            node,
        };
        let r = seed.deserialize(g).map_err(|e| e.under(&n));
        self.n += 1;
        r.map(Some)
    }
}

pub(super) struct Variant<'p, 'b, 'de, 'x> {
    cx: Cx<'p>,
    de: &'b mut Json<'de>,
    t: &'p Typed,
    bits: &'x mut u64,
}

impl<'p, 'b, 'de, 'x> Variant<'p, 'b, 'de, 'x> {
    pub(super) fn new(cx: Cx<'p>, de: &'b mut Json<'de>, t: &'p Typed, bits: &'x mut u64) -> Self {
        Self { cx, de, t, bits }
    }
}

pub(super) struct Payload<'p, 'b, 'de> {
    cx: Cx<'p>,
    de: &'b mut Json<'de>,
    t: &'p Typed,
    child: Id,
    seg: Seg,
}

impl<'p, 'b, 'de> EnumAccess<'de> for Variant<'p, 'b, 'de, '_> {
    type Error = JsonError;
    type Variant = Payload<'p, 'b, 'de>;

    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Payload<'p, 'b, 'de>), JsonError> {
        match self.de.sc.ws() {
            Some(b'"') => {}
            Some(_) => return Err(self.de.err("key must be a string")),
            None => return Err(self.de.err("EOF while parsing a value")),
        }
        let at = self.de.sc.i;
        self.de.sc.i += 1;
        let key = self.de.sc.parse_str(&mut self.de.scratch)?;
        let s = match key {
            Str::Borrowed(s) => s,
            Str::Copied(s) => s,
        };
        let (child, seg) = match lookup(self.t, 0, s) {
            Some(i) => {
                if i < 64 {
                    *self.bits |= 1 << i;
                }
                (self.t.props[i].1, Seg::Prop(i))
            }
            None => match self.t.extra {
                Extra::Deny => {
                    return Err(JsonError::at(&format!("unexpected \"{s}\""), at));
                }
                Extra::Node(n) => (n, Seg::At(at)),
                Extra::Ignore => (ANY, Seg::At(at)),
            },
        };
        let key = match key {
            Str::Borrowed(s) => Key::borrowed(s, at),
            Str::Copied(s) => Key::copied(s, at),
        };
        let tag = seed.deserialize(key)?;
        match self.de.sc.ws() {
            Some(b':') => self.de.sc.i += 1,
            Some(_) => return Err(self.de.err("expected `:`")),
            None => return Err(self.de.err("EOF while parsing an object")),
        }
        let payload = Payload {
            cx: self.cx,
            de: self.de,
            t: self.t,
            child,
            seg,
        };
        Ok((tag, payload))
    }
}

impl<'p, 'de> Payload<'p, '_, 'de> {
    fn guide(&mut self) -> Guide<'p, '_, 'de> {
        Guide {
            cx: self.cx,
            de: &mut *self.de,
            node: self.child,
        }
    }

    #[cold]
    fn under(&self, e: JsonError) -> JsonError {
        match self.seg {
            Seg::Prop(i) => e.under(&self.t.props[i].0),
            Seg::At(at) => e.under(&key_at(self.de, at)),
        }
    }
}

impl<'de> VariantAccess<'de> for Payload<'_, '_, 'de> {
    type Error = JsonError;

    fn unit_variant(mut self) -> Result<(), JsonError> {
        <()>::deserialize(self.guide()).map_err(|e| self.under(e))
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        mut self,
        seed: T,
    ) -> Result<T::Value, JsonError> {
        seed.deserialize(self.guide()).map_err(|e| self.under(e))
    }

    fn tuple_variant<V: Visitor<'de>>(mut self, _len: usize, v: V) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_seq(self.guide(), v).map_err(|e| self.under(e))
    }

    fn struct_variant<V: Visitor<'de>>(
        mut self,
        fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_struct(self.guide(), "", fields, v).map_err(|e| self.under(e))
    }
}
