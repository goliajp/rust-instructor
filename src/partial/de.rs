//! A serde `Deserializer` over the partial tree, for typed snapshots: no
//! JSON text is written and read again, and a value that fails to read
//! consumes nothing, so a snapshot can drop one field and keep the rest.

use serde::de::{
    self, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, Unexpected, VariantAccess, Visitor,
};

use super::typed::WHOLE_TOKEN;
use super::{NodeKind, PartialValue};
use crate::json::JsonError;
use crate::json::de::Deserializer as Json;
use crate::schema::{At, Key};

pub(crate) struct Tree<'a, 'p> {
    v: PartialValue<'a>,
    // where a strict answer's value sits in the schema, to read the wire
    // forms back
    at: Option<At<'p>>,
}

impl<'a, 'p> Tree<'a, 'p> {
    pub(crate) fn new(v: PartialValue<'a>, at: Option<At<'p>>) -> Self {
        let at = at.and_then(|a| a.resolve(v.is_null()));
        Self { v, at }
    }

    fn kind(&self) -> &'a NodeKind {
        &self.v.node().kind
    }

    /// A number read by the JSON deserializer from its text, so wide
    /// integers keep every digit.
    fn number<V, F>(text: &'a str, v: V, read: F) -> Result<V::Value, JsonError>
    where
        V: Visitor<'a>,
        F: FnOnce(&mut Json<'a>, V) -> Result<V::Value, JsonError>,
    {
        let mut d = Json::new(text);
        let r = read(&mut d, v)?;
        match d.sc.ws() {
            None => Ok(r),
            Some(_) => Err(de::Error::custom("an unfinished number")),
        }
    }
}

macro_rules! numbers {
    ($($method:ident)*) => {$(
        fn $method<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
            match self.kind() {
                NodeKind::Number(t) => Self::number(t, v, |d, v| de::Deserializer::$method(d, v)),
                _ => self.deserialize_any(v),
            }
        }
    )*};
}

impl<'a> de::Deserializer<'a> for Tree<'a, '_> {
    type Error = JsonError;

    fn deserialize_any<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.kind() {
            NodeKind::Null => v.visit_unit(),
            NodeKind::Bool(b) => v.visit_bool(*b),
            NodeKind::Number(t) => match Self::number(t, v, |d, v| d.deserialize_any(v)) {
                Ok(x) => Ok(x),
                // `-` or `1.` so far
                Err(_) => Err(de::Error::custom("an unfinished number")),
            },
            NodeKind::String(s) => v.visit_borrowed_str(s),
            NodeKind::Array(_) => match self.at.and_then(At::pairs) {
                Some(value) => v.visit_map(Pairs {
                    v: self.v,
                    value,
                    i: 0,
                    next: None,
                }),
                None => v.visit_seq(Items {
                    v: self.v,
                    at: self.at,
                    i: 0,
                }),
            },
            NodeKind::Object(_) if self.at.is_some_and(|a| a.slots().is_some()) => {
                v.visit_seq(Slots {
                    v: self.v,
                    at: self.at,
                    i: 0,
                })
            }
            NodeKind::Object(_) => v.visit_map(Members {
                v: self.v,
                at: self.at,
                i: 0,
                next: None,
            }),
        }
    }

    numbers! {
        deserialize_i8 deserialize_i16 deserialize_i32 deserialize_i64 deserialize_i128
        deserialize_u8 deserialize_u16 deserialize_u32 deserialize_u64 deserialize_u128
        deserialize_f32 deserialize_f64
    }

    fn deserialize_option<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.kind() {
            NodeKind::Null => v.visit_none(),
            _ => v.visit_some(self),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'a>>(
        self,
        name: &'static str,
        v: V,
    ) -> Result<V::Value, JsonError> {
        if name == WHOLE_TOKEN && !self.v.is_complete() {
            return Err(de::Error::custom("not complete yet"));
        }
        v.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'a>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        match self.kind() {
            NodeKind::String(s) => v.visit_enum(Key::borrowed(s, 0)),
            NodeKind::Object(entries) if entries.len() == 1 => {
                let (k, idx) = &entries[0];
                let value = self.v.at_idx(*idx);
                let at = self.at.and_then(|a| a.key(k));
                v.visit_enum(Variant {
                    tag: k,
                    value: Tree::new(value, at),
                })
            }
            _ => Err(de::Error::invalid_type(Unexpected::Map, &v)),
        }
    }

    // a struct only from an object, not from a list as serde would allow
    fn deserialize_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        match self.kind() {
            NodeKind::Array(_) => Err(de::Error::invalid_type(Unexpected::Seq, &v)),
            _ => self.deserialize_any(v),
        }
    }

    fn deserialize_ignored_any<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        v.visit_unit()
    }

    serde::forward_to_deserialize_any! {
        <W: Visitor<'a>>
        bool char str string bytes byte_buf unit unit_struct seq tuple tuple_struct map
        identifier
    }
}

struct Items<'a, 'p> {
    v: PartialValue<'a>,
    at: Option<At<'p>>,
    i: usize,
}

impl<'a> SeqAccess<'a> for Items<'a, '_> {
    type Error = JsonError;

    fn next_element_seed<T: DeserializeSeed<'a>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, JsonError> {
        let Some(item) = self.v.at(self.i) else {
            return Ok(None);
        };
        let at = self.at.and_then(|a| a.item(self.i));
        self.i += 1;
        seed.deserialize(Tree::new(item, at)).map(Some)
    }
}

/// A tuple sent as `{"0": .., "1": ..}`, read up to the first position
/// not yet received.
struct Slots<'a, 'p> {
    v: PartialValue<'a>,
    at: Option<At<'p>>,
    i: usize,
}

impl<'a> SeqAccess<'a> for Slots<'a, '_> {
    type Error = JsonError;

    fn next_element_seed<T: DeserializeSeed<'a>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, JsonError> {
        let Some(item) = self.v.get(&self.i.to_string()) else {
            return Ok(None);
        };
        let at = self.at.and_then(|a| a.item(self.i));
        self.i += 1;
        seed.deserialize(Tree::new(item, at)).map(Some)
    }
}

struct Members<'a, 'p> {
    v: PartialValue<'a>,
    at: Option<At<'p>>,
    i: usize,
    next: Option<Tree<'a, 'p>>,
}

impl<'a> MapAccess<'a> for Members<'a, '_> {
    type Error = JsonError;

    fn next_key_seed<K: DeserializeSeed<'a>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, JsonError> {
        let NodeKind::Object(entries) = &self.v.node().kind else {
            return Ok(None);
        };
        let Some((k, idx)) = entries.get(self.i) else {
            return Ok(None);
        };
        self.i += 1;
        let at = self.at.and_then(|a| a.key(k));
        self.next = Some(Tree::new(self.v.at_idx(*idx), at));
        seed.deserialize(Key::borrowed(k, 0)).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'a>>(&mut self, seed: V) -> Result<V::Value, JsonError> {
        let t = self.next.take().expect("a key was read");
        seed.deserialize(t)
    }
}

/// A map sent as `[{"key": .., "value": ..}]`: the pairs whose key is
/// complete and whose value has started.
struct Pairs<'a, 'p> {
    v: PartialValue<'a>,
    value: At<'p>,
    i: usize,
    next: Option<Tree<'a, 'p>>,
}

impl<'a> MapAccess<'a> for Pairs<'a, '_> {
    type Error = JsonError;

    fn next_key_seed<K: DeserializeSeed<'a>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, JsonError> {
        while let Some(pair) = self.v.at(self.i) {
            self.i += 1;
            let key = pair.get("key").filter(|k| k.is_complete());
            let (Some(k), Some(value)) = (key.and_then(|k| k.as_str()), pair.get("value")) else {
                continue;
            };
            self.next = Some(Tree::new(value, Some(self.value)));
            return seed.deserialize(Key::borrowed(k, 0)).map(Some);
        }
        Ok(None)
    }

    fn next_value_seed<V: DeserializeSeed<'a>>(&mut self, seed: V) -> Result<V::Value, JsonError> {
        let t = self.next.take().expect("a key was read");
        seed.deserialize(t)
    }
}

struct Variant<'a, 'p> {
    tag: &'a str,
    value: Tree<'a, 'p>,
}

impl<'a, 'p> EnumAccess<'a> for Variant<'a, 'p> {
    type Error = JsonError;
    type Variant = Tree<'a, 'p>;

    fn variant_seed<V: DeserializeSeed<'a>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Tree<'a, 'p>), JsonError> {
        let tag = seed.deserialize(Key::borrowed(self.tag, 0))?;
        Ok((tag, self.value))
    }
}

impl<'a> VariantAccess<'a> for Tree<'a, '_> {
    type Error = JsonError;

    fn unit_variant(self) -> Result<(), JsonError> {
        de::Deserialize::deserialize(self)
    }

    fn newtype_variant_seed<T: DeserializeSeed<'a>>(self, seed: T) -> Result<T::Value, JsonError> {
        seed.deserialize(self)
    }

    fn tuple_variant<V: Visitor<'a>>(self, _len: usize, v: V) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_seq(self, v)
    }

    fn struct_variant<V: Visitor<'a>>(
        self,
        _fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_map(self, v)
    }
}
