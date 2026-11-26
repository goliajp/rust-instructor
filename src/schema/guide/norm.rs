//! A value read through the guide into the schema tree, the leaves as
//! written, so items can be compared in their converted form.

use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};

use super::super::tree::J;
use super::de::NORM_TOKEN;

pub(super) struct Norm;

impl<'de> DeserializeSeed<'de> for Norm {
    type Value = J;

    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<J, D::Error> {
        d.deserialize_newtype_struct(NORM_TOKEN, Visit)
    }
}

struct Visit;

impl<'de> Visitor<'de> for Visit {
    type Value = J;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_borrowed_str<E: de::Error>(self, raw: &'de str) -> Result<J, E> {
        match raw.as_bytes().first() {
            Some(b'-' | b'0'..=b'9') => Ok(J::Num(raw.to_owned())),
            _ => crate::json::from_str(raw).map_err(E::custom),
        }
    }

    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
        let mut out = Vec::new();
        while let Some(k) = a.next_key::<String>()? {
            let v = a.next_value_seed(Norm)?;
            out.push((k, v));
        }
        Ok(J::Obj(out))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = a.next_element_seed(Norm)? {
            out.push(v);
        }
        Ok(J::Arr(out))
    }
}
