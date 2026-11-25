//! Access types the deserializer hands to visitors: sequences, maps,
//! enum variants and object keys.

use serde::de::{
    self, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, Unexpected, VariantAccess, Visitor,
};

use super::super::JsonError;
use super::super::scan::Str;
use super::Deserializer;

pub(super) struct Seq<'b, 'a> {
    pub(super) de: &'b mut Deserializer<'a>,
    pub(super) first: bool,
}

impl<'a> SeqAccess<'a> for Seq<'_, 'a> {
    type Error = JsonError;

    fn next_element_seed<T: DeserializeSeed<'a>>(
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
        seed.deserialize(&mut *self.de).map(Some)
    }
}

pub(super) struct Map<'b, 'a> {
    pub(super) de: &'b mut Deserializer<'a>,
    pub(super) first: bool,
}

impl<'a> MapAccess<'a> for Map<'_, 'a> {
    type Error = JsonError;

    #[inline]
    fn next_key_seed<K: DeserializeSeed<'a>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, JsonError> {
        let c = match self.de.sc.ws() {
            Some(b'}') => return Ok(None),
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
            Some(b'"') => seed.deserialize(MapKey { de: &mut *self.de }).map(Some),
            Some(b'}') => Err(self.de.err("trailing comma")),
            Some(_) => Err(self.de.err("key must be a string")),
            None => Err(self.de.err("EOF while parsing a value")),
        }
    }

    #[inline]
    fn next_value_seed<V: DeserializeSeed<'a>>(&mut self, seed: V) -> Result<V::Value, JsonError> {
        match self.de.sc.ws() {
            Some(b':') => self.de.sc.i += 1,
            Some(_) => return Err(self.de.err("expected `:`")),
            None => return Err(self.de.err("EOF while parsing an object")),
        }
        seed.deserialize(&mut *self.de)
    }
}

pub(super) struct VariantMap<'b, 'a> {
    pub(super) de: &'b mut Deserializer<'a>,
}

impl<'a> EnumAccess<'a> for VariantMap<'_, 'a> {
    type Error = JsonError;
    type Variant = Self;

    fn variant_seed<V: DeserializeSeed<'a>>(self, seed: V) -> Result<(V::Value, Self), JsonError> {
        let tag = match self.de.sc.ws() {
            Some(b'"') => seed.deserialize(MapKey { de: &mut *self.de })?,
            Some(_) => return Err(self.de.err("key must be a string")),
            None => return Err(self.de.err("EOF while parsing a value")),
        };
        match self.de.sc.ws() {
            Some(b':') => self.de.sc.i += 1,
            Some(_) => return Err(self.de.err("expected `:`")),
            None => return Err(self.de.err("EOF while parsing an object")),
        }
        Ok((tag, self))
    }
}

impl<'a> VariantAccess<'a> for VariantMap<'_, 'a> {
    type Error = JsonError;

    fn unit_variant(self) -> Result<(), JsonError> {
        de::Deserialize::deserialize(&mut *self.de)
    }

    fn newtype_variant_seed<T: DeserializeSeed<'a>>(self, seed: T) -> Result<T::Value, JsonError> {
        seed.deserialize(&mut *self.de)
    }

    fn tuple_variant<V: Visitor<'a>>(self, _len: usize, v: V) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_seq(&mut *self.de, v)
    }

    fn struct_variant<V: Visitor<'a>>(
        self,
        fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_struct(&mut *self.de, "", fields, v)
    }
}

pub(super) struct UnitVariant<'b, 'a> {
    pub(super) de: &'b mut Deserializer<'a>,
}

impl<'a> EnumAccess<'a> for UnitVariant<'_, 'a> {
    type Error = JsonError;
    type Variant = Self;

    fn variant_seed<V: DeserializeSeed<'a>>(self, seed: V) -> Result<(V::Value, Self), JsonError> {
        let v = seed.deserialize(&mut *self.de)?;
        Ok((v, self))
    }
}

impl<'a> VariantAccess<'a> for UnitVariant<'_, 'a> {
    type Error = JsonError;

    fn unit_variant(self) -> Result<(), JsonError> {
        Ok(())
    }

    fn newtype_variant_seed<T: DeserializeSeed<'a>>(self, _seed: T) -> Result<T::Value, JsonError> {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"newtype variant",
        ))
    }

    fn tuple_variant<V: Visitor<'a>>(self, _len: usize, _v: V) -> Result<V::Value, JsonError> {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"tuple variant",
        ))
    }

    fn struct_variant<V: Visitor<'a>>(
        self,
        _fields: &'static [&'static str],
        _v: V,
    ) -> Result<V::Value, JsonError> {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"struct variant",
        ))
    }
}

/// Object keys: always strings, read as numbers or bools when the target
/// type asks for one.
pub(super) struct MapKey<'b, 'a> {
    pub(super) de: &'b mut Deserializer<'a>,
}

impl MapKey<'_, '_> {
    fn parsed<T: std::str::FromStr>(self, what: &str) -> Result<T, JsonError> {
        self.de.sc.i += 1;
        let r = match self.de.sc.parse_str(&mut self.de.scratch)? {
            Str::Borrowed(s) | Str::Copied(s) => s.parse::<T>().ok(),
        };
        r.ok_or_else(|| self.de.err(what))
    }
}

macro_rules! key_number {
    ($($method:ident => $visit:ident $ty:ty,)*) => {$(
        fn $method<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
            let n: $ty = self.parsed("invalid number in map key")?;
            v.$visit(n)
        }
    )*};
}

impl<'a> de::Deserializer<'a> for MapKey<'_, 'a> {
    type Error = JsonError;

    #[inline]
    fn deserialize_any<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        self.de.string(v)
    }

    key_number! {
        deserialize_i8 => visit_i8 i8,
        deserialize_i16 => visit_i16 i16,
        deserialize_i32 => visit_i32 i32,
        deserialize_i64 => visit_i64 i64,
        deserialize_i128 => visit_i128 i128,
        deserialize_u8 => visit_u8 u8,
        deserialize_u16 => visit_u16 u16,
        deserialize_u32 => visit_u32 u32,
        deserialize_u64 => visit_u64 u64,
        deserialize_u128 => visit_u128 u128,
        deserialize_f32 => visit_f32 f32,
        deserialize_f64 => visit_f64 f64,
        deserialize_bool => visit_bool bool,
    }

    fn deserialize_option<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        v.visit_some(self)
    }

    fn deserialize_newtype_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        v: V,
    ) -> Result<V::Value, JsonError> {
        v.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'a>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_enum(&mut *self.de, name, variants, v)
    }

    fn deserialize_bytes<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_bytes(&mut *self.de, v)
    }

    fn deserialize_byte_buf<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        de::Deserializer::deserialize_bytes(&mut *self.de, v)
    }

    serde::forward_to_deserialize_any! {
        <W: Visitor<'a>>
        char str string unit unit_struct seq tuple tuple_struct map struct
        identifier ignored_any
    }
}
