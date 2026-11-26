//! A key already read from the answer, handed to the caller's type: always
//! a string, read as a number or a bool when the type asks for one, as an
//! object key is.

use serde::de::{self, DeserializeSeed, EnumAccess, Unexpected, VariantAccess, Visitor};

use crate::json::JsonError;

pub(crate) struct Key<'k, 'de> {
    borrowed: Option<&'de str>,
    copied: &'k str,
    // where the key was read, for errors
    at: usize,
}

impl<'k, 'de> Key<'k, 'de> {
    pub(crate) fn borrowed(s: &'de str, at: usize) -> Self {
        Self {
            borrowed: Some(s),
            copied: "",
            at,
        }
    }

    pub(crate) fn copied(s: &'k str, at: usize) -> Self {
        Self {
            borrowed: None,
            copied: s,
            at,
        }
    }

    fn str(&self) -> &str {
        self.borrowed.unwrap_or(self.copied)
    }

    fn visit<V: Visitor<'de>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.borrowed {
            Some(s) => v.visit_borrowed_str(s),
            None => v.visit_str(self.copied),
        }
    }

    fn parsed<T: std::str::FromStr>(&self) -> Result<T, JsonError> {
        self.str()
            .parse()
            .map_err(|_| JsonError::at("invalid number in map key", self.at))
    }
}

macro_rules! key_number {
    ($($method:ident => $visit:ident $ty:ty,)*) => {$(
        fn $method<V: Visitor<'de>>(self, v: V) -> Result<V::Value, JsonError> {
            let n: $ty = self.parsed()?;
            v.$visit(n)
        }
    )*};
}

impl<'de> de::Deserializer<'de> for Key<'_, 'de> {
    type Error = JsonError;

    fn deserialize_any<V: Visitor<'de>>(self, v: V) -> Result<V::Value, JsonError> {
        self.visit(v)
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

    fn deserialize_option<V: Visitor<'de>>(self, v: V) -> Result<V::Value, JsonError> {
        v.visit_some(self)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        v: V,
    ) -> Result<V::Value, JsonError> {
        v.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        v.visit_enum(self)
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.borrowed {
            Some(s) => v.visit_borrowed_bytes(s.as_bytes()),
            None => v.visit_bytes(self.copied.as_bytes()),
        }
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, v: V) -> Result<V::Value, JsonError> {
        self.deserialize_bytes(v)
    }

    serde::forward_to_deserialize_any! {
        <W: Visitor<'de>>
        char str string unit unit_struct seq tuple tuple_struct map struct
        identifier ignored_any
    }
}

impl<'de> EnumAccess<'de> for Key<'_, 'de> {
    type Error = JsonError;
    type Variant = UnitOnly;

    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, UnitOnly), JsonError> {
        seed.deserialize(self).map(|v| (v, UnitOnly))
    }
}

/// A variant named by a string: only a unit variant fits.
pub(crate) struct UnitOnly;

impl<'de> VariantAccess<'de> for UnitOnly {
    type Error = JsonError;

    fn unit_variant(self) -> Result<(), JsonError> {
        Ok(())
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        self,
        _seed: T,
    ) -> Result<T::Value, JsonError> {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"newtype variant",
        ))
    }

    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, _v: V) -> Result<V::Value, JsonError> {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"tuple variant",
        ))
    }

    fn struct_variant<V: Visitor<'de>>(
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
