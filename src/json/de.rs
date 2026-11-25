//! A serde `Deserializer` that reads JSON bytes directly, without a tree.

use serde::de::{
    self, DeserializeSeed, EnumAccess, Expected, MapAccess, SeqAccess, Unexpected, VariantAccess,
    Visitor,
};

use super::JsonError;
use super::scan::{Num, Scanner, Str};

pub(crate) const RAW_TOKEN: &str = "$instructors::private::Raw";

const MAX_DEPTH: u8 = 128;

pub(crate) struct Deserializer<'a> {
    pub(crate) sc: Scanner<'a>,
    scratch: Vec<u8>,
    depth: u8,
}

impl<'a> Deserializer<'a> {
    /// `input` must be valid utf-8.
    pub(crate) fn new(input: &'a str) -> Self {
        Self {
            sc: Scanner::new(input.as_bytes()),
            scratch: Vec::new(),
            depth: MAX_DEPTH,
        }
    }

    fn err(&self, msg: &str) -> JsonError {
        self.sc.err(msg)
    }

    #[inline]
    fn next(&mut self) -> Result<u8, JsonError> {
        self.sc
            .ws()
            .ok_or_else(|| self.err("EOF while parsing a value"))
    }

    #[cold]
    fn invalid_type(&mut self, exp: &dyn Expected) -> JsonError {
        let r: Result<(), JsonError> = (|| {
            let c = self.next()?;
            let unexp = match c {
                b'n' => Unexpected::Unit,
                b't' | b'f' => Unexpected::Bool(c == b't'),
                b'-' | b'0'..=b'9' => match self.sc.parse_number()? {
                    Num::U(n) => Unexpected::Unsigned(n),
                    Num::I(n) => Unexpected::Signed(n),
                    Num::F(n) => Unexpected::Float(n),
                },
                b'"' => {
                    self.sc.i += 1;
                    let s = match self.sc.parse_str(&mut self.scratch)? {
                        Str::Borrowed(s) | Str::Copied(s) => s.to_owned(),
                    };
                    return Err(de::Error::invalid_type(Unexpected::Str(&s), exp));
                }
                b'[' => Unexpected::Seq,
                b'{' => Unexpected::Map,
                _ => return Err(self.err("expected value")),
            };
            Err(de::Error::invalid_type(unexp, exp))
        })();
        let e = r.expect_err("always an error");
        e.or_at(self.sc.i)
    }

    #[inline]
    fn enter(&mut self) -> Result<(), JsonError> {
        self.depth -= 1;
        if self.depth == 0 {
            return Err(self.err("recursion limit exceeded"));
        }
        Ok(())
    }

    fn number<V: Visitor<'a>>(&mut self, v: V) -> Result<V::Value, JsonError> {
        match self.sc.parse_number()? {
            Num::U(n) => v.visit_u64(n),
            Num::I(n) => v.visit_i64(n),
            Num::F(n) => v.visit_f64(n),
        }
        .map_err(|e: JsonError| e.or_at(self.sc.i))
    }

    fn seq<V: Visitor<'a>>(&mut self, v: V) -> Result<V::Value, JsonError> {
        self.enter()?;
        self.sc.i += 1;
        let r = v.visit_seq(Seq {
            de: self,
            first: true,
        });
        self.depth += 1;
        let value = r.map_err(|e| e.or_at(self.sc.i))?;
        match self.sc.ws() {
            Some(b']') => {
                self.sc.i += 1;
                Ok(value)
            }
            Some(b',') => Err(self.err("trailing comma")),
            Some(_) => Err(self.err("trailing characters")),
            None => Err(self.err("EOF while parsing a list")),
        }
    }

    fn map<V: Visitor<'a>>(&mut self, v: V) -> Result<V::Value, JsonError> {
        self.enter()?;
        self.sc.i += 1;
        let r = v.visit_map(Map {
            de: self,
            first: true,
        });
        self.depth += 1;
        let value = r.map_err(|e| e.or_at(self.sc.i))?;
        match self.sc.ws() {
            Some(b'}') => {
                self.sc.i += 1;
                Ok(value)
            }
            Some(b',') => Err(self.err("trailing comma")),
            Some(_) => Err(self.err("trailing characters")),
            None => Err(self.err("EOF while parsing an object")),
        }
    }

    fn string<V: Visitor<'a>>(&mut self, v: V) -> Result<V::Value, JsonError> {
        self.sc.i += 1;
        let r = match self.sc.parse_str(&mut self.scratch)? {
            Str::Borrowed(s) => v.visit_borrowed_str(s),
            Str::Copied(s) => v.visit_str(s),
        };
        r.map_err(|e: JsonError| e.or_at(self.sc.i))
    }

    fn raw<V: Visitor<'a>>(&mut self, v: V) -> Result<V::Value, JsonError> {
        self.sc.ws();
        let start = self.sc.i;
        self.sc.skip_value()?;
        let b = self.sc.b;
        // SAFETY: the input is valid utf-8 and a value starts and ends on
        // ascii bytes
        let s = unsafe { std::str::from_utf8_unchecked(&b[start..self.sc.i]) };
        v.visit_borrowed_str(s)
    }
}

macro_rules! forward_number {
    ($($method:ident)*) => {$(
        #[inline]
        fn $method<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
            match self.next()? {
                b'-' | b'0'..=b'9' => self.number(v),
                _ => Err(self.invalid_type(&v)),
            }
        }
    )*};
}

impl<'a> de::Deserializer<'a> for &mut Deserializer<'a> {
    type Error = JsonError;

    fn deserialize_any<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'n' => {
                self.sc.i += 1;
                self.sc.expect_ident(b"ull")?;
                v.visit_unit()
            }
            b't' => {
                self.sc.i += 1;
                self.sc.expect_ident(b"rue")?;
                v.visit_bool(true)
            }
            b'f' => {
                self.sc.i += 1;
                self.sc.expect_ident(b"alse")?;
                v.visit_bool(false)
            }
            b'-' | b'0'..=b'9' => self.number(v),
            b'"' => self.string(v),
            b'[' => self.seq(v),
            b'{' => self.map(v),
            _ => Err(self.err("expected value")),
        }
        .map_err(|e| e.or_at(self.sc.i))
    }

    fn deserialize_bool<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b't' => {
                self.sc.i += 1;
                self.sc.expect_ident(b"rue")?;
                v.visit_bool(true)
            }
            b'f' => {
                self.sc.i += 1;
                self.sc.expect_ident(b"alse")?;
                v.visit_bool(false)
            }
            _ => Err(self.invalid_type(&v)),
        }
    }

    forward_number! {
        deserialize_i8 deserialize_i16 deserialize_i32 deserialize_i64
        deserialize_u8 deserialize_u16 deserialize_u32 deserialize_u64
        deserialize_f32 deserialize_f64
    }

    fn deserialize_char<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        self.deserialize_str(v)
    }

    #[inline]
    fn deserialize_str<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'"' => self.string(v),
            _ => Err(self.invalid_type(&v)),
        }
    }

    fn deserialize_string<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        self.deserialize_str(v)
    }

    fn deserialize_bytes<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'"' => {
                self.sc.i += 1;
                match self.sc.parse_str(&mut self.scratch)? {
                    Str::Borrowed(s) => v.visit_borrowed_bytes(s.as_bytes()),
                    Str::Copied(s) => v.visit_bytes(s.as_bytes()),
                }
            }
            b'[' => self.seq(v),
            _ => Err(self.invalid_type(&v)),
        }
    }

    fn deserialize_byte_buf<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        self.deserialize_bytes(v)
    }

    #[inline]
    fn deserialize_option<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'n' => {
                self.sc.i += 1;
                self.sc.expect_ident(b"ull")?;
                v.visit_none()
            }
            _ => v.visit_some(self),
        }
    }

    fn deserialize_unit<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'n' => {
                self.sc.i += 1;
                self.sc.expect_ident(b"ull")?;
                v.visit_unit()
            }
            _ => Err(self.invalid_type(&v)),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        v: V,
    ) -> Result<V::Value, JsonError> {
        self.deserialize_unit(v)
    }

    fn deserialize_newtype_struct<V: Visitor<'a>>(
        self,
        name: &'static str,
        v: V,
    ) -> Result<V::Value, JsonError> {
        if name == RAW_TOKEN {
            return self.raw(v);
        }
        v.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'[' => self.seq(v),
            _ => Err(self.invalid_type(&v)),
        }
    }

    fn deserialize_tuple<V: Visitor<'a>>(self, _len: usize, v: V) -> Result<V::Value, JsonError> {
        self.deserialize_seq(v)
    }

    fn deserialize_tuple_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        _len: usize,
        v: V,
    ) -> Result<V::Value, JsonError> {
        self.deserialize_seq(v)
    }

    fn deserialize_map<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'{' => self.map(v),
            _ => Err(self.invalid_type(&v)),
        }
    }

    #[inline]
    fn deserialize_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'{' => self.map(v),
            b'[' => self.seq(v),
            _ => Err(self.invalid_type(&v)),
        }
    }

    fn deserialize_enum<V: Visitor<'a>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'"' => v.visit_enum(UnitVariant { de: self }),
            b'{' => {
                self.enter()?;
                self.sc.i += 1;
                let value = v.visit_enum(VariantMap { de: self });
                self.depth += 1;
                let value = value?;
                match self.sc.ws() {
                    Some(b'}') => {
                        self.sc.i += 1;
                        Ok(value)
                    }
                    Some(_) => Err(self.err("expected `}`")),
                    None => Err(self.err("EOF while parsing an object")),
                }
            }
            _ => Err(self.err("expected value")),
        }
    }

    #[inline]
    fn deserialize_identifier<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        self.deserialize_str(v)
    }

    fn deserialize_ignored_any<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        self.sc.skip_value()?;
        v.visit_unit()
    }
}

struct Seq<'b, 'a> {
    de: &'b mut Deserializer<'a>,
    first: bool,
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

struct Map<'b, 'a> {
    de: &'b mut Deserializer<'a>,
    first: bool,
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

struct VariantMap<'b, 'a> {
    de: &'b mut Deserializer<'a>,
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

struct UnitVariant<'b, 'a> {
    de: &'b mut Deserializer<'a>,
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
struct MapKey<'b, 'a> {
    de: &'b mut Deserializer<'a>,
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

/// Accepts any value; used to check that text is JSON serde_json would
/// accept (numbers in range, surrogates paired, depth limit).
pub(crate) struct Strict;

impl<'de> de::Deserialize<'de> for Strict {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Strict;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E>(self, _: bool) -> Result<Strict, E> {
        Ok(Strict)
    }
    fn visit_i64<E>(self, _: i64) -> Result<Strict, E> {
        Ok(Strict)
    }
    fn visit_u64<E>(self, _: u64) -> Result<Strict, E> {
        Ok(Strict)
    }
    fn visit_f64<E>(self, _: f64) -> Result<Strict, E> {
        Ok(Strict)
    }
    fn visit_str<E>(self, _: &str) -> Result<Strict, E> {
        Ok(Strict)
    }
    fn visit_unit<E>(self) -> Result<Strict, E> {
        Ok(Strict)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Strict, A::Error> {
        while a.next_element::<Strict>()?.is_some() {}
        Ok(Strict)
    }
    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Strict, A::Error> {
        while a.next_entry::<de::IgnoredAny, Strict>()?.is_some() {}
        Ok(Strict)
    }
}
