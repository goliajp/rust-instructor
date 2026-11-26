//! A serde `Deserializer` that reads JSON bytes directly, without a tree.

use serde::de::{self, Visitor};

mod access;
mod scalar;

use access::{Map, Seq, UnitVariant, VariantMap};

use super::JsonError;
use super::scan::{Scanner, Str};

pub(crate) const RAW_TOKEN: &str = "$instructors::private::Raw";

const MAX_DEPTH: u8 = 128;

pub(crate) struct Deserializer<'a> {
    pub(crate) sc: Scanner<'a>,
    pub(crate) scratch: Vec<u8>,
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

    pub(crate) fn err(&self, msg: &str) -> JsonError {
        self.sc.err(msg)
    }

    #[inline]
    pub(crate) fn next(&mut self) -> Result<u8, JsonError> {
        self.sc
            .ws()
            .ok_or_else(|| self.err("EOF while parsing a value"))
    }

    #[inline]
    pub(crate) fn enter(&mut self) -> Result<(), JsonError> {
        self.depth -= 1;
        if self.depth == 0 {
            return Err(self.err("recursion limit exceeded"));
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn leave(&mut self) {
        self.depth += 1;
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

    fn deserialize_i128<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'-' | b'0'..=b'9' => self.wide(v, true),
            _ => Err(self.invalid_type(&v)),
        }
    }

    fn deserialize_u128<V: Visitor<'a>>(self, v: V) -> Result<V::Value, JsonError> {
        match self.next()? {
            b'-' | b'0'..=b'9' => self.wide(v, false),
            _ => Err(self.invalid_type(&v)),
        }
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
