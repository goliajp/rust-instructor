//! JSON reading and writing used by the request builders and response
//! parsers.
//!
//! The functions here are the only JSON entry points in the crate; the
//! dialect modules never name a JSON library.

pub(crate) mod de;
mod repair;
mod scan;
mod write;

use std::fmt;

use serde::de::{Deserialize, Deserializer, Visitor};

#[cfg(feature = "fuzzing")]
pub use repair::Repaired;
pub(crate) use repair::repair;
pub(crate) use scan::{Scanner, Str};
pub(crate) use write::write_u64;
pub(crate) use write::{write_f64, write_str};

/// A JSON syntax or data error, with the line and column where it was
/// found.
// one pointer, so every `Result` on the reading path stays small; the
// fields are allocated only when an error is built
#[derive(Clone, PartialEq, Eq)]
pub struct JsonError(Box<Repr>);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Repr {
    message: String,
    // where in the value the error was found, as `/key/0`; empty at the root
    path: String,
    offset: Option<usize>,
    // 1-based, filled in once the input is known
    line: usize,
    column: usize,
}

impl JsonError {
    #[cold]
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(Box::new(Repr {
            message: message.into(),
            path: String::new(),
            offset: None,
            line: 0,
            column: 0,
        }))
    }

    #[cold]
    pub(crate) fn at(message: &str, offset: usize) -> Self {
        let mut e = Self::new(message);
        e.0.offset = Some(offset);
        e
    }

    /// Set the position unless one is already known.
    pub(crate) fn or_at(mut self, offset: usize) -> Self {
        self.0.offset.get_or_insert(offset);
        self
    }

    /// The error happened inside `seg` of the value being read.
    pub(crate) fn under(mut self, seg: &dyn fmt::Display) -> Self {
        self.0.path.insert_str(0, &format!("/{seg}"));
        self
    }

    pub(crate) fn locate(mut self, input: &[u8]) -> Self {
        let Some(at) = self.0.offset else { return self };
        let before = &input[..at.min(input.len())];
        self.0.line = 1 + before.iter().filter(|&&b| b == b'\n').count();
        self.0.column = before.len()
            - before
                .iter()
                .rposition(|&b| b == b'\n')
                .map_or(0, |p| p + 1);
        self
    }

    /// Byte offset into the input where the error was found, when known.
    pub fn offset(&self) -> Option<usize> {
        self.0.offset
    }
}

impl fmt::Debug for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.0;
        f.debug_struct("JsonError")
            .field("message", &r.message)
            .field("path", &r.path)
            .field("offset", &r.offset)
            .field("line", &r.line)
            .field("column", &r.column)
            .finish()
    }
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.0;
        if !r.path.is_empty() {
            write!(f, "{}: ", r.path)?;
        }
        if r.line == 0 {
            f.write_str(&r.message)
        } else {
            write!(f, "{} at line {} column {}", r.message, r.line, r.column)
        }
    }
}

impl std::error::Error for JsonError {}

impl serde::de::Error for JsonError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::new(msg.to_string())
    }
}

/// Deserialize `T` from JSON bytes. Trailing whitespace is allowed, any
/// other trailing text is an error.
pub(crate) fn from_slice<'de, T: Deserialize<'de>>(bytes: &'de [u8]) -> Result<T, JsonError> {
    match std::str::from_utf8(bytes) {
        Ok(s) => from_str(s),
        Err(e) => Err(JsonError::at("invalid unicode code point", e.valid_up_to()).locate(bytes)),
    }
}

/// Deserialize `T` from a JSON string.
pub(crate) fn from_str<'de, T: Deserialize<'de>>(s: &'de str) -> Result<T, JsonError> {
    let mut d = de::Deserializer::new(s);
    let r = T::deserialize(&mut d).and_then(|v| match d.sc.ws() {
        None => Ok(v),
        Some(_) => Err(d.sc.err("trailing characters")),
    });
    r.map_err(|e| e.or_at(d.sc.i).locate(s.as_bytes()))
}

/// True when `s` is one JSON value that serde_json would accept, with
/// optional surrounding whitespace.
pub(crate) fn is_valid(s: &str) -> bool {
    let mut sc = scan::Scanner::new(s.as_bytes());
    sc.validate().is_ok() && sc.ws().is_none()
}

/// The text of one JSON value, borrowed from the input without parsing it
/// into a tree. Only this module's deserializer can produce it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Raw<'a>(pub &'a str);

impl<'de: 'a, 'a> Deserialize<'de> for Raw<'a> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Raw<'de>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a raw JSON value")
            }
            fn visit_borrowed_str<E>(self, s: &'de str) -> Result<Raw<'de>, E> {
                Ok(Raw(s))
            }
        }
        d.deserialize_newtype_struct(de::RAW_TOKEN, V)
    }
}

#[cfg(test)]
mod diff_tests;
#[cfg(test)]
mod repair_tests;
#[cfg(test)]
mod suite_tests;
#[cfg(test)]
mod tests;
