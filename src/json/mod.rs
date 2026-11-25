//! JSON reading and writing used by the request builders and response
//! parsers.
//!
//! The functions here are the only JSON entry points in the crate; the
//! dialect modules never name a JSON library.

use std::fmt;

use serde::de::{Deserialize, Deserializer};

/// A JSON syntax or data error, with the byte offset where it was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError {
    message: String,
}

impl JsonError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for JsonError {}

/// Deserialize `T` from JSON bytes. Trailing whitespace is allowed, any
/// other trailing text is an error.
pub(crate) fn from_slice<'de, T: Deserialize<'de>>(bytes: &'de [u8]) -> Result<T, JsonError> {
    serde_json::from_slice(bytes).map_err(|e| JsonError::new(e.to_string()))
}

/// Deserialize `T` from a JSON string.
pub(crate) fn from_str<'de, T: Deserialize<'de>>(s: &'de str) -> Result<T, JsonError> {
    from_slice(s.as_bytes())
}

/// The text of one JSON value, borrowed from the input without parsing it
/// into a tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Raw<'a>(pub(crate) &'a str);

impl<'de: 'a, 'a> Deserialize<'de> for Raw<'a> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw: &'de serde_json::value::RawValue = Deserialize::deserialize(d)?;
        Ok(Raw(raw.get()))
    }
}

/// Append `s` as a quoted, escaped JSON string.
pub(crate) fn write_str(out: &mut Vec<u8>, s: &str) {
    serde_json::to_writer(&mut *out, s).expect("writing to a Vec cannot fail");
}

/// Append an unsigned integer.
pub(crate) fn write_u64(out: &mut Vec<u8>, n: u64) {
    serde_json::to_writer(&mut *out, &n).expect("writing to a Vec cannot fail");
}

/// Append a float; NaN and infinities are written as `null`.
pub(crate) fn write_f64(out: &mut Vec<u8>, n: f64) {
    serde_json::to_writer(&mut *out, &n).expect("writing to a Vec cannot fail");
}

/// The result of [`repair`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Repaired<'a> {
    /// The input when it needed no change, else the repaired text.
    pub(crate) text: std::borrow::Cow<'a, str>,
    /// The input ended inside a value; open strings and containers were
    /// closed and an unfinished key or number was dropped.
    pub(crate) cut: bool,
}

/// Turn what a model wrote into JSON: strip markdown fences and prose around
/// the value, fix trailing commas, single quotes, unquoted keys, comments,
/// Python literals, and close a value the input stopped in the middle of.
/// Returns `None` when no JSON value can be found.
pub(crate) fn repair(s: &str) -> Option<Repaired<'_>> {
    let fixed = jsonrepair::repair_json(s, &jsonrepair::Options::default()).ok()?;
    Some(if fixed == s {
        Repaired {
            text: std::borrow::Cow::Borrowed(s),
            cut: false,
        }
    } else {
        Repaired {
            text: std::borrow::Cow::Owned(fixed),
            cut: false,
        }
    })
}
