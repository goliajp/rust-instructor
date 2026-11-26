//! Internals reachable for the fuzz targets and the JSON benchmark. Not a
//! public API: anything here can change in any release.

pub use crate::json::{JsonError, Raw, Repaired};
pub use crate::sse::SseDecoder;

pub fn json_from_str<'de, T: serde::Deserialize<'de>>(s: &'de str) -> Result<T, JsonError> {
    crate::json::from_str(s)
}

pub fn json_from_slice<'de, T: serde::Deserialize<'de>>(b: &'de [u8]) -> Result<T, JsonError> {
    crate::json::from_slice(b)
}

pub fn json_is_valid(s: &str) -> bool {
    crate::json::is_valid(s)
}

pub fn json_repair(s: &str) -> Option<Repaired<'_>> {
    crate::json::repair(s)
}

pub fn json_write_str(out: &mut Vec<u8>, s: &str) {
    crate::json::write_str(out, s)
}

pub fn json_write_u64(out: &mut Vec<u8>, n: u64) {
    crate::json::write_u64(out, n)
}

pub fn json_write_f64(out: &mut Vec<u8>, n: f64) {
    crate::json::write_f64(out, n)
}

/// A strict-schema answer read by the guided deserializer.
pub fn strict_guided<T>(text: &str) -> Result<T, JsonError>
where
    T: serde::de::DeserializeOwned + crate::Schema,
{
    crate::schema::entry_for::<T>(false)
        .guide()
        .answer(text, false)
}

/// The same answer rewritten into the type's own JSON text and then
/// parsed: the reference the guided reader is checked against.
pub fn strict_reference<T>(text: &str) -> Result<T, JsonError>
where
    T: serde::de::DeserializeOwned + crate::Schema,
{
    let own = crate::schema::entry_for::<T>(false).unwire(text)?;
    crate::json::from_str(&own)
}
