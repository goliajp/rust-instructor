//! Answers to a reshaped schema read straight into the caller's type.
//!
//! A strict schema sends a map as a list of `{"key": .., "value": ..}`
//! objects, a tuple as an object with the keys `"0"`, `"1"`, …, and a root
//! that is not an object inside `{"value": ..}`, and it leaves out
//! constraints some providers do not take. The deserializer here is driven
//! by the canonical schema: it reads the wire form and gives the type's
//! `Deserialize` impl its own form, and checks the left-out constraints
//! as it goes. Numbers are handed over as written, so every digit of a
//! `u64` or `u128` arrives.

mod access;
mod de;
mod key;
mod norm;
mod pairs;
mod plan;

use serde::Deserialize;
use serde::de::DeserializeOwned;

pub(crate) use plan::Plan;

use self::de::{Cx, Guide};
use self::plan::Id;
use super::strict::WRAP;
use crate::json::JsonError;
use crate::json::de::Deserializer as Json;

#[derive(Deserialize)]
struct Items<T> {
    items: T,
}

/// Read `text` into `T` from `node` on, the whole input one value.
fn read<T: DeserializeOwned>(
    plan: &Plan,
    text: &str,
    node: Id,
    root: bool,
) -> Result<T, JsonError> {
    let mut de = Json::new(text);
    let r = (|| {
        let wrapped = root && !plan.object_root && wrapped(&mut de)?;
        let v = T::deserialize(Guide {
            cx: Cx { plan, full: false },
            de: &mut de,
            node,
        })?;
        if wrapped {
            if de.next()? != b'}' {
                return Err(de.err("expected `}`"));
            }
            de.sc.i += 1;
        }
        match de.sc.ws() {
            None => Ok(v),
            Some(_) => Err(de.err("trailing characters")),
        }
    })();
    r.map_err(|e| e.or_at(de.sc.i).locate(text.as_bytes()))
}

/// True, at the inner value, when the answer is exactly `{"value": ..}`;
/// false, with nothing read, when it is anything else and is read as the
/// bare value.
fn wrapped(de: &mut Json<'_>) -> Result<bool, JsonError> {
    let start = de.sc.i;
    if de.sc.ws() != Some(b'{') {
        return Ok(false);
    }
    de.sc.i += 1;
    if de.next()? != b'"' {
        de.sc.i = start;
        return Ok(false);
    }
    de.sc.i += 1;
    let is_wrap = match de.sc.parse_str(&mut de.scratch)? {
        crate::json::Str::Borrowed(k) | crate::json::Str::Copied(k) => k == WRAP,
    };
    if !is_wrap {
        de.sc.i = start;
        return Ok(false);
    }
    if de.next()? != b':' {
        return Err(de.err("expected `:`"));
    }
    de.sc.i += 1;
    let inner = de.sc.i;
    de.sc.skip_value()?;
    if de.next()? != b'}' {
        de.sc.i = start;
        return Ok(false);
    }
    de.sc.i = inner;
    Ok(true)
}

impl Plan {
    /// An answer to the whole schema. For a list schema (`{"items": [T]}`)
    /// a bare list is taken too.
    pub(crate) fn answer<T: DeserializeOwned>(
        &self,
        text: &str,
        list: bool,
    ) -> Result<T, JsonError> {
        if !list {
            return read(self, text, self.root, true);
        }
        match read::<Items<T>>(self, text, self.root, true) {
            Ok(w) => Ok(w.items),
            Err(e) => match self.list {
                Some(node) => read(self, text, node, false).map_err(|_| e),
                None => Err(e),
            },
        }
    }

    /// One item of a list answer.
    pub(crate) fn item<T: DeserializeOwned>(&self, text: &str) -> Result<T, JsonError> {
        match self.items {
            Some(node) => read(self, text, node, false),
            None => Err(JsonError::new("not a list schema")),
        }
    }
}

#[cfg(test)]
mod tests;
