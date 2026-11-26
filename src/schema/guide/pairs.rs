//! Maps sent as `[{"key": k, "value": v}, ..]` and tuples sent as
//! `{"0": a, "1": b}`, read as the map and the sequence they stand for.

use std::borrow::Cow;

use serde::de::{DeserializeSeed, MapAccess, SeqAccess};

use super::de::{Cx, Guide};
use super::key::Key;
use super::plan::Id;
use crate::json::de::Deserializer as Json;
use crate::json::{JsonError, Str};

const PAIR: &str = "expected a string key and a value";
const EXTRA: &str = "an extra field in a key and value pair";
const MORE: &str = "expected `,` or the end of the container";

pub(super) struct Pairs<'p, 'b, 'de> {
    cx: Cx<'p>,
    de: &'b mut Json<'de>,
    value: Id,
    n: usize,
    first: bool,
    keys: Vec<Cow<'de, str>>,
    // where the value starts when it came before the key
    early: Option<usize>,
}

impl<'p, 'b, 'de> Pairs<'p, 'b, 'de> {
    pub(super) fn new(cx: Cx<'p>, de: &'b mut Json<'de>, value: Id) -> Self {
        Self {
            cx,
            de,
            value,
            n: 0,
            first: true,
            keys: Vec::new(),
            early: None,
        }
    }

    fn expect(&mut self, c: u8, what: &str) -> Result<(), JsonError> {
        if self.de.next()? == c {
            self.de.sc.i += 1;
            Ok(())
        } else {
            Err(self.de.err(what))
        }
    }

    /// One pair up to its value, or past its end when the value came
    /// first; the key, and where it starts.
    fn pair(&mut self) -> Result<(Cow<'de, str>, usize), JsonError> {
        self.expect(b'{', "expected `{`")?;
        let mut key = None;
        self.early = None;
        loop {
            self.expect(b'"', "expected `\"`")?;
            let field = match self.de.sc.parse_str(&mut self.de.scratch)? {
                Str::Borrowed("key") | Str::Copied("key") => 0,
                Str::Borrowed("value") | Str::Copied("value") => 1,
                _ => 2,
            };
            self.expect(b':', "expected `:`")?;
            match field {
                0 if key.is_none() => {
                    let at = self.de.sc.i;
                    if self.de.next()? != b'"' {
                        return Err(self.de.err(PAIR));
                    }
                    self.de.sc.i += 1;
                    let k = match self.de.sc.parse_str(&mut self.de.scratch)? {
                        Str::Borrowed(s) => Cow::Borrowed(s),
                        Str::Copied(s) => Cow::Owned(s.to_owned()),
                    };
                    key = Some((k, at));
                }
                // the reader stops at the value; `next_value_seed` reads it
                1 if self.early.is_none() && key.is_some() => {
                    return Ok(key.expect("checked"));
                }
                1 if self.early.is_none() => {
                    self.early = Some(self.de.sc.i);
                    self.de.sc.skip_value()?;
                }
                _ => return Err(self.de.err(EXTRA)),
            }
            match self.de.next()? {
                b',' => self.de.sc.i += 1,
                b'}' => {
                    self.de.sc.i += 1;
                    return match (key, self.early) {
                        (Some(k), Some(_)) => Ok(k),
                        _ => Err(self.de.err(PAIR)),
                    };
                }
                _ => return Err(self.de.err(MORE)),
            }
        }
    }
}

impl<'de> MapAccess<'de> for Pairs<'_, '_, 'de> {
    type Error = JsonError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, JsonError> {
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
        let n = self.n;
        let (k, at) = self.pair().map_err(|e| e.under(&n))?;
        if self.keys.contains(&k) {
            let e = JsonError::at(&format!("a repeated key \"{k}\""), at);
            return Err(e.under(&n));
        }
        self.keys.push(k);
        let key = match self.keys.last().expect("just pushed") {
            Cow::Borrowed(s) => Key::borrowed(s, at),
            Cow::Owned(s) => Key::copied(s, at),
        };
        seed.deserialize(key).map(Some).map_err(|e| e.under(&n))
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, JsonError> {
        let n = self.n;
        self.n += 1;
        let r = match self.early.take() {
            Some(start) => {
                let resume = self.de.sc.i;
                self.de.sc.i = start;
                let r = seed.deserialize(Guide {
                    cx: self.cx,
                    de: &mut *self.de,
                    node: self.value,
                });
                self.de.sc.i = resume;
                r
            }
            None => seed
                .deserialize(Guide {
                    cx: self.cx,
                    de: &mut *self.de,
                    node: self.value,
                })
                .and_then(|v| match self.de.next()? {
                    b'}' => {
                        self.de.sc.i += 1;
                        Ok(v)
                    }
                    b',' => Err(self.de.err(EXTRA)),
                    _ => Err(self.de.err(MORE)),
                }),
        };
        r.map_err(|e| e.under(&n))
    }
}

/// A tuple sent as an object, its values found first and then read in
/// order.
pub(super) struct Slots<'p, 'b, 'de> {
    cx: Cx<'p>,
    de: &'b mut Json<'de>,
    prefix: &'p [Id],
    starts: Vec<usize>,
    k: usize,
}

impl<'p, 'b, 'de> Slots<'p, 'b, 'de> {
    /// Find where each value starts; also where the object ends.
    pub(super) fn read(
        cx: Cx<'p>,
        de: &'b mut Json<'de>,
        prefix: &'p [Id],
    ) -> Result<(Self, usize), JsonError> {
        let mut slots: Vec<Option<usize>> = vec![None; prefix.len()];
        if de.next()? == b'}' {
            de.sc.i += 1;
        } else {
            loop {
                if de.next()? != b'"' {
                    return Err(de.err("expected `\"`"));
                }
                let at = de.sc.i;
                de.sc.i += 1;
                let i = match de.sc.parse_str(&mut de.scratch)? {
                    Str::Borrowed(s) | Str::Copied(s) => {
                        match s.parse::<usize>().ok().filter(|&i| i < prefix.len()) {
                            Some(i) if slots[i].is_some() => {
                                let e = format!("a repeated key \"{s}\"");
                                return Err(JsonError::at(&e, at));
                            }
                            Some(i) => i,
                            None => {
                                return Err(JsonError::at(&format!("unexpected \"{s}\""), at));
                            }
                        }
                    }
                };
                if de.next()? != b':' {
                    return Err(de.err("expected `:`"));
                }
                de.sc.i += 1;
                slots[i] = Some(de.sc.i);
                de.sc.skip_value()?;
                match de.next()? {
                    b',' => de.sc.i += 1,
                    b'}' => {
                        de.sc.i += 1;
                        break;
                    }
                    _ => return Err(de.err(MORE)),
                }
            }
        }
        let end = de.sc.i;
        if let Some(i) = slots.iter().position(Option::is_none) {
            return Err(de.err(&format!("missing \"{i}\"")));
        }
        let starts = slots.into_iter().flatten().collect();
        let slots = Self {
            cx,
            de,
            prefix,
            starts,
            k: 0,
        };
        Ok((slots, end))
    }
}

impl<'de> SeqAccess<'de> for Slots<'_, '_, 'de> {
    type Error = JsonError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, JsonError> {
        let k = self.k;
        let Some(&start) = self.starts.get(k) else {
            return Ok(None);
        };
        self.k += 1;
        self.de.sc.i = start;
        let g = Guide {
            cx: self.cx,
            de: &mut *self.de,
            node: self.prefix[k],
        };
        seed.deserialize(g).map(Some).map_err(|e| e.under(&k))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.starts.len() - self.k)
    }
}
