//! Key/value lists and tuple objects, rewritten as objects and arrays.

use super::super::tree::J;
use super::{Seg, Tx};

impl<'a> Tx<'a> {
    /// `[{"key": k, "value": v}, ..]` as `{k: v, ..}`.
    pub(super) fn pairs(&mut self, value: &'a J) -> Result<(), String> {
        self.expect(b'[')?;
        self.out.push(b'{');
        let mut keys: Vec<String> = Vec::new();
        if self.next()? == b']' {
            self.sc.i += 1;
        } else {
            loop {
                self.path.push(Seg::Index(keys.len()));
                self.expect(b'{')?;
                let mut key: Option<String> = None;
                let mut val: Option<Vec<u8>> = None;
                loop {
                    let field = self.key()?;
                    self.expect(b':')?;
                    match &*field {
                        "key" if key.is_none() => {
                            let raw = self.raw()?;
                            let k: String = crate::json::from_str(raw)
                                .map_err(|_| self.fail("expected a string key and a value"))?;
                            key = Some(k);
                        }
                        "value" if val.is_none() => {
                            let start = self.out.len();
                            self.value(value)?;
                            val = Some(self.out.split_off(start));
                        }
                        _ => return Err(self.fail("an extra field in a key and value pair")),
                    }
                    if !self.more(b'}')? {
                        break;
                    }
                }
                let (Some(k), Some(v)) = (key, val) else {
                    return Err(self.fail("expected a string key and a value"));
                };
                if keys.contains(&k) {
                    return Err(self.fail(&format!("a repeated key \"{k}\"")));
                }
                self.path.pop();
                if !keys.is_empty() {
                    self.out.push(b',');
                }
                crate::json::write_str(&mut self.out, &k);
                self.out.push(b':');
                self.out.extend_from_slice(&v);
                keys.push(k);
                if !self.more(b']')? {
                    break;
                }
            }
        }
        self.out.push(b'}');
        Ok(())
    }

    /// `{"0": a, "1": b}` as `[a, b]`, in any key order.
    pub(super) fn tuple_object(&mut self, items: &'a [J]) -> Result<(), String> {
        self.expect(b'{')?;
        let mut slots: Vec<Option<Vec<u8>>> = vec![None; items.len()];
        if self.next()? == b'}' {
            self.sc.i += 1;
        } else {
            loop {
                let key = self.key()?;
                self.expect(b':')?;
                let Some(i) = key.parse::<usize>().ok().filter(|&i| i < items.len()) else {
                    return Err(self.fail(&format!("unexpected \"{key}\"")));
                };
                if slots[i].is_some() {
                    return Err(self.fail(&format!("a repeated key \"{key}\"")));
                }
                self.path.push(Seg::Index(i));
                let start = self.out.len();
                self.value(&items[i])?;
                slots[i] = Some(self.out.split_off(start));
                self.path.pop();
                if !self.more(b'}')? {
                    break;
                }
            }
        }
        self.out.push(b'[');
        for (i, slot) in slots.into_iter().enumerate() {
            let Some(v) = slot else {
                return Err(self.fail(&format!("missing \"{i}\"")));
            };
            if i > 0 {
                self.out.push(b',');
            }
            self.out.extend_from_slice(&v);
        }
        self.out.push(b']');
        Ok(())
    }
}
