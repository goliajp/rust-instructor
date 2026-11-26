//! A strict answer rewritten into the type's own JSON text in one pass,
//! guided by the canonical schema: key/value lists become objects, tuple
//! objects become arrays, a `{"value": ..}` root is unwrapped, and the
//! constraints the strict schema dropped are checked on the way. Scalars are
//! copied as written, so numbers keep every digit.

use super::check::bounds;
use super::strict::{WRAP, has_type, is_map, object_root};
use super::tree::{J, Map, get};
use crate::json::{Scanner, Str};

mod pairs;

/// `answer` as the type's own JSON, or why it does not fit `root`.
pub(crate) fn answer(root: &J, answer: &str) -> Result<String, String> {
    let mut t = Tx::new(root, answer);
    t.sc.ws();
    if object_root(root) || !t.wrapper()? {
        t.value(root)?;
    }
    t.end()
}

enum Seg<'a> {
    Key(&'a str),
    Index(usize),
}

struct Tx<'a> {
    sc: Scanner<'a>,
    src: &'a str,
    out: Vec<u8>,
    root: &'a J,
    path: Vec<Seg<'a>>,
    scratch: Vec<u8>,
}

fn kind(c: u8) -> &'static str {
    match c {
        b'{' => "object",
        b'[' => "array",
        b'"' => "string",
        b't' | b'f' => "boolean",
        b'n' => "null",
        _ => "number",
    }
}

impl<'a> Tx<'a> {
    fn new(root: &'a J, src: &'a str) -> Self {
        Self {
            sc: Scanner::new(src.as_bytes()),
            src,
            out: Vec::with_capacity(src.len() + 16),
            root,
            path: Vec::new(),
            scratch: Vec::new(),
        }
    }

    fn end(mut self) -> Result<String, String> {
        if self.sc.ws().is_some() {
            return Err(self.fail("trailing characters"));
        }
        Ok(String::from_utf8(self.out).expect("copied from a str"))
    }

    fn at(&self) -> String {
        if self.path.is_empty() {
            return "the answer".into();
        }
        let mut s = String::new();
        for seg in &self.path {
            s.push('/');
            match seg {
                Seg::Key(k) => s.push_str(k),
                Seg::Index(i) => s.push_str(&i.to_string()),
            }
        }
        s
    }

    fn fail(&self, what: &str) -> String {
        format!("{}: {what}", self.at())
    }

    fn json(&self, e: crate::json::JsonError) -> String {
        self.fail(&e.to_string())
    }

    fn next(&mut self) -> Result<u8, String> {
        self.sc
            .ws()
            .ok_or_else(|| self.fail("EOF while parsing a value"))
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.next()? == c {
            self.sc.i += 1;
            Ok(())
        } else {
            Err(self.fail(&format!("expected `{}`", c as char)))
        }
    }

    /// The next raw value, skipped.
    fn raw(&mut self) -> Result<&'a str, String> {
        self.sc.ws();
        let start = self.sc.i;
        self.sc.skip_value().map_err(|e| self.json(e))?;
        Ok(&self.src[start..self.sc.i])
    }

    /// An object key, the opening quote not yet read; borrowed unless it has
    /// escapes.
    fn key(&mut self) -> Result<std::borrow::Cow<'a, str>, String> {
        self.expect(b'"')?;
        match self.sc.parse_str(&mut self.scratch) {
            Ok(Str::Borrowed(k)) => Ok(k.into()),
            Ok(Str::Copied(k)) => Ok(k.to_owned().into()),
            Err(e) => Err(self.json(e)),
        }
    }

    /// `,` between members, or the closing byte; true when there is more.
    fn more(&mut self, close: u8) -> Result<bool, String> {
        match self.next()? {
            b',' => {
                self.sc.i += 1;
                Ok(true)
            }
            c if c == close => {
                self.sc.i += 1;
                Ok(false)
            }
            _ => Err(self.fail("expected `,` or the end of the container")),
        }
    }

    /// A root wrapped as `{"value": ..}`. False, with nothing read, when
    /// the answer is not exactly that, and the caller reads it as a bare
    /// value.
    fn wrapper(&mut self) -> Result<bool, String> {
        let start = self.sc.i;
        if self.sc.peek() != Some(b'{') {
            return Ok(false);
        }
        self.sc.i += 1;
        if self.next()? != b'"' || self.key()? != WRAP {
            self.sc.i = start;
            return Ok(false);
        }
        self.expect(b':')?;
        let inner = self.sc.i;
        self.raw()?;
        if self.next()? != b'}' {
            self.sc.i = start;
            return Ok(false);
        }
        self.sc.i = inner;
        self.value(self.root)?;
        self.expect(b'}')?;
        Ok(true)
    }

    fn resolve(&self, r: &str) -> Result<&'a J, String> {
        if r == "#" {
            return Ok(self.root);
        }
        let name = r.strip_prefix("#/$defs/").ok_or("unknown $ref")?;
        match self.root {
            J::Obj(m) => match get(m, "$defs") {
                Some(J::Obj(d)) => get(d, name).ok_or_else(|| "unknown $ref".into()),
                _ => Err("unknown $ref".into()),
            },
            _ => Err("unknown $ref".into()),
        }
    }

    fn value(&mut self, schema: &'a J) -> Result<(), String> {
        let J::Obj(m) = schema else {
            let raw = self.raw()?;
            self.out.extend_from_slice(raw.as_bytes());
            return Ok(());
        };
        if let Some(J::Str(r)) = get(m, "$ref") {
            return self.value(self.resolve(r)?);
        }
        if let Some(J::Arr(alts)) = get(m, "anyOf").or_else(|| get(m, "oneOf")) {
            return self.any_of(alts);
        }
        let c = self.next()?;
        if get(m, "const").is_some() || get(m, "enum").is_some() {
            self.allowed(m)?;
        }
        if c == b'n' {
            return match get(m, "type") {
                Some(_) if !has_type(m, "null") => Err(self.fail("expected a value, got null")),
                _ => {
                    let raw = self.raw()?;
                    self.out.extend_from_slice(raw.as_bytes());
                    Ok(())
                }
            };
        }
        if is_map(m) {
            let value = get(m, "additionalProperties").expect("checked by is_map");
            return match c {
                b'[' => self.pairs(value),
                _ => self.object(m),
            };
        }
        if let Some(J::Arr(items)) = get(m, "prefixItems") {
            return match c {
                b'{' => self.tuple_object(items),
                _ => self.array(m),
            };
        }
        let want = |t: &str| has_type(m, t) || get(m, "type").is_none();
        match c {
            b'{' if want("object") => self.object(m),
            b'[' if want("array") => self.array(m),
            b'"' if want("string") => self.scalar(m, true),
            b't' | b'f' if want("boolean") => self.scalar(m, false),
            b'-' | b'0'..=b'9' if want("number") || has_type(m, "integer") => self.scalar(m, false),
            other => {
                let got = kind(other);
                let t = match get(m, "type") {
                    Some(J::Str(t)) => t.clone(),
                    _ => "another type".into(),
                };
                Err(self.fail(&format!("expected {t}, got {got}")))
            }
        }
    }

    /// The first alternative that takes the value.
    fn any_of(&mut self, alts: &'a [J]) -> Result<(), String> {
        let save = (self.sc.i, self.out.len(), self.path.len());
        let mut first = None;
        for alt in alts {
            match self.value(alt) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    first.get_or_insert(e);
                    self.sc.i = save.0;
                    self.out.truncate(save.1);
                    self.path.truncate(save.2);
                }
            }
        }
        Err(first.unwrap_or_else(|| self.fail("no variant matches")))
    }

    /// `const` and `enum`, checked on the value without consuming it.
    fn allowed(&mut self, m: &Map) -> Result<(), String> {
        let start = self.sc.i;
        let raw = self.raw()?;
        self.sc.i = start;
        let v: J = crate::json::from_str(raw).map_err(|e| self.json(e))?;
        if let Some(c) = get(m, "const")
            && *c != v
        {
            return Err(self.fail(&format!("expected {}", super::strict::text(c))));
        }
        if let Some(J::Arr(e)) = get(m, "enum")
            && !e.contains(&v)
        {
            return Err(self.fail(&format!(
                "{} is not one of the allowed values",
                crate::text::clip(raw, 80)
            )));
        }
        Ok(())
    }

    fn scalar(&mut self, m: &Map, string: bool) -> Result<(), String> {
        let raw = self.raw()?;
        let b = bounds(m);
        if !b.is_empty() {
            let check = if string {
                let s: String = crate::json::from_str(raw).map_err(|e| self.json(e))?;
                b.length(s.chars().count())
            } else {
                b.number(raw)
            };
            check.map_err(|what| self.fail(&format!("{} {what}", crate::text::clip(raw, 80))))?;
        }
        self.out.extend_from_slice(raw.as_bytes());
        Ok(())
    }

    fn object(&mut self, m: &'a Map) -> Result<(), String> {
        let props = match get(m, "properties") {
            Some(J::Obj(p)) => Some(p),
            _ => None,
        };
        let extra = get(m, "additionalProperties");
        self.expect(b'{')?;
        self.out.push(b'{');
        let mut seen: Vec<std::borrow::Cow<'a, str>> = Vec::new();
        if self.next()? == b'}' {
            self.sc.i += 1;
        } else {
            loop {
                let key = self.key()?;
                self.expect(b':')?;
                if !seen.is_empty() {
                    self.out.push(b',');
                }
                crate::json::write_str(&mut self.out, &key);
                self.out.push(b':');
                let sub = match (props.and_then(|p| get(p, &key)), extra) {
                    (Some(sc), _) => Some(sc),
                    (None, Some(J::Bool(false))) => {
                        return Err(self.fail(&format!("unexpected \"{key}\"")));
                    }
                    (None, Some(sc @ J::Obj(_))) => Some(sc),
                    (None, _) => None,
                };
                // the path borrows from the schema or the input, so a key
                // with escapes is shown by its schema name when it has one
                let seg = props
                    .and_then(|p| p.iter().find(|(k, _)| **k == *key))
                    .map(|(k, _)| k.as_str())
                    .or(match &key {
                        std::borrow::Cow::Borrowed(k) => Some(*k),
                        std::borrow::Cow::Owned(_) => None,
                    })
                    .unwrap_or("?");
                self.path.push(Seg::Key(seg));
                match sub {
                    Some(sc) => self.value(sc)?,
                    None => {
                        let raw = self.raw()?;
                        self.out.extend_from_slice(raw.as_bytes());
                    }
                }
                self.path.pop();
                seen.push(key);
                if !self.more(b'}')? {
                    break;
                }
            }
        }
        self.out.push(b'}');
        if let Some(J::Arr(req)) = get(m, "required") {
            for r in req {
                if let J::Str(name) = r
                    && !seen.iter().any(|k| k == name)
                {
                    return Err(self.fail(&format!("missing \"{name}\"")));
                }
            }
        }
        Ok(())
    }

    fn array(&mut self, m: &'a Map) -> Result<(), String> {
        let items = get(m, "items");
        let prefix = match get(m, "prefixItems") {
            Some(J::Arr(p)) => Some(p),
            _ => None,
        };
        self.expect(b'[')?;
        self.out.push(b'[');
        let unique = get(m, "uniqueItems") == Some(&J::Bool(true));
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut n = 0;
        if self.next()? == b']' {
            self.sc.i += 1;
        } else {
            loop {
                if n > 0 {
                    self.out.push(b',');
                }
                let start = self.out.len();
                self.path.push(Seg::Index(n));
                let sc = prefix.map_or(items, |p| p.get(n));
                match sc {
                    Some(sc) => self.value(sc)?,
                    None => {
                        let raw = self.raw()?;
                        self.out.extend_from_slice(raw.as_bytes());
                    }
                }
                self.path.pop();
                if unique {
                    spans.push((start, self.out.len()));
                }
                n += 1;
                if !self.more(b']')? {
                    break;
                }
            }
        }
        self.out.push(b']');
        if let Some(p) = prefix
            && p.len() != n
        {
            return Err(self.fail(&format!("expected {} items", p.len())));
        }
        bounds(m)
            .items(n)
            .map_err(|what| self.fail(&format!("the list {what}")))?;
        if unique && self.repeated(&spans) {
            return Err(self.fail("the list has repeated items"));
        }
        Ok(())
    }

    // items compared in a normal form, so spacing does not matter
    fn repeated(&self, spans: &[(usize, usize)]) -> bool {
        let norm: Vec<Vec<u8>> = spans
            .iter()
            .map(|&(a, b)| {
                let text = std::str::from_utf8(&self.out[a..b]).expect("utf-8");
                let mut out = Vec::new();
                if let Ok(v) = crate::json::from_str::<J>(text) {
                    super::tree::write(&v, &mut out);
                }
                out
            })
            .collect();
        (1..norm.len()).any(|i| norm[..i].contains(&norm[i]))
    }
}
