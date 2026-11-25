//! Strings, numbers and bare words inside a value being repaired.

use super::super::scan::{Scanner, find_stop};
use super::{Outcome, Repairer};

impl Repairer<'_> {
    /// True when the quote at `at` ends the string: what follows it,
    /// after whitespace, is a delimiter or the end of the input.
    fn ends_string(&self, at: usize) -> bool {
        let rest = &self.b[at + 1..];
        match rest.iter().find(|c| !c.is_ascii_whitespace()) {
            None => true,
            Some(c) => matches!(c, b',' | b'}' | b']' | b':'),
        }
    }

    pub(super) fn string(&mut self, quote: u8) -> Outcome {
        self.i += 1;
        if quote == b'"' {
            self.out.keep(self.i);
        } else {
            self.out.skip(self.i);
            self.out.push("\"");
        }
        loop {
            let stop = if quote == b'"' {
                find_stop(self.b, self.i)
            } else {
                self.b[self.i..]
                    .iter()
                    .position(|&c| c == b'\'' || c == b'"' || c == b'\\' || c < 0x20)
                    .map_or(self.b.len(), |p| self.i + p)
            };
            self.i = stop;
            self.out.keep(stop);
            let Some(c) = self.peek() else {
                self.cut = true;
                self.out.push("\"");
                return Outcome::Cut;
            };
            self.i += 1;
            match c {
                b'\\' => self.escape(quote),
                c if c == quote && self.ends_string(stop) => {
                    if quote == b'"' {
                        self.out.keep(self.i);
                    } else {
                        self.out.skip(self.i);
                        self.out.push("\"");
                    }
                    return Outcome::Done;
                }
                b'"' => {
                    self.out.skip(self.i);
                    self.out.push("\\\"");
                }
                b'\'' => self.out.keep(self.i),
                b'\n' => self.replace("\\n"),
                b'\r' => self.replace("\\r"),
                b'\t' => self.replace("\\t"),
                c => self.replace(&format!("\\u{c:04x}")),
            }
        }
    }

    fn replace(&mut self, with: &str) {
        self.out.skip(self.i);
        self.out.push(with);
    }

    // after a backslash inside a string
    fn escape(&mut self, quote: u8) {
        let Some(c) = self.peek() else {
            // a cut escape is dropped
            self.out.skip(self.i);
            return;
        };
        match c {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                self.i += 1;
                self.out.keep(self.i);
            }
            b'u' => self.unicode_escape(),
            b'\'' if quote == b'\'' => {
                self.i += 1;
                self.replace("'");
            }
            _ => {
                // a stray backslash stands for itself
                self.out.skip(self.i);
                self.out.push("\\\\");
            }
        }
    }

    fn hex_at(&self, at: usize) -> Option<u16> {
        let d = self.s.get(at..at + 4)?;
        if !d.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        u16::from_str_radix(d, 16).ok()
    }

    // at the `u` of `\u`
    fn unicode_escape(&mut self) {
        let Some(n) = self.hex_at(self.i + 1) else {
            if self.b.len() < self.i + 5 {
                // cut inside the escape
                self.i = self.b.len();
                self.out.skip(self.i);
            } else {
                self.out.skip(self.i);
                self.out.push("\\\\");
            }
            return;
        };
        let after = self.i + 5;
        match n {
            0xD800..=0xDBFF => {
                let low = (self.b.get(after..after + 2) == Some(b"\\u"))
                    .then(|| self.hex_at(after + 2))
                    .flatten();
                if let Some(0xDC00..=0xDFFF) = low {
                    self.i = after + 6;
                    self.out.keep(self.i);
                } else {
                    self.i = after;
                    self.replace("\\ufffd");
                }
            }
            0xDC00..=0xDFFF => {
                self.i = after;
                self.replace("\\ufffd");
            }
            _ => {
                self.i = after;
                self.out.keep(self.i);
            }
        }
    }

    pub(super) fn number(&mut self, root: bool) -> Outcome {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.i += 1;
            } else {
                break;
            }
        }
        if self.i == self.b.len() && !root {
            // the next byte might have been another digit
            self.cut = true;
            return Outcome::Dropped;
        }
        let text = &self.s[start..self.i];
        let mut check = Scanner::new(text.as_bytes());
        if check.skip_number().is_ok() && check.i == text.len() {
            self.out.keep(self.i);
            return Outcome::Done;
        }
        self.out.skip(self.i);
        match text.trim_start_matches('+').parse::<f64>() {
            Ok(v) if v.is_finite() => {
                let mut b = Vec::new();
                super::super::write_f64(&mut b, v);
                let t = String::from_utf8(b).expect("ascii");
                let t = t.strip_suffix(".0").unwrap_or(&t);
                self.out.push(t);
            }
            _ => self.quote(text),
        }
        Outcome::Done
    }

    fn quote(&mut self, text: &str) {
        let mut b = Vec::with_capacity(text.len() + 2);
        super::super::write_str(&mut b, text);
        self.out
            .push(std::str::from_utf8(&b).expect("utf-8 in, utf-8 out"));
    }

    /// A bare word: a JSON literal, a Python one, or text to quote.
    pub(super) fn word(&mut self, root: bool) -> Outcome {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.i += 1;
            } else {
                break;
            }
        }
        let w = &self.s[start..self.i];
        let at_end = self.i == self.b.len();
        let replacement = match w {
            "true" | "false" | "null" => {
                self.out.keep(self.i);
                return Outcome::Done;
            }
            "True" => "true",
            "False" => "false",
            "None" | "NaN" | "Infinity" | "undefined" | "nil" => "null",
            _ if at_end && !root => {
                // possibly the start of a literal
                self.cut = true;
                return Outcome::Dropped;
            }
            _ if root => return Outcome::Dropped,
            _ => {
                // unquoted text up to the next delimiter
                let end = self.b[start..]
                    .iter()
                    .position(|&c| matches!(c, b',' | b'}' | b']' | b'\n'))
                    .map_or(self.b.len(), |p| start + p);
                self.i = end;
                let text = self.s[start..end].trim_end();
                self.out.skip(end);
                self.quote(text);
                return Outcome::Done;
            }
        };
        self.out.skip(self.i);
        self.out.push(replacement);
        Outcome::Done
    }
}
