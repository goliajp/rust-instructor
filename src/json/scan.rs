//! Byte-level JSON scanning shared by the deserializer, the validator and
//! repair: whitespace, strings, numbers, and skipping whole values.

mod number;

use super::JsonError;

const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

#[inline(always)]
fn zero_bytes(v: u64) -> u64 {
    v.wrapping_sub(LO) & !v & HI
}

/// Bytes of `w` that end a run of plain string content: `"`, `\` or a
/// control character. Only the lowest set bit is exact.
#[inline(always)]
pub(crate) fn stop_bytes(w: u64) -> u64 {
    zero_bytes(w ^ (LO * b'"' as u64))
        | zero_bytes(w ^ (LO * b'\\' as u64))
        | (w.wrapping_sub(LO * 0x20) & !w & HI)
}

/// Index of the first `"`, `\` or control byte at or after `i`, or
/// `b.len()`.
#[inline]
pub(crate) fn find_stop(b: &[u8], mut i: usize) -> usize {
    while i + 8 <= b.len() {
        let w = u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"));
        let m = stop_bytes(w);
        if m != 0 {
            return i + (m.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < b.len() {
        let c = b[i];
        if c == b'"' || c == b'\\' || c < 0x20 {
            return i;
        }
        i += 1;
    }
    i
}

/// A string read from the input: borrowed when it has no escapes, else
/// unescaped into the caller's scratch buffer.
pub(crate) enum Str<'a, 's> {
    Borrowed(&'a str),
    Copied(&'s str),
}

/// A number as serde_json reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Num {
    U(u64),
    I(i64),
    F(f64),
}

pub(crate) struct Scanner<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) i: usize,
}

impl<'a> Scanner<'a> {
    pub(crate) fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    pub(crate) fn err(&self, msg: &str) -> JsonError {
        JsonError::at(msg, self.i)
    }

    #[inline]
    pub(crate) fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    /// Skip whitespace and return the next byte without consuming it.
    #[inline]
    pub(crate) fn ws(&mut self) -> Option<u8> {
        while let Some(&c) = self.b.get(self.i) {
            match c {
                b' ' | b'\n' | b'\t' | b'\r' => self.i += 1,
                _ => return Some(c),
            }
        }
        None
    }

    pub(crate) fn expect_ident(&mut self, rest: &[u8]) -> Result<(), JsonError> {
        for &want in rest {
            match self.peek() {
                Some(c) if c == want => self.i += 1,
                Some(_) => return Err(self.err("expected ident")),
                None => return Err(self.err("EOF while parsing a value")),
            }
        }
        Ok(())
    }

    /// Read a string whose opening quote was already consumed.
    #[inline]
    pub(crate) fn parse_str<'s>(
        &mut self,
        scratch: &'s mut Vec<u8>,
    ) -> Result<Str<'a, 's>, JsonError> {
        let start = self.i;
        let stop = find_stop(self.b, start);
        match self.b.get(stop) {
            Some(b'"') => {
                self.i = stop + 1;
                Ok(Str::Borrowed(self.slice_str(start, stop)))
            }
            _ => {
                scratch.clear();
                self.i = stop;
                self.unescape_rest(start, scratch)?;
                // only whole utf-8 sequences and ascii escapes were copied
                Ok(Str::Copied(
                    std::str::from_utf8(scratch).map_err(|_| self.err("invalid unicode"))?,
                ))
            }
        }
    }

    fn slice_str(&self, from: usize, to: usize) -> &'a str {
        let bytes = &self.b[from..to];
        // the input was checked to be utf-8 when the scanner was built
        // from a `&str`; `from_slice` validates up front
        debug_assert!(std::str::from_utf8(bytes).is_ok());
        // SAFETY: the whole input is valid utf-8 and `from`/`to` sit next to
        // ascii bytes (`"` or the start of string content), so they are char
        // boundaries
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    #[cold]
    fn unescape_rest(&mut self, mut run: usize, out: &mut Vec<u8>) -> Result<(), JsonError> {
        loop {
            let stop = find_stop(self.b, self.i);
            out.extend_from_slice(&self.b[run..stop]);
            self.i = stop;
            match self.b.get(stop) {
                None => return Err(self.err("EOF while parsing a string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(());
                }
                Some(b'\\') => {
                    self.i += 1;
                    self.escape(out)?;
                    run = self.i;
                }
                Some(_) => {
                    return Err(self
                        .err("control character (\\u0000-\\u001F) found while parsing a string"));
                }
            }
        }
    }

    fn escape(&mut self, out: &mut Vec<u8>) -> Result<(), JsonError> {
        let Some(c) = self.peek() else {
            return Err(self.err("EOF while parsing a string"));
        };
        self.i += 1;
        let byte = match c {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => return self.unicode_escape(out),
            _ => return Err(self.err("invalid escape")),
        };
        out.push(byte);
        Ok(())
    }

    fn hex4(&mut self) -> Result<u16, JsonError> {
        let Some(d) = self.b.get(self.i..self.i + 4) else {
            self.i = self.b.len();
            return Err(self.err("EOF while parsing a string"));
        };
        self.i += 4;
        let mut n = 0u16;
        for &c in d {
            let v = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(self.err("invalid escape")),
            };
            n = n * 16 + v as u16;
        }
        Ok(n)
    }

    fn unicode_escape(&mut self, out: &mut Vec<u8>) -> Result<(), JsonError> {
        let n = self.hex4()?;
        let cp = match n {
            0xDC00..=0xDFFF => return Err(self.err("lone leading surrogate in hex escape")),
            0xD800..=0xDBFF => {
                if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
                    if self.i >= self.b.len() {
                        return Err(self.err("EOF while parsing a string"));
                    }
                    self.i += 1;
                    return Err(self.err("unexpected end of hex escape"));
                }
                self.i += 2;
                let n2 = self.hex4()?;
                if !(0xDC00..=0xDFFF).contains(&n2) {
                    return Err(self.err("lone leading surrogate in hex escape"));
                }
                (((n as u32 - 0xD800) << 10) | (n2 as u32 - 0xDC00)) + 0x1_0000
            }
            n => n as u32,
        };
        let ch = char::from_u32(cp).expect("non-surrogate code point");
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        Ok(())
    }

    /// Skip a string whose opening quote was consumed, checking escapes;
    /// `strict` also checks surrogate pairing.
    pub(crate) fn skip_str(&mut self, strict: bool) -> Result<(), JsonError> {
        loop {
            let stop = find_stop(self.b, self.i);
            self.i = stop;
            match self.b.get(stop) {
                None => return Err(self.err("EOF while parsing a string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(());
                }
                Some(b'\\') => {
                    self.i += 1;
                    match self.peek() {
                        None => return Err(self.err("EOF while parsing a string")),
                        Some(b'u') => {
                            self.i += 1;
                            let n = self.hex4()?;
                            if strict && (0xD800..=0xDFFF).contains(&n) {
                                self.surrogate(n)?;
                            }
                        }
                        Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => self.i += 1,
                        Some(_) => {
                            self.i += 1;
                            return Err(self.err("invalid escape"));
                        }
                    }
                }
                Some(_) => {
                    return Err(self
                        .err("control character (\\u0000-\\u001F) found while parsing a string"));
                }
            }
        }
    }

    // the low half after a high surrogate `n`
    #[cold]
    fn surrogate(&mut self, n: u16) -> Result<(), JsonError> {
        if n >= 0xDC00 {
            return Err(self.err("lone leading surrogate in hex escape"));
        }
        if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
            return Err(self.err("unexpected end of hex escape"));
        }
        self.i += 2;
        match self.hex4()? {
            0xDC00..=0xDFFF => Ok(()),
            _ => Err(self.err("lone leading surrogate in hex escape")),
        }
    }

    /// Skip one value of any kind, nested containers included, without
    /// building anything. No depth limit: containers are tracked on a stack.
    pub(crate) fn skip_value(&mut self) -> Result<(), JsonError> {
        self.skip_value_with(false)
    }

    /// Check one value the way serde_json reads it into a tree: numbers in
    /// range, surrogates paired, at most 127 levels of nesting.
    pub(crate) fn validate(&mut self) -> Result<(), JsonError> {
        self.skip_value_with(true)
    }

    fn skip_value_with(&mut self, strict: bool) -> Result<(), JsonError> {
        let mut stack: Vec<u8> = Vec::new();
        loop {
            // a value
            match self.ws() {
                None => return Err(self.err("EOF while parsing a value")),
                Some(b'"') => {
                    self.i += 1;
                    self.skip_str(strict)?;
                }
                Some(b'-' | b'0'..=b'9') if strict => {
                    self.parse_number()?;
                }
                Some(b'-' | b'0'..=b'9') => {
                    self.skip_number()?;
                }
                Some(b't') => {
                    self.i += 1;
                    self.expect_ident(b"rue")?;
                }
                Some(b'f') => {
                    self.i += 1;
                    self.expect_ident(b"alse")?;
                }
                Some(b'n') => {
                    self.i += 1;
                    self.expect_ident(b"ull")?;
                }
                Some(open @ (b'[' | b'{')) => {
                    if strict && stack.len() >= 127 {
                        return Err(self.err("recursion limit exceeded"));
                    }
                    self.i += 1;
                    let close = if open == b'[' { b']' } else { b'}' };
                    if self.ws() == Some(close) {
                        self.i += 1;
                    } else {
                        stack.push(open);
                        if open == b'{' {
                            self.skip_key(strict)?;
                        }
                        continue;
                    }
                }
                Some(_) => return Err(self.err("expected value")),
            }
            // after a value: comma or close, possibly several closes
            loop {
                let Some(&top) = stack.last() else {
                    return Ok(());
                };
                match self.ws() {
                    Some(b',') => {
                        self.i += 1;
                        if top == b'{' {
                            self.skip_key(strict)?;
                        }
                        break;
                    }
                    Some(b']') if top == b'[' => {
                        self.i += 1;
                        stack.pop();
                    }
                    Some(b'}') if top == b'{' => {
                        self.i += 1;
                        stack.pop();
                    }
                    None => {
                        return Err(self.err(if top == b'[' {
                            "EOF while parsing a list"
                        } else {
                            "EOF while parsing an object"
                        }));
                    }
                    Some(_) => {
                        return Err(self.err(if top == b'[' {
                            "expected `,` or `]`"
                        } else {
                            "expected `,` or `}`"
                        }));
                    }
                }
            }
        }
    }

    // `"key" :` inside an object
    fn skip_key(&mut self, strict: bool) -> Result<(), JsonError> {
        match self.ws() {
            Some(b'"') => {
                self.i += 1;
                self.skip_str(strict)?;
            }
            None => return Err(self.err("EOF while parsing an object")),
            Some(_) => return Err(self.err("key must be a string")),
        }
        match self.ws() {
            Some(b':') => {
                self.i += 1;
                Ok(())
            }
            None => Err(self.err("EOF while parsing an object")),
            Some(_) => Err(self.err("expected `:`")),
        }
    }
}
