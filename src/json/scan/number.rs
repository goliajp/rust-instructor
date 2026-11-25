//! Numbers: parsing into `Num` and skipping.

use super::{JsonError, Num, Scanner};

impl Scanner<'_> {
    /// Read a number starting at the current byte (`-` or a digit).
    #[inline]
    pub(crate) fn parse_number(&mut self) -> Result<Num, JsonError> {
        let start = self.i;
        let neg = self.peek() == Some(b'-');
        if neg {
            self.i += 1;
        }
        let digits = self.i;
        let mut n: u64 = 0;
        let mut overflow = false;
        match self.peek() {
            Some(b'0') => {
                self.i += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.err("invalid number"));
                }
            }
            Some(b'1'..=b'9') => {
                while let Some(c @ b'0'..=b'9') = self.peek() {
                    self.i += 1;
                    let d = (c - b'0') as u64;
                    match n.checked_mul(10).and_then(|m| m.checked_add(d)) {
                        Some(m) => n = m,
                        None => overflow = true,
                    }
                }
            }
            None => return Err(self.err("EOF while parsing a value")),
            _ => return Err(self.err("invalid number")),
        }
        let mut float = overflow;
        if self.peek() == Some(b'.') {
            self.i += 1;
            self.digits()?;
            float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            self.digits()?;
            float = true;
        }
        if float {
            return self.float(start);
        }
        debug_assert!(self.i > digits);
        Ok(if !neg {
            Num::U(n)
        } else {
            let v = (n as i64).wrapping_neg();
            if v >= 0 {
                Num::F(-(n as f64))
            } else {
                Num::I(v)
            }
        })
    }

    fn digits(&mut self) -> Result<(), JsonError> {
        match self.peek() {
            Some(b'0'..=b'9') => {}
            None => return Err(self.err("EOF while parsing a value")),
            Some(_) => return Err(self.err("invalid number")),
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.i += 1;
        }
        Ok(())
    }

    #[cold]
    fn float(&mut self, start: usize) -> Result<Num, JsonError> {
        let text = self.slice_str(start, self.i);
        let v: f64 = text.parse().map_err(|_| self.err("invalid number"))?;
        if v.is_infinite() {
            return Err(self.err("number out of range"));
        }
        Ok(Num::F(v))
    }

    pub(crate) fn skip_number(&mut self) -> Result<(), JsonError> {
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.i += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.err("invalid number"));
                }
            }
            Some(b'1'..=b'9') => {
                while let Some(b'0'..=b'9') = self.peek() {
                    self.i += 1;
                }
            }
            None => return Err(self.err("EOF while parsing a value")),
            _ => return Err(self.err("invalid number")),
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            self.digits()?;
        }
        Ok(())
    }
}
