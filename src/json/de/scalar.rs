//! Numbers, and the error for a value of the wrong type.

use serde::de::{self, Expected, Unexpected, Visitor};

use super::Deserializer;
use crate::json::JsonError;
use crate::json::scan::{Num, Str};

impl<'a> Deserializer<'a> {
    #[cold]
    pub(crate) fn invalid_type(&mut self, exp: &dyn Expected) -> JsonError {
        let r: Result<(), JsonError> = (|| {
            let c = self.next()?;
            let unexp = match c {
                b'n' => Unexpected::Unit,
                b't' | b'f' => Unexpected::Bool(c == b't'),
                b'-' | b'0'..=b'9' => match self.sc.parse_number()? {
                    Num::U(n) => Unexpected::Unsigned(n),
                    Num::I(n) => Unexpected::Signed(n),
                    Num::F(n) => Unexpected::Float(n),
                },
                b'"' => {
                    self.sc.i += 1;
                    let s = match self.sc.parse_str(&mut self.scratch)? {
                        Str::Borrowed(s) | Str::Copied(s) => s.to_owned(),
                    };
                    return Err(de::Error::invalid_type(Unexpected::Str(&s), exp));
                }
                b'[' => Unexpected::Seq,
                b'{' => Unexpected::Map,
                _ => return Err(self.err("expected value")),
            };
            Err(de::Error::invalid_type(unexp, exp))
        })();
        let e = r.expect_err("always an error");
        e.or_at(self.sc.i)
    }

    pub(super) fn number<V: Visitor<'a>>(&mut self, v: V) -> Result<V::Value, JsonError> {
        match self.sc.parse_number()? {
            Num::U(n) => v.visit_u64(n),
            Num::I(n) => v.visit_i64(n),
            Num::F(n) => v.visit_f64(n),
        }
        .map_err(|e: JsonError| e.or_at(self.sc.i))
    }

    /// An `i128` / `u128`: integer text is parsed at full width, anything
    /// else goes through [`number`](Self::number).
    pub(super) fn wide<V: Visitor<'a>>(
        &mut self,
        v: V,
        signed: bool,
    ) -> Result<V::Value, JsonError> {
        let start = self.sc.i;
        let num = self.sc.parse_number()?;
        let text = &self.sc.b[start..self.sc.i];
        let r = if text.iter().any(|c| matches!(c, b'.' | b'e' | b'E')) {
            match num {
                Num::U(n) => v.visit_u64(n),
                Num::I(n) => v.visit_i64(n),
                Num::F(n) => v.visit_f64(n),
            }
        } else {
            let t = std::str::from_utf8(text).expect("a number is ascii");
            match (signed, num) {
                (true, _) => match t.parse::<i128>() {
                    Ok(n) => v.visit_i128(n),
                    Err(_) => Err(self.err("number out of range")),
                },
                (false, Num::I(n)) => v.visit_i64(n),
                (false, _) => match t.parse::<u128>() {
                    Ok(n) => v.visit_u128(n),
                    Err(_) => Err(self.err("number out of range")),
                },
            }
        };
        r.map_err(|e: JsonError| e.or_at(self.sc.i))
    }
}

#[cfg(test)]
mod tests {
    use crate::json::from_str;

    #[test]
    fn full_width_integers() {
        assert_eq!(from_str::<u128>(&u128::MAX.to_string()), Ok(u128::MAX));
        assert_eq!(from_str::<i128>(&i128::MIN.to_string()), Ok(i128::MIN));
        assert_eq!(from_str::<i128>("-5"), Ok(-5));
        assert_eq!(from_str::<u128>(" 7 "), Ok(7));
        assert!(from_str::<u128>("340282366920938463463374607431768211456").is_err());
        assert!(from_str::<u128>("-1").is_err());
        assert!(from_str::<u128>("1.5").is_err());
        assert!(from_str::<i128>("\"1\"").is_err());
    }
}
