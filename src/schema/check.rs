//! The constraint keywords strict providers do not take, checked on the
//! answer instead.

use super::tree::{J, Map, get};

fn num(v: &J) -> Option<f64> {
    match v {
        J::Num(n) => n.parse().ok(),
        _ => None,
    }
}

/// The constraint keywords of one subschema, read once.
#[derive(Default)]
pub(crate) struct Bounds {
    min: Option<f64>,
    max: Option<f64>,
    xmin: Option<f64>,
    xmax: Option<f64>,
    multiple: Option<f64>,
    min_len: Option<f64>,
    max_len: Option<f64>,
    min_items: Option<f64>,
    max_items: Option<f64>,
}

pub(crate) fn bounds(m: &Map) -> Bounds {
    let b = |k: &str| get(m, k).and_then(num);
    Bounds {
        min: b("minimum"),
        max: b("maximum"),
        xmin: b("exclusiveMinimum"),
        xmax: b("exclusiveMaximum"),
        multiple: b("multipleOf"),
        min_len: b("minLength"),
        max_len: b("maxLength"),
        min_items: b("minItems"),
        max_items: b("maxItems"),
    }
}

impl Bounds {
    pub(crate) fn is_empty(&self) -> bool {
        [
            self.min,
            self.max,
            self.xmin,
            self.xmax,
            self.multiple,
            self.min_len,
            self.max_len,
        ]
        .iter()
        .all(Option::is_none)
    }

    /// A number's text against the numeric bounds; the reason when it
    /// breaks one.
    pub(crate) fn number(&self, text: &str) -> Result<(), String> {
        let x: f64 = text.parse().unwrap_or(f64::NAN);
        if let Some(b) = self.min
            && x < b
        {
            return Err(format!("is less than the minimum {b}"));
        }
        if let Some(b) = self.max
            && x > b
        {
            return Err(format!("is greater than the maximum {b}"));
        }
        if let Some(b) = self.xmin
            && x <= b
        {
            return Err(format!("must be greater than {b}"));
        }
        if let Some(b) = self.xmax
            && x >= b
        {
            return Err(format!("must be less than {b}"));
        }
        if let Some(b) = self.multiple
            && b > 0.0
            && ((x / b).round() * b - x).abs() > 1e-9 * x.abs().max(1.0)
        {
            return Err(format!("is not a multiple of {b}"));
        }
        Ok(())
    }

    pub(crate) fn length(&self, n: usize) -> Result<(), String> {
        let n = n as f64;
        if let Some(b) = self.min_len
            && n < b
        {
            return Err(format!("is shorter than {b} characters"));
        }
        if let Some(b) = self.max_len
            && n > b
        {
            return Err(format!("is longer than {b} characters"));
        }
        Ok(())
    }

    pub(crate) fn items(&self, n: usize) -> Result<(), String> {
        let n = n as f64;
        if let Some(b) = self.min_items
            && n < b
        {
            return Err(format!("has fewer than {b} items"));
        }
        if let Some(b) = self.max_items
            && n > b
        {
            return Err(format!("has more than {b} items"));
        }
        Ok(())
    }
}
