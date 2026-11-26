//! The constraint keywords strict providers do not take, checked on the
//! answer instead.

use super::strict::text;
use super::tree::{J, Map, get};

fn num(v: &J) -> Option<f64> {
    match v {
        J::Num(n) => n.parse().ok(),
        _ => None,
    }
}

fn at(path: &str) -> &str {
    if path.is_empty() { "the answer" } else { path }
}

/// `v` if it meets the constraints of the subschema `m`, else why not.
pub(crate) fn constraints(m: &Map, v: J, path: &str) -> Result<J, String> {
    let bound = |k: &str| get(m, k).and_then(num);
    let fail = |what: String| Err(format!("{}: {} {what}", at(path), text(&v)));
    match &v {
        J::Num(_) => {
            let x = num(&v).unwrap_or(f64::NAN);
            if let Some(b) = bound("minimum")
                && x < b
            {
                return fail(format!("is less than the minimum {b}"));
            }
            if let Some(b) = bound("maximum")
                && x > b
            {
                return fail(format!("is greater than the maximum {b}"));
            }
            if let Some(b) = bound("exclusiveMinimum")
                && x <= b
            {
                return fail(format!("must be greater than {b}"));
            }
            if let Some(b) = bound("exclusiveMaximum")
                && x >= b
            {
                return fail(format!("must be less than {b}"));
            }
            if let Some(b) = bound("multipleOf")
                && b > 0.0
                && ((x / b).round() * b - x).abs() > 1e-9 * x.abs().max(1.0)
            {
                return fail(format!("is not a multiple of {b}"));
            }
        }
        J::Str(t) => {
            let n = t.chars().count() as f64;
            if let Some(b) = bound("minLength")
                && n < b
            {
                return fail(format!("is shorter than {b} characters"));
            }
            if let Some(b) = bound("maxLength")
                && n > b
            {
                return fail(format!("is longer than {b} characters"));
            }
        }
        J::Arr(xs) => {
            let n = xs.len() as f64;
            if let Some(b) = bound("minItems")
                && n < b
            {
                return fail(format!("has fewer than {b} items"));
            }
            if let Some(b) = bound("maxItems")
                && n > b
            {
                return fail(format!("has more than {b} items"));
            }
            if get(m, "uniqueItems") == Some(&J::Bool(true)) {
                let seen: Vec<Vec<u8>> = xs
                    .iter()
                    .map(|x| {
                        let mut out = Vec::new();
                        super::tree::write(x, &mut out);
                        out
                    })
                    .collect();
                if (1..seen.len()).any(|i| seen[..i].contains(&seen[i])) {
                    return fail("has repeated items".into());
                }
            }
        }
        _ => {}
    }
    Ok(v)
}
