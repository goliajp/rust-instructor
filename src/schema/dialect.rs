//! Per-provider adjustments of the canonical schema, then bytes.

use super::cache::Dialect;
use super::strict;
use super::tree::{J, Map, get, remove, s, set, write};

/// Visit every subschema of one schema object (not property maps
/// themselves, so a field named `title` or `format` is never mistaken for a
/// keyword), parents before children.
fn walk(v: &mut J, f: &mut impl FnMut(&mut Map)) {
    let J::Obj(m) = v else { return };
    f(m);
    for (key, val) in m.iter_mut() {
        match (key.as_str(), val) {
            ("properties" | "$defs", J::Obj(p)) => p.iter_mut().for_each(|(_, x)| walk(x, f)),
            ("anyOf" | "oneOf" | "allOf" | "prefixItems", J::Arr(a)) => {
                a.iter_mut().for_each(|x| walk(x, f))
            }
            ("items" | "additionalProperties", x) => walk(x, f),
            _ => {}
        }
    }
}

pub(crate) fn build(base: &J, dialect: Dialect) -> Box<str> {
    let mut v = base.clone();
    let wrap = |v: J| {
        if strict::object_root(base) {
            v
        } else {
            strict::wrap(v)
        }
    };
    let v = match dialect {
        Dialect::Strict => {
            walk(&mut v, &mut |m| {
                strict::reshape(m);
                remove(m, "title");
                strip_unsupported_openai(m);
                one_of_to_any_of(m);
                close_object(m);
            });
            wrap(v)
        }
        Dialect::AnthropicOutput => {
            walk(&mut v, &mut |m| {
                strict::reshape(m);
                remove(m, "title");
                strip_unsupported_anthropic(m);
                one_of_to_any_of(m);
                close_object(m);
            });
            wrap(v)
        }
        Dialect::AnthropicTool => {
            walk(&mut v, &mut |m| {
                remove(m, "title");
            });
            v
        }
        Dialect::GeminiJson => {
            walk(&mut v, &mut |m| {
                const_to_enum(m);
                for k in strict::GEMINI_DROPS {
                    remove(m, k);
                }
                // formats are documented for strings only
                if strict::has_type(m, "integer") || strict::has_type(m, "number") {
                    remove(m, "format");
                }
            });
            v
        }
    };
    let mut out = Vec::with_capacity(512);
    write(&v, &mut out);
    String::from_utf8(out)
        .expect("the writer emits utf-8")
        .into_boxed_str()
}

/// The dialects as 1.5 built them from the same canonical schema, for the
/// tests that compare the canonical form with 1.5's output.
#[cfg(test)]
pub(crate) fn build_15(base: &J, dialect: Dialect) -> String {
    let mut v = base.clone();
    walk(&mut v, &mut |m| match dialect {
        Dialect::Strict => {
            remove(m, "title");
            remove(m, "format");
            one_of_to_any_of(m);
            close_object(m);
        }
        Dialect::AnthropicOutput => {
            remove(m, "title");
            strip_unsupported_anthropic(m);
            one_of_to_any_of(m);
            close_object(m);
        }
        Dialect::AnthropicTool => {
            remove(m, "title");
        }
        Dialect::GeminiJson => {}
    });
    let mut out = Vec::new();
    write(&v, &mut out);
    String::from_utf8(out).expect("the writer emits utf-8")
}

/// Strict modes: no extra keys, every property listed in `required`.
/// Optional fields stay optional through their nullable type.
fn close_object(m: &mut Map) {
    if get(m, "type") != Some(&s("object")) {
        return;
    }
    let Some(J::Obj(props)) = get(m, "properties") else {
        return;
    };
    let keys = props.iter().map(|(k, _)| s(k)).collect();
    set(m, "required", J::Arr(keys));
    set(m, "additionalProperties", J::Bool(false));
}

// gemini's subset has `enum` but not `const`, and ignores what it does not
// support, which would drop the tag of a tagged enum
fn const_to_enum(m: &mut Map) {
    if get(m, "enum").is_none()
        && let Some(i) = m.iter().position(|(k, _)| k == "const")
    {
        let (_, value) = m.remove(i);
        m.insert(i, ("enum".into(), J::Arr(vec![value])));
    }
}

fn one_of_to_any_of(m: &mut Map) {
    if get(m, "anyOf").is_none()
        && let Some(i) = m.iter().position(|(k, _)| k == "oneOf")
    {
        m[i].0 = "anyOf".into();
    }
}

// keywords outside openai's strict subset, which rejects them
fn strip_unsupported_openai(m: &mut Map) {
    for k in strict::OPENAI_DROPS {
        remove(m, k);
    }
    const FORMATS: [&str; 9] = [
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "uuid",
    ];
    if let Some(J::Str(f)) = get(m, "format")
        && !FORMATS.contains(&f.as_str())
    {
        remove(m, "format");
    }
}

// keywords anthropic structured output rejects with a 400
fn strip_unsupported_anthropic(m: &mut Map) {
    for k in strict::ANTHROPIC_DROPS {
        remove(m, k);
    }
    if let Some(J::Num(n)) = get(m, "minItems")
        && n.parse::<u64>().is_ok_and(|n| n > 1)
    {
        remove(m, "minItems");
    }
    const FORMATS: [&str; 10] = [
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    ];
    if let Some(J::Str(f)) = get(m, "format")
        && !FORMATS.contains(&f.as_str())
    {
        remove(m, "format");
    }
}
