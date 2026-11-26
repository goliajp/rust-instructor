//! Answer text → the caller's type: direct parse, then repair, then (for a
//! cut-off list) the items that were complete.

use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::json;
use crate::partial::PartialParser;
use crate::text::clip;
use crate::wire::Stop;

#[derive(Deserialize)]
struct Items<T> {
    items: T,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Bad {
    /// Not the type's JSON; the message for the re-ask.
    Invalid(String),
    /// Cut off by the token limit and not accepted.
    Truncated,
}

/// Reads answers whose schema was reshaped for the wire, straight into
/// the caller's type.
pub(crate) type Guide = Option<crate::schema::Reader>;

fn direct<O: DeserializeOwned>(s: &str, list: bool, guide: Guide) -> Result<O, json::JsonError> {
    if let Some(g) = guide {
        return g.answer(s, list);
    }
    if list {
        // a model that ignored the wrapper may still return the bare list
        json::from_str::<Items<O>>(s)
            .map(|w| w.items)
            .or_else(|e| json::from_str::<O>(s).map_err(|_| e))
    } else {
        json::from_str(s)
    }
}

/// Parse an answer. Returns the value and whether it came from a cut-off
/// answer.
pub(crate) fn parse<O: DeserializeOwned>(
    content: &str,
    list: bool,
    stop: &Stop,
    accept_truncated: bool,
    guide: Guide,
) -> Result<(O, bool), Bad> {
    let direct_err = match direct(content, list, guide) {
        Ok(v) => return Ok((v, false)),
        Err(e) => e,
    };
    let cut = matches!(stop, Stop::Truncated(_));
    if cut && !accept_truncated {
        return Err(Bad::Truncated);
    }
    let repaired = json::repair(content);
    if let Some(r) = &repaired
        && r.text != content
        && let Ok(v) = direct(&r.text, list, guide)
    {
        return Ok((v, cut || r.cut));
    }
    if cut {
        return complete_items(content, guide).ok_or(Bad::Truncated);
    }
    Err(Bad::Invalid(match repaired {
        Some(r) if r.text != content => {
            format!("{direct_err} (repaired: {})", clip(&r.text, 200))
        }
        _ => format!("{direct_err} (raw: {})", clip(content, 200)),
    }))
}

/// The list items of a cut-off answer that were complete before the cut.
fn complete_items<O: DeserializeOwned>(content: &str, guide: Guide) -> Option<(O, bool)> {
    let mut p = PartialParser::new();
    p.feed(content);
    let root = p.root()?;
    let list = root.get("items").unwrap_or(root);
    let mut out = b"{\"items\":[".to_vec();
    for (i, item) in list.items().take_while(|v| v.is_complete()).enumerate() {
        if i > 0 {
            out.push(b',');
        }
        item.write_json(&mut out);
    }
    out.extend_from_slice(b"]}");
    let text = String::from_utf8(out).ok()?;
    direct::<O>(&text, true, guide).ok().map(|v| (v, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct P {
        a: i32,
    }

    #[test]
    fn list_wrapper_and_bare_list() {
        let done = Stop::Done;
        assert_eq!(
            parse::<Vec<P>>(r#"{"items":[{"a":1}]}"#, true, &done, false, None).unwrap(),
            (vec![P { a: 1 }], false)
        );
        assert_eq!(
            parse::<Vec<P>>(r#"[{"a":2}]"#, true, &done, false, None).unwrap(),
            (vec![P { a: 2 }], false)
        );
    }

    #[test]
    fn repairs_complete_answers_only() {
        assert_eq!(
            parse::<P>("{'a': 3,}", false, &Stop::Done, false, None).unwrap(),
            (P { a: 3 }, false)
        );
        let cut = Stop::Truncated("length".into());
        assert_eq!(
            parse::<P>("{'a': 3,}", false, &cut, false, None),
            Err(Bad::Truncated)
        );
    }

    #[test]
    fn accepted_truncation_keeps_complete_items() {
        let cut = Stop::Truncated("length".into());
        let (v, truncated) =
            parse::<Vec<P>>(r#"{"items":[{"a":1},{"a":2},{"a""#, true, &cut, true, None).unwrap();
        assert!(truncated);
        assert_eq!(v[..2], [P { a: 1 }, P { a: 2 }]);
    }

    #[test]
    fn invalid_answer_quotes_the_raw_text() {
        match parse::<P>("no json here", false, &Stop::Done, false, None) {
            Err(Bad::Invalid(m)) => assert!(m.contains("no json here"), "{m}"),
            other => panic!("{other:?}"),
        }
    }
}
