use std::borrow::Cow;

use serde::Deserialize;
use serde_json::{Value, json};

use super::diff_tests::random_doc;
use super::*;

fn fixed(s: &str) -> (Value, bool) {
    let r = repair(s).unwrap_or_else(|| panic!("no repair for {s:?}"));
    let v = serde_json::from_str(&r.text).unwrap_or_else(|e| panic!("{e}: {}", r.text));
    (v, r.cut)
}

#[test]
fn valid_input_is_borrowed() {
    for s in [r#"{"a": [1, 2.5, "x"]}"#, "[]", "  {}\n", "\"text\"", "12"] {
        let r = repair(s).unwrap();
        assert!(matches!(r.text, Cow::Borrowed(t) if t == s), "{s}");
        assert!(!r.cut);
    }
}

#[test]
fn common_defects() {
    let cases: &[(&str, Value)] = &[
        ("```json\n{\"a\": 1}\n```", json!({"a": 1})),
        ("Here you go: {\"a\": 1} hope it helps", json!({"a": 1})),
        ("{\"a\": 1,}", json!({"a": 1})),
        ("[1, 2, ]", json!([1, 2])),
        (
            "{'a': 'it\\'s', 'b': \"x\"}",
            json!({"a": "it's", "b": "x"}),
        ),
        ("{'a': 'it's ok'}", json!({"a": "it's ok"})),
        ("{a: 1, b_c: 2}", json!({"a": 1, "b_c": 2})),
        (
            "{\"a\": 1 // note\n, /* x */ \"b\": 2}",
            json!({"a": 1, "b": 2}),
        ),
        (
            "{\"a\": True, \"b\": False, \"c\": None, \"d\": NaN}",
            json!({"a": true, "b": false, "c": null, "d": null}),
        ),
        (
            "{\"a\": \"line\nbreak\ttab\"}",
            json!({"a": "line\nbreak\ttab"}),
        ),
        (
            "{\"a\": \"say \"hi\" now\", \"b\": 1}",
            json!({"a": "say \"hi\" now", "b": 1}),
        ),
        ("{\"a\": 1 \"b\": 2}", json!({"a": 1, "b": 2})),
        ("[1 2 3]", json!([1, 2, 3])),
        ("{\"a\": [1, 2}", json!({"a": [1, 2]})),
        ("{\"a\": +1, \"b\": .5}", json!({"a": 1, "b": 0.5})),
        ("{\"p\": \"C:\\dir\"}", json!({"p": "C:\\dir"})),
        ("{\"s\": \"\\ud800x\"}", json!({"s": "\u{fffd}x"})),
        ("{\"a\": 1,, \"b\": 2}", json!({"a": 1, "b": 2})),
        ("{\"a\": hello world}", json!({"a": "hello world"})),
    ];
    for (input, want) in cases {
        let (v, cut) = fixed(input);
        assert_eq!(&v, want, "{input:?}");
        assert!(!cut, "{input:?}");
    }
}

#[test]
fn truncation_closes_and_drops() {
    let cases: &[(&str, Value)] = &[
        ("{\"a\": 1, \"b\": \"hel", json!({"a": 1, "b": "hel"})),
        ("{\"a\": 1, \"b", json!({"a": 1})),
        ("{\"a\": 1, \"b\":", json!({"a": 1})),
        ("{\"a\": 1, \"b\": 12", json!({"a": 1})),
        ("{\"a\": 1, \"b\": tr", json!({"a": 1})),
        ("{\"a\": 1, \"b\": true", json!({"a": 1, "b": true})),
        ("{\"a\": [1, 2, 3", json!({"a": [1, 2]})),
        ("{\"a\": [{\"x\": 1}, {\"x\"", json!({"a": [{"x": 1}, {}]})),
        ("{\"a\": 1,", json!({"a": 1})),
        ("{", json!({})),
        ("[", json!([])),
        ("{\"a\": \"x\\", json!({"a": "x"})),
        ("{\"a\": \"x\\u00", json!({"a": "x"})),
    ];
    for (input, want) in cases {
        let (v, cut) = fixed(input);
        assert_eq!(&v, want, "{input:?}");
        assert!(cut, "{input:?}");
    }
}

#[test]
fn nothing_to_repair() {
    for s in ["", "   ", "no json here", "}"] {
        assert!(repair(s).is_none(), "{s:?}");
    }
}

#[test]
fn properties_on_random_documents() {
    for seed in 0..3000 {
        let doc = random_doc(seed);
        let r = repair(&doc).unwrap();
        assert!(matches!(r.text, Cow::Borrowed(_)), "{doc}");
        // every prefix: no panic; when repaired, valid and stable
        let mut cuts = 0;
        for (i, _) in doc.char_indices().step_by(3) {
            let Some(r) = repair(&doc[..i]) else { continue };
            assert!(serde_json::from_str::<Value>(&r.text).is_ok(), "{}", r.text);
            let again = repair(&r.text).unwrap();
            assert_eq!(again.text, r.text);
            assert!(!again.cut);
            cuts += 1;
        }
        let _ = cuts;
    }
}

#[derive(Debug, Deserialize)]
#[expect(dead_code, reason = "fields are only filled by deserialization")]
struct Invoice {
    invoice_number: String,
    vendor: String,
    items: Vec<Item>,
    notes: Option<String>,
    total: f64,
}

#[derive(Debug, Deserialize)]
#[expect(dead_code, reason = "fields are only filled by deserialization")]
struct Item {
    description: String,
    quantity: u32,
    sku: Option<String>,
}

const INVOICE: &str = r#"{"invoice_number":"INV-1","vendor":"GOLIA","items":[{"description":"Widget with \"quoted\" spec 東京","quantity":1,"sku":"SKU-0"},{"description":"Other","quantity":2,"sku":null}],"notes":"Payment due.\nThanks.","total":98765.25}"#;

#[test]
fn malformed_invoices_parse_after_repair() {
    let trailing = INVOICE.replace("}]", "},]").replacen('}', ",}", 1);
    let single = INVOICE.replace('"', "'");
    let fenced = format!("Here is the JSON:\n```json\n{INVOICE}\n```\n");
    for s in [INVOICE.to_string(), trailing, single, fenced] {
        let r = repair(&s).unwrap_or_else(|| panic!("{s}"));
        from_str::<Invoice>(&r.text).unwrap_or_else(|e| panic!("{e}: {}", r.text));
        assert!(!r.cut);
    }
    let end = INVOICE.floor_char_boundary(INVOICE.len() * 2 / 3);
    let r = repair(&INVOICE[..end]).unwrap();
    assert!(r.cut);
    assert!(serde_json::from_str::<Value>(&r.text).is_ok());
}
