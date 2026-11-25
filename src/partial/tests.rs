use super::*;
use serde_json::{Value, json};

fn parse_in_pieces(doc: &str, size: usize) -> PartialParser {
    let mut p = PartialParser::new();
    let mut i = 0;
    while i < doc.len() {
        let mut j = (i + size).min(doc.len());
        while !doc.is_char_boundary(j) {
            j += 1;
        }
        p.feed(&doc[i..j]);
        i = j;
    }
    p
}

const DOCS: &[&str] = &[
    r#"{"a":1,"b":[true,false,null],"c":{"d":"e"}}"#,
    r#"[1, -2.5e3, 0, 1E-2, "x", [], {}]"#,
    r#"{"s":"q\"uote \\ slash\/ \b\f\n\r\t end","u":"\u00e9\u6771\ud83d\ude00"}"#,
    r#"{"cjk":"東京 ✓","nested":[{"x":[[[]]]},{"y":{}}]}"#,
    "{ \"spaced\" :\n [ 1 ,\t2 ] }",
];

#[test]
fn matches_serde_json_for_every_chunk_size() {
    for doc in DOCS {
        let want: Value = serde_json::from_str(doc).unwrap();
        for size in [1, 2, 3, 4, 7, 64, 4096] {
            let p = parse_in_pieces(doc, size);
            assert!(p.is_complete(), "{doc} size {size}");
            assert_eq!(p.root().unwrap().to_value(), want, "{doc} size {size}");
        }
    }
}

#[test]
fn every_prefix_is_a_consistent_partial() {
    let doc = r#"{"name":"Ada","tags":["x","y"],"n":12.5,"ok":true}"#;
    for i in 0..doc.len() {
        let mut p = PartialParser::new();
        p.feed(&doc[..i]);
        assert!(!p.is_error(), "prefix {i}");
        assert!(!p.is_complete(), "prefix {i}");
        if let Some(root) = p.root() {
            assert_eq!(root.kind(), Kind::Object);
        }
    }
}

#[test]
fn partial_strings_and_numbers() {
    let mut p = PartialParser::new();
    p.feed(r#"{"title":"Hel"#);
    let r = p.root().unwrap();
    let t = r.get("title").unwrap();
    assert_eq!(t.as_str(), Some("Hel"));
    assert!(!t.is_complete());

    p.feed(r#"lo","total":12"#);
    let r = p.root().unwrap();
    assert!(r.get("title").unwrap().is_complete());
    let total = r.get("total").unwrap();
    assert_eq!(total.as_number_str(), Some("12"));
    assert!(!total.is_complete());

    p.feed(".5}");
    assert_eq!(p.root().unwrap().get("total").unwrap().as_f64(), Some(12.5));
    assert!(p.is_complete());
}

#[test]
fn key_is_not_visible_until_its_value_starts() {
    let mut p = PartialParser::new();
    p.feed(r#"{"a":1,"bc"#);
    assert_eq!(p.root().unwrap().len(), 1);
    p.feed(r#"":"#);
    assert_eq!(p.root().unwrap().len(), 1);
    p.feed("[");
    assert_eq!(p.root().unwrap().len(), 2);
    assert_eq!(p.root().unwrap().get("bc").unwrap().kind(), Kind::Array);
}

#[test]
fn surrogate_pair_split_across_feeds() {
    let doc = r#"{"e":"\ud83d\ude00"}"#;
    for i in 0..doc.len() {
        let mut p = PartialParser::new();
        p.feed(&doc[..i]);
        p.feed(&doc[i..]);
        assert_eq!(p.root().unwrap().get("e").unwrap().as_str(), Some("😀"));
    }
}

#[test]
fn lone_surrogate_becomes_replacement() {
    let mut p = PartialParser::new();
    p.feed(r#"["\ud83dx", "\ude00"]"#);
    let r = p.root().unwrap();
    assert_eq!(r.at(0).unwrap().as_str(), Some("\u{fffd}x"));
    assert_eq!(r.at(1).unwrap().as_str(), Some("\u{fffd}"));
}

#[test]
fn skips_text_around_the_root() {
    let mut p = PartialParser::new();
    p.feed("Here you go:\n```json\n{\"a\": [1, 2]}\n```\n");
    assert!(p.is_complete());
    assert_eq!(p.root().unwrap().to_value(), json!({"a": [1, 2]}));
}

#[test]
fn stops_on_invalid_input() {
    let mut p = PartialParser::new();
    p.feed(r#"{"a": 1 "b": 2}"#);
    assert!(p.is_error());
    assert_eq!(p.root().unwrap().to_value(), json!({"a": 1}));

    let mut p = PartialParser::new();
    p.feed(r#"{"a": [1}"#);
    assert!(p.is_error());

    let mut p = PartialParser::new();
    p.feed(r#"{"a": tru"#);
    p.feed("x");
    assert!(p.is_error());
}

#[test]
fn incomplete_number_converts_to_null() {
    let mut p = PartialParser::new();
    p.feed(r#"{"x": -"#);
    assert_eq!(p.root().unwrap().to_value(), json!({"x": null}));
}

#[test]
fn reset_allows_reuse() {
    let mut p = PartialParser::new();
    p.feed("{\"a\":1}");
    p.reset();
    assert!(p.root().is_none());
    p.feed("[2]");
    assert_eq!(p.root().unwrap().to_value(), json!([2]));
}

#[test]
fn iterators_and_accessors() {
    let mut p = PartialParser::new();
    p.feed(r#"{"k":[1,"two",false,null]}"#);
    let r = p.root().unwrap();
    let keys: Vec<&str> = r.entries().map(|(k, _)| k).collect();
    assert_eq!(keys, ["k"]);
    let arr = r.get("k").unwrap();
    assert_eq!(arr.items().count(), 4);
    assert_eq!(arr.at(1).unwrap().as_str(), Some("two"));
    assert_eq!(arr.at(2).unwrap().as_bool(), Some(false));
    assert!(arr.at(3).unwrap().is_null());
    assert!(arr.at(9).is_none());
    assert!(r.get("missing").is_none());
    assert_eq!(format!("{r:?}"), r#"{"k":[1,"two",false,null]}"#);
}
