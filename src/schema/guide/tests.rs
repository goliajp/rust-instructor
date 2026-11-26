// The guided reader against the reference path it replaces: rewrite the
// answer into the type's own JSON text, then parse that. Both must accept
// the same answers with the same values, on valid answers and on every
// answer one edit away from one.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;

use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::Schema;
use crate::schema::entry_for;

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Inventory {
    counts: HashMap<String, u32>,
    by_id: BTreeMap<u32, String>,
    pair: (u8, String),
    tags: HashSet<String>,
    flags: Option<BTreeMap<String, bool>>,
    #[schema(min = 1, max = 5)]
    rating: u8,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
enum Shape {
    Dot,
    Circle(f64),
    Rect { w: u32, h: u32 },
    Line(u8, u8),
    Tagged(HashMap<String, u8>),
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[serde(tag = "kind")]
enum Figure {
    Circle {
        r: f64,
    },
    Square {
        side: f64,
        labels: BTreeMap<String, String>,
    },
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[serde(untagged)]
enum Loose {
    Pair((u8, u8)),
    Map(BTreeMap<String, u8>),
    Text(String),
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[serde(tag = "t", content = "c")]
enum Adjacent {
    A(BTreeMap<String, u8>),
    B,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Outer {
    name: String,
    #[serde(flatten)]
    inner: Inner,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Inner {
    extra: HashMap<String, (u8, bool)>,
    n: u64,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
enum Expr {
    Num(u32),
    Add(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Big {
    exact: u64,
    wide: u128,
    m: BTreeMap<String, u128>,
    neg: i128,
}

type Rows = Vec<HashMap<String, Vec<(u8, Option<String>)>>>;

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Deep {
    rows: Rows,
    opt: Option<Vec<(bool, i8)>>,
    shapes: Vec<Shape>,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Holder {
    figure: Figure,
    loose: Vec<Loose>,
    adjacent: Adjacent,
    outer: Outer,
    expr: Option<Expr>,
    maybe: Option<Inner>,
}

fn both<T: DeserializeOwned + Schema>(text: &str) -> (Result<T, String>, Result<T, String>) {
    let e = entry_for::<T>(false);
    let old = e
        .unwire(text)
        .and_then(|t| crate::json::from_str::<T>(&t))
        .map_err(|e| e.to_string());
    let new = e
        .guide()
        .answer::<T>(text, false)
        .map_err(|e| e.to_string());
    (old, new)
}

fn agree<T: DeserializeOwned + Schema + PartialEq + Debug>(text: &str) -> bool {
    match both::<T>(text) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "{text}");
            true
        }
        (Err(_), Err(_)) => false,
        (o, n) => panic!("{text}\nreference: {o:?}\nguided: {n:?}"),
    }
}

/// A valid answer, and every answer one edit away from it.
fn agree_near<T: DeserializeOwned + Schema + PartialEq + Debug>(text: &str) {
    assert!(agree::<T>(text), "not accepted: {text}");
    let b = text.as_bytes();
    let mut edits = Vec::new();
    for i in 0..b.len() {
        let mut d = b.to_vec();
        d.remove(i);
        edits.push(d);
        let mut d = b.to_vec();
        d.insert(i, b[i]);
        edits.push(d);
        if i + 1 < b.len() {
            let mut d = b.to_vec();
            d.swap(i, i + 1);
            edits.push(d);
        }
        for &c in b"{}[]\",:0n" {
            let mut d = b.to_vec();
            d[i] = c;
            edits.push(d);
        }
    }
    for e in edits {
        if let Ok(s) = String::from_utf8(e) {
            agree::<T>(&s);
        }
    }
}

#[test]
fn maps_tuples_sets_and_bounds() {
    agree_near::<Inventory>(
        r#"{"counts": [{"key": "a", "value": 3}, {"value": 0, "key": "bA"}], "by_id": [{"key": "7", "value": "x"}], "pair": {"1": "two", "0": 2}, "tags": ["a", "b"], "flags": [{"key": "f", "value": true}], "rating": 4}"#,
    );
    agree_near::<Inventory>(
        r#"{"counts": {"a": 1}, "by_id": {"7": "x"}, "pair": [2, "y"], "tags": [], "flags": null, "rating": 1}"#,
    );
    // one edit from a repeated item
    agree_near::<Inventory>(
        r#"{"counts": [], "by_id": [], "pair": [2, "y"], "tags": ["a", "b", "a "], "flags": null, "rating": 1}"#,
    );
}

#[test]
fn enums_of_every_tagging() {
    agree_near::<Holder>(
        r#"{"figure": {"kind": "Square", "side": 2.5, "labels": [{"key": "k", "value": "v"}]}, "loose": [{"0": 1, "1": 2}, [{"key": "m", "value": 3}], "t"], "adjacent": {"t": "A", "c": [{"key": "z", "value": 9}]}, "outer": {"name": "n", "extra": [{"key": "e", "value": {"0": 1, "1": true}}], "n": 5}, "expr": {"Add": {"0": {"Num": 1}, "1": {"Num": 2}}}, "maybe": null}"#,
    );
    agree_near::<Holder>(
        r#"{"figure": {"kind": "Circle", "r": 1}, "loose": [], "adjacent": {"t": "B"}, "outer": {"name": "", "extra": [], "n": 0}, "expr": null, "maybe": {"extra": [{"key": "x", "value": [1, false]}], "n": 1}}"#,
    );
}

#[test]
fn nested_lists_and_options() {
    agree_near::<Deep>(
        r#"{"rows": [[{"key": "r", "value": [{"0": 1, "1": null}, {"0": 2, "1": "s"}]}]], "opt": [{"0": true, "1": -3}], "shapes": ["Dot", {"Circle": 1.5}, {"Rect": {"w": 1, "h": 2}}, {"Line": {"0": 1, "1": 2}}, {"Tagged": [{"key": "q", "value": 1}]}]}"#,
    );
}

#[test]
fn numbers_keep_every_digit() {
    let text = r#"{"exact": 18446744073709551615, "wide": 340282366920938463463374607431768211455, "m": [{"key": "a", "value": 123456789012345678901234567890}], "neg": -170141183460469231731687303715884105728}"#;
    agree_near::<Big>(text);
    let v: Big = entry_for::<Big>(false).guide().answer(text, false).unwrap();
    assert_eq!(v.exact, u64::MAX);
    assert_eq!(v.wide, u128::MAX);
    assert_eq!(v.m["a"], 123456789012345678901234567890);
    assert_eq!(v.neg, i128::MIN);
}

#[test]
fn roots_that_are_not_objects() {
    agree_near::<Vec<(u8, String)>>(r#"{"value": [{"0": 1, "1": "a"}, [2, "b"]]}"#);
    agree_near::<BTreeMap<String, Shape>>(r#"{"value": [{"key": "s", "value": "Dot"}]}"#);
    agree_near::<Shape>(r#"{"value": {"Rect": {"w": 3, "h": 4}}}"#);
    agree_near::<Shape>(r#""Dot""#);
}

#[test]
fn messages_name_the_place() {
    let g = entry_for::<Inventory>(false).guide();
    let base = |counts: &str, tags: &str, rating: u8| {
        format!(
            r#"{{"counts": {counts}, "by_id": [], "pair": {{"0": 1, "1": "x"}}, "tags": {tags}, "flags": null, "rating": {rating}}}"#
        )
    };
    let e = g
        .answer::<Inventory>(&base("[]", r#"["a", "a"]"#, 3), false)
        .unwrap_err()
        .to_string();
    assert!(e.starts_with("/tags: the list has repeated items"), "{e}");
    let e = g
        .answer::<Inventory>(&base("[]", "[]", 9), false)
        .unwrap_err()
        .to_string();
    assert!(
        e.starts_with("/rating: 9 is greater than the maximum 5"),
        "{e}"
    );
    let dup = r#"[{"key": "a", "value": 1}, {"key": "a", "value": 2}]"#;
    let e = g
        .answer::<Inventory>(&base(dup, "[]", 3), false)
        .unwrap_err()
        .to_string();
    assert!(e.starts_with(r#"/counts/1: a repeated key "a""#), "{e}");
}

#[test]
fn lists_and_items() {
    let e = entry_for::<Shape>(true);
    let g = e.guide();
    let want = vec![Shape::Dot, Shape::Line(1, 2)];
    let wrapped = r#"{"items": ["Dot", {"Line": {"0": 1, "1": 2}}]}"#;
    assert_eq!(g.answer::<Vec<Shape>>(wrapped, true).unwrap(), want);
    let bare = r#"["Dot", {"Line": {"0": 1, "1": 2}}]"#;
    assert_eq!(g.answer::<Vec<Shape>>(bare, true).unwrap(), want);
    assert_eq!(
        g.item::<Shape>(r#"{"Line": {"1": 2, "0": 1}}"#).unwrap(),
        Shape::Line(1, 2)
    );
}
