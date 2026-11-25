#![no_main]

use std::collections::BTreeMap;

use instructors::__private as ours;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, PartialEq, Deserialize)]
enum Kind {
    Unit,
    New(i32),
    Tuple(u8, bool),
    Named { a: Option<String> },
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(untagged)]
enum Loose {
    N(u64),
    S(String),
    L(Vec<Loose>),
}

#[derive(Debug, PartialEq, Deserialize)]
struct Inner {
    x: Option<f64>,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Typed {
    a: Option<u32>,
    #[serde(default)]
    b: Vec<i64>,
    kind: Option<Kind>,
    loose: Option<Loose>,
    map: Option<BTreeMap<String, bool>>,
    #[serde(flatten)]
    inner: Inner,
}

fuzz_target!(|data: &[u8]| {
    let theirs = serde_json::from_slice::<Value>(data);
    let mine = ours::json_from_slice::<Value>(data);
    match (&mine, &theirs) {
        (Ok(a), Ok(b)) => assert_eq!(a, b),
        (Err(_), Err(_)) => {}
        _ => panic!("value: ours {mine:?} serde_json {theirs:?}"),
    }
    if let Ok(s) = std::str::from_utf8(data) {
        assert_eq!(ours::json_is_valid(s), theirs.is_ok());
        let mine = ours::json_from_str::<Typed>(s);
        let theirs = serde_json::from_str::<Typed>(s);
        match (&mine, &theirs) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(_), Err(_)) => {}
            _ => panic!("typed: ours {mine:?} serde_json {theirs:?}"),
        }
    }
});
