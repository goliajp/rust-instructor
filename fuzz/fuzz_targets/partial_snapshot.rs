#![no_main]
#![allow(dead_code, reason = "the fuzz types are only read")]

// Typed snapshots read from the partial tree: never a panic on any prefix
// of any input, and on a complete document the tree reads the same value
// as serde_json.

use std::collections::BTreeMap;

use instructors::Schema;
use instructors::partial::PartialParser;
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize, Schema)]
#[schema(partial)]
struct Inner {
    name: String,
    n: Option<u64>,
}

#[derive(Debug, Deserialize, Schema)]
#[schema(partial)]
enum Kind {
    A,
    B(u8),
}

#[derive(Debug, Deserialize, Schema)]
#[schema(partial)]
struct Doc {
    title: String,
    items: Vec<Inner>,
    by: BTreeMap<String, Inner>,
    kind: Kind,
    pair: (i8, bool),
    wide: Option<u128>,
}

fuzz_target!(|data: &[u8]| {
    let Some((&seed, rest)) = data.split_first() else {
        return;
    };
    let Ok(s) = std::str::from_utf8(rest) else {
        return;
    };
    let size = seed as usize % 7 + 1;
    let mut p = PartialParser::new();
    let mut at = 0;
    while at < s.len() {
        let mut end = (at + size).min(s.len());
        while !s.is_char_boundary(end) {
            end += 1;
        }
        p.feed(&s[at..end]);
        if let Some(r) = p.root() {
            let _ = r.snapshot::<Doc>();
            let _ = r.snapshot::<Vec<Kind>>();
            let _ = r.snapshot::<Value>();
        }
        at = end;
    }
    if let Ok(v @ (Value::Object(_) | Value::Array(_))) = serde_json::from_str::<Value>(s) {
        let root = p.root().expect("a complete document has a root");
        assert_eq!(root.snapshot::<Value>().as_ref(), Some(&v), "{s:?}");
    }
});
