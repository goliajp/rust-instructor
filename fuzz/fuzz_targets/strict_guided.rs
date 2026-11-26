#![no_main]

// Strict answers read by the guided deserializer against the reference:
// rewrite into the type's own JSON text, then parse. Both must accept the
// same inputs and give the same values.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use instructors::Schema;
use instructors::__private::{strict_guided, strict_reference};
use libfuzzer_sys::fuzz_target;
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Inventory {
    counts: BTreeMap<String, u32>,
    by_id: BTreeMap<u8, String>,
    pair: (u8, String),
    tags: BTreeSet<String>,
    flags: Option<BTreeMap<String, bool>>,
    #[schema(min = 1, max = 5)]
    rating: u8,
    big: u128,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
enum Shape {
    Dot,
    Circle(f64),
    Rect { w: u32, h: u32 },
    Line(u8, u8),
    Tagged(BTreeMap<String, u8>),
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[serde(tag = "kind")]
enum Figure {
    Circle { r: f64 },
    Square { labels: BTreeMap<String, String> },
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[serde(untagged)]
enum Loose {
    Pair((u8, u8)),
    Map(BTreeMap<String, u8>),
    Text(String),
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Holder {
    shapes: Vec<Shape>,
    figure: Option<Figure>,
    loose: Vec<Loose>,
    nested: Option<Vec<BTreeMap<String, (bool, Option<i8>)>>>,
}

fn check<T: DeserializeOwned + Schema + PartialEq + Debug>(s: &str) {
    match (strict_reference::<T>(s), strict_guided::<T>(s)) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{s:?}"),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("{s:?}\nreference: {a:?}\nguided: {b:?}"),
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&which, rest)) = data.split_first() else {
        return;
    };
    let Ok(s) = std::str::from_utf8(rest) else {
        return;
    };
    match which % 4 {
        0 => check::<Inventory>(s),
        1 => check::<Holder>(s),
        2 => check::<Vec<(u8, Shape)>>(s),
        _ => check::<BTreeMap<String, Loose>>(s),
    }
});
