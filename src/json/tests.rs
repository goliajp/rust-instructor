use std::borrow::Cow;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::*;

/// Values equal, floats within one ulp (serde_json's default float parser
/// is not always correctly rounded; ours is).
pub(super) fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            _ if x.is_u64() || x.is_i64() || y.is_u64() || y.is_i64() => x == y,
            (Some(p), Some(q)) => p == q || (p.to_bits() as i64 - q.to_bits() as i64).abs() <= 1,
            _ => false,
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|((k1, v1), (k2, v2))| k1 == k2 && same(v1, v2))
        }
        _ => a == b,
    }
}

pub(super) fn agree(s: &str) {
    let ours = from_str::<Value>(s);
    let theirs = serde_json::from_str::<Value>(s);
    match (&ours, &theirs) {
        (Ok(a), Ok(b)) => assert!(same(a, b), "{s:?}: ours {a:?} serde_json {b:?}"),
        (Err(_), Err(_)) => {}
        _ => panic!("{s:?}: ours {ours:?} serde_json {theirs:?}"),
    }
    let ours = from_slice::<Value>(s.as_bytes()).is_ok();
    assert_eq!(ours, theirs.is_ok(), "{s:?} from_slice");
    assert_eq!(is_valid(s), theirs.is_ok(), "{s:?} is_valid");
}

const EDGE: &[&str] = &[
    "0",
    "-0",
    "-0.0",
    "1.5",
    "-1.5e10",
    "1E5",
    "1e-5",
    "1e400",
    "-1e400",
    "5e-324",
    "2.2250738585072014e-308",
    "18446744073709551615",
    "18446744073709551616",
    "-9223372036854775808",
    "-9223372036854775809",
    "123456789012345678901234567890",
    "01",
    "1.",
    ".5",
    "-",
    "+1",
    "1e",
    "1e+",
    "true",
    "tru",
    "nul",
    r#""a\"b\\c\/d\b\f\n\r\t""#,
    r#""é中""#,
    r#""😀""#,
    r#""\ud83d""#,
    r#""\ud83dx""#,
    r#""\ude00""#,
    "\"a\u{1}b\"",
    r#""\x""#,
    r#""\u12""#,
    "\"abc",
    "[1,2,]",
    r#"{"a":1,}"#,
    "[1 2]",
    r#"{"a" 1}"#,
    r#"{"a":1 "b":2}"#,
    "[[[]]]",
    "{}",
    "[]",
    "  [1]  ",
    "[1] x",
    "{1:2}",
    "",
    " ",
    "[",
    r#"{"a":"#,
    r#"{"a":1,"a":2}"#,
    "nulll",
    "[true,false,null]",
    "\u{feff}1",
];

#[test]
fn edge_cases_agree_with_serde_json() {
    for s in EDGE {
        agree(s);
    }
}

#[test]
fn depth_limit_matches() {
    for n in [126, 127, 128, 129] {
        agree(&format!("{}{}", "[".repeat(n), "]".repeat(n)));
        agree(&format!("{}{}", "{\"a\":".repeat(n), "}".repeat(n)));
    }
}

#[test]
fn numbers_take_serde_json_shapes() {
    let v: Value = from_str("[-0, 18446744073709551615, -1, 1.0]").unwrap();
    assert!(v[0].is_f64() && v[0].as_f64().unwrap().is_sign_negative());
    assert_eq!(v[1].as_u64(), Some(u64::MAX));
    assert_eq!(v[2].as_i64(), Some(-1));
    assert!(v[3].is_f64());
}

#[test]
fn strings_borrow_without_escapes() {
    #[derive(Deserialize)]
    struct S<'a> {
        #[serde(borrow)]
        a: Cow<'a, str>,
        #[serde(borrow)]
        b: Cow<'a, str>,
    }
    let s: S<'_> = from_str(r#"{"a":"plain","b":"esc\naped"}"#).unwrap();
    assert!(matches!(s.a, Cow::Borrowed("plain")));
    assert!(matches!(s.b, Cow::Owned(_)));
    assert_eq!(s.b, "esc\naped");
}

#[test]
fn raw_borrows_the_value_text() {
    #[derive(Deserialize)]
    struct Block<'a> {
        #[serde(rename = "type")]
        kind: &'a str,
        #[serde(borrow, default)]
        input: Option<Raw<'a>>,
    }
    #[derive(Deserialize)]
    struct Resp<'a> {
        #[serde(borrow)]
        content: Vec<Block<'a>>,
    }
    let body = r#"{"id":"x","content":[{"type":"text","text":"hi"},
        {"type":"tool_use","input": {"a": [1, "}"], "b": null} }],"usage":{}}"#;
    let r: Resp<'_> = from_str(body).unwrap();
    assert_eq!(r.content[0].kind, "text");
    assert_eq!(r.content[0].input, None);
    assert_eq!(
        r.content[1].input,
        Some(Raw(r#"{"a": [1, "}"], "b": null}"#))
    );
}

#[test]
fn errors_carry_positions() {
    let e = from_str::<Value>("{\n  \"a\": tru\n}").unwrap_err();
    assert!(e.to_string().contains("line 2"), "{e}");
    assert!(e.offset().is_some());
    #[derive(Debug, Deserialize)]
    #[expect(dead_code, reason = "only the error is checked")]
    struct T {
        a: u8,
    }
    let e = from_str::<T>(r#"{"a": 300}"#).unwrap_err();
    assert!(e.to_string().contains("300"), "{e}");
    let e = from_str::<T>("{}").unwrap_err();
    assert!(e.to_string().contains("missing field `a`"), "{e}");
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Kind {
    Plain,
    Pair(u8, i16),
    Wrapped(String),
    Named { x: f64, tag: Option<char> },
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum Loose {
    Num(i64),
    Text(String),
    List(Vec<bool>),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Extra {
    note: Option<String>,
    weight: f32,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", content = "c")]
enum Adjacent {
    A,
    B(u32),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Doc {
    id: u64,
    neg: i32,
    name: String,
    kind: Kind,
    kinds: Vec<Kind>,
    loose: Vec<Loose>,
    #[serde(flatten)]
    extra: Extra,
    map: HashMap<String, Vec<u16>>,
    by_num: HashMap<u32, bool>,
    tuple: (bool, String),
    unit: (),
    adjacent: Adjacent,
    #[serde(default)]
    missing: Vec<u8>,
    #[serde(with = "serde_bytes_like")]
    bytes: Vec<u8>,
}

mod serde_bytes_like {
    use serde::{Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Vec<u8>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("bytes")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Vec<u8>, A::Error> {
                let mut v = Vec::new();
                while let Some(b) = a.next_element()? {
                    v.push(b);
                }
                Ok(v)
            }
            fn visit_bytes<E>(self, b: &[u8]) -> Result<Vec<u8>, E> {
                Ok(b.to_vec())
            }
        }
        d.deserialize_bytes(V)
    }
}

pub(super) fn sample_doc(seed: u64) -> Doc {
    let mut r = super::diff_tests::Rng(seed | 1);
    let mut map = HashMap::new();
    map.insert(r.string(), vec![r.next() as u16, 7]);
    let mut by_num = HashMap::new();
    by_num.insert(r.next() as u32, true);
    Doc {
        id: r.next(),
        neg: r.next() as i32,
        name: r.string(),
        kind: Kind::Named {
            x: r.float(),
            tag: Some('é'),
        },
        kinds: vec![
            Kind::Plain,
            Kind::Pair(r.next() as u8, r.next() as i16),
            Kind::Wrapped(r.string()),
        ],
        loose: vec![
            Loose::Num(r.next() as i64),
            Loose::Text(r.string()),
            Loose::List(vec![true, false]),
        ],
        extra: Extra {
            note: if r.next() % 2 == 0 {
                Some(r.string())
            } else {
                None
            },
            weight: 1.5,
        },
        map,
        by_num,
        tuple: (true, r.string()),
        unit: (),
        adjacent: Adjacent::B(r.next() as u32),
        missing: vec![],
        bytes: vec![1, 2, 255],
    }
}

#[test]
fn typed_round_trip_matches_serde_json() {
    for seed in 0..300 {
        let doc = sample_doc(seed);
        let text = serde_json::to_string_pretty(&doc).unwrap();
        let ours: Doc = from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        // serde_json's default float parser can be one ulp off, so only
        // ours is compared with the original
        serde_json::from_str::<Doc>(&text).unwrap();
        assert_eq!(ours, doc);
    }
}

#[test]
fn typed_errors_agree_on_mutations() {
    let mut r = super::diff_tests::Rng(99);
    for seed in 0..200 {
        let text = serde_json::to_string(&sample_doc(seed)).unwrap();
        let mut bytes = text.into_bytes();
        for _ in 0..1 + r.next() % 3 {
            let at = (r.next() as usize) % bytes.len();
            match r.next() % 3 {
                0 => {
                    bytes.remove(at);
                }
                1 => bytes.insert(at, b"{}[],:\"0a-"[(r.next() % 10) as usize]),
                _ => bytes[at] = b"{}[],:\"0a-"[(r.next() % 10) as usize],
            }
        }
        let Ok(s) = std::str::from_utf8(&bytes) else {
            continue;
        };
        let ours = from_str::<Doc>(s);
        let theirs = serde_json::from_str::<Doc>(s);
        assert_eq!(ours.is_ok(), theirs.is_ok(), "{s}\n{ours:?}\n{theirs:?}");
        if let (Ok(a), Ok(b)) = (ours, theirs) {
            assert_eq!(a, b, "{s}");
        }
    }
}

#[test]
fn writer_matches_serde_json() {
    let mut all = String::new();
    for c in (0u32..0x80).filter_map(char::from_u32) {
        all.push(c);
    }
    all.push_str("é東京😀\u{7f}\u{2028}");
    for s in [
        "",
        "plain ascii that is longer than eight bytes",
        "\"",
        "\\",
        all.as_str(),
    ] {
        let mut ours = Vec::new();
        write_str(&mut ours, s);
        assert_eq!(
            String::from_utf8(ours).unwrap(),
            serde_json::to_string(s).unwrap()
        );
    }
    for n in [0, 9, 10, 99, 100, 12345, u32::MAX as u64, u64::MAX] {
        let mut ours = Vec::new();
        write_u64(&mut ours, n);
        assert_eq!(ours, serde_json::to_vec(&n).unwrap());
    }
    for n in [
        0.0,
        -0.0,
        0.7,
        1.0,
        1e16,
        1.5e-7,
        123.456,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        let mut ours = Vec::new();
        write_f64(&mut ours, n);
        let back: f64 = serde_json::from_slice(&ours).unwrap();
        assert_eq!(
            back.to_bits(),
            n.to_bits(),
            "{}",
            String::from_utf8_lossy(&ours)
        );
    }
    let mut ours = Vec::new();
    write_f64(&mut ours, f64::NAN);
    assert_eq!(ours, b"null");
    let mut ours = Vec::new();
    write_f64(&mut ours, 0.0);
    assert_eq!(ours, b"0.0");
}
