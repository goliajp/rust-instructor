#![no_main]

use instructors::partial::PartialParser;
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

fn state(p: &PartialParser) -> (bool, bool, Option<Value>) {
    (
        p.is_complete(),
        p.is_error(),
        p.root().map(|r| r.to_value()),
    )
}

fuzz_target!(|data: &[u8]| {
    let Some((&seed, rest)) = data.split_first() else {
        return;
    };
    let Ok(s) = std::str::from_utf8(rest) else {
        return;
    };
    let mut one = PartialParser::new();
    one.feed(s);
    let want = state(&one);

    let size = seed as usize % 9 + 1;
    let mut pieces = PartialParser::new();
    let mut at = 0;
    while at < s.len() {
        let mut end = (at + size).min(s.len());
        while !s.is_char_boundary(end) {
            end += 1;
        }
        pieces.feed(&s[at..end]);
        // every intermediate view must be readable
        if let Some(r) = pieces.root() {
            let _ = r.to_value();
        }
        at = end;
    }
    assert_eq!(state(&pieces), want);

    // a complete object or array parses to what serde_json reads
    if let Ok(v @ (Value::Object(_) | Value::Array(_))) = serde_json::from_str::<Value>(s) {
        assert!(want.0, "complete json not complete: {s:?}");
        assert_eq!(want.2.as_ref(), Some(&v), "{s:?}");
    }
});
