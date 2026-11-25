#![no_main]

use std::borrow::Cow;

use instructors::__private as ours;
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    let valid = serde_json::from_str::<Value>(s).is_ok();
    let Some(r) = ours::json_repair(s) else {
        assert!(!valid, "valid input not returned: {s:?}");
        return;
    };
    if valid {
        assert!(matches!(r.text, Cow::Borrowed(t) if t == s), "{s:?}");
        assert!(!r.cut);
    }
    if let Err(e) = serde_json::from_str::<Value>(&r.text) {
        panic!("{e}: {s:?} -> {:?}", r.text);
    }
    let again = ours::json_repair(&r.text).expect("repaired text repairs");
    assert_eq!(again.text, r.text);
    assert!(!again.cut);
});
