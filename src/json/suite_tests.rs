//! The JSONTestSuite files (nst/JSONTestSuite at 1ef36fa, MIT), parsed by
//! this module and by serde_json. The two large nesting files are generated
//! here instead of stored.

use std::fs;
use std::path::Path;

use serde_json::Value;

use super::tests::same;
use super::*;

fn check(name: &str, bytes: &[u8]) {
    let ours = from_slice::<Value>(bytes);
    let theirs = serde_json::from_slice::<Value>(bytes);
    match (&ours, &theirs) {
        (Ok(a), Ok(b)) => assert!(same(a, b), "{name}: ours {a:?} serde_json {b:?}"),
        (Err(_), Err(_)) => {}
        _ => panic!("{name}: ours {ours:?} serde_json {theirs:?}"),
    }
    if let Ok(s) = std::str::from_utf8(bytes) {
        assert_eq!(is_valid(s), theirs.is_ok(), "{name} is_valid");
    }
    match name.as_bytes()[0] {
        b'y' => assert!(ours.is_ok(), "{name}: {ours:?}"),
        b'n' => assert!(ours.is_err(), "{name}: {ours:?}"),
        _ => {}
    }
}

#[test]
fn json_test_suite_agrees_with_serde_json() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/json_suite");
    // the published crate does not carry the files
    if !dir.exists() {
        return;
    }
    let mut seen = 0;
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            let name = path.file_name().unwrap().to_string_lossy();
            check(&name, &fs::read(&path).unwrap());
            seen += 1;
        }
    }
    assert_eq!(seen, 316);
}

#[test]
fn json_test_suite_deep_nesting() {
    check("n_structure_100000_opening_arrays", &b"[".repeat(100_000));
    check("n_structure_open_array_object", &b"[{\"\":".repeat(50_000));
}
