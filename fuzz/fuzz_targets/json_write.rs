#![no_main]

use instructors::__private as ours;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let s = String::from_utf8_lossy(data);
    let mut out = Vec::new();
    ours::json_write_str(&mut out, &s);
    assert_eq!(out, serde_json::to_vec(&s).unwrap());
    let back: String = serde_json::from_slice(&out).unwrap();
    assert_eq!(back, s);
    if data.len() >= 8 {
        let n = u64::from_le_bytes(data[..8].try_into().unwrap());
        let mut out = Vec::new();
        ours::json_write_u64(&mut out, n);
        assert_eq!(out, serde_json::to_vec(&n).unwrap());
        let f = f64::from_bits(n);
        let mut out = Vec::new();
        ours::json_write_f64(&mut out, f);
        let back: Option<f64> = serde_json::from_slice(&out).unwrap();
        match back {
            Some(b) => assert_eq!(b.to_bits(), f.to_bits()),
            None => assert!(!f.is_finite()),
        }
    }
});
