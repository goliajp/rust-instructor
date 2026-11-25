//! Random documents written with random whitespace and escapes, parsed by
//! this module and by serde_json, must give the same result.

use super::tests::{agree, same};
use super::*;

pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub(super) fn float(&mut self) -> f64 {
        loop {
            let f = f64::from_bits(self.next());
            if f.is_finite() {
                return f;
            }
        }
    }

    pub(super) fn string(&mut self) -> String {
        const POOL: &[&str] = &[
            "a", "Z", " ", "\"", "\\", "/", "\n", "\t", "\u{1}", "\u{1f}", "é", "東", "😀",
            "\u{7f}", "\u{2028}", "}", "{", ",", ":", "]", "0",
        ];
        let n = self.below(12);
        (0..n)
            .map(|_| POOL[self.below(POOL.len() as u64) as usize])
            .collect()
    }
}

fn ws(r: &mut Rng, out: &mut String) {
    const WS: [&str; 5] = ["", "", " ", "\n  ", "\t\r\n"];
    out.push_str(WS[r.below(5) as usize]);
}

fn write_string(r: &mut Rng, s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        let pick = r.below(4);
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            '/' if pick == 0 => out.push_str("\\/"),
            c if pick == 1 => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04X}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_number(r: &mut Rng, out: &mut String) {
    match r.below(6) {
        0 => out.push_str(&r.next().to_string()),
        1 => out.push_str(&(r.next() as i64).to_string()),
        2 => out.push_str(&(r.below(1000) as i64 - 500).to_string()),
        3 => out.push_str(&format!("{:e}", r.float())),
        4 => out.push_str(&format!("{:?}", r.float())),
        _ => out.push_str(&format!(
            "{}.{}e{}{}",
            r.below(100),
            r.below(1000),
            ["", "+", "-"][r.below(3) as usize],
            r.below(300)
        )),
    }
}

fn write_value(r: &mut Rng, depth: u32, out: &mut String) {
    let pick = if depth > 4 { r.below(4) } else { r.below(6) };
    match pick {
        0 => out.push_str(["null", "true", "false"][r.below(3) as usize]),
        1 => write_number(r, out),
        2 | 3 => {
            let s = r.string();
            write_string(r, &s, out);
        }
        4 => {
            out.push('[');
            let n = r.below(5);
            for i in 0..n {
                if i > 0 {
                    out.push(',');
                }
                ws(r, out);
                write_value(r, depth + 1, out);
                ws(r, out);
            }
            out.push(']');
        }
        _ => {
            out.push('{');
            let n = r.below(5);
            for i in 0..n {
                if i > 0 {
                    out.push(',');
                }
                ws(r, out);
                let k = r.string();
                write_string(r, &k, out);
                ws(r, out);
                out.push(':');
                ws(r, out);
                write_value(r, depth + 1, out);
                ws(r, out);
            }
            out.push('}');
        }
    }
}

pub(super) fn random_doc(seed: u64) -> String {
    let mut r = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = String::new();
    ws(&mut r, &mut out);
    write_value(&mut r, 0, &mut out);
    ws(&mut r, &mut out);
    out
}

#[test]
fn random_documents_agree() {
    for seed in 0..5000 {
        let doc = random_doc(seed);
        let ours: serde_json::Value = from_str(&doc).unwrap_or_else(|e| panic!("{e}: {doc}"));
        let theirs: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert!(same(&ours, &theirs), "{doc}");
    }
}

#[test]
fn mutated_documents_agree() {
    let mut r = Rng(7);
    for seed in 0..5000 {
        let mut b = random_doc(seed).into_bytes();
        if b.is_empty() {
            continue;
        }
        for _ in 0..1 + r.below(2) {
            let at = r.below(b.len() as u64) as usize;
            let byte = b"{}[]\",:\\u0e.-+ tfn\x01"[r.below(19) as usize];
            match r.below(3) {
                0 => {
                    b.remove(at);
                }
                1 => b.insert(at, byte),
                _ => b[at] = byte,
            }
            if b.is_empty() {
                break;
            }
        }
        if let Ok(s) = std::str::from_utf8(&b) {
            agree(s);
        }
    }
}

#[test]
fn written_strings_parse_back() {
    let mut r = Rng(11);
    for _ in 0..2000 {
        let s = r.string();
        let mut out = Vec::new();
        write_str(&mut out, &s);
        assert_eq!(out, serde_json::to_vec(&s).unwrap());
        let back: String = from_slice(&out).unwrap();
        assert_eq!(back, s);
    }
}
