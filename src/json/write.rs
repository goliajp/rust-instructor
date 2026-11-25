//! JSON output: escaped strings and numbers appended to a byte buffer.

use super::scan::stop_bytes;

const UU: u8 = b'u';
const BB: u8 = b'b';
const TT: u8 = b't';
const NN: u8 = b'n';
const FF: u8 = b'f';
const RR: u8 = b'r';
const QU: u8 = b'"';
const BS: u8 = b'\\';
const __: u8 = 0;

// the escape letter for each byte, 0 when the byte is written as is
static ESCAPE: [u8; 256] = {
    let mut t = [__; 256];
    let mut i = 0;
    while i < 0x20 {
        t[i] = UU;
        i += 1;
    }
    t[0x08] = BB;
    t[0x09] = TT;
    t[0x0a] = NN;
    t[0x0c] = FF;
    t[0x0d] = RR;
    t[b'"' as usize] = QU;
    t[b'\\' as usize] = BS;
    t
};

static HEX: &[u8; 16] = b"0123456789abcdef";

/// Append `s` quoted, escaping `"`, `\` and control characters the way
/// serde_json does.
#[inline]
pub(crate) fn write_str(out: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    out.reserve(b.len() + 2);
    out.push(b'"');
    let mut run = 0;
    let mut i = 0;
    loop {
        // skip 8 plain bytes at a time
        while i + 8 <= b.len() {
            let w = u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"));
            let m = stop_bytes(w);
            if m != 0 {
                i += (m.trailing_zeros() / 8) as usize;
                break;
            }
            i += 8;
        }
        while i < b.len() && ESCAPE[b[i] as usize] == 0 {
            i += 1;
        }
        if i >= b.len() {
            break;
        }
        let esc = ESCAPE[b[i] as usize];
        out.extend_from_slice(&b[run..i]);
        if esc == UU {
            let c = b[i];
            out.extend_from_slice(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX[(c >> 4) as usize],
                HEX[(c & 0xf) as usize],
            ]);
        } else {
            out.extend_from_slice(&[b'\\', esc]);
        }
        i += 1;
        run = i;
    }
    out.extend_from_slice(&b[run..]);
    out.push(b'"');
}

static DIGITS2: &[u8; 200] = b"\
0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

/// Append an unsigned integer in decimal.
#[inline]
pub(crate) fn write_u64(out: &mut Vec<u8>, mut n: u64) {
    let mut buf = [0u8; 20];
    let mut at = buf.len();
    while n >= 100 {
        let d = (n % 100) as usize * 2;
        n /= 100;
        at -= 2;
        buf[at..at + 2].copy_from_slice(&DIGITS2[d..d + 2]);
    }
    if n >= 10 {
        let d = n as usize * 2;
        at -= 2;
        buf[at..at + 2].copy_from_slice(&DIGITS2[d..d + 2]);
    } else {
        at -= 1;
        buf[at] = b'0' + n as u8;
    }
    out.extend_from_slice(&buf[at..]);
}

/// Append a float in its shortest round-trip form; NaN and infinities
/// are written as `null`, integral values keep a `.0`.
pub(crate) fn write_f64(out: &mut Vec<u8>, n: f64) {
    use std::io::Write;
    if !n.is_finite() {
        out.extend_from_slice(b"null");
        return;
    }
    // `{:?}` is the shortest representation that round-trips, with `.0` on
    // integral values and an exponent for very large or small ones
    write!(out, "{n:?}").expect("writing to a Vec cannot fail");
}
