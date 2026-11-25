#![no_main]

use instructors::__private::SseDecoder;
use libfuzzer_sys::fuzz_target;

fn events(chunks: &[&[u8]]) -> Vec<String> {
    let mut d = SseDecoder::default();
    let mut out = Vec::new();
    for c in chunks {
        d.feed(c, |e| out.push(e.to_string()));
    }
    d.finish(|e| out.push(e.to_string()));
    out
}

fuzz_target!(|data: &[u8]| {
    let Some((&seed, body)) = data.split_first() else {
        return;
    };
    let whole = events(&[body]);
    // cut at positions derived from the seed, so multi-byte characters and
    // `\r\n` pairs get split too
    let step = seed as usize % 7 + 1;
    let mut chunks = Vec::new();
    let mut at = 0;
    let mut n = 0;
    while at < body.len() {
        n += 1;
        let len = (step * n) % 11 + 1;
        let end = (at + len).min(body.len());
        chunks.push(&body[at..end]);
        at = end;
    }
    assert_eq!(events(&chunks), whole);
    let bytes: Vec<&[u8]> = body.chunks(1).collect();
    assert_eq!(events(&bytes), whole);
});
