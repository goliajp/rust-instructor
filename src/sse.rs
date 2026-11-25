//! Byte-level server-sent events decoder.
//!
//! Bytes are buffered until a full line is available, so a multi-byte UTF-8
//! character split across two network reads is decoded intact. Each `data:`
//! line is delivered as its own payload: the providers this crate talks to
//! put one JSON document per line, and some compatible servers omit the blank
//! line that would end an event.

#[derive(Default)]
pub(crate) struct SseDecoder {
    buf: Vec<u8>,
    // bytes of `buf` already known to hold no newline
    scanned: usize,
}

impl SseDecoder {
    /// Feed raw bytes; `on_data` receives each complete `data:` payload.
    pub(crate) fn feed(&mut self, bytes: &[u8], mut on_data: impl FnMut(&str)) {
        self.buf.extend_from_slice(bytes);
        let mut start = 0;
        let mut from = self.scanned;
        while let Some(nl) = self.buf[from..].iter().position(|&b| b == b'\n') {
            let end = from + nl;
            dispatch(&self.buf[start..end], &mut on_data);
            start = end + 1;
            from = start;
        }
        self.buf.drain(..start);
        self.scanned = self.buf.len();
    }

    /// Deliver a final line that had no trailing newline.
    pub(crate) fn finish(&mut self, mut on_data: impl FnMut(&str)) {
        let rest = std::mem::take(&mut self.buf);
        self.scanned = 0;
        if !rest.is_empty() {
            dispatch(&rest, &mut on_data);
        }
    }
}

fn dispatch(line: &[u8], on_data: &mut impl FnMut(&str)) {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let Some(rest) = line.strip_prefix(b"data:") else {
        return;
    };
    let rest = rest.strip_prefix(b" ").unwrap_or(rest);
    match std::str::from_utf8(rest) {
        Ok(s) => on_data(s),
        Err(_) => on_data(&String::from_utf8_lossy(rest)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(chunks: &[&[u8]]) -> Vec<String> {
        let mut d = SseDecoder::default();
        let mut out = Vec::new();
        for c in chunks {
            d.feed(c, |s| out.push(s.to_string()));
        }
        d.finish(|s| out.push(s.to_string()));
        out
    }

    #[test]
    fn utf8_split_across_reads() {
        let all = "data: {\"t\":\"東京\"}\n\n".as_bytes();
        // split inside the first character's three bytes
        let at = all.iter().position(|&b| b == 0xe6).unwrap() + 1;
        let out = collect(&[&all[..at], &all[at..]]);
        assert_eq!(out, vec!["{\"t\":\"東京\"}"]);
    }

    #[test]
    fn every_split_point_gives_the_same_events() {
        let all = "event: x\r\ndata: 一\r\n\r\n: comment\ndata:二\n\ndata: [DONE]\n".as_bytes();
        let want = vec!["一".to_string(), "二".to_string(), "[DONE]".to_string()];
        for i in 0..=all.len() {
            assert_eq!(collect(&[&all[..i], &all[i..]]), want, "split at {i}");
        }
    }

    #[test]
    fn final_line_without_newline() {
        assert_eq!(collect(&[b"data: a\ndata: b"]), vec!["a", "b"]);
    }

    #[test]
    fn ignores_other_fields() {
        assert!(collect(&[b"id: 1\nevent: ping\nretry: 10\n\n"]).is_empty());
    }
}
