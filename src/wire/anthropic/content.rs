//! Request message content for Anthropic Messages.

use crate::message::Part;
use crate::wire::{Arr, Obj};

pub(super) fn content(out: &mut Vec<u8>, parts: &[Part]) {
    if let [Part::Text(t)] = parts {
        crate::json::write_str(out, t);
        return;
    }
    // images and documents first, then text (anthropic convention)
    let mut a = Arr::new(out);
    let media = parts.iter().filter(|p| !matches!(p, Part::Text(_)));
    let texts = parts.iter().filter(|p| matches!(p, Part::Text(_)));
    for p in media.chain(texts) {
        let mut o = Obj::new(a.item());
        match p {
            Part::Text(t) => {
                o.str("type", "text");
                o.str("text", t);
            }
            Part::ImageUrl(u) => {
                o.str("type", "image");
                let mut s = Obj::new(o.key("source"));
                s.str("type", "url");
                s.str("url", u);
                s.end();
            }
            Part::Image { media_type, data } | Part::Document { media_type, data } => {
                let kind = if matches!(p, Part::Image { .. }) {
                    "image"
                } else {
                    "document"
                };
                o.str("type", kind);
                let mut s = Obj::new(o.key("source"));
                s.str("type", "base64");
                s.str("media_type", media_type);
                s.str("data", data);
                s.end();
            }
        }
        o.end();
    }
    a.end();
}
