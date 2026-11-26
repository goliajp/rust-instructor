//! Request message content for Chat Completions.

use crate::message::Part;
use crate::wire::{Arr, Obj};

pub(super) fn content(out: &mut Vec<u8>, parts: &[Part]) {
    if let [Part::Text(t)] = parts {
        crate::json::write_str(out, t);
        return;
    }
    let mut a = Arr::new(out);
    for p in parts {
        let mut o = Obj::new(a.item());
        match p {
            Part::Text(t) => {
                o.str("type", "text");
                o.str("text", t);
            }
            Part::ImageUrl(u) => {
                o.str("type", "image_url");
                let mut i = Obj::new(o.key("image_url"));
                i.str("url", u);
                i.end();
            }
            Part::Image { media_type, data } => {
                o.str("type", "image_url");
                let mut i = Obj::new(o.key("image_url"));
                i.str("url", &format!("data:{media_type};base64,{data}"));
                i.end();
            }
            Part::Document { media_type, data } => {
                o.str("type", "file");
                let mut f = Obj::new(o.key("file"));
                f.str("filename", "document.pdf");
                f.str("file_data", &format!("data:{media_type};base64,{data}"));
                f.end();
            }
        }
        o.end();
    }
    a.end();
}
