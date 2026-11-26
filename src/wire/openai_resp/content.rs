//! Request message content for the Responses API.

use crate::message::{Part, Role};
use crate::wire::{Arr, Obj};

pub(super) fn content(out: &mut Vec<u8>, role: Role, parts: &[Part]) {
    let mut a = Arr::new(out);
    for p in parts {
        let mut o = Obj::new(a.item());
        match (role, p) {
            (Role::Assistant, Part::Text(t)) => {
                o.str("type", "output_text");
                o.str("text", t);
            }
            (_, Part::Text(t)) => {
                o.str("type", "input_text");
                o.str("text", t);
            }
            (_, Part::ImageUrl(u)) => {
                o.str("type", "input_image");
                o.str("image_url", u);
            }
            (_, Part::Image { media_type, data }) => {
                o.str("type", "input_image");
                o.str("image_url", &format!("data:{media_type};base64,{data}"));
            }
            (_, Part::Document { media_type, data }) => {
                o.str("type", "input_file");
                o.str("filename", "document.pdf");
                o.str("file_data", &format!("data:{media_type};base64,{data}"));
            }
        }
        o.end();
    }
    a.end();
}
