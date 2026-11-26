//! Request message content for the Responses API.

use crate::message::{Part, Role};
use crate::wire::{Arr, Obj};

pub(super) fn content(out: &mut Vec<u8>, role: Role, parts: &[Part]) {
    // an earlier answer goes as string content: the input-message form
    // takes only `input_*` parts, and an output message needs an id
    if role == Role::Assistant {
        let text: String = parts
            .iter()
            .filter_map(|p| match p {
                Part::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        crate::json::write_str(out, &text);
        return;
    }
    let mut a = Arr::new(out);
    for p in parts {
        let mut o = Obj::new(a.item());
        match p {
            Part::Text(t) => {
                o.str("type", "input_text");
                o.str("text", t);
            }
            Part::ImageUrl(u) => image(&mut o, u),
            Part::Image { media_type, data } => {
                image(&mut o, &format!("data:{media_type};base64,{data}"))
            }
            Part::Document { media_type, data } => {
                o.str("type", "input_file");
                o.str("filename", "document.pdf");
                o.str("file_data", &format!("data:{media_type};base64,{data}"));
            }
        }
        o.end();
    }
    a.end();
}

fn image(o: &mut Obj<'_>, url: &str) {
    o.str("type", "input_image");
    o.str("image_url", url);
    o.str("detail", "auto");
}
