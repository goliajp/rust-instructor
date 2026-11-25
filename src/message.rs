/// Who wrote a message.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One piece of a message.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    Text(String),
    /// An image by URL. Gemini receives it as `fileData`, so it must be a
    /// URI Gemini can read (a Files API or `gs://` URI).
    ImageUrl(String),
    /// An image as base64 data with its media type (`"image/png"`).
    Image {
        media_type: String,
        data: String,
    },
    /// A document (PDF) as base64 data with its media type.
    Document {
        media_type: String,
        data: String,
    },
}

impl Part {
    /// An image by URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl(url.into())
    }

    /// An image as base64 data.
    pub fn image(media_type: impl Into<String>, base64: impl Into<String>) -> Self {
        Self::Image {
            media_type: media_type.into(),
            data: base64.into(),
        }
    }

    /// A document as base64 data, such as `application/pdf`.
    pub fn document(media_type: impl Into<String>, base64: impl Into<String>) -> Self {
        Self::Document {
            media_type: media_type.into(),
            data: base64.into(),
        }
    }
}

/// One message of conversation history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    role: Role,
    parts: Vec<Part>,
}

impl Message {
    /// A user message with text.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            parts: vec![Part::Text(text.into())],
        }
    }

    /// An assistant message with text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![Part::Text(text.into())],
        }
    }

    /// Add a part (an image, a document, more text).
    pub fn with(mut self, part: Part) -> Self {
        self.parts.push(part);
        self
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn parts(&self) -> &[Part] {
        &self.parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors() {
        let m = Message::user("hi").with(Part::image_url("u"));
        assert_eq!(m.role(), Role::User);
        assert_eq!(m.parts().len(), 2);
        let a = Message::assistant("ok");
        assert_eq!(a.role(), Role::Assistant);
        assert_eq!(
            Part::image("image/png", "AA"),
            Part::Image {
                media_type: "image/png".into(),
                data: "AA".into()
            }
        );
    }
}
