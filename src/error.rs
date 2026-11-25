use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("api error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("extraction failed after {retries} retries: {message}")]
    ExtractionFailed { retries: u32, message: String },

    #[error("validation failed after {retries} retries: {message}")]
    ValidationFailed { retries: u32, message: String },

    #[error("request timed out after {0:?}")]
    Timeout(Duration),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

const REFUSED: &str = "model refused (";
const TRUNCATED: &str = "output truncated (";

impl Error {
    pub(crate) fn refused(provider: &str, text: &str) -> Self {
        Self::Other(format!("{REFUSED}{provider}): {text}"))
    }

    pub(crate) fn truncated(provider: &str, reason: &str) -> Self {
        Self::Other(format!(
            "{TRUNCATED}{provider}, {reason}): the answer hit the output token limit; \
             raise max_tokens or lower the reasoning effort"
        ))
    }

    /// Append the fallbacks' errors to the primary provider's error, keeping
    /// the primary's variant where it has a message to extend.
    pub(crate) fn with_fallback_failures(self, failures: &str) -> Self {
        let note = format!(" [fallbacks failed: {failures}]");
        match self {
            Self::Api { status, message } => Self::Api {
                status,
                message: message + &note,
            },
            Self::ExtractionFailed { retries, message } => Self::ExtractionFailed {
                retries,
                message: message + &note,
            },
            Self::ValidationFailed { retries, message } => Self::ValidationFailed {
                retries,
                message: message + &note,
            },
            Self::Other(message) => Self::Other(message + &note),
            other => Self::Other(format!("{other}{note}")),
        }
    }

    /// The refusal text when the model declined to answer: an OpenAI
    /// `refusal`, an Anthropic `stop_reason: refusal`, or a Gemini safety
    /// block. Refusals are not retried.
    ///
    /// The error itself is an [`Error::Other`] so that matching on `Error`
    /// keeps compiling; use this method to tell refusals apart.
    pub fn refusal(&self) -> Option<&str> {
        match self {
            Self::Other(m) => m
                .strip_prefix(REFUSED)
                .and_then(|rest| rest.split_once("): "))
                .map(|(_, text)| text),
            _ => None,
        }
    }

    /// True when the answer was cut off by the output token limit (OpenAI
    /// `finish_reason: length`, Anthropic `stop_reason: max_tokens`, Gemini
    /// `finishReason: MAX_TOKENS`) and did not parse. Not retried, since the
    /// same limit would cut the next answer too.
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Other(m) if m.starts_with(TRUNCATED))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_api_error() {
        let err = Error::Api {
            status: 429,
            message: "rate limited".into(),
        };
        let s = err.to_string();
        assert!(s.contains("429"));
        assert!(s.contains("rate limited"));
    }

    #[test]
    fn display_extraction_failed() {
        let err = Error::ExtractionFailed {
            retries: 3,
            message: "invalid json".into(),
        };
        let s = err.to_string();
        assert!(s.contains("3 retries"));
        assert!(s.contains("invalid json"));
    }

    #[test]
    fn display_validation_failed() {
        let err = Error::ValidationFailed {
            retries: 2,
            message: "age too high".into(),
        };
        let s = err.to_string();
        assert!(s.contains("2 retries"));
        assert!(s.contains("age too high"));
    }

    #[test]
    fn display_other() {
        let err = Error::Other("something went wrong".into());
        assert_eq!(err.to_string(), "something went wrong");
    }

    #[test]
    fn refusal_and_truncation_are_recognizable() {
        let r = Error::refused("openai", "I can't help with that.");
        assert_eq!(r.refusal(), Some("I can't help with that."));
        assert!(!r.is_truncated());
        assert!(r.to_string().contains("openai"));

        let t = Error::truncated("gemini", "finishReason: MAX_TOKENS");
        assert!(t.is_truncated());
        assert_eq!(t.refusal(), None);
        assert!(t.to_string().contains("MAX_TOKENS"));

        assert_eq!(Error::Other("x".into()).refusal(), None);

        let r = Error::refused("openai", "no").with_fallback_failures("gemini (m): boom");
        assert_eq!(r.refusal(), Some("no [fallbacks failed: gemini (m): boom]"));
        assert!(!Error::Timeout(Duration::from_secs(1)).is_truncated());
    }

    #[test]
    fn json_error_conversion() {
        let json_err = serde_json::from_str::<String>("invalid").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn display_timeout() {
        let err = Error::Timeout(Duration::from_secs(60));
        let s = err.to_string();
        assert!(s.contains("timed out"));
        assert!(s.contains("60"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn debug_format() {
        let err = Error::Api {
            status: 500,
            message: "internal".into(),
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("Api"));
        assert!(debug.contains("500"));
    }
}
