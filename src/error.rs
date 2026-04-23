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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_http_error() {
        // reqwest errors can't be easily constructed, skip direct test
    }

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
