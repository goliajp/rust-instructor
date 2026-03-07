/// Validation error with a human-readable message that gets fed back to the LLM on retry.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValidationError {}

impl From<String> for ValidationError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<&str> for ValidationError {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

/// Trait for types that can validate themselves after extraction.
///
/// Implement this trait and use `.validated()` on the builder to enable
/// automatic validation with retry.
///
/// # Example
///
/// ```rust
/// use instructors::{Validate, ValidationError};
/// use serde::Deserialize;
/// use schemars::JsonSchema;
///
/// #[derive(Debug, Deserialize, JsonSchema)]
/// struct User {
///     name: String,
///     age: u32,
///     email: String,
/// }
///
/// impl Validate for User {
///     fn validate(&self) -> Result<(), ValidationError> {
///         if self.name.is_empty() {
///             return Err("name must not be empty".into());
///         }
///         if self.age > 150 {
///             return Err(format!("age {} is unrealistic", self.age).into());
///         }
///         if !self.email.contains('@') {
///             return Err(format!("'{}' is not a valid email", self.email).into());
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait Validate {
    fn validate(&self) -> Result<(), ValidationError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ValidUser;

    impl Validate for ValidUser {
        fn validate(&self) -> Result<(), ValidationError> {
            Ok(())
        }
    }

    struct InvalidUser;

    impl Validate for InvalidUser {
        fn validate(&self) -> Result<(), ValidationError> {
            Err("always fails".into())
        }
    }

    #[test]
    fn valid_passes() {
        assert!(ValidUser.validate().is_ok());
    }

    #[test]
    fn invalid_fails() {
        let err = InvalidUser.validate().unwrap_err();
        assert_eq!(err.message, "always fails");
        assert_eq!(err.to_string(), "always fails");
    }

    #[test]
    fn error_from_string() {
        let err = ValidationError::from("test".to_string());
        assert_eq!(err.message, "test");
    }

    #[test]
    fn error_from_str() {
        let err = ValidationError::from("test");
        assert_eq!(err.message, "test");
    }

    #[test]
    fn error_display() {
        let err = ValidationError::from("display test");
        assert_eq!(format!("{err}"), "display test");
    }

    #[test]
    fn error_debug() {
        let err = ValidationError::from("debug test");
        let debug = format!("{err:?}");
        assert!(debug.contains("debug test"));
    }
}
