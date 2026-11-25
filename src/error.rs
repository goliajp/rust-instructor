use std::fmt;
use std::time::Duration;

use crate::usage::Usage;

/// Why an extraction failed.
///
/// Every variant that comes after a request carries the [`Usage`] spent so
/// far, so a failed extraction still reports what it cost
/// ([`Error::usage`]).
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// The request could not be sent or the response could not be read.
    Transport {
        provider: &'static str,
        source: TransportError,
        usage: Usage,
    },
    /// The provider answered with a non-success HTTP status.
    Api {
        provider: &'static str,
        status: u16,
        message: String,
        retry_after: Option<Duration>,
        usage: Usage,
    },
    /// The model declined to answer: an OpenAI `refusal`, an Anthropic
    /// `stop_reason: refusal`, a Gemini safety stop or blocked prompt.
    Refusal {
        provider: &'static str,
        text: String,
        usage: Usage,
    },
    /// The answer hit the output token limit and did not parse. `raw` is
    /// what arrived.
    Truncated {
        provider: &'static str,
        reason: String,
        raw: String,
        usage: Usage,
    },
    /// The answer did not parse into the type after every attempt.
    Parse {
        provider: &'static str,
        attempts: u32,
        message: String,
        raw: String,
        usage: Usage,
    },
    /// The answer parsed but validation rejected it on every attempt.
    Validation {
        provider: &'static str,
        attempts: u32,
        message: String,
        usage: Usage,
    },
    /// A setting the provider has no parameter for, such as
    /// `Reasoning::Budget` on OpenAI. Nothing was sent.
    Unsupported {
        provider: &'static str,
        what: String,
    },
    /// The overall timeout passed.
    Timeout { after: Duration, usage: Usage },
    /// The client has no transport: the crate was built without the
    /// `reqwest` feature and no `with_transport` was given.
    NoTransport,
    /// Every provider failed. `error` is the last fallback's error and
    /// `previous` the one before it, down to the primary's.
    Fallback {
        provider: &'static str,
        error: Box<Error>,
        previous: Box<Error>,
    },
}

/// Shorthand for `Result<T, instructors::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure below HTTP: connecting, sending, reading the body.
#[derive(Debug)]
pub struct TransportError {
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
    retryable: bool,
}

impl TransportError {
    /// An error with a message only.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
            retryable: false,
        }
    }

    /// Wrap another error.
    pub fn from_source(source: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self {
            message: source.to_string(),
            source: Some(Box::new(source)),
            retryable: false,
        }
    }

    /// Mark the error as worth retrying (a refused or reset connection).
    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }

    pub fn is_retryable(&self) -> bool {
        self.retryable
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport {
                provider, source, ..
            } => write!(f, "{provider}: transport error: {source}"),
            Self::Api {
                provider,
                status,
                message,
                ..
            } => write!(f, "{provider}: api error ({status}): {message}"),
            Self::Refusal { provider, text, .. } => write!(f, "{provider}: model refused: {text}"),
            Self::Truncated {
                provider, reason, ..
            } => write!(
                f,
                "{provider}: output truncated ({reason}): the answer hit the output token \
                 limit; raise max_tokens, lower the reasoning effort, or accept_truncated()"
            ),
            Self::Parse {
                provider,
                attempts,
                message,
                ..
            } => write!(
                f,
                "{provider}: answer did not parse after {attempts} attempt(s): {message}"
            ),
            Self::Validation {
                provider,
                attempts,
                message,
                ..
            } => write!(
                f,
                "{provider}: validation failed after {attempts} attempt(s): {message}"
            ),
            Self::Unsupported { provider, what } => write!(f, "{provider}: unsupported: {what}"),
            Self::Timeout { after, .. } => write!(f, "timed out after {after:?}"),
            Self::NoTransport => f.write_str(
                "no transport: enable the `reqwest` feature or call Client::with_transport",
            ),
            Self::Fallback {
                provider, error, ..
            } => write!(f, "all providers failed; last ({provider}): {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport { source, .. } => Some(source),
            Self::Fallback { previous, .. } => Some(previous.as_ref()),
            _ => None,
        }
    }
}

impl Error {
    /// What the failed extraction cost, summed over every request it sent,
    /// fallbacks included.
    pub fn usage(&self) -> Option<&Usage> {
        match self {
            Self::Transport { usage, .. }
            | Self::Api { usage, .. }
            | Self::Refusal { usage, .. }
            | Self::Truncated { usage, .. }
            | Self::Parse { usage, .. }
            | Self::Validation { usage, .. }
            | Self::Timeout { usage, .. } => Some(usage),
            Self::Fallback { error, .. } => error.usage(),
            Self::Unsupported { .. } | Self::NoTransport => None,
        }
    }

    /// The provider that produced the error.
    pub fn provider(&self) -> Option<&'static str> {
        match self {
            Self::Transport { provider, .. }
            | Self::Api { provider, .. }
            | Self::Refusal { provider, .. }
            | Self::Truncated { provider, .. }
            | Self::Parse { provider, .. }
            | Self::Validation { provider, .. }
            | Self::Unsupported { provider, .. }
            | Self::Fallback { provider, .. } => Some(provider),
            Self::Timeout { .. } | Self::NoTransport => None,
        }
    }

    /// Each provider's error, the last provider tried first.
    pub fn chain(&self) -> impl Iterator<Item = &Error> {
        let mut next = Some(self);
        std::iter::from_fn(move || {
            let cur = next?;
            match cur {
                Self::Fallback {
                    error, previous, ..
                } => {
                    next = Some(previous.as_ref());
                    Some(error.as_ref())
                }
                other => {
                    next = None;
                    Some(other)
                }
            }
        })
    }

    /// The refusal text when the model declined to answer.
    pub fn refusal(&self) -> Option<&str> {
        match self.chain().next()? {
            Self::Refusal { text, .. } => Some(text),
            _ => None,
        }
    }

    /// True when the answer was cut off by the output token limit.
    pub fn is_truncated(&self) -> bool {
        matches!(self.chain().next(), Some(Self::Truncated { .. }))
    }

    pub(crate) fn usage_mut(&mut self) -> Option<&mut Usage> {
        match self {
            Self::Transport { usage, .. }
            | Self::Api { usage, .. }
            | Self::Refusal { usage, .. }
            | Self::Truncated { usage, .. }
            | Self::Parse { usage, .. }
            | Self::Validation { usage, .. }
            | Self::Timeout { usage, .. } => Some(usage),
            Self::Fallback { error, .. } => error.usage_mut(),
            Self::Unsupported { .. } | Self::NoTransport => None,
        }
    }
}

#[cfg(test)]
mod tests;
