use std::time::Duration;

use super::*;

fn usage(requests: u32) -> Usage {
    Usage {
        requests,
        ..Usage::default()
    }
}

#[test]
fn display_names_the_provider() {
    let e = Error::Api {
        provider: "openai",
        status: 429,
        message: "rate limited".into(),
        retry_after: None,
        usage: usage(0),
    };
    assert_eq!(e.to_string(), "openai: api error (429): rate limited");
    let e = Error::Timeout {
        after: Duration::from_secs(60),
        usage: usage(2),
    };
    assert!(e.to_string().contains("60s"));
    assert_eq!(e.usage().unwrap().requests, 2);
    assert_eq!(e.provider(), None);
    assert!(Error::NoTransport.to_string().contains("with_transport"));
}

#[test]
fn chain_and_source() {
    let first = Error::Parse {
        provider: "openai",
        attempts: 3,
        message: "m".into(),
        raw: "r".into(),
        usage: usage(3),
    };
    let second = Error::Refusal {
        provider: "anthropic",
        text: "no".into(),
        usage: usage(4),
    };
    let e = Error::Fallback {
        provider: "anthropic",
        error: Box::new(second),
        previous: Box::new(first),
    };
    let names: Vec<_> = e.chain().map(|x| x.provider().unwrap()).collect();
    assert_eq!(names, ["anthropic", "openai"]);
    assert_eq!(e.refusal(), Some("no"));
    assert!(!e.is_truncated());
    assert_eq!(e.usage().unwrap().requests, 4);
    let src = std::error::Error::source(&e).unwrap();
    assert!(src.to_string().starts_with("openai:"));
}

#[test]
fn transport_error_source() {
    let io = std::io::Error::other("reset");
    let t = TransportError::from_source(io).retryable();
    assert!(t.is_retryable());
    assert_eq!(t.to_string(), "reset");
    let e = Error::Transport {
        provider: "gemini",
        source: t,
        usage: usage(0),
    };
    assert_eq!(std::error::Error::source(&e).unwrap().to_string(), "reset");
}

#[test]
fn error_is_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<Error>();
}
