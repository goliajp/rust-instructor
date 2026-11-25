//! The HTTP layer, as a trait the async driver calls.

#[cfg(feature = "reqwest")]
mod reqwest;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

#[cfg(feature = "reqwest")]
pub use self::reqwest::ReqwestTransport;
#[cfg(feature = "reqwest")]
pub(crate) use self::reqwest::shared;
use crate::error::TransportError;
use crate::wire::{Headers, HttpRequest};

/// A boxed future; `Send` except on `wasm32`, where a browser transport
/// cannot be.
#[cfg(not(target_arch = "wasm32"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
/// A boxed future; `Send` except on `wasm32`, where a browser transport
/// cannot be.
#[cfg(target_arch = "wasm32")]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Sends requests and sleeps. The default, with the `reqwest` feature, is a
/// shared `reqwest::Client`; implement this to use another HTTP client or
/// runtime.
///
/// `send` resolves once the status and headers have arrived; the body may
/// still be streaming.
#[cfg(not(target_arch = "wasm32"))]
pub trait Transport: Send + Sync {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>>;
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()>;
}

/// Sends requests and sleeps.
#[cfg(target_arch = "wasm32")]
pub trait Transport {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>>;
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()>;
}

/// A response: status, headers and the body, whole or streaming.
pub struct HttpResponse {
    pub status: u16,
    pub headers: Headers,
    pub body: Body,
}

/// A response body.
pub enum Body {
    Full(Vec<u8>),
    Stream(Pin<Box<dyn BodyStream>>),
}

/// A body that arrives in pieces.
#[cfg(not(target_arch = "wasm32"))]
pub trait BodyStream: Send {
    /// Append the next chunk to `buf` and return its length; `Ok(0)` at the
    /// end.
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut Vec<u8>,
    ) -> Poll<Result<usize, TransportError>>;
}

/// A body that arrives in pieces.
#[cfg(target_arch = "wasm32")]
pub trait BodyStream {
    /// Append the next chunk to `buf` and return its length; `Ok(0)` at the
    /// end.
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut Vec<u8>,
    ) -> Poll<Result<usize, TransportError>>;
}
