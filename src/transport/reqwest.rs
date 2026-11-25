use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use super::{Body, BodyStream, BoxFuture, HttpResponse, Transport};
use crate::error::TransportError;
use crate::wire::{Headers, HttpRequest};

/// [`Transport`] over a [`reqwest::Client`] on tokio (feature `reqwest`).
///
/// The named constructors share one process-wide instance. Build your own
/// from a configured `reqwest::Client` for proxies or custom TLS roots:
///
/// ```rust,no_run
/// use instructors::transport::ReqwestTransport;
///
/// let http = ReqwestTransport::client_builder()
///     .proxy(reqwest::Proxy::all("http://proxy.internal:3128").unwrap())
///     .build()
///     .unwrap();
/// let client = instructors::Client::openai("sk-...")
///     .with_transport(ReqwestTransport::from(http));
/// ```
#[derive(Debug, Clone)]
pub struct ReqwestTransport {
    http: reqwest::Client,
}

impl ReqwestTransport {
    /// A transport with its own connection pool.
    pub fn new() -> Self {
        let http = Self::client_builder()
            .build()
            .expect("a reqwest client with default settings builds");
        Self { http }
    }

    /// A `reqwest::ClientBuilder` with TLS through rustls and the ring
    /// provider. Build reqwest clients through this (or install a rustls
    /// crypto provider yourself first): reqwest is enabled without a bundled
    /// provider and panics when it finds none.
    pub fn client_builder() -> reqwest::ClientBuilder {
        // a provider the application installed first wins
        let _ = rustls::crypto::ring::default_provider().install_default();
        reqwest::Client::builder()
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl From<reqwest::Client> for ReqwestTransport {
    fn from(http: reqwest::Client) -> Self {
        Self { http }
    }
}

/// The process-wide instance used by the named constructors.
pub(crate) fn shared() -> Arc<dyn Transport> {
    static SHARED: OnceLock<Arc<ReqwestTransport>> = OnceLock::new();
    SHARED
        .get_or_init(|| Arc::new(ReqwestTransport::new()))
        .clone()
}

fn error(e: reqwest::Error) -> TransportError {
    let retryable = e.is_connect() || e.is_timeout();
    let t = TransportError::from_source(e);
    if retryable { t.retryable() } else { t }
}

type Chunk = (reqwest::Response, reqwest::Result<Option<bytes::Bytes>>);

struct ChunkStream {
    // `None` while the response is inside the pending chunk future
    resp: Option<reqwest::Response>,
    pending: Option<BoxFuture<'static, Chunk>>,
}

impl BodyStream for ChunkStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut Vec<u8>,
    ) -> Poll<Result<usize, TransportError>> {
        let this = &mut *self;
        if this.pending.is_none() {
            let Some(mut resp) = this.resp.take() else {
                return Poll::Ready(Ok(0));
            };
            this.pending = Some(Box::pin(async move {
                let r = resp.chunk().await;
                (resp, r)
            }));
        }
        let fut = this.pending.as_mut().expect("set above");
        let (resp, r) = std::task::ready!(fut.as_mut().poll(cx));
        this.pending = None;
        this.resp = Some(resp);
        Poll::Ready(match r {
            Ok(Some(b)) => {
                buf.extend_from_slice(&b);
                Ok(b.len())
            }
            Ok(None) => Ok(0),
            Err(e) => Err(error(e)),
        })
    }
}

impl Transport for ReqwestTransport {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        let mut b = self.http.post(req.url);
        for (k, v) in req.headers.iter() {
            b = b.header(k, v);
        }
        let fut = b.body(req.body).send();
        let stream = req.stream;
        Box::pin(async move {
            let resp = fut.await.map_err(error)?;
            let status = resp.status().as_u16();
            let mut headers = Headers::new();
            for (k, v) in resp.headers() {
                if let Ok(v) = v.to_str() {
                    headers.push(k.as_str(), v);
                }
            }
            let body = if stream && (200..300).contains(&status) {
                Body::Stream(Box::pin(ChunkStream {
                    resp: Some(resp),
                    pending: None,
                }))
            } else {
                Body::Full(resp.bytes().await.map_err(error)?.to_vec())
            };
            Ok(HttpResponse {
                status,
                headers,
                body,
            })
        })
    }

    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep(d))
    }
}
