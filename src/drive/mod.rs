//! Runs an [`Exchange`] over a [`Transport`], with the overall timeout.
//! std only: no async runtime of its own.

pub(crate) mod batch;

use std::future::{Future, poll_fn};
use std::pin::pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use serde::de::DeserializeOwned;

use crate::error::{Error, Result};
use crate::exchange::{Exchange, Extracted, Step};
use crate::transport::{Body, Transport};

/// Run `work` until it finishes or `timer` fires; `None` when the timer won.
/// `work` is polled first, so a result ready at the deadline is kept.
pub(crate) async fn race<T>(
    work: impl Future<Output = T>,
    timer: impl Future<Output = ()>,
) -> Option<T> {
    let mut work = pin!(work);
    let mut timer = pin!(timer);
    poll_fn(|cx| {
        if let Poll::Ready(v) = work.as_mut().poll(cx) {
            return Poll::Ready(Some(v));
        }
        if timer.as_mut().poll(cx).is_ready() {
            return Poll::Ready(None);
        }
        Poll::Pending
    })
    .await
}

/// One transport per provider, in the exchange's provider order.
pub(crate) type Transports = Vec<Option<Arc<dyn Transport>>>;

async fn steps<O: DeserializeOwned>(
    ex: &mut Exchange<O>,
    transports: &Transports,
) -> Result<Extracted<O>> {
    loop {
        let t = transports[ex.provider_index()].as_ref();
        match ex.next() {
            Step::Done(r) => return r,
            Step::Wait(d) => match t {
                Some(t) => t.sleep(d).await,
                None => return Err(Error::NoTransport),
            },
            Step::Send(req) => {
                let Some(t) = t else {
                    return Err(Error::NoTransport);
                };
                let resp = match t.send(req).await {
                    Ok(r) => r,
                    Err(e) => {
                        ex.receive_error(e);
                        continue;
                    }
                };
                match resp.body {
                    Body::Full(b) => ex.receive(resp.status, &resp.headers, &b),
                    Body::Stream(mut s) => {
                        ex.receive_head(resp.status, &resp.headers);
                        let mut buf = Vec::with_capacity(8192);
                        loop {
                            buf.clear();
                            match poll_fn(|cx| s.as_mut().poll_read(cx, &mut buf)).await {
                                Ok(0) => break,
                                Ok(_) => ex.receive_chunk(&buf),
                                Err(e) => {
                                    ex.receive_error(e);
                                    break;
                                }
                            }
                        }
                        ex.receive_end();
                    }
                }
            }
        }
    }
}

/// Drive `ex` to completion, giving up after `timeout`.
pub(crate) async fn run<O: DeserializeOwned>(
    mut ex: Exchange<O>,
    transports: Transports,
    timeout: Duration,
) -> Result<Extracted<O>> {
    let Some(timer_t) = transports[0].clone() else {
        return Err(Error::NoTransport);
    };
    let out = race(steps(&mut ex, &transports), timer_t.sleep(timeout)).await;
    out.unwrap_or_else(|| {
        Err(Error::Timeout {
            after: timeout,
            usage: Box::new(ex.usage()),
        })
    })
}
