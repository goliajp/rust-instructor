//! The extraction as a state machine with no I/O: it says what to send and
//! how long to wait, and is told what came back.

mod answer;
mod parse;
mod retry;
mod stream;
#[cfg(test)]
mod tests;

use std::time::Duration;

use serde::de::DeserializeOwned;

pub use retry::RetryPolicy;

use crate::error::{Error, Result, TransportError};
use crate::message::Message;
use crate::partial::PartialValue;
use crate::reasoning::Reasoning;
use crate::schema::SchemaEntry;
use crate::usage::Usage;
use crate::validate::ValidationError;
use crate::wire::{Fail, Headers, HttpRequest, OutputMode, Provider, Req};
use stream::Inbound;

/// A successful extraction.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Extracted<T> {
    pub value: T,
    /// Every request the extraction sent, fallbacks included.
    pub usage: Usage,
    /// The answer was cut off by the token limit and repaired; only with
    /// `accept_truncated`.
    pub truncated: bool,
}

/// What the caller of [`Exchange::next`] does next.
#[derive(Debug)]
pub enum Step<T> {
    /// Send this request, then call [`Exchange::receive`] (or the
    /// `receive_head` / `receive_chunk` / `receive_end` trio for a stream).
    Send(HttpRequest),
    /// Wait this long, then call [`Exchange::next`] again.
    Wait(Duration),
    /// Finished.
    Done(Result<Extracted<T>>),
}

pub(crate) type Validator<O> =
    std::sync::Arc<dyn Fn(&O) -> std::result::Result<(), ValidationError> + Send + Sync>;

type Hook<F> = Option<Box<F>>;

/// Called with each complete list item's JSON, and the guide when it is in
/// a strict schema's wire form.
pub(crate) type ItemHook = dyn Fn(&str, Option<&crate::schema::Plan>) + Send + Sync;

/// Called with the streaming view, and where it sits in the schema plan
/// when the answer is in a strict schema's wire form.
pub(crate) type SnapshotHook =
    dyn Fn(PartialValue<'_>, Option<crate::schema::At<'_>>) + Send + Sync;

#[derive(Default)]
pub(crate) struct Hooks {
    pub(crate) on_request: Hook<dyn Fn(&HttpRequest) + Send + Sync>,
    pub(crate) on_response: Hook<dyn Fn(&Usage) + Send + Sync>,
    pub(crate) on_delta: Hook<dyn Fn(&str) + Send + Sync>,
    pub(crate) on_partial: Hook<dyn Fn(PartialValue<'_>) + Send + Sync>,
    pub(crate) on_snapshot: Hook<SnapshotHook>,
    /// Receives each list item's JSON once it is complete.
    pub(crate) on_item: Hook<ItemHook>,
}

impl Hooks {
    fn streaming(&self) -> bool {
        self.on_delta.is_some()
            || self.on_partial.is_some()
            || self.on_snapshot.is_some()
            || self.on_item.is_some()
    }
}

/// One provider to try, with its model and output mode.
pub(crate) struct Target {
    pub(crate) provider: Provider,
    pub(crate) model: String,
    pub(crate) mode: OutputMode,
}

/// Everything an extraction needs, gathered by the builder.
pub(crate) struct Spec<O> {
    pub(crate) targets: Vec<Target>,
    pub(crate) system: Option<String>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) reasoning: Option<Reasoning>,
    pub(crate) messages: Vec<Message>,
    pub(crate) schema: &'static SchemaEntry,
    pub(crate) list: bool,
    pub(crate) max_attempts: u32,
    pub(crate) retry: RetryPolicy,
    pub(crate) accept_truncated: bool,
    pub(crate) validator: Option<Validator<O>>,
    pub(crate) hooks: Hooks,
}

enum State<O> {
    Ready,
    Awaiting,
    Wait(Duration),
    Done(Option<Result<Extracted<O>>>),
}

/// An extraction driven by the caller: no transport, no runtime.
///
/// ```no_run
/// # use instructors::prelude::*;
/// # #[derive(Deserialize, Schema)] struct Contact { name: String }
/// # fn my_http(r: &HttpRequest) -> (u16, Headers, Vec<u8>) { unimplemented!() }
/// let client = Client::openai("sk-...");
/// let mut ex = client.extract::<Contact>("John Doe").exchange();
/// let result = loop {
///     match ex.next() {
///         Step::Send(req) => {
///             let (status, headers, body) = my_http(&req);
///             ex.receive(status, &headers, &body);
///         }
///         Step::Wait(d) => std::thread::sleep(d),
///         Step::Done(r) => break r,
///     }
/// };
/// ```
pub struct Exchange<O> {
    spec: Spec<O>,
    base_len: usize,
    idx: usize,
    attempt: u32,
    http_retry: u32,
    usage: Usage,
    total: Usage,
    failed: Option<Error>,
    state: State<O>,
    rng: retry::Rng,
    inbound: Option<Inbound>,
}

impl<O: DeserializeOwned> Exchange<O> {
    pub(crate) fn new(spec: Spec<O>) -> Self {
        Self {
            base_len: spec.messages.len(),
            spec,
            idx: 0,
            attempt: 0,
            http_retry: 0,
            usage: Usage::default(),
            total: Usage::default(),
            failed: None,
            state: State::Ready,
            rng: retry::Rng::new(),
            inbound: None,
        }
    }

    /// Which provider (0 = the client's own, then its fallbacks in order) the
    /// pending request is for.
    pub fn provider_index(&self) -> usize {
        self.idx
    }

    #[cfg(feature = "tracing")]
    pub(crate) fn provider_name(&self) -> &'static str {
        self.name()
    }

    #[cfg(feature = "tracing")]
    pub(crate) fn model(&self) -> &str {
        &self.spec.targets[self.idx].model
    }

    /// Usage of every response received so far.
    pub fn usage(&self) -> Usage {
        let mut u = self.total.clone();
        u.merge(&self.usage);
        u
    }

    fn req(&self) -> Req<'_> {
        let t = &self.spec.targets[self.idx];
        Req {
            model: &t.model,
            system: self.spec.system.as_deref(),
            messages: &self.spec.messages,
            schema: self.spec.schema,
            temperature: self.spec.temperature,
            max_tokens: self.spec.max_tokens,
            reasoning: self.spec.reasoning,
            mode: t.mode,
            stream: self.spec.hooks.streaming(),
        }
    }

    fn name(&self) -> &'static str {
        self.spec.targets[self.idx].provider.name()
    }

    /// The next thing to do.
    #[expect(
        clippy::should_implement_trait,
        reason = "a step machine, not an iterator: it never ends with None"
    )]
    pub fn next(&mut self) -> Step<O> {
        match std::mem::replace(&mut self.state, State::Ready) {
            State::Ready => self.send(),
            State::Wait(d) => Step::Wait(d),
            State::Awaiting => {
                self.state = State::Awaiting;
                Step::Done(Err(Error::Transport {
                    provider: self.name(),
                    source: TransportError::new("next() called before the response was received"),
                    usage: Box::new(self.usage()),
                }))
            }
            State::Done(r) => {
                self.state = State::Done(None);
                Step::Done(r.unwrap_or_else(|| {
                    Err(Error::Transport {
                        provider: self.name(),
                        source: TransportError::new("the exchange already finished"),
                        usage: Box::new(self.usage()),
                    })
                }))
            }
        }
    }

    fn send(&mut self) -> Step<O> {
        let target = &self.spec.targets[self.idx];
        match target.provider.request(&self.req()) {
            Ok(req) => {
                if let Some(h) = &self.spec.hooks.on_request {
                    h(&req);
                }
                self.state = State::Awaiting;
                Step::Send(req)
            }
            Err(f) => {
                self.fail_with(f, 0);
                self.next()
            }
        }
    }

    /// A complete response.
    pub fn receive(&mut self, status: u16, headers: &Headers, body: &[u8]) {
        if (200..300).contains(&status) && !self.spec.hooks.streaming() {
            self.state = State::Ready;
            let answer = self.spec.targets[self.idx]
                .provider
                .response(&self.req(), body);
            self.answer(answer, status);
            return;
        }
        self.receive_head(status, headers);
        self.receive_chunk(body);
        self.receive_end();
    }

    /// The request failed below HTTP.
    pub fn receive_error(&mut self, error: TransportError) {
        self.inbound = None;
        self.state = State::Ready;
        if error.is_retryable() && self.http_retry < self.spec.retry.max_retries {
            let d = self.spec.retry.delay(self.http_retry, None, &mut self.rng);
            self.http_retry += 1;
            self.state = State::Wait(d);
            return;
        }
        self.fail(Error::Transport {
            provider: self.name(),
            source: error,
            usage: Box::default(),
        });
    }

    fn http_error(&mut self, status: u16, headers: &Headers, body: &[u8]) {
        let server = retry::retry_after(headers);
        if retry::retryable(status, headers) && self.http_retry < self.spec.retry.max_retries {
            let d = self
                .spec
                .retry
                .delay(self.http_retry, server, &mut self.rng);
            self.http_retry += 1;
            self.state = State::Wait(d);
            return;
        }
        self.fail(Error::Api {
            provider: self.name(),
            status,
            message: String::from_utf8_lossy(body).into_owned(),
            retry_after: server,
            usage: Box::default(),
        });
    }

    fn succeed(&mut self, value: O, truncated: bool) {
        #[cfg(feature = "tracing")]
        tracing::info!(requests = self.usage.requests, "extraction succeeded");
        self.total.merge(&self.usage);
        self.usage = Usage::default();
        if let Some(h) = &self.spec.hooks.on_response {
            h(&self.total);
        }
        self.state = State::Done(Some(Ok(Extracted {
            value,
            usage: std::mem::take(&mut self.total),
            truncated,
        })));
    }

    fn fail_with(&mut self, f: Fail, status: u16) {
        let provider = self.name();
        let err = match f {
            Fail::Refusal(text, u) => {
                let model = self.spec.targets[self.idx].model.clone();
                self.usage.add(&u, &model);
                Error::Refusal {
                    provider,
                    text,
                    usage: Box::default(),
                }
            }
            Fail::Malformed(message) | Fail::Api(message) => Error::Api {
                provider,
                status,
                message,
                retry_after: None,
                usage: Box::default(),
            },
            Fail::Unsupported(what) => Error::Unsupported { provider, what },
        };
        self.fail(err);
    }

    /// The current provider is done for; try the next one or finish.
    fn fail(&mut self, mut err: Error) {
        #[cfg(feature = "tracing")]
        tracing::info!(error = %err, "provider failed");
        self.total.merge(&self.usage);
        self.usage = Usage::default();
        if let Some(u) = err.usage_mut() {
            u.clone_from(&self.total);
        }
        let err = match self.failed.take() {
            None => err,
            Some(prev) => Error::Fallback {
                provider: self.name(),
                error: Box::new(err),
                previous: Box::new(prev),
            },
        };
        if self.idx + 1 < self.spec.targets.len() {
            self.failed = Some(err);
            self.idx += 1;
            self.attempt = 0;
            self.http_retry = 0;
            self.spec.messages.truncate(self.base_len);
            self.state = State::Ready;
        } else {
            self.state = State::Done(Some(Err(err)));
        }
    }
}
