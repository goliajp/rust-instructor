use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;

use crate::client::Client;
use crate::drive::{self, Transports};
use crate::error::Result;
use crate::exchange::{Exchange, Extracted, Hooks, RetryPolicy, Spec, Target, Validator};
use crate::message::{Message, Part};
use crate::partial::PartialValue;
use crate::reasoning::Reasoning;
use crate::schema::SchemaEntry;
use crate::transport::BoxFuture;
use crate::usage::Usage;
use crate::validate::{Validate, ValidationError};
use crate::wire::{HttpRequest, OutputMode};

/// One extraction, configured per request. `.await` it, or take the
/// sans-IO [`Exchange`] with [`exchange`](Self::exchange).
#[must_use = "an extraction does nothing until it is awaited"]
pub struct Extract<'a, T> {
    client: &'a Client,
    prompt: String,
    parts: Vec<Part>,
    context: Option<String>,
    history: Vec<Message>,
    model: Option<String>,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    reasoning: Option<Reasoning>,
    mode: Option<OutputMode>,
    retry: RetryPolicy,
    max_attempts: u32,
    timeout: Duration,
    accept_truncated: bool,
    validator: Option<Validator<T>>,
    hooks: Hooks,
    schema: &'static SchemaEntry,
    list: bool,
}

impl<'a, T: DeserializeOwned> Extract<'a, T> {
    pub(crate) fn new(
        client: &'a Client,
        prompt: String,
        schema: &'static SchemaEntry,
        list: bool,
    ) -> Self {
        let c = &client.inner;
        Self {
            client,
            prompt,
            parts: Vec::new(),
            context: None,
            history: Vec::new(),
            model: c.model.clone(),
            system: c.system.clone(),
            temperature: c.temperature,
            max_tokens: c.max_tokens,
            reasoning: c.reasoning,
            mode: None,
            retry: c.retry.clone(),
            max_attempts: c.max_attempts,
            timeout: c.timeout,
            accept_truncated: false,
            validator: None,
            hooks: Hooks::default(),
            schema,
            list,
        }
    }

    /// The model for this request. Fallback clients keep their own.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// See [`Client::with_max_tokens`].
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// See [`Reasoning`].
    pub fn reasoning(mut self, r: Reasoning) -> Self {
        self.reasoning = Some(r);
        self
    }

    /// The output mode for this request's first provider.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn retry(mut self, p: RetryPolicy) -> Self {
        self.retry = p;
        self
    }

    /// See [`Client::with_max_attempts`].
    pub fn max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n.max(1);
        self
    }

    /// See [`Client::with_timeout`].
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    /// Text appended to the prompt under "Additional context:".
    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    /// Attach an image or document to the prompt.
    pub fn part(mut self, p: Part) -> Self {
        self.parts.push(p);
        self
    }

    /// Conversation history sent before the prompt.
    pub fn messages(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Check each parsed answer; an error is sent back to the model, which
    /// is asked again.
    pub fn validate<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> std::result::Result<(), ValidationError> + Send + Sync + 'static,
    {
        self.validator = Some(Arc::new(f));
        self
    }

    pub(crate) fn validator(mut self, v: Validator<T>) -> Self {
        self.validator = Some(v);
        self
    }

    /// Validate with `T`'s [`Validate`] impl.
    pub fn validated(mut self) -> Self
    where
        T: Validate,
    {
        self.validator = Some(Arc::new(|v: &T| v.validate()));
        self
    }

    /// When the answer is cut off by the token limit, close it, parse it and
    /// return it with `truncated: true` instead of failing. It parses only
    /// when every missing field is optional; `extract_many` returns the
    /// items that were complete. Nothing is made up: an unfinished string or
    /// number is dropped.
    pub fn accept_truncated(mut self) -> Self {
        self.accept_truncated = true;
        self
    }

    /// Called with each request before it is sent.
    pub fn on_request(mut self, f: impl Fn(&HttpRequest) + Send + Sync + 'static) -> Self {
        self.hooks.on_request = Some(Box::new(f));
        self
    }

    /// Called with the total usage after a successful extraction.
    pub fn on_response(mut self, f: impl Fn(&Usage) + Send + Sync + 'static) -> Self {
        self.hooks.on_response = Some(Box::new(f));
        self
    }

    /// Stream the response and call `f` with each text delta.
    pub fn on_delta(mut self, f: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.hooks.on_delta = Some(Box::new(f));
        self
    }

    /// Stream the response and call `f` after each delta with the JSON
    /// parsed so far. For `extract_many` the view is the list. A re-ask
    /// starts a new value.
    pub fn on_partial(mut self, f: impl Fn(PartialValue<'_>) + Send + Sync + 'static) -> Self {
        self.hooks.on_partial = Some(Box::new(f));
        self
    }

    /// The sans-IO state machine for this extraction.
    pub fn exchange(self) -> Exchange<T> {
        self.split().0
    }

    fn split(self) -> (Exchange<T>, Transports, Duration) {
        let c = &self.client.inner;
        let mut targets = vec![Target {
            provider: c.provider.clone(),
            model: self
                .model
                .unwrap_or_else(|| c.provider.default_model().to_owned()),
            mode: self.mode.unwrap_or(c.mode),
        }];
        let mut transports = vec![c.transport.clone()];
        for f in &c.fallbacks {
            let f = &f.inner;
            targets.push(Target {
                provider: f.provider.clone(),
                model: f
                    .model
                    .clone()
                    .unwrap_or_else(|| f.provider.default_model().to_owned()),
                mode: f.mode,
            });
            transports.push(f.transport.clone());
        }
        let mut text = self.prompt;
        if let Some(ctx) = self.context {
            text = format!("{text}\n\nAdditional context:\n{ctx}");
        }
        let mut user = Message::user(text);
        for p in self.parts {
            user = user.with(p);
        }
        let mut messages = self.history;
        messages.push(user);
        let spec = Spec {
            targets,
            system: self.system,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            reasoning: self.reasoning,
            messages,
            schema: self.schema,
            list: self.list,
            max_attempts: self.max_attempts,
            retry: self.retry,
            accept_truncated: self.accept_truncated,
            validator: self.validator,
            hooks: self.hooks,
        };
        (Exchange::new(spec), transports, self.timeout)
    }
}

impl<T: DeserializeOwned> Extract<'_, Vec<T>> {
    /// Stream the response and call `f` with each list item as soon as it
    /// is complete. Items that do not parse into `T` are skipped here; the
    /// final result still reports them.
    pub fn on_item(mut self, f: impl Fn(T) + Send + Sync + 'static) -> Self {
        self.hooks.on_item = Some(Box::new(move |json: &str, guide| {
            let v = match guide {
                Some(g) => g.item::<T>(json),
                None => crate::json::from_str::<T>(json),
            };
            if let Ok(v) = v {
                f(v);
            }
        }));
        self
    }
}

impl<'a, T> IntoFuture for Extract<'a, T>
where
    T: DeserializeOwned + Send + 'a,
{
    type Output = Result<Extracted<T>>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let (ex, transports, timeout) = self.split();
        #[cfg(feature = "tracing")]
        {
            use tracing::Instrument;
            let span = tracing::info_span!(
                "instructors.extract",
                provider = ex.provider_name(),
                model = ex.model(),
            );
            Box::pin(drive::run(ex, transports, timeout).instrument(span))
        }
        #[cfg(not(feature = "tracing"))]
        Box::pin(drive::run(ex, transports, timeout))
    }
}
