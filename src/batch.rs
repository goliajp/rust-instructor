use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;

use crate::client::Client;
use crate::drive::batch::ordered;
use crate::error::Result;
use crate::exchange::{Extracted, Validator};
use crate::extract::Extract;
use crate::reasoning::Reasoning;
use crate::schema::Schema;
use crate::transport::BoxFuture;
use crate::validate::ValidationError;

/// Many prompts, each extracted into a `T`, with bounded concurrency.
/// Results come back in prompt order.
///
/// ```rust,no_run
/// use instructors::prelude::*;
///
/// #[derive(Deserialize, Schema)]
/// struct Contact { name: String }
///
/// # async fn run() {
/// let client = Client::openai("sk-...");
/// let results = client
///     .extract_batch::<Contact>(vec!["John Doe".into(), "Jane Smith".into()])
///     .concurrency(5)
///     .await;
/// for r in results {
///     match r {
///         Ok(c) => println!("{}", c.value.name),
///         Err(e) => eprintln!("{e}"),
///     }
/// }
/// # }
/// ```
#[must_use = "a batch does nothing until it is awaited"]
pub struct Batch<'a, T> {
    client: &'a Client,
    prompts: Vec<String>,
    concurrency: usize,
    model: Option<String>,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    reasoning: Option<Reasoning>,
    max_attempts: Option<u32>,
    timeout: Option<Duration>,
    validator: Option<Validator<T>>,
}

impl<'a, T> Batch<'a, T> {
    pub(crate) fn new(client: &'a Client, prompts: Vec<String>) -> Self {
        Self {
            client,
            prompts,
            concurrency: 5,
            model: None,
            system: None,
            temperature: None,
            max_tokens: None,
            reasoning: None,
            max_attempts: None,
            timeout: None,
            validator: None,
        }
    }
}

impl<'a, T: DeserializeOwned + Schema + Send + 'a> Batch<'a, T> {
    /// Requests in flight at once. Default 5.
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

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

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn reasoning(mut self, r: Reasoning) -> Self {
        self.reasoning = Some(r);
        self
    }

    pub fn max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = Some(n);
        self
    }

    /// The timeout of each extraction.
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    pub fn validate<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> std::result::Result<(), ValidationError> + Send + Sync + 'static,
    {
        self.validator = Some(Arc::new(f));
        self
    }

    fn one(&self, prompt: String) -> Extract<'a, T> {
        let mut e = self.client.extract::<T>(prompt);
        if let Some(m) = &self.model {
            e = e.model(m.clone());
        }
        if let Some(s) = &self.system {
            e = e.system(s.clone());
        }
        if let Some(t) = self.temperature {
            e = e.temperature(t);
        }
        if let Some(n) = self.max_tokens {
            e = e.max_tokens(n);
        }
        if let Some(r) = self.reasoning {
            e = e.reasoning(r);
        }
        if let Some(n) = self.max_attempts {
            e = e.max_attempts(n);
        }
        if let Some(d) = self.timeout {
            e = e.timeout(d);
        }
        if let Some(v) = &self.validator {
            e = e.validator(v.clone());
        }
        e
    }

    /// Run every prompt; one result per prompt, in order.
    pub async fn run(mut self) -> Vec<Result<Extracted<T>>> {
        let prompts = std::mem::take(&mut self.prompts);
        let futs: Vec<BoxFuture<'a, Result<Extracted<T>>>> = prompts
            .into_iter()
            .map(|p| self.one(p).into_future())
            .collect();
        ordered(futs, self.concurrency).await
    }
}

impl<'a, T: DeserializeOwned + Schema + Send + 'a> IntoFuture for Batch<'a, T> {
    type Output = Vec<Result<Extracted<T>>>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.run())
    }
}
