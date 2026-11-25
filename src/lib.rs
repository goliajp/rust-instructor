//! # instructors
//!
//! Typed structured output from LLMs.
//!
//! Derive `Deserialize` and [`Schema`] on a Rust type and call
//! [`Client::extract`]. The crate sends the type's JSON schema in the form
//! each provider enforces, parses the answer into the type, and when parsing
//! or your validation fails, shows the model its answer and the error and
//! asks again.
//!
//! ```rust,no_run
//! use instructors::prelude::*;
//!
//! #[derive(Debug, Deserialize, Schema)]
//! struct Contact {
//!     /// Full name as written.
//!     name: String,
//!     email: Option<String>,
//! }
//!
//! # async fn run() -> instructors::Result<()> {
//! let client = Client::openai("sk-...");
//! let out = client
//!     .extract::<Contact>("Contact John Doe at john@example.com")
//!     .await?;
//! println!("{}: {:?}", out.value.name, out.value.email);
//! println!("{} tokens, {:?} USD", out.usage.total_tokens(), out.usage.cost);
//! # Ok(())
//! # }
//! ```
//!
//! ## Layers
//!
//! - The core builds request bytes and parses response bytes; it does no
//!   I/O and needs no async runtime. [`Extract::exchange`] hands you that
//!   state machine ([`Exchange`]) to drive with any HTTP client.
//! - `.await` on an extraction drives the exchange over a
//!   [`Transport`](transport::Transport):
//!   reqwest on tokio with the default `reqwest` feature, or your own
//!   through [`Client::with_transport`].
//!
//! ## Features
//!
//! - `derive` (default): `#[derive(Schema)]`.
//! - `reqwest` (default): the reqwest transport, TLS through rustls with the
//!   ring provider. Without it the crate makes no network calls by itself.
//! - `cost-tracking` (default): cost estimates from `tiktoken`'s price table.
//! - `tracing`: spans around each extraction.
//! - `schemars`: [`Client::extract_schemars`] for types that derive
//!   `schemars::JsonSchema` instead of [`Schema`].

#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate self as instructors;

mod batch;
mod client;
mod drive;
mod error;
mod exchange;
mod extract;
mod json;
mod message;
pub mod partial;
mod reasoning;
pub mod schema;
mod sse;
mod text;
pub mod transport;
mod usage;
mod validate;
mod wire;

pub use batch::Batch;
pub use client::Client;
pub use error::{Error, Result, TransportError};
pub use exchange::{Exchange, Extracted, RetryPolicy, Step};
pub use extract::Extract;
#[cfg(feature = "derive")]
pub use instructors_derive::Schema;
// unit tests derive schemas in every feature combination
#[cfg(all(test, not(feature = "derive")))]
use instructors_derive::Schema;
pub use message::{Message, Part, Role};
pub use partial::PartialValue;
pub use reasoning::{Effort, Reasoning};
#[doc(no_inline)]
pub use schema::Schema;
pub use serde::{self, Deserialize};
pub use usage::Usage;
pub use validate::{Validate, ValidationError};
pub use wire::{Headers, HttpRequest, OpenAiApi, OutputMode};

/// The 1.x name of [`Extracted`].
#[deprecated(since = "2.0.0", note = "renamed to `Extracted`")]
pub type ExtractResult<T> = Extracted<T>;

/// Common imports.
///
/// ```rust
/// use instructors::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        Client, Deserialize, Effort, Error, Exchange, Extracted, Headers, HttpRequest, Message,
        OutputMode, Part, PartialValue, Reasoning, RetryPolicy, Schema, Step, Usage, Validate,
        ValidationError,
    };
}
