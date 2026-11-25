//! `#[derive(Schema)]` for the [instructors](https://docs.rs/instructors)
//! crate. Use it through `instructors`, which re-exports it.

mod attr;
mod case;
mod expand;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derive `instructors::schema::Schema`.
///
/// Reads the type's serde attributes so the schema matches what its
/// `Deserialize` impl accepts, doc comments as descriptions, and
/// `#[schema(...)]`: `description = ".."`, `format = ".."`, `min = N`,
/// `max = N` on fields, and `like = Type` to describe a field or type by
/// another type's JSON.
#[proc_macro_derive(Schema, attributes(schema))]
pub fn derive_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::derive(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
