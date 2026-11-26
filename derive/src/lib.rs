//! `#[derive(Schema)]` for the [instructors](https://docs.rs/instructors)
//! crate. Use it through `instructors`, which re-exports it.

mod attr;
mod case;
mod expand;
mod partial;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derive `instructors::schema::Schema`.
///
/// Reads the type's serde attributes so the schema matches what its
/// `Deserialize` impl accepts, doc comments as descriptions, and
/// `#[schema(...)]`: `description = ".."`, `format = ".."`, `min = N`,
/// `max = N` on fields, and `like = Type` to describe a field or type by
/// another type's JSON.
///
/// `#[schema(partial)]` on the type also implements
/// `instructors::partial::Partial` and generates the type its value has
/// while it streams: for a struct `Name` with named fields, a struct
/// `NamePartial` (or the name given as `partial = "Other"`) whose fields
/// are all `Option`. `#[schema(whole)]` on a field keeps it out until it is
/// complete; a field's type needs `Partial` unless it has that attribute.
#[proc_macro_derive(Schema, attributes(schema))]
pub fn derive_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::derive(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
