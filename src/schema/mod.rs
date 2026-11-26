//! What a type's JSON looks like, as data the request builders turn into
//! each provider's schema dialect.
//!
//! `#[derive(Schema)]` writes a [`Schema`] impl that returns a `static`
//! [`Shape`]. The provider-specific JSON Schema bytes are built from it once
//! per type and cached next to the shape.

mod cache;
mod check;
mod dialect;
mod guide;
mod impls;
mod intern;
mod lower;
#[cfg(feature = "schemars")]
pub(crate) mod schemars;
mod strict;
#[cfg(any(test, feature = "fuzzing"))]
mod transcode;
mod tree;
mod variant;

pub use cache::Cache;
pub(crate) use cache::{Dialect, SchemaEntry, entry_for};
pub(crate) use guide::{At, Key, Plan};
use intern::lock;
#[doc(hidden)]
pub use intern::{intern, leak};
pub use variant::{Variant, VariantKind};

/// A type whose JSON form can be described to a model.
///
/// Derive it with `#[derive(Schema)]`. Implement it by hand for a type the
/// derive cannot see into, returning the shape of the JSON the type's
/// `Deserialize` impl accepts:
///
/// ```
/// use instructors::schema::{Schema, Shape};
///
/// struct Timestamp(String);
///
/// impl Schema for Timestamp {
///     fn schema() -> &'static Shape {
///         static S: Shape = Shape::string_format("date-time");
///         &S
///     }
/// }
/// ```
pub trait Schema {
    /// The shape of this type's JSON.
    fn schema() -> &'static Shape;
}

/// Returns a type's [`Shape`]; nested types are referred to this way so a
/// type can contain itself.
pub type ShapeFn = fn() -> &'static Shape;

/// The JSON a type accepts.
#[non_exhaustive]
pub enum Shape {
    /// `null` (`()` and unit structs).
    Null,
    /// `true` / `false`.
    Bool,
    /// An integer with its JSON Schema `format` and optional bounds, written
    /// as JSON number text.
    Integer {
        format: &'static str,
        min: Option<&'static str>,
        max: Option<&'static str>,
    },
    /// A number with its JSON Schema `format` (`float`, `double`).
    Number { format: &'static str },
    /// A string; `len` fixes its length in characters (`char` is 1).
    String {
        format: Option<&'static str>,
        len: Option<u32>,
    },
    /// The inner value or `null`.
    Option(ShapeFn),
    /// A list; `len` fixes its length (arrays), `unique` marks sets.
    Array {
        items: ShapeFn,
        len: Option<u32>,
        unique: bool,
    },
    /// A fixed-length list with one shape per position.
    Tuple(&'static [ShapeFn]),
    /// An object with string keys and values of one shape.
    Map(ShapeFn),
    /// Any JSON value.
    Any,
    /// An object with named fields.
    Struct(&'static StructShape),
    /// One of several variants.
    Enum(&'static EnumShape),
}

impl Shape {
    /// A string with a JSON Schema `format`, such as `"date-time"`.
    pub const fn string_format(format: &'static str) -> Self {
        Self::String {
            format: Some(format),
            len: None,
        }
    }
}

/// A struct: named fields, in declaration order.
pub struct StructShape {
    name: &'static str,
    doc: Option<&'static str>,
    fields: &'static [Field],
    closed: bool,
    cache: Cache,
}

impl StructShape {
    pub const fn new(name: &'static str, fields: &'static [Field]) -> Self {
        Self {
            name,
            doc: None,
            fields,
            closed: false,
            cache: Cache::new(),
        }
    }

    /// The description sent with the schema.
    pub const fn doc(mut self, doc: &'static str) -> Self {
        self.doc = Some(doc);
        self
    }

    /// `#[serde(deny_unknown_fields)]`.
    pub const fn deny_unknown_fields(mut self) -> Self {
        self.closed = true;
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn fields(&self) -> &'static [Field] {
        self.fields
    }
}

/// One field of a struct or struct variant.
pub struct Field {
    name: &'static str,
    doc: Option<&'static str>,
    shape: ShapeFn,
    default: bool,
    flatten: bool,
    format: Option<&'static str>,
    min: Option<&'static str>,
    max: Option<&'static str>,
}

impl Field {
    /// A required field (an `Option` field is optional by its shape).
    pub const fn new(name: &'static str, shape: ShapeFn) -> Self {
        Self {
            name,
            doc: None,
            shape,
            default: false,
            flatten: false,
            format: None,
            min: None,
            max: None,
        }
    }

    pub const fn doc(mut self, doc: &'static str) -> Self {
        self.doc = Some(doc);
        self
    }

    /// `#[serde(default)]`: the field may be left out.
    pub const fn default(mut self) -> Self {
        self.default = true;
        self
    }

    /// `#[serde(flatten)]`.
    pub const fn flatten(mut self) -> Self {
        self.flatten = true;
        self
    }

    /// `#[schema(format = "..")]`.
    pub const fn format(mut self, format: &'static str) -> Self {
        self.format = Some(format);
        self
    }

    /// `#[schema(min = ..)]`, as JSON number text.
    pub const fn min(mut self, min: &'static str) -> Self {
        self.min = Some(min);
        self
    }

    /// `#[schema(max = ..)]`, as JSON number text.
    pub const fn max(mut self, max: &'static str) -> Self {
        self.max = Some(max);
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn shape(&self) -> &'static Shape {
        (self.shape)()
    }
}

/// How an enum's variants are told apart in JSON, as in serde.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tagging {
    External,
    Internal {
        tag: &'static str,
    },
    Adjacent {
        tag: &'static str,
        content: &'static str,
    },
    Untagged,
}

/// An enum.
pub struct EnumShape {
    name: &'static str,
    doc: Option<&'static str>,
    tagging: Tagging,
    variants: &'static [Variant],
    cache: Cache,
}

impl EnumShape {
    pub const fn new(name: &'static str, tagging: Tagging, variants: &'static [Variant]) -> Self {
        Self {
            name,
            doc: None,
            tagging,
            variants,
            cache: Cache::new(),
        }
    }

    pub const fn doc(mut self, doc: &'static str) -> Self {
        self.doc = Some(doc);
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod oracle_15;
#[cfg(test)]
mod tests;

/// The derive rejects what it cannot describe.
///
/// ```compile_fail
/// #[derive(serde::Deserialize, instructors::Schema)]
/// struct S {
///     #[serde(deserialize_with = "f")]
///     a: u8,
/// }
/// ```
///
/// ```compile_fail
/// #[derive(serde::Deserialize, instructors::Schema)]
/// #[serde(rename_all = "Train-Case")]
/// struct S { a: u8 }
/// ```
///
/// ```compile_fail
/// #[derive(serde::Deserialize, instructors::Schema)]
/// #[serde(try_from = "u8")]
/// struct S { a: u8 }
/// ```
///
/// ```compile_fail
/// #[derive(instructors::Schema)]
/// struct S {
///     #[schema(minimum = 1)]
///     a: u8,
/// }
/// ```
///
/// ```compile_fail
/// #[derive(instructors::Schema)]
/// union U { a: u8 }
/// ```
///
/// The same types are accepted with `like`:
///
/// ```
/// #[derive(serde::Deserialize, instructors::Schema)]
/// #[serde(try_from = "u8")]
/// #[schema(like = u8)]
/// struct S { a: u8 }
/// # impl TryFrom<u8> for S { type Error = String; fn try_from(a: u8) -> Result<Self, String> { Ok(S { a }) } }
/// ```
#[cfg(doctest)]
pub struct DeriveErrors;
