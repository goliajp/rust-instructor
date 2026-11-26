//! Enum variants.

use super::{Field, ShapeFn};

/// One enum variant.
pub struct Variant {
    pub(super) name: &'static str,
    pub(super) doc: Option<&'static str>,
    pub(super) kind: VariantKind,
}

impl Variant {
    pub const fn new(name: &'static str, kind: VariantKind) -> Self {
        Self {
            name,
            doc: None,
            kind,
        }
    }

    pub const fn doc(mut self, doc: &'static str) -> Self {
        self.doc = Some(doc);
        self
    }
}

/// What a variant carries.
#[non_exhaustive]
pub enum VariantKind {
    Unit,
    Newtype(ShapeFn),
    Tuple(&'static [ShapeFn]),
    Struct(&'static [Field]),
}
