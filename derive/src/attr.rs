//! serde, `schema` and doc attributes.
//!
//! Every serde attribute that changes what JSON a type accepts is either
//! understood here or rejected, so the schema never disagrees with the
//! `Deserialize` impl. Serialization-only attributes are ignored.

use proc_macro2::{Span, TokenTree};
use syn::meta::ParseNestedMeta;
use syn::{Attribute, Expr, ExprLit, Lit, LitStr, Result, Token, Type};

use crate::case::Rule;

#[derive(Default)]
pub(crate) struct Container {
    pub(crate) rename: Option<String>,
    pub(crate) rename_all: Option<Rule>,
    pub(crate) rename_all_fields: Option<Rule>,
    pub(crate) tag: Option<String>,
    pub(crate) content: Option<String>,
    pub(crate) untagged: bool,
    pub(crate) transparent: bool,
    pub(crate) deny_unknown: bool,
    pub(crate) default: bool,
    pub(crate) doc: Option<String>,
    pub(crate) like: Option<Type>,
    /// `#[schema(partial)]`: generate the streaming snapshot type, under
    /// this name when one is given.
    pub(crate) partial: Option<Option<String>>,
}

#[derive(Default)]
pub(crate) struct Field {
    pub(crate) rename: Option<String>,
    pub(crate) default: bool,
    pub(crate) skip: bool,
    pub(crate) flatten: bool,
    pub(crate) doc: Option<String>,
    pub(crate) like: Option<Type>,
    pub(crate) format: Option<String>,
    pub(crate) min: Option<String>,
    pub(crate) max: Option<String>,
    /// `#[schema(whole)]`: in a snapshot only once complete.
    pub(crate) whole: bool,
    // `with` / `deserialize_with` seen, so the field type says nothing
    pub(crate) custom_de: Option<Span>,
}

#[derive(Default)]
pub(crate) struct Variant {
    pub(crate) rename: Option<String>,
    pub(crate) rename_all: Option<Rule>,
    pub(crate) skip: bool,
    pub(crate) doc: Option<String>,
}

fn err(span: Span, msg: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(span, msg)
}

/// Consume `= value` or `(...)` after a key this derive ignores.
fn skip(meta: &ParseNestedMeta<'_>) -> Result<()> {
    if meta.input.peek(Token![=]) {
        meta.input.parse::<Token![=]>()?;
    }
    while !meta.input.is_empty() && !meta.input.peek(Token![,]) {
        meta.input.parse::<TokenTree>()?;
    }
    Ok(())
}

fn string(meta: &ParseNestedMeta<'_>) -> Result<String> {
    Ok(meta.value()?.parse::<LitStr>()?.value())
}

/// `rename = "x"` or `rename(deserialize = "x", serialize = "y")`: the
/// deserialize name, since that is what the model must write.
fn de_name(meta: &ParseNestedMeta<'_>) -> Result<Option<String>> {
    if meta.input.peek(Token![=]) {
        return Ok(Some(string(meta)?));
    }
    let mut out = None;
    meta.parse_nested_meta(|m| {
        if m.path.is_ident("deserialize") {
            out = Some(string(&m)?);
        } else {
            skip(&m)?;
        }
        Ok(())
    })?;
    Ok(out)
}

fn rule(meta: &ParseNestedMeta<'_>) -> Result<Option<Rule>> {
    let Some(name) = de_name(meta)? else {
        return Ok(None);
    };
    match Rule::parse(&name) {
        Some(r) => Ok(Some(r)),
        None => Err(err(
            meta.path.span_ident(),
            format!(
                "unknown rename rule `{name}`, expected one of {}",
                Rule::names()
            ),
        )),
    }
}

trait SpanIdent {
    fn span_ident(&self) -> Span;
}

impl SpanIdent for syn::Path {
    fn span_ident(&self) -> Span {
        self.segments
            .last()
            .map_or_else(Span::call_site, |s| s.ident.span())
    }
}

fn key(meta: &ParseNestedMeta<'_>) -> String {
    meta.path
        .get_ident()
        .map_or_else(String::new, ToString::to_string)
}

fn unsupported(meta: &ParseNestedMeta<'_>, what: &str) -> syn::Error {
    err(
        meta.path.span_ident(),
        format!(
            "instructors::Schema does not support `#[serde({})]`: {what}",
            key(meta)
        ),
    )
}

/// Doc comment lines joined by `\n`, one leading space stripped per line.
pub(crate) fn doc(attrs: &[Attribute]) -> Option<String> {
    let mut lines = Vec::new();
    for a in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        if let Ok(nv) = a.meta.require_name_value()
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
        {
            let v = s.value();
            lines.push(v.strip_prefix(' ').map(str::to_owned).unwrap_or(v));
        }
    }
    let joined = lines.join("\n");
    let trimmed = joined.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn each(
    attrs: &[Attribute],
    name: &str,
    mut f: impl FnMut(&ParseNestedMeta<'_>) -> Result<()>,
) -> Result<()> {
    for a in attrs.iter().filter(|a| a.path().is_ident(name)) {
        a.parse_nested_meta(|m| f(&m))?;
    }
    Ok(())
}

pub(crate) fn container(attrs: &[Attribute]) -> Result<Container> {
    let mut c = Container {
        doc: doc(attrs),
        ..Container::default()
    };
    each(attrs, "schema", |m| {
        match key(m).as_str() {
            "description" => c.doc = Some(string(m)?),
            "like" => c.like = Some(m.value()?.parse::<Type>()?),
            "partial" if m.input.peek(Token![=]) => c.partial = Some(Some(string(m)?)),
            "partial" => c.partial = Some(None),
            _ => {
                return Err(err(
                    m.path.span_ident(),
                    "expected `description`, `like` or `partial`",
                ));
            }
        }
        Ok(())
    })?;
    let has_like = c.like.is_some();
    each(attrs, "serde", |m| {
        match key(m).as_str() {
            "rename" => c.rename = de_name(m)?,
            "rename_all" => c.rename_all = rule(m)?,
            "rename_all_fields" => c.rename_all_fields = rule(m)?,
            "tag" => c.tag = Some(string(m)?),
            "content" => c.content = Some(string(m)?),
            "untagged" => c.untagged = true,
            "transparent" => c.transparent = true,
            "deny_unknown_fields" => c.deny_unknown = true,
            "default" => {
                c.default = true;
                skip(m)?;
            }
            "from" | "try_from" | "remote" if !has_like => {
                skip(m)?;
                return Err(unsupported(
                    m,
                    "the JSON shape is another type's; add #[schema(like = ThatType)]",
                ));
            }
            "variant_identifier" | "field_identifier" => {
                return Err(unsupported(m, "identifier enums are not data types"));
            }
            // serialization only, or no effect on the accepted JSON
            "from" | "try_from" | "remote" | "bound" | "crate" | "expecting" | "into" => skip(m)?,
            _ => {
                skip(m)?;
                return Err(unsupported(m, "unknown attribute"));
            }
        }
        Ok(())
    })?;
    Ok(c)
}

fn number(m: &ParseNestedMeta<'_>) -> Result<String> {
    let input = m.value()?;
    let neg = input.peek(Token![-]);
    if neg {
        input.parse::<Token![-]>()?;
    }
    let digits = match input.parse::<Lit>()? {
        Lit::Int(i) => i.base10_digits().to_owned(),
        Lit::Float(f) => f.base10_digits().to_owned(),
        other => return Err(err(other.span(), "expected a number")),
    };
    Ok(if neg { format!("-{digits}") } else { digits })
}

pub(crate) fn field(attrs: &[Attribute]) -> Result<Field> {
    let mut f = Field {
        doc: doc(attrs),
        ..Field::default()
    };
    each(attrs, "serde", |m| {
        match key(m).as_str() {
            "rename" => f.rename = de_name(m)?,
            "default" => {
                f.default = true;
                skip(m)?;
            }
            "skip" | "skip_deserializing" => f.skip = true,
            "flatten" => f.flatten = true,
            "with" | "deserialize_with" => {
                f.custom_de = Some(m.path.span_ident());
                skip(m)?;
            }
            "alias"
            | "skip_serializing"
            | "skip_serializing_if"
            | "serialize_with"
            | "borrow"
            | "bound"
            | "getter" => skip(m)?,
            _ => {
                skip(m)?;
                return Err(unsupported(m, "unknown attribute"));
            }
        }
        Ok(())
    })?;
    each(attrs, "schema", |m| {
        match key(m).as_str() {
            "description" => f.doc = Some(string(m)?),
            "format" => f.format = Some(string(m)?),
            "min" => f.min = Some(number(m)?),
            "max" => f.max = Some(number(m)?),
            "like" => f.like = Some(m.value()?.parse::<Type>()?),
            "whole" => f.whole = true,
            _ => {
                return Err(err(
                    m.path.span_ident(),
                    "expected `description`, `format`, `min`, `max`, `like` or `whole`",
                ));
            }
        }
        Ok(())
    })?;
    if let (Some(span), None) = (f.custom_de, &f.like) {
        return Err(err(
            span,
            "the field's JSON shape is set by its deserialize function; \
             add #[schema(like = TypeWithThatShape)]",
        ));
    }
    Ok(f)
}

pub(crate) fn variant(attrs: &[Attribute]) -> Result<Variant> {
    let mut v = Variant {
        doc: doc(attrs),
        ..Variant::default()
    };
    each(attrs, "serde", |m| {
        match key(m).as_str() {
            "rename" => v.rename = de_name(m)?,
            "rename_all" => v.rename_all = rule(m)?,
            "skip" | "skip_deserializing" => v.skip = true,
            "alias" | "skip_serializing" | "serialize_with" | "bound" | "borrow" | "other" => {
                skip(m)?
            }
            "with" | "deserialize_with" => {
                skip(m)?;
                return Err(unsupported(m, "the variant's JSON shape is unknown"));
            }
            "untagged" => return Err(unsupported(m, "mixed tagging is not supported")),
            _ => {
                skip(m)?;
                return Err(unsupported(m, "unknown attribute"));
            }
        }
        Ok(())
    })?;
    each(attrs, "schema", |m| {
        if m.path.is_ident("description") {
            v.doc = Some(string(m)?);
            Ok(())
        } else {
            Err(err(m.path.span_ident(), "expected `description`"))
        }
    })?;
    Ok(v)
}
