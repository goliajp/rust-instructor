//! `#[schema(partial)]`: the type a value has while it streams, and the
//! `Partial` impl that names it.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident, Result, Type, WherePredicate};

use crate::attr::{self, Container};
use crate::case::Rule;

const P: &str = "::instructors::partial::Partial";

fn partial_path() -> TokenStream {
    quote!(::instructors::partial::Partial)
}

pub(crate) fn expand(
    input: &DeriveInput,
    c: &Container,
    name: Option<&str>,
) -> Result<TokenStream> {
    if let Some(l) = input.generics.lifetimes().next() {
        return Err(syn::Error::new_spanned(
            l,
            "#[schema(partial)] needs a type without lifetime parameters: a snapshot owns its data",
        ));
    }
    if let Some(like) = &c.like {
        return Ok(delegate(input, like));
    }
    match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) if c.transparent => {
                for f in &n.named {
                    let a = attr::field(&f.attrs)?;
                    if !a.skip {
                        return Ok(delegate(input, a.like.as_ref().unwrap_or(&f.ty)));
                    }
                }
                Ok(whole(input))
            }
            Fields::Named(n) => named(input, c, n, name),
            Fields::Unnamed(u) => {
                let mut live = Vec::new();
                for f in &u.unnamed {
                    let a = attr::field(&f.attrs)?;
                    if !a.skip {
                        live.push(a.like.clone().unwrap_or_else(|| f.ty.clone()));
                    }
                }
                if live.len() == 1 && (u.unnamed.len() == 1 || c.transparent) {
                    return Ok(delegate(input, &live[0]));
                }
                Ok(whole(input))
            }
            Fields::Unit => Ok(whole(input)),
        },
        _ => Ok(whole(input)),
    }
}

/// The where predicates of the input, plus `T: Partial` for each type
/// parameter.
fn bounds(input: &DeriveInput) -> Vec<WherePredicate> {
    let mut out: Vec<WherePredicate> = input
        .generics
        .where_clause
        .iter()
        .flat_map(|w| w.predicates.iter().cloned())
        .collect();
    let p = partial_path();
    for t in input.generics.type_params() {
        let id = &t.ident;
        out.push(syn::parse_quote!(#id: #p));
    }
    out
}

/// A type whose JSON is another type's: the other type's snapshot.
fn delegate(input: &DeriveInput, inner: &Type) -> TokenStream {
    let name = &input.ident;
    let (impl_g, ty_g, _) = input.generics.split_for_impl();
    let preds = bounds(input);
    let p = partial_path();
    quote! {
        #[automatically_derived]
        impl #impl_g #p for #name #ty_g where #(#preds,)* #inner: #p {
            type Partial = <#inner as #p>::Partial;
            fn read<'de, D: ::instructors::serde::Deserializer<'de>>(
                d: D,
            ) -> ::std::result::Result<::std::option::Option<Self::Partial>, D::Error> {
                <#inner as #p>::read(d)
            }
        }
    }
}

/// A type shown only once complete: enums, tuple and unit structs.
fn whole(input: &DeriveInput) -> TokenStream {
    let name = &input.ident;
    let (impl_g, ty_g, where_g) = input.generics.split_for_impl();
    let preds = where_g.map(|w| w.predicates.iter().cloned().collect::<Vec<_>>());
    let preds = preds.unwrap_or_default();
    let p = partial_path();
    quote! {
        #[automatically_derived]
        impl #impl_g #p for #name #ty_g
        where
            #(#preds,)*
            Self: ::instructors::serde::de::DeserializeOwned,
        {
            type Partial = Self;
            fn read<'de, D: ::instructors::serde::Deserializer<'de>>(
                d: D,
            ) -> ::std::result::Result<::std::option::Option<Self>, D::Error> {
                ::instructors::partial::whole(d)
            }
        }
    }
}

struct Out {
    ident: Ident,
    vis: syn::Visibility,
    attrs: TokenStream,
    ty: TokenStream,
}

fn named(
    input: &DeriveInput,
    c: &Container,
    n: &syn::FieldsNamed,
    name: Option<&str>,
) -> Result<TokenStream> {
    let ident = &input.ident;
    let pname = match name {
        Some(s) => format_ident!("{}", s),
        None => format_ident!("{}Partial", ident),
    };
    let rule = c.rename_all.unwrap_or(Rule::None);
    let p = partial_path();
    let mut fields = Vec::new();
    for f in &n.named {
        let a = attr::field(&f.attrs)?;
        if a.skip {
            continue;
        }
        let id = f.ident.clone().expect("named field");
        let json = a
            .rename
            .clone()
            .unwrap_or_else(|| rule.field(&crate::expand::unraw(&id)));
        let ty = a.like.as_ref().unwrap_or(&f.ty);
        let (attrs, ty) = if a.flatten {
            (quote!(#[serde(flatten)]), quote!(<#ty as #p>::Partial))
        } else if a.whole {
            (
                quote!(#[serde(rename = #json, default, deserialize_with = "::instructors::partial::whole")]),
                quote!(::std::option::Option<#ty>),
            )
        } else {
            let read = format!("<{} as {P}>::read", quote!(#ty));
            (
                quote!(#[serde(rename = #json, default, deserialize_with = #read)]),
                quote!(::std::option::Option<<#ty as #p>::Partial>),
            )
        };
        fields.push(Out {
            ident: id,
            vis: f.vis.clone(),
            attrs,
            ty,
        });
    }
    let vis = &input.vis;
    let generics = &input.generics;
    let (impl_g, ty_g, _) = generics.split_for_impl();
    let preds = bounds(input);
    let decl = fields.iter().map(|f| {
        let (attrs, vis, id, ty) = (&f.attrs, &f.vis, &f.ident, &f.ty);
        quote!(#attrs #vis #id: #ty)
    });
    let ids: Vec<&Ident> = fields.iter().map(|f| &f.ident).collect();
    let names: Vec<String> = ids.iter().map(|i| crate::expand::unraw(i)).collect();
    let tys: Vec<&TokenStream> = fields.iter().map(|f| &f.ty).collect();
    let pname_str = pname.to_string();
    let doc = format!(
        "The value of [`{ident}`] as far as it has streamed: each field is `None` until \
         something usable for it has arrived. Generated by `#[schema(partial)]`."
    );
    // a binder keeps each field bound from being checked on its own, so an
    // impl whose field type lacks the trait is left out instead of failing
    Ok(quote! {
        #[doc = #doc]
        #[derive(::instructors::serde::Deserialize)]
        #[serde(crate = "::instructors::serde", bound(deserialize = ""))]
        #vis struct #pname #generics where #(#preds,)* {
            #(#decl,)*
        }

        #[automatically_derived]
        impl #impl_g #p for #ident #ty_g where #(#preds,)* {
            type Partial = #pname #ty_g;
        }

        #[automatically_derived]
        impl #impl_g ::std::fmt::Debug for #pname #ty_g
        where
            #(#preds,)*
            #(for<'__p> #tys: ::std::fmt::Debug,)*
        {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.debug_struct(#pname_str)
                    #(.field(#names, &self.#ids))*
                    .finish()
            }
        }

        #[automatically_derived]
        impl #impl_g ::std::clone::Clone for #pname #ty_g
        where
            #(#preds,)*
            #(for<'__p> #tys: ::std::clone::Clone,)*
        {
            fn clone(&self) -> Self {
                Self { #(#ids: ::std::clone::Clone::clone(&self.#ids),)* }
            }
        }

        #[automatically_derived]
        impl #impl_g ::std::default::Default for #pname #ty_g
        where
            #(#preds,)*
            #(for<'__p> #tys: ::std::default::Default,)*
        {
            fn default() -> Self {
                Self { #(#ids: ::std::default::Default::default(),)* }
            }
        }
    })
}
