//! Code generation: a `Schema` impl returning a `static` shape, or for
//! generic types, a shape interned per instantiation.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Generics, Ident, Result, Type};

use crate::attr;
use crate::case::Rule;

/// Field and variant data, either as statics or built at run time.
struct Gen {
    generic: bool,
    statics: Vec<TokenStream>,
    counter: usize,
}

impl Gen {
    fn next(&mut self, prefix: &str) -> Ident {
        self.counter += 1;
        format_ident!("__{}_{}", prefix, self.counter)
    }

    /// A `&'static [elem]` expression for `items`.
    fn slice(&mut self, prefix: &str, elem: TokenStream, items: Vec<TokenStream>) -> TokenStream {
        if self.generic {
            return quote!(::instructors::schema::leak(::std::vec![#(#items),*]));
        }
        let name = self.next(prefix);
        let n = items.len();
        self.statics
            .push(quote!(static #name: [#elem; #n] = [#(#items),*];));
        quote!(&#name)
    }

    /// A `&'static T` for a value of a type with interior state.
    fn place(&mut self, prefix: &str, ty: TokenStream, value: TokenStream) -> TokenStream {
        if self.generic {
            return quote!(::std::boxed::Box::leak(::std::boxed::Box::new(#value)));
        }
        let name = self.next(prefix);
        self.statics.push(quote!(static #name: #ty = #value;));
        quote!(&#name)
    }
}

fn shape_fn(ty: &Type) -> TokenStream {
    quote!(<#ty as ::instructors::schema::Schema>::schema)
}

pub(crate) fn unraw(id: &Ident) -> String {
    let s = id.to_string();
    s.strip_prefix("r#").map(str::to_owned).unwrap_or(s)
}

fn opt_doc(doc: Option<&String>) -> TokenStream {
    match doc {
        Some(d) => quote!(.doc(#d)),
        None => quote!(),
    }
}

pub(crate) fn derive(input: DeriveInput) -> Result<TokenStream> {
    let c = attr::container(&input.attrs)?;
    let name = &input.ident;
    let generics = with_bounds(&input.generics);
    let (impl_g, ty_g, where_g) = generics.split_for_impl();
    let generic = !input.generics.params.is_empty();
    let type_name = c.rename.clone().unwrap_or_else(|| unraw(name));

    let mut g = Gen {
        generic,
        statics: Vec::new(),
        counter: 0,
    };
    let body = if let Some(like) = &c.like {
        let f = shape_fn(like);
        quote!(#f())
    } else {
        let shape = match &input.data {
            Data::Struct(s) => strukt(&mut g, &c, &type_name, &s.fields)?,
            Data::Enum(e) => enumeration(&mut g, &c, &type_name, e)?,
            Data::Union(_) => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "instructors::Schema cannot be derived for unions",
                ));
            }
        };
        match shape {
            Body::Delegate(t) => t,
            Body::Shape(shape) if generic => quote! {
                ::instructors::schema::intern(<Self as ::instructors::schema::Schema>::schema, || #shape)
            },
            Body::Shape(shape) => {
                let statics = &g.statics;
                quote! {
                    #(#statics)*
                    static __SHAPE: ::instructors::schema::Shape = #shape;
                    &__SHAPE
                }
            }
        }
    };
    let partial = match &c.partial {
        Some(pname) => crate::partial::expand(&input, &c, pname.as_deref())?,
        None => quote!(),
    };
    Ok(quote! {
        #[automatically_derived]
        impl #impl_g ::instructors::schema::Schema for #name #ty_g #where_g {
            fn schema() -> &'static ::instructors::schema::Shape {
                #body
            }
        }
        #partial
    })
}

fn with_bounds(g: &Generics) -> Generics {
    let mut g = g.clone();
    let params: Vec<Ident> = g.type_params().map(|p| p.ident.clone()).collect();
    let wc = g.make_where_clause();
    for p in params {
        wc.predicates
            .push(syn::parse_quote!(#p: ::instructors::schema::Schema));
    }
    g
}

enum Body {
    /// The type's JSON is another type's.
    Delegate(TokenStream),
    Shape(TokenStream),
}

fn strukt(g: &mut Gen, c: &attr::Container, type_name: &str, fields: &Fields) -> Result<Body> {
    match fields {
        Fields::Unit => Ok(Body::Shape(quote!(::instructors::schema::Shape::Null))),
        Fields::Unnamed(u) => {
            let live = live_unnamed(u)?;
            if live.len() == 1 && (u.unnamed.len() == 1 || c.transparent) {
                let f = &live[0];
                return Ok(Body::Delegate(quote!(#f())));
            }
            let t = g.slice("TUPLE", quote!(::instructors::schema::ShapeFn), live);
            Ok(Body::Shape(quote!(::instructors::schema::Shape::Tuple(#t))))
        }
        Fields::Named(n) => {
            if c.transparent {
                for f in &n.named {
                    let a = attr::field(&f.attrs)?;
                    if !a.skip {
                        let sf = shape_fn(a.like.as_ref().unwrap_or(&f.ty));
                        return Ok(Body::Delegate(quote!(#sf())));
                    }
                }
            }
            let list = named(g, n, c.rename_all.unwrap_or(Rule::None), c.default)?;
            let doc = opt_doc(c.doc.as_ref());
            let deny = if c.deny_unknown {
                quote!(.deny_unknown_fields())
            } else {
                quote!()
            };
            let st = g.place(
                "STRUCT",
                quote!(::instructors::schema::StructShape),
                quote!(::instructors::schema::StructShape::new(#type_name, #list)#doc #deny),
            );
            Ok(Body::Shape(
                quote!(::instructors::schema::Shape::Struct(#st)),
            ))
        }
    }
}

fn live_unnamed(u: &syn::FieldsUnnamed) -> Result<Vec<TokenStream>> {
    let mut out = Vec::new();
    for f in &u.unnamed {
        let a = attr::field(&f.attrs)?;
        if !a.skip {
            out.push(shape_fn(a.like.as_ref().unwrap_or(&f.ty)));
        }
    }
    Ok(out)
}

/// A `&'static [Field]` for named fields.
fn named(g: &mut Gen, n: &syn::FieldsNamed, rule: Rule, all_default: bool) -> Result<TokenStream> {
    let mut items = Vec::new();
    for f in &n.named {
        let a = attr::field(&f.attrs)?;
        if a.skip {
            continue;
        }
        let ident = f.ident.as_ref().expect("named field");
        let name = a
            .rename
            .clone()
            .unwrap_or_else(|| rule.field(&unraw(ident)));
        let sf = shape_fn(a.like.as_ref().unwrap_or(&f.ty));
        let mut e = quote!(::instructors::schema::Field::new(#name, #sf));
        if let Some(d) = &a.doc {
            e.extend(quote!(.doc(#d)));
        }
        if a.default || all_default {
            e.extend(quote!(.default()));
        }
        if a.flatten {
            e.extend(quote!(.flatten()));
        }
        if let Some(v) = &a.format {
            e.extend(quote!(.format(#v)));
        }
        if let Some(v) = &a.min {
            e.extend(quote!(.min(#v)));
        }
        if let Some(v) = &a.max {
            e.extend(quote!(.max(#v)));
        }
        items.push(e);
    }
    Ok(g.slice("FIELDS", quote!(::instructors::schema::Field), items))
}

fn enumeration(
    g: &mut Gen,
    c: &attr::Container,
    type_name: &str,
    e: &syn::DataEnum,
) -> Result<Body> {
    let tagging = match (&c.tag, &c.content, c.untagged) {
        (_, _, true) => quote!(::instructors::schema::Tagging::Untagged),
        (Some(t), Some(ct), false) => {
            quote!(::instructors::schema::Tagging::Adjacent { tag: #t, content: #ct })
        }
        (Some(t), None, false) => quote!(::instructors::schema::Tagging::Internal { tag: #t }),
        (None, _, false) => quote!(::instructors::schema::Tagging::External),
    };
    let rule = c.rename_all.unwrap_or(Rule::None);
    let mut variants = Vec::new();
    for v in &e.variants {
        let a = attr::variant(&v.attrs)?;
        if a.skip {
            continue;
        }
        let name = a
            .rename
            .clone()
            .unwrap_or_else(|| rule.variant(&unraw(&v.ident)));
        let kind = match &v.fields {
            Fields::Unit => quote!(::instructors::schema::VariantKind::Unit),
            Fields::Unnamed(u) => {
                let live = live_unnamed(u)?;
                if live.len() == 1 {
                    let f = &live[0];
                    quote!(::instructors::schema::VariantKind::Newtype(#f))
                } else {
                    let t = g.slice("TUPLE", quote!(::instructors::schema::ShapeFn), live);
                    quote!(::instructors::schema::VariantKind::Tuple(#t))
                }
            }
            Fields::Named(n) => {
                let field_rule = a
                    .rename_all
                    .unwrap_or(Rule::None)
                    .or(c.rename_all_fields.unwrap_or(Rule::None));
                let list = named(g, n, field_rule, false)?;
                quote!(::instructors::schema::VariantKind::Struct(#list))
            }
        };
        let doc = opt_doc(a.doc.as_ref());
        variants.push(quote!(::instructors::schema::Variant::new(#name, #kind)#doc));
    }
    let list = g.slice("VARIANTS", quote!(::instructors::schema::Variant), variants);
    let doc = opt_doc(c.doc.as_ref());
    let en = g.place(
        "ENUM",
        quote!(::instructors::schema::EnumShape),
        quote!(::instructors::schema::EnumShape::new(#type_name, #tagging, #list)#doc),
    );
    Ok(Body::Shape(quote!(::instructors::schema::Shape::Enum(#en))))
}
