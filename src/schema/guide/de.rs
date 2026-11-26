//! The guided `Deserializer`: reads the answer in its wire form and hands
//! the caller's type the type's own form, checking on the way what the
//! wire schema could not say.

use serde::de::{self, Deserialize, IgnoredAny, Visitor};

use super::access::{Arr, Obj, Pairs, Slots, Variant};
use super::plan::{ARRAY, BOOLEAN, INTEGER, Id, NULL, NUMBER, Node, OBJECT, Plan, STRING, Typed};
use crate::json::de::{Deserializer as Json, RAW_TOKEN};
use crate::json::{JsonError, Str};
use crate::text::clip;

/// Asks a [`Guide`] for a value as raw JSON text at the leaves, so that
/// list items can be compared in their converted form.
pub(super) const NORM_TOKEN: &str = "$instructors::private::Norm";

/// The plan, and whether every node is checked: a choice between
/// alternatives is made by reading the value fully against each.
#[derive(Clone, Copy)]
pub(super) struct Cx<'p> {
    pub(super) plan: &'p Plan,
    pub(super) full: bool,
    pub(super) fill: bool,
}

pub(super) struct Guide<'p, 'b, 'de> {
    pub(super) cx: Cx<'p>,
    pub(super) de: &'b mut Json<'de>,
    pub(super) node: Id,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ask {
    Option,
    Enum,
    Other,
}

pub(super) fn kind(c: u8) -> &'static str {
    match c {
        b'{' => "object",
        b'[' => "array",
        b'"' => "string",
        b't' | b'f' => "boolean",
        b'n' => "null",
        _ => "number",
    }
}

/// The input between two offsets of a deserializer.
pub(super) fn text<'de>(de: &Json<'de>, from: usize, to: usize) -> &'de str {
    std::str::from_utf8(&de.sc.b[from..to]).expect("the input is a str and values end on ascii")
}

impl<'p, 'de> Guide<'p, '_, 'de> {
    fn fail(&self, what: &str) -> JsonError {
        self.de.err(what)
    }

    /// The node that reads the value here, choices resolved.
    fn resolve(&mut self) -> Result<Id, JsonError> {
        let mut id = self.node;
        loop {
            match self.cx.plan.node(id) {
                Node::AnyOf { alts, option } => {
                    id = match option {
                        Some(_) if self.de.next()? == b'n' => alts[1],
                        Some(first) => *first,
                        None => self.pick(alts)?,
                    };
                }
                Node::Fail(m) => return Err(self.fail(m)),
                _ => return Ok(id),
            }
        }
    }

    /// The first alternative that takes the value, found by reading it
    /// once per alternative without building anything.
    fn pick(&mut self, alts: &[Id]) -> Result<Id, JsonError> {
        let start = self.de.sc.i;
        let mut first = None;
        let cx = Cx {
            full: true,
            ..self.cx
        };
        for &alt in alts {
            let r = IgnoredAny::deserialize(Guide {
                cx,
                de: &mut *self.de,
                node: alt,
            });
            self.de.sc.i = start;
            match r {
                Ok(_) => return Ok(alt),
                Err(e) => {
                    first.get_or_insert(e);
                }
            }
        }
        Err(first.unwrap_or_else(|| self.fail("no variant matches")))
    }

    fn route<V, F>(mut self, ask: Ask, v: V, plain: F) -> Result<V::Value, JsonError>
    where
        V: Visitor<'de>,
        F: FnOnce(&mut Json<'de>, V) -> Result<V::Value, JsonError>,
    {
        if !self.cx.full && self.cx.plan.is_plain(self.node) {
            return plain(self.de, v);
        }
        let id = self.resolve()?;
        let Node::Typed(t) = self.cx.plan.node(id) else {
            return plain(self.de, v);
        };
        let t: &'p Typed = t;
        let c = self.de.next()?;
        if ask == Ask::Option && c != b'n' {
            return v.visit_some(Guide {
                cx: self.cx,
                de: self.de,
                node: id,
            });
        }
        if t.allowed {
            self.allowed(t)?;
        }
        if c == b'n' {
            if t.typed && t.types & NULL == 0 {
                return Err(self.fail("expected a value, got null"));
            }
            return plain(self.de, v);
        }
        if let Some(value) = t.map {
            return match c {
                b'[' => self.pairs(value, v),
                _ => self.object(t, ask, v),
            };
        }
        if let Some(prefix) = &t.prefix {
            return match c {
                b'{' => self.tuple(prefix, v),
                _ => self.array(t, v),
            };
        }
        match c {
            b'{' if t.wants(OBJECT) => self.object(t, ask, v),
            b'[' if t.wants(ARRAY) => self.array(t, v),
            b'"' if t.wants(STRING) => {
                if t.scalar_bounds {
                    self.length(t)?;
                }
                plain(self.de, v)
            }
            b't' | b'f' if t.wants(BOOLEAN) => {
                if t.scalar_bounds {
                    self.number(t)?;
                }
                plain(self.de, v)
            }
            b'-' | b'0'..=b'9' if t.wants(NUMBER) || t.types & INTEGER != 0 => {
                if t.scalar_bounds {
                    self.number(t)?;
                }
                plain(self.de, v)
            }
            other => {
                let want = t.name.as_deref().unwrap_or("another type");
                Err(self.fail(&format!("expected {want}, got {}", kind(other))))
            }
        }
    }

    /// The raw text of the next value, which is not consumed.
    fn peek_raw(&mut self) -> Result<&'de str, JsonError> {
        let start = self.de.sc.i;
        self.de.sc.skip_value()?;
        let raw = text(self.de, start, self.de.sc.i);
        self.de.sc.i = start;
        Ok(raw)
    }

    /// `const` and `enum`.
    #[cold]
    fn allowed(&mut self, t: &Typed) -> Result<(), JsonError> {
        let raw = self.peek_raw()?;
        let v: super::super::tree::J = crate::json::from_str(raw)?;
        if let Some(c) = &t.constant
            && *c != v
        {
            let want = super::super::strict::text(c);
            return Err(self.fail(&format!("expected {want}")));
        }
        if let Some(e) = &t.choices
            && !e.contains(&v)
        {
            let raw = clip(raw, 80);
            return Err(self.fail(&format!("{raw} is not one of the allowed values")));
        }
        Ok(())
    }

    fn number(&mut self, t: &Typed) -> Result<(), JsonError> {
        let start = self.de.sc.i;
        let raw = match self.de.sc.peek() {
            Some(b'-' | b'0'..=b'9') => {
                self.de.sc.skip_number()?;
                let raw = text(self.de, start, self.de.sc.i);
                self.de.sc.i = start;
                raw
            }
            _ => self.peek_raw()?,
        };
        t.bounds
            .number(raw)
            .map_err(|what| self.fail(&format!("{} {what}", clip(raw, 80))))
    }

    fn length(&mut self, t: &Typed) -> Result<(), JsonError> {
        let start = self.de.sc.i;
        self.de.sc.i += 1;
        let n = match self.de.sc.parse_str(&mut self.de.scratch)? {
            Str::Borrowed(s) | Str::Copied(s) => s.chars().count(),
        };
        let raw = text(self.de, start, self.de.sc.i);
        self.de.sc.i = start;
        t.bounds
            .length(n)
            .map_err(|what| self.fail(&format!("{} {what}", clip(raw, 80))))
    }

    /// After a container's visitor: its closing byte.
    fn close<T>(&mut self, value: Result<T, JsonError>, end: u8) -> Result<T, JsonError> {
        let value = value.map_err(|e| e.or_at(self.de.sc.i))?;
        match self.de.sc.ws() {
            Some(c) if c == end => {
                self.de.sc.i += 1;
                Ok(value)
            }
            Some(b',') => Err(self.fail("trailing comma")),
            Some(_) => Err(self.fail("trailing characters")),
            None if end == b']' => Err(self.fail("EOF while parsing a list")),
            None => Err(self.fail("EOF while parsing an object")),
        }
    }

    fn object<V: Visitor<'de>>(self, t: &'p Typed, ask: Ask, v: V) -> Result<V::Value, JsonError> {
        if self.de.next()? != b'{' {
            return Err(self.fail("expected `{`"));
        }
        if ask == Ask::Enum {
            return self.variant(t, v);
        }
        self.de.enter()?;
        self.de.sc.i += 1;
        let r = v.visit_map(Obj::new(self.cx, &mut *self.de, t));
        self.de.leave();
        let mut this = self;
        this.close(r, b'}')
    }

    fn variant<V: Visitor<'de>>(self, t: &'p Typed, v: V) -> Result<V::Value, JsonError> {
        self.de.enter()?;
        self.de.sc.i += 1;
        let mut bits = 0;
        let r = v.visit_enum(Variant::new(self.cx, &mut *self.de, t, &mut bits));
        self.de.leave();
        let value = r.map_err(|e| e.or_at(self.de.sc.i))?;
        match self.de.sc.ws() {
            Some(b'}') => self.de.sc.i += 1,
            Some(_) => return Err(self.fail("expected `}`")),
            None => return Err(self.fail("EOF while parsing an object")),
        }
        if bits & t.required_mask != t.required_mask || t.by_name {
            let missing = t.required.iter().find(|r| {
                !t.props
                    .iter()
                    .position(|(k, _)| k == *r)
                    .is_some_and(|i| i < 64 && bits & (1 << i) != 0)
            });
            if let Some(name) = missing {
                return Err(self.fail(&format!("missing \"{name}\"")));
            }
        }
        Ok(value)
    }

    fn array<V: Visitor<'de>>(mut self, t: &'p Typed, v: V) -> Result<V::Value, JsonError> {
        if self.de.next()? != b'[' {
            return Err(self.fail("expected `[`"));
        }
        self.de.enter()?;
        self.de.sc.i += 1;
        let mut seq = Arr::new(self.cx, &mut *self.de, t);
        let r = v.visit_seq(&mut seq);
        let (n, norms) = seq.done();
        self.de.leave();
        let value = self.close(r, b']')?;
        if let Some(p) = &t.prefix
            && p.len() != n
        {
            return Err(self.fail(&format!("expected {} items", p.len())));
        }
        t.bounds
            .items(n)
            .map_err(|what| self.fail(&format!("the list {what}")))?;
        if (1..norms.len()).any(|i| norms[..i].contains(&norms[i])) {
            return Err(self.fail("the list has repeated items"));
        }
        Ok(value)
    }

    fn pairs<V: Visitor<'de>>(mut self, value: Id, v: V) -> Result<V::Value, JsonError> {
        self.de.enter()?;
        self.de.sc.i += 1;
        let r = v.visit_map(Pairs::new(self.cx, &mut *self.de, value));
        self.de.leave();
        self.close(r, b']')
    }

    fn tuple<V: Visitor<'de>>(self, prefix: &'p [Id], v: V) -> Result<V::Value, JsonError> {
        self.de.enter()?;
        self.de.sc.i += 1;
        let r = match Slots::read(self.cx, &mut *self.de, prefix) {
            Ok((slots, end)) => {
                let r = v.visit_seq(slots);
                self.de.sc.i = end;
                r
            }
            Err(e) => Err(e),
        };
        self.de.leave();
        r.map_err(|e| e.or_at(self.de.sc.i))
    }
}

macro_rules! guided {
    ($($method:ident($($arg:ident: $ty:ty),*);)*) => {$(
        fn $method<V: Visitor<'de>>(self, $($arg: $ty,)* v: V) -> Result<V::Value, JsonError> {
            self.route(Ask::Other, v, |de, v| de::Deserializer::$method(de, $($arg,)* v))
        }
    )*};
}

impl<'de> de::Deserializer<'de> for Guide<'_, '_, 'de> {
    type Error = JsonError;

    guided! {
        deserialize_any();
        deserialize_bool();
        deserialize_i8(); deserialize_i16(); deserialize_i32(); deserialize_i64(); deserialize_i128();
        deserialize_u8(); deserialize_u16(); deserialize_u32(); deserialize_u64(); deserialize_u128();
        deserialize_f32(); deserialize_f64();
        deserialize_char(); deserialize_str(); deserialize_string();
        deserialize_bytes(); deserialize_byte_buf();
        deserialize_unit();
        deserialize_unit_struct(name: &'static str);
        deserialize_seq();
        deserialize_tuple(len: usize);
        deserialize_tuple_struct(name: &'static str, len: usize);
        deserialize_map();
        deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_identifier();
        deserialize_ignored_any();
    }

    fn deserialize_option<V: Visitor<'de>>(self, v: V) -> Result<V::Value, JsonError> {
        self.route(Ask::Option, v, |de, v| {
            de::Deserializer::deserialize_option(de, v)
        })
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, JsonError> {
        self.route(Ask::Enum, v, |de, v| {
            de::Deserializer::deserialize_enum(de, name, variants, v)
        })
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        v: V,
    ) -> Result<V::Value, JsonError> {
        match name {
            // the leaves of a converted value as written
            NORM_TOKEN => self.route(Ask::Other, v, |de, v| {
                de::Deserializer::deserialize_newtype_struct(de, RAW_TOKEN, v)
            }),
            RAW_TOKEN => de::Deserializer::deserialize_newtype_struct(self.de, name, v),
            _ => v.visit_newtype_struct(self),
        }
    }
}
