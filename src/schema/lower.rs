//! Shape → JSON Schema, then per-dialect adjustments, then bytes.
//!
//! The canonical form follows what schemars 1 generates for the same type,
//! with every non-recursive reference inlined and `$defs` kept only for
//! types that contain themselves.

use super::cache::Dialect;
use super::{EnumShape, Field, Shape, StructShape, Tagging, VariantKind};
use crate::json;

/// A JSON value with keys in insertion order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum J {
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

type Map = Vec<(String, J)>;

fn s(v: &str) -> J {
    J::Str(v.to_owned())
}

fn obj(pairs: impl IntoIterator<Item = (&'static str, J)>) -> J {
    J::Obj(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

fn get<'a>(m: &'a Map, key: &str) -> Option<&'a J> {
    m.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn remove(m: &mut Map, key: &str) -> Option<J> {
    let i = m.iter().position(|(k, _)| k == key)?;
    Some(m.remove(i).1)
}

/// Insert or replace, keeping the key's position when it exists.
fn set(m: &mut Map, key: &str, v: J) {
    match m.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = v,
        None => m.push((key.to_owned(), v)),
    }
}

pub(crate) fn write(v: &J, out: &mut Vec<u8>) {
    match v {
        J::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        J::Num(n) => out.extend_from_slice(n.as_bytes()),
        J::Str(t) => json::write_str(out, t),
        J::Arr(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write(x, out);
            }
            out.push(b']');
        }
        J::Obj(m) => {
            out.push(b'{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                json::write_str(out, k);
                out.push(b':');
                write(x, out);
            }
            out.push(b'}');
        }
    }
}

/// The name schemars would give the type, for `json_schema.name`.
pub(crate) fn name_of(shape: &Shape) -> String {
    match shape {
        Shape::Null => "null".into(),
        Shape::Bool => "boolean".into(),
        Shape::Integer { format, .. } | Shape::Number { format } => (*format).into(),
        Shape::String { .. } => "string".into(),
        Shape::Option(f) => format!("Nullable_{}", name_of(f())),
        Shape::Array { items, len, unique } => match (len, unique) {
            (_, true) => format!("Set_of_{}", name_of(items())),
            (Some(n), _) => format!("Array_size_{n}_of_{}", name_of(items())),
            _ => format!("Array_of_{}", name_of(items())),
        },
        Shape::Tuple(fs) => {
            let parts: Vec<String> = fs.iter().map(|f| name_of(f())).collect();
            format!("Tuple_of_{}", parts.join("_and_"))
        }
        Shape::Map(v) => format!("Map_of_{}", name_of(v())),
        Shape::Any => "AnyValue".into(),
        Shape::Struct(st) => st.name.into(),
        Shape::Enum(e) => e.name.into(),
    }
}

struct Lower {
    root: Option<usize>,
    stack: Vec<usize>,
    // address, name used in $defs, lowered yet
    defs: Vec<(usize, String, &'static Shape, bool)>,
    root_ref: bool,
}

fn addr(shape: &'static Shape) -> Option<usize> {
    match shape {
        Shape::Struct(s) => Some(*s as *const StructShape as usize),
        Shape::Enum(e) => Some(*e as *const EnumShape as usize),
        _ => None,
    }
}

/// Lower `shape` to the canonical schema; true when it refers to itself.
pub(crate) fn canonical(shape: &'static Shape, list: bool) -> (J, bool) {
    let mut l = Lower {
        root: if list { None } else { addr(shape) },
        stack: Vec::new(),
        defs: Vec::new(),
        root_ref: false,
    };
    let mut root = l.shape(shape);
    if let (J::Obj(m), false) = (&mut root, list) {
        m.insert(0, ("title".into(), s(&name_of(shape))));
    }
    let mut out_defs = Vec::new();
    while let Some(i) = l.defs.iter().position(|d| !d.3) {
        l.defs[i].3 = true;
        let (a, name, sh, _) = l.defs[i].clone();
        l.stack.push(a);
        let v = l.inner(sh);
        l.stack.pop();
        out_defs.push((name, v));
    }
    let recursive = l.root_ref || !out_defs.is_empty();
    let mut root = if list {
        let title = J::Str(format!("Array_of_{}", name_of(shape)));
        let array = obj([("title", title), ("type", s("array")), ("items", root)]);
        obj([
            ("type", s("object")),
            ("properties", obj([("items", array)])),
            ("required", J::Arr(vec![s("items")])),
        ])
    } else {
        root
    };
    if !out_defs.is_empty()
        && let J::Obj(m) = &mut root
    {
        m.push(("$defs".into(), J::Obj(out_defs)));
    }
    (root, recursive)
}

impl Lower {
    fn shape(&mut self, shape: &'static Shape) -> J {
        let Some(a) = addr(shape) else {
            return self.inner(shape);
        };
        if self.stack.contains(&a) {
            if Some(a) == self.root {
                self.root_ref = true;
                return obj([("$ref", s("#"))]);
            }
            let name = self.def_name(a, shape);
            return obj([("$ref", J::Str(format!("#/$defs/{name}")))]);
        }
        self.stack.push(a);
        let v = self.inner(shape);
        self.stack.pop();
        v
    }

    fn def_name(&mut self, a: usize, shape: &'static Shape) -> String {
        if let Some(d) = self.defs.iter().find(|d| d.0 == a) {
            return d.1.clone();
        }
        let base = name_of(shape);
        let mut name = base.clone();
        let mut n = 2;
        while self.defs.iter().any(|d| d.1 == name) {
            name = format!("{base}{n}");
            n += 1;
        }
        self.defs.push((a, name.clone(), shape, false));
        name
    }

    fn inner(&mut self, shape: &'static Shape) -> J {
        match shape {
            Shape::Null => obj([("type", s("null"))]),
            Shape::Bool => obj([("type", s("boolean"))]),
            Shape::Integer { format, min, max } => {
                let mut m = vec![("type", s("integer")), ("format", s(format))];
                if let Some(v) = min {
                    m.push(("minimum", J::Num((*v).into())));
                }
                if let Some(v) = max {
                    m.push(("maximum", J::Num((*v).into())));
                }
                obj(m)
            }
            Shape::Number { format } => obj([("type", s("number")), ("format", s(format))]),
            Shape::String { format, len } => {
                let mut m = vec![("type", s("string"))];
                if let Some(f) = format {
                    m.push(("format", s(f)));
                }
                if let Some(n) = len {
                    m.push(("minLength", J::Num(n.to_string())));
                    m.push(("maxLength", J::Num(n.to_string())));
                }
                obj(m)
            }
            Shape::Option(f) => self.option(f()),
            Shape::Array { items, len, unique } => {
                let mut m = vec![("type", s("array"))];
                if *unique {
                    m.push(("uniqueItems", J::Bool(true)));
                }
                m.push(("items", self.shape(items())));
                if let Some(n) = len {
                    m.push(("minItems", J::Num(n.to_string())));
                    m.push(("maxItems", J::Num(n.to_string())));
                }
                obj(m)
            }
            Shape::Tuple(fs) => self.tuple(fs),
            Shape::Map(v) => obj([
                ("type", s("object")),
                ("additionalProperties", self.shape(v())),
            ]),
            Shape::Any => J::Bool(true),
            Shape::Struct(st) => self.strukt(st),
            Shape::Enum(e) => self.enumeration(e),
        }
    }

    fn option(&mut self, inner: &'static Shape) -> J {
        let referenced = matches!(inner, Shape::Struct(_) | Shape::Enum(_));
        let mut v = self.shape(inner);
        if !referenced && let J::Obj(m) = &mut v {
            match get(m, "type") {
                Some(J::Str(t)) => {
                    let t = J::Str(t.clone());
                    set(m, "type", J::Arr(vec![t, s("null")]));
                    return v;
                }
                Some(J::Arr(ts)) => {
                    let mut ts = ts.clone();
                    if !ts.contains(&s("null")) {
                        ts.push(s("null"));
                    }
                    set(m, "type", J::Arr(ts));
                    return v;
                }
                _ => {}
            }
        }
        obj([("anyOf", J::Arr(vec![v, obj([("type", s("null"))])]))])
    }

    fn tuple(&mut self, fs: &'static [super::ShapeFn]) -> J {
        let items = fs.iter().map(|f| self.shape(f())).collect();
        let n = J::Num(fs.len().to_string());
        obj([
            ("type", s("array")),
            ("prefixItems", J::Arr(items)),
            ("minItems", n.clone()),
            ("maxItems", n),
        ])
    }

    /// `properties` and `required` of a field list, flattening as serde does.
    fn fields(&mut self, fields: &'static [Field], all_default: bool) -> (Map, Vec<J>, Option<J>) {
        let mut props = Map::new();
        let mut required = Vec::new();
        let mut extra = None;
        for f in fields {
            let shape = f.shape();
            if f.flatten {
                let (inner, optional) = match shape {
                    Shape::Option(i) => (i(), true),
                    other => (other, false),
                };
                if let J::Obj(m) = self.shape(inner) {
                    if let Some(J::Obj(p)) = get(&m, "properties") {
                        props.extend(p.iter().cloned());
                    }
                    if !optional && let Some(J::Arr(r)) = get(&m, "required") {
                        required.extend(r.iter().cloned());
                    }
                    if let Some(ap) = get(&m, "additionalProperties") {
                        extra = Some(ap.clone());
                    }
                }
                continue;
            }
            let mut v = self.shape(shape);
            if let J::Obj(m) = &mut v {
                if let Some(d) = f.doc {
                    set(m, "description", s(d));
                }
                if let Some(fmt) = f.format {
                    set(m, "format", s(fmt));
                }
                if let Some(n) = f.min {
                    set(m, "minimum", J::Num(n.into()));
                }
                if let Some(n) = f.max {
                    set(m, "maximum", J::Num(n.into()));
                }
            }
            let optional = f.default || all_default || matches!(shape, Shape::Option(_));
            if !optional {
                required.push(s(f.name));
            }
            props.push((f.name.into(), v));
        }
        (props, required, extra)
    }

    fn object(&mut self, fields: &'static [Field], closed: bool, doc: Option<&str>) -> Map {
        let (props, required, extra) = self.fields(fields, false);
        let mut m = Map::new();
        if let Some(d) = doc {
            m.push(("description".into(), s(d)));
        }
        m.push(("type".into(), s("object")));
        m.push(("properties".into(), J::Obj(props)));
        if let Some(ap) = extra {
            m.push(("additionalProperties".into(), ap));
        } else if closed {
            m.push(("additionalProperties".into(), J::Bool(false)));
        }
        if !required.is_empty() {
            m.push(("required".into(), J::Arr(required)));
        }
        m
    }

    fn strukt(&mut self, st: &'static StructShape) -> J {
        J::Obj(self.object(st.fields, st.closed, st.doc))
    }

    fn enumeration(&mut self, e: &'static EnumShape) -> J {
        let mut out = match e.tagging {
            Tagging::External => self.external(e),
            Tagging::Untagged => {
                let vs = e.variants.iter().map(|v| self.payload(&v.kind)).collect();
                obj([("anyOf", J::Arr(vs))])
            }
            Tagging::Internal { tag } => self.tagged(e, tag, None),
            Tagging::Adjacent { tag, content } => self.tagged(e, tag, Some(content)),
        };
        if let (Some(d), J::Obj(m)) = (e.doc, &mut out) {
            m.insert(0, ("description".into(), s(d)));
        }
        out
    }

    /// The JSON of a variant's data alone.
    fn payload(&mut self, kind: &VariantKind) -> J {
        match kind {
            VariantKind::Unit => obj([("type", s("null"))]),
            VariantKind::Newtype(f) => self.shape(f()),
            VariantKind::Tuple(fs) => self.tuple(fs),
            VariantKind::Struct(fields) => J::Obj(self.object(fields, false, None)),
        }
    }

    fn external(&mut self, e: &'static EnumShape) -> J {
        let plain: Vec<J> = e
            .variants
            .iter()
            .filter(|v| matches!(v.kind, VariantKind::Unit) && v.doc.is_none())
            .map(|v| s(v.name))
            .collect();
        if plain.len() == e.variants.len() {
            return obj([("type", s("string")), ("enum", J::Arr(plain))]);
        }
        let mut one_of = Vec::new();
        if !plain.is_empty() {
            one_of.push(obj([("type", s("string")), ("enum", J::Arr(plain))]));
        }
        for v in e.variants {
            let mut m = match &v.kind {
                VariantKind::Unit if v.doc.is_none() => continue,
                VariantKind::Unit => {
                    vec![("type".into(), s("string")), ("const".into(), s(v.name))]
                }
                kind => {
                    let inner = self.payload(kind);
                    vec![
                        ("type".into(), s("object")),
                        ("properties".into(), J::Obj(vec![(v.name.into(), inner)])),
                        ("required".into(), J::Arr(vec![s(v.name)])),
                        ("additionalProperties".into(), J::Bool(false)),
                    ]
                }
            };
            if let Some(d) = v.doc {
                m.insert(0, ("description".into(), s(d)));
            }
            one_of.push(J::Obj(m));
        }
        obj([("oneOf", J::Arr(one_of))])
    }

    fn tagged(&mut self, e: &'static EnumShape, tag: &str, content: Option<&str>) -> J {
        let mut one_of = Vec::new();
        for v in e.variants {
            let tag_schema = obj([("type", s("string")), ("const", s(v.name))]);
            let mut props: Map = vec![(tag.into(), tag_schema)];
            let mut required = vec![s(tag)];
            match (&v.kind, content) {
                (VariantKind::Unit, _) => {}
                (kind, Some(c)) => {
                    props.push((c.into(), self.payload(kind)));
                    required.push(s(c));
                }
                (VariantKind::Struct(fields), None) => {
                    let (p, r, _) = self.fields(fields, false);
                    props.extend(p);
                    required.extend(r);
                }
                (VariantKind::Newtype(f), None) => {
                    if let J::Obj(m) = self.shape(f()) {
                        if let Some(J::Obj(p)) = get(&m, "properties") {
                            props.extend(p.iter().cloned());
                        }
                        if let Some(J::Arr(r)) = get(&m, "required") {
                            required.extend(r.iter().cloned());
                        }
                    }
                }
                // serde rejects internally tagged tuple variants at compile time
                (VariantKind::Tuple(_), None) => {}
            }
            let mut m: Map = Vec::new();
            if let Some(d) = v.doc {
                m.push(("description".into(), s(d)));
            }
            m.push(("type".into(), s("object")));
            m.push(("properties".into(), J::Obj(props)));
            m.push(("required".into(), J::Arr(required)));
            one_of.push(J::Obj(m));
        }
        obj([("oneOf", J::Arr(one_of))])
    }
}

/// Visit every subschema of one schema object (not property maps
/// themselves, so a field named `title` or `format` is never mistaken for a
/// keyword), parents before children.
fn walk(v: &mut J, f: &mut impl FnMut(&mut Map)) {
    let J::Obj(m) = v else { return };
    f(m);
    for (key, val) in m.iter_mut() {
        match (key.as_str(), val) {
            ("properties" | "$defs", J::Obj(p)) => p.iter_mut().for_each(|(_, x)| walk(x, f)),
            ("anyOf" | "oneOf" | "allOf" | "prefixItems", J::Arr(a)) => {
                a.iter_mut().for_each(|x| walk(x, f))
            }
            ("items" | "additionalProperties", x) => walk(x, f),
            _ => {}
        }
    }
}

pub(crate) fn build(base: &J, dialect: Dialect) -> Box<str> {
    let mut v = base.clone();
    let strict = |v: &mut J| {
        walk(v, &mut |m| {
            remove(m, "title");
            remove(m, "format");
            one_of_to_any_of(m);
            close_object(m);
        })
    };
    let v = match dialect {
        Dialect::Strict => {
            strict(&mut v);
            v
        }
        Dialect::AnthropicOutput => {
            walk(&mut v, &mut |m| {
                remove(m, "title");
                strip_unsupported_anthropic(m);
                one_of_to_any_of(m);
                close_object(m);
            });
            v
        }
        Dialect::AnthropicTool => {
            walk(&mut v, &mut |m| {
                remove(m, "title");
            });
            v
        }
        Dialect::GeminiJson => v,
    };
    let mut out = Vec::with_capacity(512);
    write(&v, &mut out);
    String::from_utf8(out)
        .expect("the writer emits utf-8")
        .into_boxed_str()
}

/// Strict modes: no extra keys, every property listed in `required`.
/// Optional fields stay optional through their nullable type.
fn close_object(m: &mut Map) {
    if get(m, "type") != Some(&s("object")) {
        return;
    }
    let Some(J::Obj(props)) = get(m, "properties") else {
        return;
    };
    let keys = props.iter().map(|(k, _)| s(k)).collect();
    set(m, "required", J::Arr(keys));
    set(m, "additionalProperties", J::Bool(false));
}

fn one_of_to_any_of(m: &mut Map) {
    if get(m, "anyOf").is_none()
        && let Some(i) = m.iter().position(|(k, _)| k == "oneOf")
    {
        m[i].0 = "anyOf".into();
    }
}

// keywords anthropic structured output rejects with a 400
fn strip_unsupported_anthropic(m: &mut Map) {
    for k in [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    ] {
        remove(m, k);
    }
    if let Some(J::Num(n)) = get(m, "minItems")
        && n.parse::<u64>().is_ok_and(|n| n > 1)
    {
        remove(m, "minItems");
    }
    const FORMATS: [&str; 10] = [
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    ];
    if let Some(J::Str(f)) = get(m, "format")
        && !FORMATS.contains(&f.as_str())
    {
        remove(m, "format");
    }
}
