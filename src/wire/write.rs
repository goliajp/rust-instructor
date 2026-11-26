//! Request bodies written member by member, without a JSON tree.

/// Writes one JSON object member by member.
pub(crate) struct Obj<'a> {
    out: &'a mut Vec<u8>,
    first: bool,
}

impl<'a> Obj<'a> {
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        out.push(b'{');
        Self { out, first: true }
    }

    /// Write `"key":` and return the buffer for the value.
    pub(crate) fn key(&mut self, key: &str) -> &mut Vec<u8> {
        if !self.first {
            self.out.push(b',');
        }
        self.first = false;
        crate::json::write_str(self.out, key);
        self.out.push(b':');
        self.out
    }

    pub(crate) fn str(&mut self, key: &str, v: &str) {
        crate::json::write_str(self.key(key), v);
    }

    pub(crate) fn raw(&mut self, key: &str, json: &str) {
        self.key(key).extend_from_slice(json.as_bytes());
    }

    pub(crate) fn u64(&mut self, key: &str, v: u64) {
        crate::json::write_u64(self.key(key), v);
    }

    pub(crate) fn f64(&mut self, key: &str, v: f64) {
        crate::json::write_f64(self.key(key), v);
    }

    pub(crate) fn bool(&mut self, key: &str, v: bool) {
        self.raw(key, if v { "true" } else { "false" });
    }

    pub(crate) fn end(self) {
        self.out.push(b'}');
    }
}

/// Writes one JSON array item by item.
pub(crate) struct Arr<'a> {
    out: &'a mut Vec<u8>,
    first: bool,
}

impl<'a> Arr<'a> {
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        out.push(b'[');
        Self { out, first: true }
    }

    pub(crate) fn item(&mut self) -> &mut Vec<u8> {
        if !self.first {
            self.out.push(b',');
        }
        self.first = false;
        self.out
    }

    pub(crate) fn end(self) {
        self.out.push(b']');
    }
}

/// `{"type": t}` objects and the like, written inline.
pub(crate) fn type_obj(out: &mut Vec<u8>, t: &str) {
    let mut o = Obj::new(out);
    o.str("type", t);
    o.end();
}
