//! The canonical schema compiled once into nodes the guided reader walks:
//! references resolved to node ids, keywords read into fields, so reading
//! an answer never searches the schema tree.

use super::super::check::{Bounds, bounds};
use super::super::strict::{fills, is_map, object_root};
use super::super::tree::{J, Map, get};

pub(super) type Id = u32;

/// A node that takes any value as written.
pub(super) const ANY: Id = 0;

pub(super) const OBJECT: u8 = 1;
pub(super) const ARRAY: u8 = 2;
pub(super) const STRING: u8 = 4;
pub(super) const BOOLEAN: u8 = 8;
pub(super) const NUMBER: u8 = 16;
pub(super) const INTEGER: u8 = 32;
pub(super) const NULL: u8 = 64;

pub(crate) struct Plan {
    pub(super) nodes: Vec<Node>,
    /// Per node: nothing in its value is reshaped or checked, so the
    /// caller's type reads it as written.
    pub(super) plain: Vec<bool>,
    pub(super) root: Id,
    /// The list node of a list schema (`{"items": [T]}`) and its item node.
    pub(super) list: Option<Id>,
    pub(super) items: Option<Id>,
    pub(super) object_root: bool,
}

pub(super) enum Node {
    Any,
    Fail(&'static str),
    /// The first alternative that takes the value. `option` is set when
    /// the second alternative takes only `null` and the first never does,
    /// so the choice can be made from the first byte.
    AnyOf {
        alts: Box<[Id]>,
        option: Option<Id>,
    },
    Typed(Box<Typed>),
}

pub(super) struct Typed {
    /// One bit per type keyword value.
    pub(super) types: u8,
    /// A `type` keyword is present.
    pub(super) typed: bool,
    /// The `type` keyword when it is a single name, for messages.
    pub(super) name: Option<Box<str>>,
    pub(super) constant: Option<J>,
    pub(super) choices: Option<Vec<J>>,
    pub(super) allowed: bool,
    /// A map: the node of its values.
    pub(super) map: Option<Id>,
    pub(super) prefix: Option<Box<[Id]>>,
    pub(super) props: Box<[(Box<str>, Id)]>,
    pub(super) extra: Extra,
    pub(super) required: Box<[Box<str>]>,
    /// Bits of the required properties among the first 64.
    pub(super) required_mask: u64,
    /// Some required name is not a property or sits past the 64th, so
    /// keys are tracked by name.
    pub(super) by_name: bool,
    pub(super) items: Id,
    /// Required recursive lists and maps that the Gemini schema leaves
    /// optional, by property bit, and which of them are maps.
    pub(super) fill: u64,
    pub(super) fill_map: u64,
    pub(super) bounds: Bounds,
    /// Numeric or length bounds to check on a scalar.
    pub(super) scalar_bounds: bool,
    pub(super) unique: bool,
}

/// What happens to a key that is not a property.
pub(super) enum Extra {
    Deny,
    Node(Id),
    Ignore,
}

impl Typed {
    #[inline]
    pub(super) fn wants(&self, t: u8) -> bool {
        !self.typed || self.types & t != 0
    }
}

impl Plan {
    pub(crate) fn new(base: &J) -> Self {
        let mut c = Compiler {
            base,
            nodes: vec![Node::Any],
            memo: Vec::new(),
        };
        let root = c.reserve(base);
        c.nodes[root as usize] = c.build(base);
        c.options();
        let plain = c.plain();
        let list = match &c.nodes[root as usize] {
            Node::Typed(t) => t.props.iter().find(|(k, _)| &**k == "items").map(|p| p.1),
            _ => None,
        };
        let items = list.and_then(|l| match &c.nodes[l as usize] {
            Node::Typed(t) => Some(t.items),
            _ => None,
        });
        Plan {
            nodes: c.nodes,
            plain,
            root,
            list,
            items,
            object_root: object_root(base),
        }
    }

    #[inline]
    pub(super) fn node(&self, id: Id) -> &Node {
        &self.nodes[id as usize]
    }

    #[inline]
    pub(super) fn is_plain(&self, id: Id) -> bool {
        self.plain[id as usize]
    }
}

struct Compiler<'a> {
    base: &'a J,
    nodes: Vec<Node>,
    // referenced schemas already given a node, by address
    memo: Vec<(*const J, Id)>,
}

fn type_bit(t: &str) -> u8 {
    match t {
        "object" => OBJECT,
        "array" => ARRAY,
        "string" => STRING,
        "boolean" => BOOLEAN,
        "number" => NUMBER,
        "integer" => INTEGER,
        "null" => NULL,
        _ => 0,
    }
}

impl<'a> Compiler<'a> {
    fn reserve(&mut self, target: &J) -> Id {
        let id = self.nodes.len() as Id;
        self.nodes.push(Node::Any);
        self.memo.push((target as *const J, id));
        id
    }

    fn resolve(&self, r: &str) -> Option<&'a J> {
        if r == "#" {
            return Some(self.base);
        }
        let name = r.strip_prefix("#/$defs/")?;
        match self.base {
            J::Obj(m) => match get(m, "$defs") {
                Some(J::Obj(d)) => get(d, name),
                _ => None,
            },
            _ => None,
        }
    }

    fn compile(&mut self, j: &'a J) -> Id {
        let J::Obj(m) = j else { return ANY };
        if let Some(J::Str(r)) = get(m, "$ref") {
            let Some(target) = self.resolve(r) else {
                return self.push(Node::Fail("unknown $ref"));
            };
            if let Some(&(_, id)) = self.memo.iter().find(|(p, _)| std::ptr::eq(*p, target)) {
                return id;
            }
            let id = self.reserve(target);
            self.nodes[id as usize] = self.build(target);
            return id;
        }
        let node = self.build(j);
        self.push(node)
    }

    fn push(&mut self, node: Node) -> Id {
        self.nodes.push(node);
        (self.nodes.len() - 1) as Id
    }

    fn build(&mut self, j: &'a J) -> Node {
        let J::Obj(m) = j else { return Node::Any };
        if get(m, "$ref").is_some() {
            // a reference to a reference
            let id = self.compile(j);
            return Node::AnyOf {
                alts: Box::new([id]),
                option: None,
            };
        }
        if let Some(J::Arr(alts)) = get(m, "anyOf").or_else(|| get(m, "oneOf")) {
            let alts = alts.iter().map(|a| self.compile(a)).collect();
            return Node::AnyOf { alts, option: None };
        }
        Node::Typed(Box::new(self.typed(m)))
    }

    fn typed(&mut self, m: &'a Map) -> Typed {
        let (types, name) = match get(m, "type") {
            Some(J::Str(t)) => (type_bit(t), Some(t.as_str().into())),
            Some(J::Arr(ts)) => (
                ts.iter()
                    .map(|t| match t {
                        J::Str(t) => type_bit(t),
                        _ => 0,
                    })
                    .fold(0, |a, b| a | b),
                None,
            ),
            _ => (0, None),
        };
        let typed = get(m, "type").is_some();
        let map = is_map(m).then(|| match get(m, "additionalProperties") {
            Some(v @ J::Obj(_)) => self.compile(v),
            _ => ANY,
        });
        let prefix = match get(m, "prefixItems") {
            Some(J::Arr(p)) => Some(p.iter().map(|x| self.compile(x)).collect()),
            _ => None,
        };
        let props: Box<[(Box<str>, Id)]> = match get(m, "properties") {
            Some(J::Obj(p)) => p
                .iter()
                .map(|(k, v)| (k.as_str().into(), self.compile(v)))
                .collect(),
            _ => Box::new([]),
        };
        let extra = match get(m, "additionalProperties") {
            Some(J::Bool(false)) => Extra::Deny,
            Some(v @ J::Obj(_)) => Extra::Node(self.compile(v)),
            _ => Extra::Ignore,
        };
        let required: Box<[Box<str>]> = match get(m, "required") {
            Some(J::Arr(r)) => r
                .iter()
                .filter_map(|n| match n {
                    J::Str(n) => Some(n.as_str().into()),
                    _ => None,
                })
                .collect(),
            _ => Box::new([]),
        };
        let mut required_mask = 0u64;
        let mut by_name = false;
        for r in &required {
            match props.iter().position(|(k, _)| k == r) {
                Some(i) if i < 64 => required_mask |= 1 << i,
                _ => by_name = true,
            }
        }
        let items = match get(m, "items") {
            Some(v) => self.compile(v),
            None => ANY,
        };
        let (mut fill, mut fill_map) = (0u64, 0u64);
        for name in fills(m) {
            if let Some(i) = props
                .iter()
                .position(|(k, _)| &**k == name)
                .filter(|&i| i < 64)
            {
                fill |= 1 << i;
                if let Some(J::Obj(p)) = get(m, "properties")
                    && let Some(J::Obj(f)) = get(p, name)
                    && is_map(f)
                {
                    fill_map |= 1 << i;
                }
            }
        }
        let b = bounds(m);
        Typed {
            types,
            typed,
            name,
            constant: get(m, "const").cloned(),
            choices: match get(m, "enum") {
                Some(J::Arr(e)) => Some(e.clone()),
                _ => None,
            },
            allowed: get(m, "const").is_some() || get(m, "enum").is_some(),
            map,
            prefix,
            props,
            extra,
            required,
            required_mask,
            by_name,
            items,
            fill,
            fill_map,
            scalar_bounds: !b.is_empty(),
            bounds: b,
            unique: get(m, "uniqueItems") == Some(&J::Bool(true)),
        }
    }

    /// Mark each two-way choice between a type that never takes `null` and
    /// one that takes only `null`.
    fn options(&mut self) {
        for i in 0..self.nodes.len() {
            let Node::AnyOf { alts, .. } = &self.nodes[i] else {
                continue;
            };
            let [first, second] = alts[..] else { continue };
            let rejects_null =
                matches!(self.target(first), Node::Typed(t) if t.typed && t.types & NULL == 0);
            let only_null = matches!(self.target(second), Node::Typed(t)
                if t.typed && t.types == NULL && t.prefix.is_none());
            if rejects_null && only_null {
                let Node::AnyOf { option, .. } = &mut self.nodes[i] else {
                    unreachable!()
                };
                *option = Some(first);
            }
        }
    }

    /// Which nodes read as written: no choice, reshape or check in them
    /// or below. Starts from all and drops nodes until nothing changes,
    /// so a cycle of such nodes stays plain.
    fn plain(&self) -> Vec<bool> {
        let mut plain = vec![true; self.nodes.len()];
        loop {
            let mut changed = false;
            for (i, n) in self.nodes.iter().enumerate() {
                if !plain[i] {
                    continue;
                }
                let keep = match n {
                    Node::Any => true,
                    Node::Fail(_) | Node::AnyOf { .. } => false,
                    Node::Typed(t) => {
                        t.map.is_none()
                            && t.prefix.is_none()
                            && t.fill == 0
                            && !t.allowed
                            && !t.unique
                            && t.bounds.is_none()
                            && plain[t.items as usize]
                            && t.props.iter().all(|(_, c)| plain[*c as usize])
                            && match t.extra {
                                Extra::Node(c) => plain[c as usize],
                                _ => true,
                            }
                    }
                };
                if !keep {
                    plain[i] = false;
                    changed = true;
                }
            }
            if !changed {
                return plain;
            }
        }
    }

    fn target(&self, id: Id) -> &Node {
        &self.nodes[id as usize]
    }
}
