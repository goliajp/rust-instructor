//! `Schema` for std types, with the formats and bounds schemars uses.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, LinkedList, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};

use super::{Schema, Shape, ShapeFn, intern, leak};

macro_rules! fixed {
    ($($t:ty => $shape:expr;)*) => {$(
        impl Schema for $t {
            fn schema() -> &'static Shape {
                static S: Shape = $shape;
                &S
            }
        }
    )*};
}

const fn int(format: &'static str, min: Option<&'static str>, max: Option<&'static str>) -> Shape {
    Shape::Integer { format, min, max }
}

fixed! {
    () => Shape::Null;
    bool => Shape::Bool;
    u8 => int("uint8", Some("0"), Some("255"));
    u16 => int("uint16", Some("0"), Some("65535"));
    u32 => int("uint32", Some("0"), None);
    u64 => int("uint64", Some("0"), None);
    u128 => int("uint128", Some("0"), None);
    usize => int("uint", Some("0"), None);
    i8 => int("int8", Some("-128"), Some("127"));
    i16 => int("int16", Some("-32768"), Some("32767"));
    i32 => int("int32", None, None);
    i64 => int("int64", None, None);
    i128 => int("int128", None, None);
    isize => int("int", None, None);
    std::num::NonZeroU8 => int("uint8", Some("1"), Some("255"));
    std::num::NonZeroU16 => int("uint16", Some("1"), Some("65535"));
    std::num::NonZeroU32 => int("uint32", Some("1"), None);
    std::num::NonZeroU64 => int("uint64", Some("1"), None);
    std::num::NonZeroUsize => int("uint", Some("1"), None);
    f32 => Shape::Number { format: "float" };
    f64 => Shape::Number { format: "double" };
    String => Shape::String { format: None, len: None };
    str => Shape::String { format: None, len: None };
    char => Shape::String { format: None, len: Some(1) };
    std::path::PathBuf => Shape::String { format: None, len: None };
    std::net::IpAddr => Shape::String { format: Some("ip"), len: None };
    std::net::Ipv4Addr => Shape::String { format: Some("ipv4"), len: None };
    std::net::Ipv6Addr => Shape::String { format: Some("ipv6"), len: None };
}

macro_rules! same_as {
    ($($t:ident),*) => {$(
        impl<T: Schema + ?Sized> Schema for $t<T> {
            fn schema() -> &'static Shape {
                T::schema()
            }
        }
    )*};
}

same_as!(Box, Rc, Arc, Cell, RefCell, Mutex, RwLock);

impl<T: Schema + ?Sized> Schema for &T {
    fn schema() -> &'static Shape {
        T::schema()
    }
}

impl<T: Schema + ?Sized + ToOwned> Schema for Cow<'_, T> {
    fn schema() -> &'static Shape {
        T::schema()
    }
}

impl<T: Schema> Schema for Option<T> {
    fn schema() -> &'static Shape {
        intern(Self::schema, || Shape::Option(T::schema))
    }
}

macro_rules! list {
    ($($t:ident: $unique:expr),*) => {$(
        impl<T: Schema> Schema for $t<T> {
            fn schema() -> &'static Shape {
                intern(Self::schema, || Shape::Array { items: T::schema, len: None, unique: $unique })
            }
        }
    )*};
}

list!(Vec: false, VecDeque: false, LinkedList: false, BTreeSet: true);

impl<T: Schema, S> Schema for HashSet<T, S> {
    fn schema() -> &'static Shape {
        intern(Self::schema, || Shape::Array {
            items: T::schema,
            len: None,
            unique: true,
        })
    }
}

impl<T: Schema> Schema for [T] {
    fn schema() -> &'static Shape {
        Vec::<T>::schema()
    }
}

impl<T: Schema, const N: usize> Schema for [T; N] {
    fn schema() -> &'static Shape {
        intern(Self::schema, || Shape::Array {
            items: T::schema,
            len: Some(u32::try_from(N).unwrap_or(u32::MAX)),
            unique: false,
        })
    }
}

impl<K, V: Schema> Schema for BTreeMap<K, V> {
    fn schema() -> &'static Shape {
        intern(Self::schema, || Shape::Map(V::schema))
    }
}

impl<K, V: Schema, S> Schema for HashMap<K, V, S> {
    fn schema() -> &'static Shape {
        intern(Self::schema, || Shape::Map(V::schema))
    }
}

macro_rules! tuple {
    ($($n:ident),+) => {
        impl<$($n: Schema),+> Schema for ($($n,)+) {
            fn schema() -> &'static Shape {
                intern(Self::schema, || {
                    let fs: Vec<ShapeFn> = vec![$($n::schema),+];
                    Shape::Tuple(leak(fs))
                })
            }
        }
    };
}

tuple!(A);
tuple!(A, B);
tuple!(A, B, C);
tuple!(A, B, C, D);
tuple!(A, B, C, D, E);
tuple!(A, B, C, D, E, F);
