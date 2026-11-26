//! Typed snapshots of a value that is still streaming.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, LinkedList, VecDeque};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use serde::de::{
    Deserialize, DeserializeOwned, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor,
};

/// A type that can be shown while its JSON is still arriving.
///
/// `Partial` is the type of the value so far. `#[derive(Schema)]` with
/// `#[schema(partial)]` implements it and generates that type: a struct
/// named after yours with `Partial` appended, every field an `Option`. The
/// std types have it too: a `String` holds the characters received so far,
/// a `Vec` the items so far, a map the entries whose value is usable, and
/// numbers, `bool`, `char`, tuples, sets and enums appear only once they
/// are complete.
///
/// A type the derive does not cover can implement it in one line, as a
/// value that appears only once complete:
///
/// ```
/// use instructors::partial::Partial;
/// # #[derive(serde::Deserialize)] struct Timestamp(String);
/// impl Partial for Timestamp {
///     type Partial = Self;
///     fn read<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Self>, D::Error> {
///         instructors::partial::whole(d)
///     }
/// }
/// ```
pub trait Partial {
    /// The value as far as it has arrived.
    type Partial: DeserializeOwned;

    /// Read the value so far from a partial view; `None` when nothing
    /// usable has arrived yet. A field that does not fit yet gives `None`
    /// rather than an error, which is safe because the view is a tree:
    /// a failed read consumes nothing.
    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self::Partial>, D::Error> {
        Ok(Self::Partial::deserialize(d).ok())
    }
}

/// The partial type of `T`.
pub type PartialOf<T> = <T as Partial>::Partial;

/// Asks a partial view whether a value is complete.
pub(crate) const WHOLE_TOKEN: &str = "$instructors::private::Whole";

/// A [`Partial::read`] for a value that is shown only once it is complete.
pub fn whole<'de, D: Deserializer<'de>, T: DeserializeOwned>(d: D) -> Result<Option<T>, D::Error> {
    Ok(
        d.deserialize_newtype_struct(WHOLE_TOKEN, Whole(PhantomData))
            .ok(),
    )
}

struct Whole<T>(PhantomData<T>);

impl<'de, T: DeserializeOwned> Visitor<'de> for Whole<T> {
    type Value = T;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a complete value")
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<T, D::Error> {
        T::deserialize(d)
    }
}

macro_rules! whole {
    ($($t:ty),*) => {$(
        impl Partial for $t {
            type Partial = Self;
            fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self>, D::Error> {
                whole(d)
            }
        }
    )*};
}

whole!(
    (),
    bool,
    char,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    std::num::NonZeroU8,
    std::num::NonZeroU16,
    std::num::NonZeroU32,
    std::num::NonZeroU64,
    std::num::NonZeroUsize,
    std::path::PathBuf,
    std::net::IpAddr,
    std::net::Ipv4Addr,
    std::net::Ipv6Addr
);

impl Partial for String {
    type Partial = String;
}

impl<T: Partial> Partial for Option<T> {
    type Partial = T::Partial;

    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<T::Partial>, D::Error> {
        struct V<T>(PhantomData<T>);
        impl<'de, T: Partial> Visitor<'de> for V<T> {
            type Value = Option<T::Partial>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an optional value")
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
                T::read(d)
            }
        }
        Ok(d.deserialize_option(V::<T>(PhantomData)).ok().flatten())
    }
}

impl<T: Partial> Partial for Box<T> {
    type Partial = Box<T::Partial>;

    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self::Partial>, D::Error> {
        T::read(d).map(|v| v.map(Box::new))
    }
}

/// Reads one item or value through [`Partial::read`].
struct Read<T>(PhantomData<T>);

impl<'de, T: Partial> DeserializeSeed<'de> for Read<T> {
    type Value = Option<T::Partial>;

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        T::read(d)
    }
}

/// The items so far; an item with nothing usable yet is left out.
struct Items<T, C>(PhantomData<(T, C)>);

impl<'de, T: Partial, C: Default + Extend<T::Partial>> Visitor<'de> for Items<T, C> {
    type Value = C;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a list")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<C, A::Error> {
        let mut out = C::default();
        while let Some(item) = a.next_element_seed(Read::<T>(PhantomData))? {
            out.extend(item);
        }
        Ok(out)
    }
}

macro_rules! list {
    ($($c:ident),*) => {$(
        impl<T: Partial> Partial for $c<T> {
            type Partial = $c<T::Partial>;
            fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self::Partial>, D::Error> {
                Ok(d.deserialize_seq(Items::<T, $c<T::Partial>>(PhantomData)).ok())
            }
        }
    )*};
}

list!(Vec, VecDeque, LinkedList);

impl<T: DeserializeOwned + Ord> Partial for BTreeSet<T> {
    type Partial = Self;
    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self>, D::Error> {
        whole(d)
    }
}

impl<T: DeserializeOwned + Eq + Hash, S: BuildHasher + Default> Partial for HashSet<T, S> {
    type Partial = Self;
    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self>, D::Error> {
        whole(d)
    }
}

/// The entries so far whose value has something usable.
struct Entries<K, V, M>(PhantomData<(K, V, M)>);

impl<'de, K, V, M> Visitor<'de> for Entries<K, V, M>
where
    K: DeserializeOwned,
    V: Partial,
    M: Default + Extend<(K, V::Partial)>,
{
    type Value = M;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a map")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<M, A::Error> {
        let mut out = M::default();
        while let Some(k) = a.next_key::<K>()? {
            if let Some(v) = a.next_value_seed(Read::<V>(PhantomData))? {
                out.extend([(k, v)]);
            }
        }
        Ok(out)
    }
}

impl<K: DeserializeOwned + Ord, V: Partial> Partial for BTreeMap<K, V> {
    type Partial = BTreeMap<K, V::Partial>;

    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self::Partial>, D::Error> {
        Ok(
            d.deserialize_map(Entries::<K, V, Self::Partial>(PhantomData))
                .ok(),
        )
    }
}

impl<K, V, S> Partial for HashMap<K, V, S>
where
    K: DeserializeOwned + Eq + Hash,
    V: Partial,
    S: BuildHasher + Default,
{
    type Partial = HashMap<K, V::Partial, S>;

    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self::Partial>, D::Error> {
        Ok(
            d.deserialize_map(Entries::<K, V, Self::Partial>(PhantomData))
                .ok(),
        )
    }
}

macro_rules! tuple {
    ($($n:ident),+) => {
        impl<$($n: DeserializeOwned),+> Partial for ($($n,)+) {
            type Partial = Self;
            fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self>, D::Error> {
                whole(d)
            }
        }
    };
}

tuple!(A);
tuple!(A, B);
tuple!(A, B, C);
tuple!(A, B, C, D0);
tuple!(A, B, C, D0, E);
tuple!(A, B, C, D0, E, F);
tuple!(A, B, C, D0, E, F, G);
tuple!(A, B, C, D0, E, F, G, H);

impl<T, const N: usize> Partial for [T; N]
where
    [T; N]: DeserializeOwned,
{
    type Partial = Self;
    fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Self>, D::Error> {
        whole(d)
    }
}

#[cfg(feature = "serde_json")]
impl Partial for serde_json::Value {
    type Partial = Self;
}
