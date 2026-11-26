//! Shapes of generic types, built once per instantiation.

use std::sync::{Mutex, OnceLock};

use super::{Shape, ShapeFn};

/// For generic types: one shape per instantiation, built on first use and
/// kept for the life of the process. `key` is the instantiation's
/// `<T as Schema>::schema` function address.
#[doc(hidden)]
pub fn intern(key: ShapeFn, build: impl FnOnce() -> Shape) -> &'static Shape {
    static MAP: OnceLock<Mutex<Vec<(usize, &'static Shape)>>> = OnceLock::new();
    let key = key as usize;
    let map = MAP.get_or_init(Default::default);
    if let Some(s) = lock(map).iter().find(|(k, _)| *k == key) {
        return s.1;
    }
    // built outside the lock: building never calls another `schema()`, but a
    // user impl might
    let shape: &'static Shape = Box::leak(Box::new(build()));
    let mut m = lock(map);
    if let Some(s) = m.iter().find(|(k, _)| *k == key) {
        return s.1;
    }
    m.push((key, shape));
    shape
}

/// Leak a slice for a generic type's interned shape.
#[doc(hidden)]
pub fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

pub(super) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // the data is a cache that stays consistent even if a holder panicked
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}
