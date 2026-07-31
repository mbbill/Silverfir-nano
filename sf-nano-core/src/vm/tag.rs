//! Runtime tag identities for WebAssembly exception handling.
//!
//! Tags compare by handle identity, not by signature — two tags declared with
//! the same type in different modules must not alias.

use core::sync::atomic::{AtomicUsize, Ordering};

/// Opaque, process-globally-unique tag identity. Zero is reserved as an
/// invalid/null sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagIdentity(usize);

impl TagIdentity {
    #[inline]
    pub fn mint_fresh() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        let raw = COUNTER.fetch_add(1, Ordering::Relaxed);
        debug_assert!(raw != 0, "TagIdentity counter overflowed to zero sentinel");
        Self(raw)
    }
}
