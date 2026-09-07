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
    pub(crate) fn mint_fresh() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        Self(next_identity(&COUNTER))
    }
}

fn next_identity(counter: &AtomicUsize) -> usize {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current
            .checked_add(1)
            .expect("tag identity space exhausted");
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return current,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_tag_counter_never_wraps_or_reuses_an_identity() {
        let counter = AtomicUsize::new(usize::MAX - 1);
        assert_eq!(next_identity(&counter), usize::MAX - 1);
        for _ in 0..2 {
            assert!(std::panic::catch_unwind(|| next_identity(&counter)).is_err());
            assert_eq!(counter.load(Ordering::Relaxed), usize::MAX);
        }
    }
}
