//! Lightweight scratch register pool with round-robin allocation and
//! scoped RAII guards.
//!
//! The pool owns the physical register table. Callers never see the raw
//! table — they get registers only through `ScratchGuard`, which frees
//! the slot on drop.
//!
//! Uses `Cell` for interior mutability so that `alloc`/`free` take `&self`,
//! avoiding borrow conflicts when the pool is a sibling field of the emitter.

use core::cell::Cell;

/// A pool of `N` scratch registers of type `R`.
///
/// `R` is the physical register type (e.g. `Arm64Reg` or `u32` for FP D-regs).
/// The pool hands out registers via `scoped_alloc()` which returns a guard
/// that frees the slot on drop.
///
/// Round-robin allocation distributes scratch usage across registers to
/// reduce WAW false-dependency chains in tight loops.
pub(crate) struct ScratchPool<R: Copy, const N: usize> {
    regs: [R; N],
    in_use: Cell<u8>,
    next: Cell<u8>,
}

impl<R: Copy, const N: usize> ScratchPool<R, N> {
    const _ASSERT_CAPACITY: () = assert!(N <= 8, "ScratchPool supports at most 8 slots");

    #[inline]
    pub const fn new(regs: [R; N]) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::_ASSERT_CAPACITY;
        Self {
            regs,
            in_use: Cell::new(0),
            next: Cell::new(0),
        }
    }

    /// Allocate a scratch register and return a guard that frees it on drop.
    ///
    /// The guard dereferences to `R`, so you can use `*guard` anywhere a
    /// physical register is needed.
    ///
    /// ```ignore
    /// let s = pool.scoped_alloc();
    /// text.emit_u32(enc::mov(*s, ...));
    /// // s freed here on drop
    /// ```
    #[inline]
    pub fn scoped_alloc(&self) -> ScratchGuard<'_, R, N> {
        let idx = self.alloc_idx();
        ScratchGuard {
            pool: self,
            idx,
            reg: self.regs[idx],
        }
    }

    /// Raw index allocation. Prefer `scoped_alloc()` for automatic cleanup.
    #[inline]
    pub fn alloc_idx(&self) -> usize {
        let bits = self.in_use.get();
        let start = self.next.get() as usize;
        for i in 0..N {
            let idx = (start + i) % N;
            if bits & (1 << idx) == 0 {
                self.in_use.set(bits | (1 << idx));
                self.next.set(((idx + 1) % N) as u8);
                return idx;
            }
        }
        panic!(
            "scratch pool exhausted (capacity {}, in_use 0b{:0width$b})",
            N,
            bits,
            width = N
        );
    }

    /// Return a scratch slot to the pool by index.
    #[inline]
    pub fn free_idx(&self, idx: usize) {
        debug_assert!(idx < N, "scratch index {} out of range (capacity {})", idx, N);
        let bits = self.in_use.get();
        debug_assert!(
            bits & (1 << idx) != 0,
            "double-free of scratch index {}",
            idx
        );
        self.in_use.set(bits & !(1 << idx));
    }

    /// Free all scratch slots. Preserves the round-robin cursor.
    #[inline]
    pub fn free_all(&self) {
        self.in_use.set(0);
    }

    /// Assert all slots are free. Debug check at instruction boundaries.
    #[inline]
    pub fn assert_all_free(&self) {
        debug_assert_eq!(
            self.in_use.get(),
            0,
            "scratch pool leak: in_use 0b{:0width$b}",
            self.in_use.get(),
            width = N
        );
    }

    /// Get the physical register for a given pool index.
    /// Use only for well-known dedicated slots (e.g. parallel-move temp).
    #[inline]
    pub fn reg_at(&self, idx: usize) -> R {
        self.regs[idx]
    }
}

impl<R: Copy, const N: usize> core::fmt::Debug for ScratchPool<R, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ScratchPool")
            .field("capacity", &N)
            .field("in_use", &format_args!("0b{:0width$b}", self.in_use.get(), width = N))
            .field("next", &self.next.get())
            .finish()
    }
}

// ── ScratchGuard ─────────────────────────────────────────────────────────────

/// RAII guard that frees a scratch slot when dropped.
///
/// Dereferences to the physical register `R`, so `*guard` gives the register
/// for use in instruction encoding.
pub(crate) struct ScratchGuard<'a, R: Copy, const N: usize> {
    pool: &'a ScratchPool<R, N>,
    idx: usize,
    reg: R,
}

impl<R: Copy, const N: usize> ScratchGuard<'_, R, N> {
    /// Explicitly free the pool slot and return the physical register value.
    ///
    /// Use this at the point where the scratch register's last read occurs.
    /// The returned `R` is a `Copy` value — safe to keep for later patching
    /// (which only needs the register identity, not a live hardware register).
    ///
    /// Consumes the guard so it cannot be used after release. If you forget
    /// to call this, `Drop` frees the slot at scope end as a safety net.
    #[inline]
    pub fn release(self) -> R {
        let reg = self.reg;
        // Free the slot now, then suppress the Drop.
        self.pool.free_idx(self.idx);
        core::mem::forget(self);
        reg
    }
}

impl<R: Copy, const N: usize> core::ops::Deref for ScratchGuard<'_, R, N> {
    type Target = R;
    #[inline]
    fn deref(&self) -> &R {
        &self.reg
    }
}

impl<R: Copy, const N: usize> Drop for ScratchGuard<'_, R, N> {
    #[inline]
    fn drop(&mut self) {
        self.pool.free_idx(self.idx);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_two_slots() {
        let pool = ScratchPool::new([10u32, 20]);
        let s = pool.scoped_alloc();
        assert_eq!(*s, 10);
        drop(s);
        let s = pool.scoped_alloc();
        assert_eq!(*s, 20); // cursor advanced
        drop(s);
        let s = pool.scoped_alloc();
        assert_eq!(*s, 10); // wraps
        drop(s);
    }

    #[test]
    fn simultaneous_alloc_gives_different_regs() {
        let pool = ScratchPool::new([10u32, 20]);
        let a = pool.scoped_alloc();
        let b = pool.scoped_alloc();
        assert_ne!(*a, *b);
    }

    #[test]
    #[should_panic(expected = "scratch pool exhausted")]
    fn exhaustion_panics() {
        let pool = ScratchPool::new([10u32, 20]);
        let _a = pool.scoped_alloc();
        let _b = pool.scoped_alloc();
        let _c = pool.scoped_alloc(); // panics
    }

    #[test]
    fn scoped_guard_frees_on_drop() {
        let pool = ScratchPool::new([10u32, 20]);
        {
            let _s = pool.scoped_alloc();
            let _t = pool.scoped_alloc();
            // both live — pool full
        }
        // both dropped — pool empty
        pool.assert_all_free();
    }

    #[test]
    fn free_all_preserves_cursor() {
        let pool = ScratchPool::new([10u32, 20, 30]);
        let _ = pool.scoped_alloc(); // gets 10
        pool.free_all();
        let s = pool.scoped_alloc(); // cursor at 1 → gets 20
        assert_eq!(*s, 20);
        pool.free_all();
        let s = pool.scoped_alloc(); // cursor at 2 → gets 30
        assert_eq!(*s, 30);
    }

    #[test]
    fn release_frees_and_returns_reg() {
        let pool = ScratchPool::new([10u32, 20]);
        let s = pool.scoped_alloc();
        assert_eq!(*s, 10);
        let reg = s.release(); // explicitly free
        assert_eq!(reg, 10);
        pool.assert_all_free(); // slot is freed
        // reg is still usable as a Copy value
        assert_eq!(reg, 10);
    }

    #[test]
    fn release_then_realloc() {
        let pool = ScratchPool::new([10u32, 20]);
        let s = pool.scoped_alloc(); // gets 10
        let _ = s.release();
        // Slot freed, cursor at 1. Next alloc gives 20.
        let s = pool.scoped_alloc();
        assert_eq!(*s, 20);
    }
}
