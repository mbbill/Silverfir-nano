//! Rotating scratch register pool with explicit ownership.
//!
//! Uses interior mutability (`Cell`) so that `scoped_alloc` takes `&self`,
//! avoiding borrow conflicts when the backend needs `&mut self` for emission
//! while scratch guards are alive.

use core::cell::Cell;
use core::fmt;
use core::ops::Deref;
use core::marker::PhantomData;

/// A pool of `N` scratch registers of type `R`.
///
/// Allocation rotates: even with alloc/free/alloc/free, consecutive allocs
/// return different registers.
pub(crate) struct ScratchPool<R: Copy, const N: usize> {
    regs: [R; N],
    in_use: Cell<u8>,
    cursor: Cell<u8>,
}

impl<R: Copy + fmt::Debug, const N: usize> fmt::Debug for ScratchPool<R, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScratchPool")
            .field("regs", &self.regs)
            .field("in_use", &self.in_use.get())
            .field("cursor", &self.cursor.get())
            .finish()
    }
}

impl<R: Copy, const N: usize> ScratchPool<R, N> {
    const _ASSERT_SIZE: () = assert!(N <= 8, "ScratchPool supports at most 8 registers");

    pub(crate) fn new(regs: [R; N]) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::_ASSERT_SIZE;
        Self {
            regs,
            in_use: Cell::new(0),
            cursor: Cell::new(0),
        }
    }

    /// Allocate a scratch register, returning an RAII guard.
    ///
    /// Takes `&self` (not `&mut self`) thanks to interior mutability.
    /// Panics if all registers are in use.
    pub(crate) fn scoped_alloc(&self) -> ScratchGuard<'_, R, N> {
        let mask = self.in_use.get();
        let mut cursor = self.cursor.get() as usize;
        for _ in 0..N {
            let idx = cursor % N;
            cursor += 1;
            if mask & (1 << idx) == 0 {
                self.in_use.set(mask | (1 << idx));
                self.cursor.set(cursor as u8);
                return ScratchGuard {
                    pool: self,
                    idx: idx as u8,
                    reg: self.regs[idx],
                };
            }
        }
        panic!("ScratchPool: all {} registers are in use", N);
    }

    /// Allocate a scratch register without RAII, returning its pool index.
    /// The caller must later call `free_index()` with the same index.
    ///
    /// Use this only for protocol-scoped allocations (e.g. cycle-break temps)
    /// where the lifetime spans multiple trait method calls and RAII guards
    /// cannot cross the borrow boundary. Prefer `scoped_alloc()` everywhere else.
    pub(crate) fn alloc(&self) -> u8 {
        let mask = self.in_use.get();
        let mut cursor = self.cursor.get() as usize;
        for _ in 0..N {
            let idx = cursor % N;
            cursor += 1;
            if mask & (1 << idx) == 0 {
                self.in_use.set(mask | (1 << idx));
                self.cursor.set(cursor as u8);
                return idx as u8;
            }
        }
        panic!("ScratchPool: all {} registers are in use", N);
    }

    /// Free a scratch register allocated by `alloc()`.
    pub(crate) fn free_index(&self, idx: u8) {
        self.free(idx);
    }

    /// Get the physical register at a pool index (allocated by `alloc()`).
    pub(crate) fn reg(&self, idx: u8) -> R {
        self.regs[idx as usize]
    }

    /// Assert that all scratch registers have been freed.
    /// Called between instructions to catch leaks.
    #[inline]
    pub(crate) fn assert_all_free(&self) {
        debug_assert_eq!(
            self.in_use.get(),
            0,
            "ScratchPool: {} register(s) still in use after instruction",
            self.in_use.get().count_ones()
        );
    }

    fn free(&self, idx: u8) {
        let mask = self.in_use.get();
        debug_assert!(
            mask & (1 << idx) != 0,
            "ScratchPool: double-free of slot {}",
            idx
        );
        self.in_use.set(mask & !(1 << idx));
    }
}

/// RAII guard for an allocated scratch register. Derefs to `R`.
/// Frees the pool slot on drop.
pub(crate) struct ScratchGuard<'a, R: Copy, const N: usize> {
    pool: &'a ScratchPool<R, N>,
    idx: u8,
    reg: R,
}

impl<R: Copy + fmt::Debug, const N: usize> fmt::Debug for ScratchGuard<'_, R, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScratchGuard")
            .field("idx", &self.idx)
            .field("reg", &self.reg)
            .finish()
    }
}

impl<R: Copy, const N: usize> Deref for ScratchGuard<'_, R, N> {
    type Target = R;

    #[inline]
    fn deref(&self) -> &R {
        &self.reg
    }
}

impl<R: Copy, const N: usize> Drop for ScratchGuard<'_, R, N> {
    fn drop(&mut self) {
        self.pool.free(self.idx);
    }
}

/// Owned scratch reservation that no longer borrows the pool.
///
/// This is the escape hatch for protocol-scoped temps that still want RAII
/// cleanup without keeping a Rust borrow of the backend field alive.
pub(crate) struct DetachedScratch<R: Copy, const N: usize> {
    pool: *const ScratchPool<R, N>,
    idx: u8,
    reg: R,
    _marker: PhantomData<ScratchPool<R, N>>,
}

impl<R: Copy + fmt::Debug, const N: usize> fmt::Debug for DetachedScratch<R, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DetachedScratch")
            .field("idx", &self.idx)
            .field("reg", &self.reg)
            .finish()
    }
}

impl<R: Copy, const N: usize> Deref for DetachedScratch<R, N> {
    type Target = R;

    #[inline]
    fn deref(&self) -> &R {
        &self.reg
    }
}

impl<R: Copy, const N: usize> Drop for DetachedScratch<R, N> {
    fn drop(&mut self) {
        unsafe {
            (*self.pool).free(self.idx);
        }
    }
}

impl<R: Copy, const N: usize> ScratchGuard<'_, R, N> {
    /// Convert a lexical guard into an owned reservation.
    ///
    /// Unlike the old `release()` pattern, this keeps the pool slot reserved
    /// until the returned token is dropped.
    #[inline]
    pub(crate) fn detach(self) -> DetachedScratch<R, N> {
        let detached = DetachedScratch {
            pool: self.pool as *const ScratchPool<R, N>,
            idx: self.idx,
            reg: self.reg,
            _marker: PhantomData,
        };
        core::mem::forget(self);
        detached
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestReg(u8);

    #[test]
    fn alloc_returns_different_regs_on_consecutive_alloc_free() {
        let pool = ScratchPool::new([TestReg(10), TestReg(11)]);

        let g1 = pool.scoped_alloc();
        assert_eq!(*g1, TestReg(10));
        drop(g1);

        let g2 = pool.scoped_alloc();
        assert_eq!(*g2, TestReg(11));
        drop(g2);

        let g3 = pool.scoped_alloc();
        assert_eq!(*g3, TestReg(10));
        drop(g3);
    }

    #[test]
    fn alloc_two_simultaneously() {
        let pool = ScratchPool::new([TestReg(10), TestReg(11)]);

        let g1 = pool.scoped_alloc();
        let g2 = pool.scoped_alloc();
        assert_ne!(*g1, *g2);
        drop(g1);
        drop(g2);
        pool.assert_all_free();
    }

    #[test]
    #[should_panic(expected = "all 2 registers are in use")]
    fn alloc_exhaustion_panics() {
        let pool = ScratchPool::new([TestReg(10), TestReg(11)]);
        let _g1 = pool.scoped_alloc();
        let _g2 = pool.scoped_alloc();
        let _g3 = pool.scoped_alloc(); // should panic
    }

    #[test]
    fn dropping_guard_frees_slot() {
        let pool = ScratchPool::new([TestReg(10), TestReg(11)]);
        let g = pool.scoped_alloc();
        assert_eq!(*g, TestReg(10));
        drop(g);
        pool.assert_all_free();
    }

    #[test]
    fn detached_guard_keeps_slot_reserved_until_drop() {
        let pool = ScratchPool::new([TestReg(10), TestReg(11)]);
        let g = pool.scoped_alloc().detach();
        assert_eq!(*g, TestReg(10));

        let g2 = pool.scoped_alloc();
        assert_eq!(*g2, TestReg(11));
        drop(g2);

        drop(g);
        pool.assert_all_free();
    }

    #[test]
    fn rotation_wraps_around() {
        let pool = ScratchPool::new([TestReg(0), TestReg(1), TestReg(2)]);
        for expected in [0, 1, 2, 0, 1, 2, 0] {
            let g = pool.scoped_alloc();
            assert_eq!(*g, TestReg(expected));
            drop(g);
        }
    }

}
