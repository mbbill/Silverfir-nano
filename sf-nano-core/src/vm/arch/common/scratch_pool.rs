//! Rotating scratch register pool with explicit ownership.
//!
//! Uses interior mutability (`Cell`) so that `scoped_alloc` takes `&self`,
//! avoiding borrow conflicts when the backend needs `&mut self` for emission
//! while scratch guards are alive.

use core::cell::Cell;
use core::fmt;
use core::ops::Deref;

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
/// Frees the pool slot on drop, or explicitly via `release()`.
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

impl<R: Copy, const N: usize> ScratchGuard<'_, R, N> {
    /// Consume the guard, free the pool slot, return the register value.
    ///
    /// Use when you need the register value to outlive the guard
    /// (e.g. for patching a previously-emitted instruction).
    #[inline]
    pub(crate) fn release(self) -> R {
        let reg = self.reg;
        // Free before forgetting so drop doesn't double-free.
        self.pool.free(self.idx);
        core::mem::forget(self);
        reg
    }
}

impl<R: Copy, const N: usize> Drop for ScratchGuard<'_, R, N> {
    fn drop(&mut self) {
        self.pool.free(self.idx);
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
    fn release_returns_reg_and_frees_slot() {
        let pool = ScratchPool::new([TestReg(10), TestReg(11)]);
        let g = pool.scoped_alloc();
        let reg = g.release();
        assert_eq!(reg, TestReg(10));
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
