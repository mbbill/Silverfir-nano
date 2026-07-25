use core::cell::Cell;
use core::fmt;
use core::marker::PhantomData;
use core::ops::Deref;

use super::reg::X86Reg;

/// x86_64 GP scratch ownership.
///
/// Unlike the shared `ScratchPool`, these registers are not interchangeable
/// arch-agnostic temps. x86_64 ordinary lowering sometimes requires these
/// exact registers:
///
/// - `RAX` / `RDX` for `div` / `idiv`
/// - `RCX` for variable shifts / rotates
///
/// They are therefore backend-owned and tracked locally by the x86_64 backend.
#[derive(Debug)]
pub(super) struct GpScratchPool {
    regs: [X86Reg; 3],
    in_use: Cell<u8>,
    cursor: Cell<u8>,
}

impl GpScratchPool {
    pub(super) fn new(regs: [X86Reg; 3]) -> Self {
        Self {
            regs,
            in_use: Cell::new(0),
            cursor: Cell::new(0),
        }
    }

    pub(super) fn scoped_alloc(&self) -> GpScratchGuard<'_> {
        let mask = self.in_use.get();
        let mut cursor = self.cursor.get() as usize;
        for _ in 0..self.regs.len() {
            let idx = cursor % self.regs.len();
            cursor += 1;
            if mask & (1 << idx) == 0 {
                self.in_use.set(mask | (1 << idx));
                self.cursor.set(cursor as u8);
                return GpScratchGuard {
                    pool: self,
                    idx: idx as u8,
                    reg: self.regs[idx],
                };
            }
        }
        panic!(
            "x86_64 GP scratch pool: all {} registers are in use",
            self.regs.len()
        );
    }

    pub(super) fn claim_rax(&self) -> GpScratchGuard<'_> {
        self.claim_named(X86Reg::RAX)
    }

    #[cfg(not(sf_os_windows))]
    pub(super) fn try_claim_rax(&self) -> Option<GpScratchGuard<'_>> {
        self.try_claim_named(X86Reg::RAX)
    }

    pub(super) fn claim_rcx(&self) -> GpScratchGuard<'_> {
        self.claim_named(X86Reg::RCX)
    }

    #[cfg(not(sf_os_windows))]
    pub(super) fn try_claim_rcx(&self) -> Option<GpScratchGuard<'_>> {
        self.try_claim_named(X86Reg::RCX)
    }

    pub(super) fn claim_rdx(&self) -> GpScratchGuard<'_> {
        self.claim_named(X86Reg::RDX)
    }

    pub(super) fn alloc(&self) -> u8 {
        let guard = self.scoped_alloc();
        let idx = guard.idx;
        core::mem::forget(guard);
        idx
    }

    pub(super) fn free_index(&self, idx: u8) {
        self.free(idx);
    }

    pub(super) fn reg(&self, idx: u8) -> X86Reg {
        debug_assert!(
            self.in_use.get() & (1 << idx) != 0,
            "x86_64 GP scratch pool: slot {} read without ownership",
            idx
        );
        self.regs[idx as usize]
    }

    pub(super) fn assert_all_free(&self) {
        debug_assert_eq!(
            self.in_use.get(),
            0,
            "x86_64 GP scratch pool: {} register(s) still in use after instruction",
            self.in_use.get().count_ones()
        );
    }

    fn claim_named(&self, reg: X86Reg) -> GpScratchGuard<'_> {
        self.try_claim_named(reg)
            .unwrap_or_else(|| panic!("x86_64 GP scratch pool: {:?} already in use", reg))
    }

    fn try_claim_named(&self, reg: X86Reg) -> Option<GpScratchGuard<'_>> {
        let idx = self
            .regs
            .iter()
            .position(|&candidate| candidate == reg)
            .unwrap_or_else(|| panic!("x86_64 GP scratch pool: {:?} is not backend-owned", reg));
        let idx = idx as u8;
        let mask = self.in_use.get();
        if mask & (1 << idx) != 0 {
            return None;
        }
        self.in_use.set(mask | (1 << idx));
        Some(GpScratchGuard {
            pool: self,
            idx,
            reg,
        })
    }

    fn free(&self, idx: u8) {
        let mask = self.in_use.get();
        debug_assert!(
            mask & (1 << idx) != 0,
            "x86_64 GP scratch pool: double-free of slot {}",
            idx
        );
        self.in_use.set(mask & !(1 << idx));
    }
}

pub(super) struct GpScratchGuard<'a> {
    pool: &'a GpScratchPool,
    idx: u8,
    reg: X86Reg,
}

impl fmt::Debug for GpScratchGuard<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpScratchGuard")
            .field("idx", &self.idx)
            .field("reg", &self.reg)
            .finish()
    }
}

impl Deref for GpScratchGuard<'_> {
    type Target = X86Reg;

    fn deref(&self) -> &Self::Target {
        &self.reg
    }
}

impl Drop for GpScratchGuard<'_> {
    fn drop(&mut self) {
        self.pool.free(self.idx);
    }
}

pub(super) struct DetachedGpScratch {
    pool: *const GpScratchPool,
    idx: u8,
    reg: X86Reg,
    _marker: PhantomData<GpScratchPool>,
}

impl fmt::Debug for DetachedGpScratch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DetachedGpScratch")
            .field("idx", &self.idx)
            .field("reg", &self.reg)
            .finish()
    }
}

impl Deref for DetachedGpScratch {
    type Target = X86Reg;

    fn deref(&self) -> &Self::Target {
        &self.reg
    }
}

impl Drop for DetachedGpScratch {
    fn drop(&mut self) {
        unsafe {
            (*self.pool).free(self.idx);
        }
    }
}

impl GpScratchGuard<'_> {
    pub(super) fn detach(self) -> DetachedGpScratch {
        let detached = DetachedGpScratch {
            pool: self.pool as *const GpScratchPool,
            idx: self.idx,
            reg: self.reg,
            _marker: PhantomData,
        };
        core::mem::forget(self);
        detached
    }
}
