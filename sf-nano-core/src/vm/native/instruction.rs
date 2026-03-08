//! Native instruction header and layout.

use alloc::boxed::Box;
use alloc::vec::Vec;

pub type NativeEntry = unsafe extern "C" fn();

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeInst {
    pub entry: NativeEntry,
    pub imm0: u64,
    pub imm1: u64,
    pub imm2: u64,
    pub tos_slots: u64,
}

pub const INVALID_TOS_SLOT: u16 = u16::MAX;
pub const EMPTY_TOS_SLOTS: u64 = u64::MAX;

const _: [(); 40] = [(); core::mem::size_of::<NativeInst>()];

impl NativeInst {
    #[inline]
    pub fn new(entry: NativeEntry, imm0: u64, imm1: u64, imm2: u64) -> Self {
        Self {
            entry,
            imm0,
            imm1,
            imm2,
            tos_slots: EMPTY_TOS_SLOTS,
        }
    }

    #[inline]
    pub fn new_with_tos_slots(
        entry: NativeEntry,
        imm0: u64,
        imm1: u64,
        imm2: u64,
        tos_slots: u64,
    ) -> Self {
        Self {
            entry,
            imm0,
            imm1,
            imm2,
            tos_slots,
        }
    }

    #[inline]
    pub fn new_entry_only(entry: NativeEntry) -> Self {
        Self::new(entry, 0, 0, 0)
    }
}

#[derive(Default)]
pub struct NativeInstArena {
    insts: Vec<NativeInst>,
}

impl NativeInstArena {
    pub fn new() -> Self {
        Self { insts: Vec::new() }
    }

    pub fn push(&mut self, inst: NativeInst) -> *mut NativeInst {
        self.insts.push(inst);
        unsafe { self.insts.as_mut_ptr().add(self.insts.len() - 1) }
    }

    pub fn into_boxed_slice(self) -> Box<[NativeInst]> {
        self.insts.into_boxed_slice()
    }
}
