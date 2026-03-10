//! Shared lowering state for CFG/SSA construction.

use alloc::vec::Vec;

use crate::vm::lir::ir::{LirBlockParams, LirInst, LirValue};

#[derive(Clone, Debug, Default)]
pub(super) struct ValueAlloc {
    next: u32,
}

impl ValueAlloc {
    #[inline]
    pub(super) fn fresh(&mut self) -> LirValue {
        let value = LirValue(self.next);
        self.next += 1;
        value
    }

    pub(super) fn many(&mut self, count: usize) -> Vec<LirValue> {
        (0..count).map(|_| self.fresh()).collect()
    }
}

#[derive(Clone, Debug)]
pub(super) struct BlockState {
    pub(super) height: u16,
    pub(super) tos: Vec<LirValue>,
    pub(super) ops: Vec<LirInst>,
}

impl BlockState {
    pub(super) fn from_params(height: u16, params: &LirBlockParams) -> Self {
        Self {
            height,
            tos: params.tos.clone(),
            ops: Vec::new(),
        }
    }
}

pub(super) fn make_block_params(
    height: u16,
    tos_register_count: u8,
    values: &mut ValueAlloc,
) -> LirBlockParams {
    LirBlockParams {
        tos: values.many(core::cmp::min(height as usize, tos_register_count as usize)),
    }
}
