//! Shared lowering state for CFG/SSA construction.

use alloc::vec::Vec;

use crate::vm::{
    lir::{
        ir::{LirBlockParams, LirInst, LirValue},
        slot::FrameSlot,
    },
    plan::frame::FrameLayoutPlan,
};

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
    pub(super) stack: Vec<StackValue>,
    pub(super) ops: Vec<LirInst>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StackValue {
    Frame(FrameSlot),
    Ssa(LirValue),
}

impl BlockState {
    pub(super) fn from_params(
        height: u16,
        params: &LirBlockParams,
        frame: FrameLayoutPlan,
    ) -> Self {
        let cached = params.tos.len();
        let spilled = height as usize - cached;
        let mut stack = Vec::with_capacity(height as usize);
        for index in 0..spilled {
            stack.push(StackValue::Frame(frame.operand_slot(index as u16)));
        }
        stack.extend(params.tos.iter().copied().map(StackValue::Ssa));

        Self {
            stack,
            ops: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn height(&self) -> u16 {
        self.stack.len() as u16
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
