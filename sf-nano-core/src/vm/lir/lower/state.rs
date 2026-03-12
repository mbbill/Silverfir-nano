//! Shared lowering state for CFG/SSA construction.

use alloc::vec::Vec;

use crate::error::WasmError;
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
    stack: Vec<LirValue>,
    pub(super) ops: Vec<LirInst>,
}

impl BlockState {
    pub(super) fn from_params(params: &LirBlockParams) -> Self {
        Self {
            stack: params.values.clone(),
            ops: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn height(&self) -> u16 {
        self.stack.len() as u16
    }

    #[inline]
    pub(super) fn values(&self) -> &[LirValue] {
        &self.stack
    }

    pub(super) fn top_values(&self, count: usize) -> Result<Vec<LirValue>, WasmError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let height = self.stack.len();
        if count > height {
            return Err(WasmError::internal(alloc::format!(
                "LIR stack underflow: requested top {} values from height {}",
                count, height,
            )));
        }
        Ok(self.stack[height - count..].to_vec())
    }

    pub(super) fn value_at(&self, index: usize) -> Result<LirValue, WasmError> {
        self.stack.get(index).copied().ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "LIR stack index {} is out of range for height {}",
                index,
                self.stack.len(),
            ))
        })
    }

    pub(super) fn pop_one(&mut self) -> Result<LirValue, WasmError> {
        self.stack
            .pop()
            .ok_or_else(|| WasmError::internal("LIR stack underflow".into()))
    }

    pub(super) fn consume_top(&mut self, count: usize) -> Result<(), WasmError> {
        if count > self.stack.len() {
            return Err(WasmError::internal(alloc::format!(
                "LIR stack underflow: tried to consume {} values from height {}",
                count,
                self.stack.len(),
            )));
        }
        let new_len = self.stack.len().saturating_sub(count);
        self.stack.truncate(new_len);
        Ok(())
    }

    pub(super) fn push_results(&mut self, results: Vec<LirValue>) {
        self.stack.extend(results);
    }
}
