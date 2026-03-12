//! Shared lowering state for prepared LIR construction.

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct EntryState {
    pub(super) stack_height: u16,
    pub(super) spill_depth: u16,
}

impl EntryState {
    #[inline]
    pub(super) const fn live_tos_count(self) -> u16 {
        self.stack_height.saturating_sub(self.spill_depth)
    }
}

#[derive(Clone, Debug)]
pub(super) struct BlockState {
    tos_limit: u8,
    stack_height: u16,
    spill_depth: u16,
    tos: Vec<LirValue>,
    pub(super) ops: Vec<LirInst>,
}

impl BlockState {
    pub(super) fn from_params(params: &LirBlockParams, tos_limit: u8) -> Result<Self, WasmError> {
        let spill_depth = params
            .stack_height
            .saturating_sub(params.tos.len() as u16);
        let state = Self {
            tos_limit,
            stack_height: params.stack_height,
            spill_depth,
            tos: params.tos.clone(),
            ops: Vec::new(),
        };
        state.ensure_tos_fit("block entry")?;
        Ok(state)
    }

    #[inline]
    pub(super) fn height(&self) -> u16 {
        self.stack_height
    }

    #[inline]
    pub(super) fn tos(&self) -> &[LirValue] {
        &self.tos
    }

    pub(super) fn top_values(&self, count: usize) -> Result<Vec<LirValue>, WasmError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > self.tos.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR TOS underflow: requested top {} values from live window {} (stack_height={}, spill_depth={})",
                count,
                self.tos.len(),
                self.stack_height,
                self.spill_depth,
            )));
        }
        Ok(self.tos[self.tos.len() - count..].to_vec())
    }

    pub(super) fn pop_one(&mut self) -> Result<LirValue, WasmError> {
        let value = self.tos.pop().ok_or_else(|| {
            WasmError::internal("prepared LIR TOS underflow".into())
        })?;
        self.stack_height = self.stack_height.saturating_sub(1);
        self.spill_depth = self.spill_depth.min(self.stack_height);
        Ok(value)
    }

    pub(super) fn consume_top(&mut self, count: usize) -> Result<(), WasmError> {
        if count > self.tos.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR TOS underflow: tried to consume {} values from live window {}",
                count,
                self.tos.len(),
            )));
        }
        let new_len = self.tos.len().saturating_sub(count);
        self.tos.truncate(new_len);
        self.stack_height = self.stack_height.saturating_sub(count as u16);
        self.spill_depth = self.spill_depth.min(self.stack_height);
        Ok(())
    }

    pub(super) fn push_results(&mut self, results: Vec<LirValue>) -> Result<(), WasmError> {
        self.stack_height = self.stack_height.saturating_add(results.len() as u16);
        self.tos.extend(results);
        self.ensure_tos_fit("value push")
    }

    pub(super) fn spill_prefix(&mut self, count: u16) -> Result<Vec<LirValue>, WasmError> {
        let count = count as usize;
        if count > self.tos.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR spill requested {} values from live window {}",
                count,
                self.tos.len(),
            )));
        }
        let spilled = self.tos.drain(..count).collect::<Vec<_>>();
        self.spill_depth = self.spill_depth.saturating_add(count as u16);
        Ok(spilled)
    }

    pub(super) fn fill_prefix(&mut self, values: Vec<LirValue>) -> Result<(), WasmError> {
        self.spill_depth = self.spill_depth.saturating_sub(values.len() as u16);
        let mut new_tos = values;
        new_tos.extend(self.tos.drain(..));
        self.tos = new_tos;
        self.ensure_tos_fit("prefix fill")
    }

    pub(super) fn finish_call(&mut self, consumed: u16, produced: u16) {
        self.stack_height = self
            .stack_height
            .saturating_sub(consumed)
            .saturating_add(produced);
        self.spill_depth = self.stack_height;
        self.tos.clear();
    }

    #[inline]
    pub(super) fn validate_tos_fit(&self, context: &'static str) -> Result<(), WasmError> {
        self.ensure_tos_fit(context)
    }

    fn ensure_tos_fit(&self, context: &'static str) -> Result<(), WasmError> {
        if self.tos.len() > self.tos_limit as usize {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR exceeds configured TOS width during {context}: live window {} > limit {} (stack_height={}, spill_depth={})",
                self.tos.len(),
                self.tos_limit,
                self.stack_height,
                self.spill_depth,
            )));
        }
        Ok(())
    }
}
