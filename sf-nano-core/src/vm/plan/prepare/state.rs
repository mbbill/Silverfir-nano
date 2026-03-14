//! Shared preparation state.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::lir::ir::{LirInst, LirValue};

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
    pub(super) const fn live_value_count(self) -> u16 {
        self.stack_height.saturating_sub(self.spill_depth)
    }
}

#[derive(Clone, Debug)]
pub(super) struct BlockState {
    tos_limit: u8,
    stack_height: u16,
    spill_depth: u16,
    live: Vec<LirValue>,
    pub(super) ops: Vec<LirInst>,
}

impl BlockState {
    pub(super) fn from_entry(
        entry: EntryState,
        params: &[LirValue],
        tos_limit: u8,
    ) -> Result<Self, WasmError> {
        let state = Self {
            tos_limit,
            stack_height: entry.stack_height,
            spill_depth: entry.spill_depth,
            live: params.to_vec(),
            ops: Vec::new(),
        };
        state.ensure_live_fit("block entry")?;
        Ok(state)
    }

    #[inline]
    pub(super) fn height(&self) -> u16 {
        self.stack_height
    }

    #[inline]
    pub(super) fn spill_depth(&self) -> u16 {
        self.spill_depth
    }

    #[inline]
    pub(super) fn live(&self) -> &[LirValue] {
        &self.live
    }

    pub(super) fn top_values(&self, count: usize) -> Result<Vec<LirValue>, WasmError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > self.live.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR transient underflow: requested {} values from live window {} (stack_height={}, spill_depth={})",
                count,
                self.live.len(),
                self.stack_height,
                self.spill_depth,
            )));
        }
        Ok(self.live[self.live.len() - count..].to_vec())
    }

    pub(super) fn pop_one(&mut self) -> Result<LirValue, WasmError> {
        let value = self
            .live
            .pop()
            .ok_or_else(|| WasmError::internal("prepared LIR transient underflow".into()))?;
        self.stack_height = self.stack_height.saturating_sub(1);
        self.spill_depth = self.spill_depth.min(self.stack_height);
        Ok(value)
    }

    pub(super) fn consume_top(&mut self, count: usize) -> Result<(), WasmError> {
        if count > self.live.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR transient underflow: tried to consume {} values from live window {}",
                count,
                self.live.len(),
            )));
        }
        let new_len = self.live.len().saturating_sub(count);
        self.live.truncate(new_len);
        self.stack_height = self.stack_height.saturating_sub(count as u16);
        self.spill_depth = self.spill_depth.min(self.stack_height);
        Ok(())
    }

    pub(super) fn push_results(&mut self, results: Vec<LirValue>) -> Result<(), WasmError> {
        self.stack_height = self.stack_height.saturating_add(results.len() as u16);
        self.live.extend(results);
        self.ensure_live_fit("value push")
    }

    pub(super) fn spill_prefix(&mut self, count: u16) -> Result<Vec<LirValue>, WasmError> {
        let count = count as usize;
        if count > self.live.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR spill requested {} values from live window {}",
                count,
                self.live.len(),
            )));
        }
        let spilled = self.live.drain(..count).collect::<Vec<_>>();
        self.spill_depth = self.spill_depth.saturating_add(count as u16);
        Ok(spilled)
    }

    pub(super) fn fill_prefix(&mut self, values: Vec<LirValue>) -> Result<(), WasmError> {
        self.spill_depth = self.spill_depth.saturating_sub(values.len() as u16);
        let mut new_live = values;
        new_live.extend(self.live.drain(..));
        self.live = new_live;
        self.ensure_live_fit("prefix fill")
    }

    pub(super) fn finish_boundary(&mut self, consumed: u16, produced: u16) {
        self.stack_height = self
            .stack_height
            .saturating_sub(consumed)
            .saturating_add(produced);
        self.spill_depth = self.stack_height;
        self.live.clear();
    }

    #[cfg(any(debug_assertions, test))]
    #[inline]
    pub(super) fn validate_live_fit(&self, context: &'static str) -> Result<(), WasmError> {
        self.ensure_live_fit(context)
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub(super) fn validate_live_fit(&self, _context: &'static str) -> Result<(), WasmError> {
        Ok(())
    }

    fn ensure_live_fit(&self, context: &'static str) -> Result<(), WasmError> {
        if self.live.len() > self.tos_limit as usize {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR exceeds configured transient width during {context}: live window {} > limit {} (stack_height={}, spill_depth={})",
                self.live.len(),
                self.tos_limit,
                self.stack_height,
                self.spill_depth,
            )));
        }
        Ok(())
    }
}
