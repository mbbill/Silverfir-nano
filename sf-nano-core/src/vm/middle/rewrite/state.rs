//! Rewrite-local mutable state.
//!
//! These types intentionally live on the rewrite side of the boundary:
//! - `ValueAlloc` owns SSA value allocation
//! - `BlockState` owns the live transient window and emitted ops
//!
//! The planner never mutates these structures directly. It only returns
//! decisions that the rewriter validates and applies.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::middle::{
        budget::count_live_bank_budget_units,
        joint_plan::TransientContract,
        ssa_ir::ir::{SsaInst, SsaValue},
    },
};

#[derive(Clone, Debug, Default)]
pub(super) struct ValueAlloc {
    next: u32,
    types: Vec<ValueType>,
}

impl ValueAlloc {
    pub(super) fn fresh_typed(&mut self, ty: ValueType) -> SsaValue {
        let value = SsaValue(self.next);
        self.next += 1;
        self.types.push(ty);
        value
    }

    pub(super) fn many_typed(&mut self, types: &[ValueType]) -> Vec<SsaValue> {
        types.iter().map(|ty| self.fresh_typed(*ty)).collect()
    }

    pub(super) fn value_type(&self, value: SsaValue) -> ValueType {
        self.types
            .get(value.0 as usize)
            .copied()
            .unwrap_or(ValueType::I64)
    }

    pub(super) fn take_types(&mut self) -> Vec<ValueType> {
        core::mem::take(&mut self.types)
    }
}

pub(super) fn make_block_params(
    entry_live_types: &[ValueType],
    values: &mut ValueAlloc,
) -> Vec<SsaValue> {
    values.many_typed(entry_live_types)
}

#[derive(Clone, Debug)]
pub(super) struct BlockState {
    pub(super) gp_unit_bytes: u8,
    pub(super) gp_live_budget: u8,
    pub(super) fp_live_budget: u8,
    stack_height: u16,
    spill_depth: u16,
    live: Vec<SsaValue>,
    pub(super) live_types: Vec<ValueType>,
    pub(super) ops: Vec<SsaInst>,
}

impl BlockState {
    pub(super) fn from_entry(
        entry: TransientContract<'_>,
        params: &[SsaValue],
        gp_unit_bytes: u8,
        gp_live_budget: u8,
        fp_live_budget: u8,
    ) -> Result<Self, WasmError> {
        let state = Self {
            gp_unit_bytes,
            gp_live_budget,
            fp_live_budget,
            stack_height: entry.stack_height,
            spill_depth: entry.spill_depth,
            live: params.to_vec(),
            live_types: entry.live_types.to_vec(),
            ops: Vec::new(),
        };
        state.ensure_live_fit("block entry")?;
        Ok(state)
    }

    pub(super) fn height(&self) -> u16 {
        self.stack_height
    }

    pub(super) fn spill_depth(&self) -> u16 {
        self.spill_depth
    }

    pub(super) fn live(&self) -> &[SsaValue] {
        &self.live
    }

    pub(super) fn top_values(&self, count: usize) -> Result<Vec<SsaValue>, WasmError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > self.live.len() {
            return Err(WasmError::internal(alloc::format!(
                "transient underflow: requested {} values from live window {} (stack_height={}, spill_depth={})",
                count,
                self.live.len(),
                self.stack_height,
                self.spill_depth,
            )));
        }
        Ok(self.live[self.live.len() - count..].to_vec())
    }

    pub(super) fn pop_one(&mut self) -> Result<SsaValue, WasmError> {
        let value = self
            .live
            .pop()
            .ok_or_else(|| WasmError::internal("transient underflow".into()))?;
        self.live_types.pop();
        self.stack_height = self.stack_height.saturating_sub(1);
        self.spill_depth = self.spill_depth.min(self.stack_height);
        Ok(value)
    }

    pub(super) fn consume_top(&mut self, count: usize) -> Result<(), WasmError> {
        if count > self.live.len() {
            return Err(WasmError::internal(alloc::format!(
                "transient underflow: tried to consume {} values from live window {}",
                count,
                self.live.len(),
            )));
        }
        let new_len = self.live.len().saturating_sub(count);
        self.live.truncate(new_len);
        self.live_types.truncate(new_len);
        self.stack_height = self.stack_height.saturating_sub(count as u16);
        self.spill_depth = self.spill_depth.min(self.stack_height);
        Ok(())
    }

    pub(super) fn push_results(
        &mut self,
        results: Vec<SsaValue>,
        result_types: Vec<ValueType>,
    ) -> Result<(), WasmError> {
        self.stack_height = self.stack_height.saturating_add(results.len() as u16);
        self.live.extend(results);
        self.live_types.extend(result_types);
        self.ensure_live_fit("value push")
    }

    pub(super) fn spill_prefix(&mut self, count: u16) -> Result<Vec<SsaValue>, WasmError> {
        let count = count as usize;
        if count > self.live.len() {
            return Err(WasmError::internal(alloc::format!(
                "spill requested {} values from live window {}",
                count,
                self.live.len(),
            )));
        }
        let spilled = self.live.drain(..count).collect::<Vec<_>>();
        self.live_types.drain(..count);
        self.spill_depth = self.spill_depth.saturating_add(count as u16);
        Ok(spilled)
    }

    pub(super) fn fill_prefix(
        &mut self,
        values: Vec<SsaValue>,
        value_types: Vec<ValueType>,
    ) -> Result<(), WasmError> {
        self.spill_depth = self.spill_depth.saturating_sub(values.len() as u16);
        let mut new_live = values;
        new_live.extend(self.live.drain(..));
        self.live = new_live;
        let mut new_live_types = value_types;
        new_live_types.extend(self.live_types.drain(..));
        self.live_types = new_live_types;
        self.ensure_live_fit("prefix fill")
    }

    pub(super) fn finish_call(&mut self, consumed: u16, produced: u16) {
        self.stack_height = self
            .stack_height
            .saturating_sub(consumed)
            .saturating_add(produced);
        self.spill_depth = self.stack_height;
        self.live.clear();
        self.live_types.clear();
    }

    fn ensure_live_fit(&self, context: &str) -> Result<(), WasmError> {
        let (gp_live, fp_live) = count_live_bank_budget_units(&self.live_types, self.gp_unit_bytes);
        if gp_live > self.gp_live_budget as usize || fp_live > self.fp_live_budget as usize {
            return Err(WasmError::internal(alloc::format!(
                "SSA-IR exceeds configured dynamic bank budget during {context}: gp {} > {} or fp {} > {}",
                gp_live, self.gp_live_budget, fp_live, self.fp_live_budget
            )));
        }
        Ok(())
    }
}
