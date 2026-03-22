//! Shared preparation state.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::value_type::ValueType;
use crate::vm::middle::ssa_ir::ir::{SsaInst, SsaValue};

#[derive(Clone, Debug, Default)]
pub(super) struct ValueAlloc {
    next: u32,
    /// Per-value type table indexed by `SsaValue.0`.
    types: Vec<ValueType>,
}

impl ValueAlloc {
    #[inline]
    pub(super) fn fresh(&mut self) -> SsaValue {
        self.fresh_typed(ValueType::I64)
    }

    #[inline]
    pub(super) fn fresh_typed(&mut self, ty: ValueType) -> SsaValue {
        let value = SsaValue(self.next);
        self.next += 1;
        self.types.push(ty);
        value
    }

    pub(super) fn many(&mut self, count: usize) -> Vec<SsaValue> {
        (0..count).map(|_| self.fresh()).collect()
    }

    pub(super) fn many_typed(&mut self, types: &[ValueType]) -> Vec<SsaValue> {
        types.iter().map(|ty| self.fresh_typed(*ty)).collect()
    }

    pub(super) fn value_type(&self, value: SsaValue) -> ValueType {
        let ty = self.types.get(value.0 as usize).copied();
        debug_assert!(
            ty.is_some(),
            "SsaValue({}) has no entry in the value-type table",
            value.0,
        );
        ty.unwrap_or(ValueType::I64)
    }

    pub(super) fn take_types(&mut self) -> Vec<ValueType> {
        core::mem::take(&mut self.types)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct EntryState {
    pub(super) stack_height: u16,
    pub(super) spill_depth: u16,
    /// Value types for the live (non-spilled) suffix at block entry.
    /// `live_types.len() == stack_height - spill_depth`.
    pub(super) live_types: Vec<ValueType>,
}

impl EntryState {
    #[inline]
    pub(super) const fn live_value_count(&self) -> u16 {
        self.stack_height.saturating_sub(self.spill_depth)
    }
}

#[derive(Clone, Debug)]
pub(super) struct BlockState {
    gp_unit_bytes: u8,
    gp_transient_budget: u8,
    fp_transient_budget: u8,
    stack_height: u16,
    spill_depth: u16,
    live: Vec<SsaValue>,
    live_types: Vec<ValueType>,
    pub(super) ops: Vec<SsaInst>,
}

impl BlockState {
    pub(super) fn from_entry(
        entry: &EntryState,
        params: &[SsaValue],
        gp_unit_bytes: u8,
        gp_transient_budget: u8,
        fp_transient_budget: u8,
    ) -> Result<Self, WasmError> {
        debug_assert_eq!(
            entry.live_types.len(),
            params.len(),
            "EntryState.live_types must line up with block params",
        );
        let state = Self {
            gp_unit_bytes,
            gp_transient_budget,
            fp_transient_budget,
            stack_height: entry.stack_height,
            spill_depth: entry.spill_depth,
            live: params.to_vec(),
            live_types: entry.live_types.clone(),
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
    pub(super) fn live(&self) -> &[SsaValue] {
        &self.live
    }

    #[inline]
    pub(super) fn live_types(&self) -> &[ValueType] {
        &self.live_types
    }

    pub(super) fn top_values(&self, count: usize) -> Result<Vec<SsaValue>, WasmError> {
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

    pub(super) fn pop_one(&mut self) -> Result<SsaValue, WasmError> {
        let value = self
            .live
            .pop()
            .ok_or_else(|| WasmError::internal("prepared LIR transient underflow".into()))?;
        self.live_types.pop().ok_or_else(|| {
            WasmError::internal("prepared LIR lost type tracking for a live value".into())
        })?;
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
        if results.len() != result_types.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR result/value type length mismatch: {} values, {} types",
                results.len(),
                result_types.len(),
            )));
        }
        self.stack_height = self.stack_height.saturating_add(results.len() as u16);
        self.live.extend(results);
        self.live_types.extend(result_types);
        self.ensure_live_fit("value push")
    }

    pub(super) fn spill_prefix(&mut self, count: u16) -> Result<Vec<SsaValue>, WasmError> {
        let count = count as usize;
        if count > self.live.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR spill requested {} values from live window {}",
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
        if values.len() != value_types.len() {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR fill/value type length mismatch: {} values, {} types",
                values.len(),
                value_types.len(),
            )));
        }
        self.spill_depth = self.spill_depth.saturating_sub(values.len() as u16);
        let mut new_live = values;
        new_live.extend(self.live.drain(..));
        self.live = new_live;
        let mut new_live_types = value_types;
        new_live_types.extend(self.live_types.drain(..));
        self.live_types = new_live_types;
        self.ensure_live_fit("prefix fill")
    }

    pub(super) fn finish_boundary(&mut self, consumed: u16, produced: u16) {
        self.stack_height = self
            .stack_height
            .saturating_sub(consumed)
            .saturating_add(produced);
        self.spill_depth = self.stack_height;
        self.live.clear();
        self.live_types.clear();
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
        let (gp_live, fp_live) = count_live_bank_budget_units(&self.live_types, self.gp_unit_bytes);
        if gp_live > self.gp_transient_budget as usize
            || fp_live > self.fp_transient_budget as usize
        {
            return Err(WasmError::internal(alloc::format!(
                "prepared LIR exceeds configured transient bank budget during {context}: gp units {} > {} or fp live {} > {} (stack_height={}, spill_depth={})",
                gp_live,
                self.gp_transient_budget,
                fp_live,
                self.fp_transient_budget,
                self.stack_height,
                self.spill_depth,
            )));
        }
        Ok(())
    }
}

pub(crate) fn count_live_bank_budget_units(
    types: &[ValueType],
    gp_unit_bytes: u8,
) -> (usize, usize) {
    let mut gp = 0usize;
    let mut fp = 0usize;
    for ty in types {
        match ty {
            ValueType::F32 | ValueType::F64 => fp += 1,
            _ => gp += gp_value_budget_units(*ty, gp_unit_bytes),
        }
    }
    (gp, fp)
}

#[inline]
pub(crate) fn gp_value_budget_units(ty: ValueType, gp_unit_bytes: u8) -> usize {
    debug_assert!(gp_unit_bytes != 0, "gp_unit_bytes must be non-zero");
    match ty {
        ValueType::I64 => {
            let width = usize::from(gp_unit_bytes.max(1));
            8usize.div_ceil(width)
        }
        _ => 1,
    }
}
