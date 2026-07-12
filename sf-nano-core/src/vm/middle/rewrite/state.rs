//! Rewrite-local mutable state.
//!
//! These types intentionally live on the rewrite side of the boundary:
//! - `ValueAlloc` owns SSA value allocation
//! - `BlockState` owns the live transient window and emitted ops
//!
//! The planner never mutates these structures directly. It only returns
//! decisions that the rewriter validates and applies.

use crate::collections;

use tracked_alloc::collections::BTreeSet;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::middle::{
        budget::count_live_bank_budget_units,
        discipline::{BankBudget, CapacityMode, DisciplineDriver, Window},
        frame::{FrameLayoutPlan, FrameSlot},
        joint_plan::TransientContract,
        ssa_ir::ir::{SsaInst, SsaOperand, SsaValue},
    },
};

#[derive(Clone, Debug, Default)]
pub(super) struct ValueAlloc {
    next: u32,
    types: collections::Vec<ValueType>,
}

impl ValueAlloc {
    pub(super) fn fresh_typed(&mut self, ty: ValueType) -> SsaValue {
        let value = SsaValue(self.next);
        self.next += 1;
        self.types.push(ty);
        value
    }

    pub(super) fn many_typed(&mut self, types: &[ValueType]) -> collections::Vec<SsaValue> {
        types.iter().map(|ty| self.fresh_typed(*ty)).collect()
    }

    pub(super) fn value_type(&self, value: SsaValue) -> ValueType {
        self.types
            .get(value.0 as usize)
            .copied()
            .unwrap_or(ValueType::I64)
    }

    pub(super) fn take_types(&mut self) -> collections::Vec<ValueType> {
        core::mem::take(&mut self.types)
    }
}

pub(super) fn make_block_params(
    entry_live_types: &[ValueType],
    values: &mut ValueAlloc,
) -> collections::Vec<SsaValue> {
    values.many_typed(entry_live_types)
}

/// Emit-side [`DisciplineDriver`]: owns SSA-value identity and op emission for
/// the rewriter. The engine calls these hooks to keep this window in lockstep
/// with its typed core.
struct EmitDriver<'a> {
    live: &'a mut collections::Vec<SsaValue>,
    ops: &'a mut collections::Vec<SsaInst>,
    values: &'a mut ValueAlloc,
}

impl DisciplineDriver for EmitDriver<'_> {
    fn on_spill(&mut self, base_slot: FrameSlot, count: usize) {
        let spilled = self.live.drain(..count).collect::<collections::Vec<_>>();
        for (offset, src) in spilled.into_iter().enumerate() {
            self.ops
                .push(SsaInst::spill(base_slot.advance(offset as u16), src));
        }
    }

    fn on_fill(&mut self, base_slot: FrameSlot, types: &[ValueType]) {
        let mut reloaded = collections::Vec::with_capacity(types.len());
        for (offset, ty) in types.iter().copied().enumerate() {
            let dst = self.values.fresh_typed(ty);
            self.ops
                .push(SsaInst::fill(base_slot.advance(offset as u16), dst));
            reloaded.push(dst);
        }
        let mut new_live = reloaded;
        new_live.extend(self.live.drain(..));
        *self.live = new_live;
    }

    fn on_drop_cache(&mut self, slot: FrameSlot) {
        self.ops.push(SsaInst::local_drop_cache(slot));
    }

    fn on_call_boundary(&mut self) {
        self.live.clear();
    }

    fn on_call_boundary_scalar(&mut self, result: SsaValue) {
        self.live.clear();
        self.live.push(result);
    }
}

#[derive(Clone, Debug)]
pub(super) struct BlockState {
    pub(super) gp_unit_bytes: u8,
    pub(super) gp_live_budget: u8,
    pub(super) fp_live_budget: u8,
    /// Typed operand-window state evolution (height, spill depth, types,
    /// aliases). All core mutation goes through this engine; `BlockState` only
    /// owns SSA-value identity and emitted ops.
    window: Window,
    live: collections::Vec<SsaValue>,
    pub(super) ops: collections::Vec<SsaInst>,
    /// Overflow storage for primitive operands beyond the first two. The
    /// primitive variant publishes the start index via [`SsaInst.meta`].
    /// Transferred into the finalized [`SsaBlock.extra_args`] when the block
    /// is closed.
    pub(super) extra_args: collections::Vec<SsaOperand>,
}

impl BlockState {
    pub(super) fn from_entry(
        entry: TransientContract<'_>,
        full_stack_types: &[ValueType],
        params: &[SsaValue],
        gp_unit_bytes: u8,
        gp_live_budget: u8,
        fp_live_budget: u8,
    ) -> Result<Self, WasmError> {
        debug_assert!(
            full_stack_types.len() >= entry.stack_height as usize,
            "from_entry: stack_types.len()={} < stack_height={}",
            full_stack_types.len(),
            entry.stack_height,
        );
        let window = Window::new(
            entry.stack_height,
            entry.spill_depth,
            full_stack_types.to_vec().into(),
            entry.live_types.to_vec().into(),
            collections::vec![None; params.len()],
        );
        let state = Self {
            gp_unit_bytes,
            gp_live_budget,
            fp_live_budget,
            window,
            live: params.to_vec().into(),
            ops: collections::Vec::new(),
            extra_args: collections::Vec::new(),
        };
        state.ensure_live_fit("block entry")?;
        Ok(state)
    }

    pub(super) fn height(&self) -> u16 {
        self.window.height()
    }

    pub(super) fn spill_depth(&self) -> u16 {
        self.window.spill_depth()
    }

    pub(super) fn live(&self) -> &[SsaValue] {
        &self.live
    }

    pub(super) fn top_values(&self, count: usize) -> Result<collections::Vec<SsaValue>, WasmError> {
        if count == 0 {
            return Ok(collections::Vec::new());
        }
        if count > self.live.len() {
            return Err(WasmError::internal("transient underflow: requested values from live window (stack_height=, spill_depth=)"));
        }
        Ok(self.live[self.live.len() - count..].to_vec().into())
    }

    pub(super) fn pop_one(&mut self) -> Result<SsaValue, WasmError> {
        let value = self
            .live
            .pop()
            .ok_or_else(|| WasmError::internal("transient underflow"))?;
        self.window.pop_core();
        Ok(value)
    }

    pub(super) fn consume_top(&mut self, count: usize) -> Result<(), WasmError> {
        if count > self.live.len() {
            return Err(WasmError::internal(
                "transient underflow: tried to consume values from live window",
            ));
        }
        let new_len = self.live.len().saturating_sub(count);
        self.live.truncate(new_len);
        self.window.consume_core(count);
        Ok(())
    }

    pub(super) fn push_results(
        &mut self,
        results: collections::Vec<SsaValue>,
        result_types: collections::Vec<ValueType>,
    ) -> Result<(), WasmError> {
        let alias_len = results.len();
        self.push_results_with_aliases(results, result_types, collections::vec![None; alias_len])
    }

    pub(super) fn push_results_with_aliases(
        &mut self,
        results: collections::Vec<SsaValue>,
        result_types: collections::Vec<ValueType>,
        result_aliases: collections::Vec<Option<FrameSlot>>,
    ) -> Result<(), WasmError> {
        debug_assert_eq!(results.len(), result_types.len());
        debug_assert_eq!(results.len(), result_aliases.len());
        self.live.extend(results);
        self.window.push_core(&result_types, &result_aliases);
        self.ensure_live_fit("value push")
    }

    pub(super) fn value_at_stack_index(&self, stack_index: u16) -> Option<SsaValue> {
        self.window
            .live_position(stack_index)
            .and_then(|pos| self.live.get(pos).copied())
    }

    pub(super) fn type_at_stack_index(&self, stack_index: u16) -> Option<ValueType> {
        self.window.type_at_stack_index(stack_index)
    }

    pub(super) fn live_aliases(&self) -> &[Option<FrameSlot>] {
        self.window.live_aliases()
    }

    pub(super) fn live_types(&self) -> &[ValueType] {
        self.window.live_types()
    }

    // -- Discipline transitions (engine-owned; emit driver realizes values) ---

    /// Fill so at least `operand_count` values are live.
    pub(super) fn fill_operands(
        &mut self,
        values: &mut ValueAlloc,
        frame: FrameLayoutPlan,
        operand_count: u16,
    ) {
        let Self {
            window, live, ops, ..
        } = self;
        let mut driver = EmitDriver { live, ops, values };
        window.fill(&mut driver, frame, operand_count);
    }

    /// Spill every live transient.
    pub(super) fn spill_all(&mut self, values: &mut ValueAlloc, frame: FrameLayoutPlan) {
        let Self {
            window, live, ops, ..
        } = self;
        let mut driver = EmitDriver { live, ops, values };
        window.spill_all(&mut driver, frame);
    }

    /// Spill every live transient except the top `keep_top`.
    pub(super) fn spill_except_top(
        &mut self,
        values: &mut ValueAlloc,
        frame: FrameLayoutPlan,
        keep_top: u16,
    ) {
        let Self {
            window, live, ops, ..
        } = self;
        let mut driver = EmitDriver { live, ops, values };
        window.spill_except_top(&mut driver, frame, keep_top);
    }

    /// Spill the bottom `count` live transients (boundary publish).
    pub(super) fn spill_bottom(
        &mut self,
        values: &mut ValueAlloc,
        frame: FrameLayoutPlan,
        count: u16,
    ) {
        let Self {
            window, live, ops, ..
        } = self;
        let mut driver = EmitDriver { live, ops, values };
        window.spill(&mut driver, frame, count);
    }

    /// Publish live transients below the top `consumed` call-operand suffix,
    /// keeping that suffix live while spilling older values that cannot cross
    /// the call boundary.
    pub(super) fn spill_live_below_top(
        &mut self,
        values: &mut ValueAlloc,
        frame: FrameLayoutPlan,
        consumed: u16,
    ) {
        self.spill_except_top(values, frame, consumed);
    }

    /// Extra GP/FP budget needed to fill `operand_count` operands and push
    /// `result_types` (see [`Window::required_capacity`]).
    pub(super) fn required_capacity(
        &self,
        operand_count: u16,
        result_types: &[ValueType],
    ) -> (usize, usize) {
        self.window
            .required_capacity(operand_count, result_types, self.gp_unit_bytes)
    }

    /// Clamp the live window against the dynamic-bank budget, spilling then
    /// evicting via the resident cache.
    pub(super) fn ensure_capacity(
        &mut self,
        values: &mut ValueAlloc,
        frame: FrameLayoutPlan,
        mode: CapacityMode,
        resident: &mut BTreeSet<FrameSlot>,
        local_slot_types: &[ValueType],
        extra: (usize, usize),
    ) -> Result<(), WasmError> {
        let budget = BankBudget {
            gp_unit_bytes: self.gp_unit_bytes,
            gp_live_budget: self.gp_live_budget,
            fp_live_budget: self.fp_live_budget,
        };
        let Self {
            window, live, ops, ..
        } = self;
        let mut driver = EmitDriver { live, ops, values };
        window.ensure_capacity(
            &mut driver,
            frame,
            mode,
            resident,
            local_slot_types,
            budget,
            extra,
        )
    }

    pub(super) fn finish_call(
        &mut self,
        values: &mut ValueAlloc,
        consumed: u16,
        produced: u16,
        result_types: &[ValueType],
    ) {
        let Self {
            window, live, ops, ..
        } = self;
        let mut driver = EmitDriver { live, ops, values };
        window.finish_call(&mut driver, consumed, produced, result_types);
    }

    pub(super) fn finish_call_with_live_scalar(
        &mut self,
        values: &mut ValueAlloc,
        consumed: u16,
        result: SsaValue,
        result_ty: ValueType,
    ) -> Result<(), WasmError> {
        {
            let Self {
                window, live, ops, ..
            } = self;
            let mut driver = EmitDriver { live, ops, values };
            window.finish_call_scalar(&mut driver, consumed, result, result_ty);
        }
        self.ensure_live_fit("scalar call result")
    }

    fn ensure_live_fit(&self, _context: &str) -> Result<(), WasmError> {
        let effective_live_types = self
            .window
            .live_types()
            .iter()
            .zip(self.window.live_aliases().iter())
            .filter_map(|(ty, alias)| alias.is_none().then_some(*ty))
            .collect::<collections::Vec<_>>();
        let (gp_live, fp_live) =
            count_live_bank_budget_units(&effective_live_types, self.gp_unit_bytes);
        if gp_live > self.gp_live_budget as usize || fp_live > self.fp_live_budget as usize {
            return Err(WasmError::internal(
                "SSA-IR exceeds configured dynamic bank budget during : gp > or fp >",
            ));
        }
        Ok(())
    }
}
