//! The shared stack-discipline engine.
//!
//! Both the planner's cache-free measure pass and the rewriter's residency-aware
//! emit pass evolve the same typed operand window: a `stack_height`, a
//! `spill_depth` splitting the window into a spilled prefix and a live suffix,
//! the per-slot `type_stack`, and an alias tag per live entry. This module owns
//! that evolution — one implementation of fill / spill / capacity / eviction —
//! so the two passes (and the exact-row pass D walker) cannot drift.
//!
//! The engine ([`Window`]) owns the typed state. A [`DisciplineDriver`] owns
//! only emission and SSA-value identity: `on_spill` / `on_fill` / `on_drop_cache`
//! maintain the rewriter's parallel `SsaValue` window and push ops. A measure
//! driver whose hooks do nothing (pass D) arrives with the exact-plan phase.
//! Value identities never live in the engine.

use tracked_alloc::collections::BTreeSet;

use crate::collections;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        middle::{
            budget::count_live_bank_budget_units,
            cell::CellId,
            frame::{FrameLayoutPlan, FrameSlot},
            ssa_ir::ir::SsaValue,
        },
        wasm::{
            primitive_op::{self, PrimitiveOpKind},
            semantic_ir::SemanticOpKind,
        },
    },
};

// --- Per-op structural table -------------------------------------------------

/// The structural fill/spill discipline for one semantic op, driver-agnostic.
///
/// This is the single per-op table both passes read. Where the planner and
/// rewriter interpret the same op differently, the divergence is named here
/// (a dedicated variant) and each driver applies its own reading — the
/// differences stay visible in one place instead of two parallel matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StructuralAction {
    /// No fill or spill (`block`, `end`, `try_table`, `unreachable`).
    None,
    /// Make `pop` operands live, reserving room for `push` results.
    ///
    /// Primitives and local get/set/tee. The planner just fills (skipping while
    /// unreachable); the rewriter reserves capacity for the operands and the
    /// results before filling.
    PrimitiveFill { pop: u16, push: u16 },
    /// Fill `keep` operands live, then spill every live value except the top
    /// `keep`. `if`, the branch family, `throw`, `throw_ref`.
    FillKeepSpillRest(u16),
    /// Spill every live transient. `loop`, `alloc_exn_ref`, `return` (void or
    /// multi-value).
    SpillAll,
    /// A wasm call. The planner spills everything; the rewriter keeps the live
    /// `params` argument suffix and publishes the rest.
    PrepareCall { params: u16 },
    /// A single-result return (`return` of one value).
    ///
    /// The planner treats it as [`SpillAll`](Self::SpillAll); the rewriter
    /// fills one value and spills the rest. The asymmetry is intentional and
    /// harmless: `mark_unreachable` overwrites `spill_depth = height` right
    /// after every return in the planner's semantic step, so the planner's
    /// spill-all can never reach a captured block-entry state. Do NOT normalize
    /// the two sides into one action.
    ReturnScalar,
    /// `else`. The planner fills the enclosing frame's result arity (control-
    /// stack aware); the rewriter does nothing (the semantic CFG accounts for
    /// the boundary).
    ElsePlannerFill,
}

/// Classify one semantic op's structural discipline.
///
/// `unreachable` returns [`StructuralAction::None`]; callers special-case it
/// before consulting this table because both passes skip it entirely (the
/// rewriter must not even run its trailing capacity check for it).
pub(crate) fn structural_action(op: &SemanticOpKind) -> StructuralAction {
    match op {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => StructuralAction::None,
        SemanticOpKind::Primitive(kind) => {
            let (pop, push) = primitive_op::stack_effect(kind);
            StructuralAction::PrimitiveFill {
                pop: pop as u16,
                push: push as u16,
            }
        }
        SemanticOpKind::LocalGet { .. } => StructuralAction::PrimitiveFill { pop: 0, push: 1 },
        SemanticOpKind::LocalSet { .. } => StructuralAction::PrimitiveFill { pop: 1, push: 0 },
        SemanticOpKind::LocalTee { .. } => StructuralAction::PrimitiveFill { pop: 1, push: 1 },
        SemanticOpKind::Block { .. } | SemanticOpKind::End | SemanticOpKind::TryTable { .. } => {
            StructuralAction::None
        }
        SemanticOpKind::Loop { .. } | SemanticOpKind::AllocExnRef { .. } => {
            StructuralAction::SpillAll
        }
        SemanticOpKind::If { params, .. } => {
            StructuralAction::FillKeepSpillRest(params.saturating_add(1))
        }
        SemanticOpKind::Else { .. } => StructuralAction::ElsePlannerFill,
        SemanticOpKind::Br { arity, .. } => StructuralAction::FillKeepSpillRest(*arity),
        SemanticOpKind::BrIf { arity, .. } | SemanticOpKind::BrOnNull { arity, .. } => {
            StructuralAction::FillKeepSpillRest(arity.saturating_add(1))
        }
        SemanticOpKind::BrOnNonNull { arity, .. }
        | SemanticOpKind::BrOnCast { arity, .. }
        | SemanticOpKind::BrOnCastFail { arity, .. } => StructuralAction::FillKeepSpillRest(*arity),
        SemanticOpKind::BrTable { entries } => {
            let arity = entries.first().map(|entry| entry.arity).unwrap_or(0);
            StructuralAction::FillKeepSpillRest(arity.saturating_add(1))
        }
        SemanticOpKind::CallDirect { params, .. }
        | SemanticOpKind::ReturnCallDirect { params, .. } => {
            StructuralAction::PrepareCall { params: *params }
        }
        SemanticOpKind::CallIndirect { params, .. }
        | SemanticOpKind::CallRef { params, .. }
        | SemanticOpKind::ReturnCallIndirect { params, .. }
        | SemanticOpKind::ReturnCallRef { params, .. } => StructuralAction::PrepareCall {
            params: params.saturating_add(1),
        },
        SemanticOpKind::ReturnVoid => StructuralAction::SpillAll,
        SemanticOpKind::ReturnOne => StructuralAction::ReturnScalar,
        SemanticOpKind::Return { arity } if *arity == 1 => StructuralAction::ReturnScalar,
        SemanticOpKind::Return { .. } => StructuralAction::SpillAll,
        SemanticOpKind::Throw { arity, .. } => StructuralAction::FillKeepSpillRest(*arity),
        SemanticOpKind::ThrowRef => StructuralAction::FillKeepSpillRest(1),
    }
}

// --- Pure window arithmetic --------------------------------------------------

/// New `spill_depth` after filling so at least `operand_count` values are live.
///
/// Filling only ever lowers `spill_depth` (restores spilled values); it never
/// raises it.
#[inline]
pub(crate) fn fill_target(height: u16, spill_depth: u16, operand_count: u16) -> u16 {
    spill_depth.min(height.saturating_sub(operand_count))
}

/// New `spill_depth` after spilling every live value.
#[inline]
pub(crate) fn spill_all_target(height: u16) -> u16 {
    height
}

/// New `spill_depth` after spilling every live value except the top `keep`.
#[inline]
pub(crate) fn spill_except_top_target(height: u16, spill_depth: u16, keep: u16) -> u16 {
    let live_count = height.saturating_sub(spill_depth);
    let count = live_count.saturating_sub(keep);
    spill_depth.saturating_add(count)
}

// --- Emission / identity driver ----------------------------------------------

/// Owns emission and SSA-value identity for one pass. The engine calls these
/// hooks so the driver's parallel `SsaValue` window and op list stay in lockstep
/// with the engine's typed window.
pub(crate) trait DisciplineDriver {
    /// Spill the bottom `count` live values, starting at `base_slot`. The
    /// emitter drains its live-value window and pushes `Spill` ops; the measure
    /// driver does nothing.
    fn on_spill(&mut self, base_slot: FrameSlot, count: usize);

    /// Restore `types` spilled values into the bottom of the live window,
    /// starting at `base_slot`. The emitter allocates fresh SSA values, pushes
    /// `Fill` ops, and prepends the values to its live window; the measure
    /// driver does nothing. Called in bottom-to-top slot order.
    fn on_fill(&mut self, base_slot: FrameSlot, types: &[ValueType]);

    /// Drop a cached local. The emitter pushes `CellDropCache`; the measure
    /// driver does nothing.
    fn on_drop_cache(&mut self, slot: CellId);

    /// A call boundary consumed the whole live window (results return in frame
    /// slots). The emitter clears its live-value window; the measure driver does
    /// nothing.
    fn on_call_boundary(&mut self);

    /// A call boundary left exactly one live scalar `result`. The emitter clears
    /// its live window and pushes `result`; the measure driver does nothing.
    fn on_call_boundary_scalar(&mut self, result: SsaValue);
}

/// Measure-side [`DisciplineDriver`] for pass D's exact-simulation walker.
///
/// The walker evolves only the typed [`Window`] (heights, spill depth, types,
/// alias tags) and observes cache eviction through the resident set the engine
/// mutates in place. It owns no `SsaValue` identity and emits nothing, so every
/// hook is a no-op.
pub(crate) struct NoEmit;

impl DisciplineDriver for NoEmit {
    fn on_spill(&mut self, _base_slot: FrameSlot, _count: usize) {}
    fn on_fill(&mut self, _base_slot: FrameSlot, _types: &[ValueType]) {}
    fn on_drop_cache(&mut self, _slot: CellId) {}
    fn on_call_boundary(&mut self) {}
    fn on_call_boundary_scalar(&mut self, _result: SsaValue) {}
}

/// Whether a cached `local.get` / `local.tee` result can alias the cached slot
/// instead of occupying its own dynamic-bank unit.
///
/// The one exception is a 64-bit local on a 32-bit GP bank: its cached copy is a
/// real linear pair, so the loaded value needs its own registers and cannot be
/// discounted against the resident cache.
#[inline]
pub(crate) fn cached_cell_get_can_source_alias(ty: ValueType, gp_unit_bytes: u8) -> bool {
    !(matches!(ty, ValueType::I64) && gp_unit_bytes == 4)
}

/// Dynamic-bank budget configuration for capacity clamping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BankBudget {
    pub gp_unit_bytes: u8,
    pub gp_live_budget: u8,
    pub fp_live_budget: u8,
}

// --- The engine --------------------------------------------------------------

/// The typed operand window evolved by both passes.
///
/// Invariants: `type_stack.len() == stack_height`, `live_types` and
/// `live_aliases` are the top `stack_height - spill_depth` entries, and
/// `type_stack[spill_depth..] == live_types`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Window {
    stack_height: u16,
    spill_depth: u16,
    type_stack: collections::Vec<ValueType>,
    live_types: collections::Vec<ValueType>,
    live_aliases: collections::Vec<Option<CellId>>,
}

impl Window {
    pub(crate) fn new(
        stack_height: u16,
        spill_depth: u16,
        type_stack: collections::Vec<ValueType>,
        live_types: collections::Vec<ValueType>,
        live_aliases: collections::Vec<Option<CellId>>,
    ) -> Self {
        Self {
            stack_height,
            spill_depth,
            type_stack,
            live_types,
            live_aliases,
        }
    }

    #[inline]
    pub(crate) fn height(&self) -> u16 {
        self.stack_height
    }

    #[inline]
    pub(crate) fn spill_depth(&self) -> u16 {
        self.spill_depth
    }

    #[inline]
    pub(crate) fn live_types(&self) -> &[ValueType] {
        &self.live_types
    }

    #[inline]
    pub(crate) fn live_aliases(&self) -> &[Option<CellId>] {
        &self.live_aliases
    }

    /// Types of the spilled values in `[new_spill_depth, spill_depth)`, i.e. the
    /// values a fill down to `new_spill_depth` would restore, bottom-to-top.
    pub(crate) fn spilled_types_for_fill(
        &self,
        new_spill_depth: u16,
    ) -> collections::Vec<ValueType> {
        let start = new_spill_depth as usize;
        let end = self.spill_depth as usize;
        if start >= end || start >= self.type_stack.len() {
            return collections::Vec::new();
        }
        let end = end.min(self.type_stack.len());
        self.type_stack[start..end].to_vec().into()
    }

    // -- Semantic core mutation (value identity handled by the caller) --------

    /// Pop the top live value's type/alias, lowering height by one.
    pub(crate) fn pop_core(&mut self) {
        self.live_types.pop();
        self.live_aliases.pop();
        self.type_stack.pop();
        self.stack_height = self.stack_height.saturating_sub(1);
        self.spill_depth = self.spill_depth.min(self.stack_height);
    }

    /// Consume the top `count` live values' types/aliases.
    pub(crate) fn consume_core(&mut self, count: usize) {
        let new_len = self.live_types.len().saturating_sub(count);
        self.live_types.truncate(new_len);
        self.live_aliases.truncate(new_len);
        self.stack_height = self.stack_height.saturating_sub(count as u16);
        self.type_stack.truncate(self.stack_height as usize);
        self.spill_depth = self.spill_depth.min(self.stack_height);
    }

    /// Push `types` results with matching `aliases` onto the live window.
    pub(crate) fn push_core(&mut self, types: &[ValueType], aliases: &[Option<CellId>]) {
        debug_assert_eq!(types.len(), aliases.len());
        self.stack_height = self.stack_height.saturating_add(types.len() as u16);
        self.type_stack.extend_from_slice(types);
        self.live_types.extend_from_slice(types);
        self.live_aliases.extend_from_slice(aliases);
    }

    /// The type at absolute stack position `stack_index`, if any.
    #[inline]
    pub(crate) fn type_at_stack_index(&self, stack_index: u16) -> Option<ValueType> {
        self.type_stack.get(stack_index as usize).copied()
    }

    /// Position of `stack_index` within the live-value window, or `None` when it
    /// falls in the spilled prefix or above the top. The driver resolves the
    /// position to an actual SSA value.
    #[inline]
    pub(crate) fn live_position(&self, stack_index: u16) -> Option<usize> {
        if stack_index < self.spill_depth || stack_index >= self.stack_height {
            return None;
        }
        Some((stack_index - self.spill_depth) as usize)
    }

    /// A call consuming `consumed` operands and producing `produced` results
    /// that return in frame slots (nothing left live). `result_types` types the
    /// produced slots (padded with `i64` when short, matching the emit path).
    pub(crate) fn finish_call(
        &mut self,
        driver: &mut impl DisciplineDriver,
        consumed: u16,
        produced: u16,
        result_types: &[ValueType],
    ) {
        let base = self.stack_height.saturating_sub(consumed) as usize;
        self.type_stack.truncate(base);
        self.type_stack.extend_from_slice(result_types);
        debug_assert!(
            result_types.len() >= produced as usize,
            "finish_call: result_types.len()={} but produced={}",
            result_types.len(),
            produced,
        );
        for _ in result_types.len()..(produced as usize) {
            self.type_stack.push(ValueType::I64);
        }
        self.stack_height = base as u16 + produced;
        self.spill_depth = self.stack_height;
        self.live_types.clear();
        self.live_aliases.clear();
        driver.on_call_boundary();
    }

    /// A call consuming `consumed` operands and leaving exactly one live scalar
    /// `result` of type `result_ty`.
    pub(crate) fn finish_call_scalar(
        &mut self,
        driver: &mut impl DisciplineDriver,
        consumed: u16,
        result: SsaValue,
        result_ty: ValueType,
    ) {
        let base = self.stack_height.saturating_sub(consumed);
        self.type_stack.truncate(base as usize);
        self.type_stack.push(result_ty);
        self.stack_height = base.saturating_add(1);
        self.spill_depth = base;
        self.live_types.clear();
        self.live_aliases.clear();
        self.live_types.push(result_ty);
        self.live_aliases.push(None);
        driver.on_call_boundary_scalar(result);
    }

    // -- Discipline transitions ----------------------------------------------

    /// Fill so at least `operand_count` values are live, notifying the driver of
    /// the restored values.
    pub(crate) fn fill(
        &mut self,
        driver: &mut impl DisciplineDriver,
        frame: FrameLayoutPlan,
        operand_count: u16,
    ) {
        let new_spill_depth = fill_target(self.stack_height, self.spill_depth, operand_count);
        if new_spill_depth >= self.spill_depth {
            return;
        }
        let fill_types = self.spilled_types_for_fill(new_spill_depth);
        let base_slot = frame.operand_slot(new_spill_depth);
        driver.on_fill(base_slot, &fill_types);
        self.spill_depth = new_spill_depth;
        let mut new_live_types = fill_types.clone();
        new_live_types.extend_from_slice(&self.live_types);
        self.live_types = new_live_types;
        let mut new_live_aliases = collections::vec![None; fill_types.len()];
        new_live_aliases.extend_from_slice(&self.live_aliases);
        self.live_aliases = new_live_aliases;
    }

    /// Spill the bottom `count` live values, notifying the driver.
    pub(crate) fn spill(
        &mut self,
        driver: &mut impl DisciplineDriver,
        frame: FrameLayoutPlan,
        count: u16,
    ) {
        let count = (count as usize).min(self.live_types.len());
        if count == 0 {
            return;
        }
        let base_slot = frame.operand_slot(self.spill_depth);
        driver.on_spill(base_slot, count);
        self.live_types.drain(..count);
        self.live_aliases.drain(..count);
        self.spill_depth = self.spill_depth.saturating_add(count as u16);
    }

    /// Spill every live value.
    pub(crate) fn spill_all(&mut self, driver: &mut impl DisciplineDriver, frame: FrameLayoutPlan) {
        let count = spill_all_target(self.stack_height).saturating_sub(self.spill_depth);
        self.spill(driver, frame, count);
    }

    /// Spill every live value except the top `keep`.
    pub(crate) fn spill_except_top(
        &mut self,
        driver: &mut impl DisciplineDriver,
        frame: FrameLayoutPlan,
        keep: u16,
    ) {
        let target = spill_except_top_target(self.stack_height, self.spill_depth, keep);
        let count = target.saturating_sub(self.spill_depth);
        self.spill(driver, frame, count);
    }

    /// Reserve room for an op that consumes `operand_count` operands and pushes
    /// `result_types`: the extra GP/FP budget units the op needs beyond what the
    /// already-live operands cover, or that filling the operands demands.
    pub(crate) fn required_capacity(
        &self,
        operand_count: u16,
        result_types: &[ValueType],
        gp_unit_bytes: u8,
    ) -> (usize, usize) {
        let min_spill_depth = self.stack_height.saturating_sub(operand_count);
        let fill_types = self.spilled_types_for_fill(min_spill_depth);
        let (fill_gp, fill_fp) = count_live_bank_budget_units(&fill_types, gp_unit_bytes);

        let already_live = self.live_types.len().min(operand_count as usize);
        let existing = &self.live_types[self.live_types.len().saturating_sub(already_live)..];
        let (existing_gp, existing_fp) = count_live_bank_budget_units(existing, gp_unit_bytes);
        let (result_gp, result_fp) = count_live_bank_budget_units(result_types, gp_unit_bytes);

        (
            fill_gp.max(result_gp.saturating_sub(existing_gp)),
            fill_fp.max(result_fp.saturating_sub(existing_fp)),
        )
    }

    /// Spill bottom transients (and, if none remain, evict the highest-numbered
    /// resident cached local) until the alias-discounted live window plus the
    /// resident cache plus `extra_gp`/`extra_fp` fits the dynamic-bank budget.
    ///
    /// The planner's lightweight measure pass never clamps, but it also never
    /// drives the engine — every engine consumer wants the clamp.
    pub(crate) fn ensure_capacity(
        &mut self,
        driver: &mut impl DisciplineDriver,
        frame: FrameLayoutPlan,
        resident: &mut BTreeSet<CellId>,
        cell_types: &[ValueType],
        budget: BankBudget,
        extra: (usize, usize),
    ) -> Result<(), WasmError> {
        let (extra_gp, extra_fp) = extra;
        loop {
            let (gp_live, fp_live) = self.effective_live_units(resident, budget.gp_unit_bytes);
            let (gp_cache, fp_cache) =
                count_cached_cell_budget_units(resident, cell_types, budget.gp_unit_bytes);
            if gp_live + gp_cache + extra_gp <= budget.gp_live_budget as usize
                && fp_live + fp_cache + extra_fp <= budget.fp_live_budget as usize
            {
                return Ok(());
            }
            // Prefer spilling a transient over dropping a cache.
            if !self.live_types.is_empty() {
                self.spill(driver, frame, 1);
                continue;
            }
            // No transients to spill — drop the weakest cached local.
            if resident.is_empty() {
                return Err(WasmError::internal(
                    "discipline capacity: budget exceeded with nothing to evict",
                ));
            }
            let victim = *resident.iter().next_back().unwrap();
            resident.remove(&victim);
            driver.on_drop_cache(victim);
        }
    }

    /// Alias-discounted live budget: live entries whose alias tag is a resident
    /// cached local do not count against the dynamic bank.
    fn effective_live_units(
        &self,
        resident_cache: &BTreeSet<CellId>,
        gp_unit_bytes: u8,
    ) -> (usize, usize) {
        let effective = self
            .live_types
            .iter()
            .zip(self.live_aliases.iter())
            .filter_map(|(ty, alias)| {
                alias
                    .and_then(|slot| resident_cache.contains(&slot).then_some(()))
                    .is_none()
                    .then_some(*ty)
            })
            .collect::<collections::Vec<_>>();
        count_live_bank_budget_units(&effective, gp_unit_bytes)
    }
}

/// GP/FP budget units held by a resident cached-local set.
pub(crate) fn count_cached_cell_budget_units(
    resident_cache: &BTreeSet<CellId>,
    cell_types: &[ValueType],
    gp_unit_bytes: u8,
) -> (usize, usize) {
    let mut gp = 0usize;
    let mut fp = 0usize;
    for &slot in resident_cache {
        let ty = cell_types
            .get(slot.0 as usize)
            .copied()
            .unwrap_or(ValueType::I64);
        if ty.is_fp() {
            fp += 1;
        } else {
            gp += crate::vm::middle::budget::gp_value_budget_units(ty, gp_unit_bytes);
        }
    }
    (gp, fp)
}

/// Apply one op's structural discipline in the rewriter's clamp-On sequence,
/// driven entirely through the engine. Both the rewriter (emit driver) and the
/// exact-simulation walker (measure driver) call THIS ONE function, so their
/// per-op fill/spill/capacity/eviction can never drift.
///
/// `unreachable` is skipped whole (no prefix, no trailing clamp), matching the
/// rewriter's early return. Every other op runs its structural action followed
/// by a trailing capacity clamp.
pub(crate) fn apply_op_discipline<D: DisciplineDriver>(
    op: &SemanticOpKind,
    window: &mut Window,
    driver: &mut D,
    frame: FrameLayoutPlan,
    resident: &mut BTreeSet<CellId>,
    cell_types: &[ValueType],
    budget: BankBudget,
) -> Result<(), WasmError> {
    if matches!(op, SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)) {
        return Ok(());
    }
    match structural_action(op) {
        StructuralAction::None => {}
        StructuralAction::PrimitiveFill { pop, .. } => {
            // Ensure capacity BEFORE filling (the fill must not be undone by a
            // subsequent spill), reserving room for the pushed results.
            let result_types = emit_result_types(op, cell_types);
            let extra = window.required_capacity(pop, &result_types, budget.gp_unit_bytes);
            window.ensure_capacity(driver, frame, resident, cell_types, budget, extra)?;
            window.fill(driver, frame, pop);
        }
        StructuralAction::FillKeepSpillRest(keep) => {
            window.fill(driver, frame, keep);
            window.spill_except_top(driver, frame, keep);
        }
        StructuralAction::SpillAll => window.spill_all(driver, frame),
        // Publish live values below the top `params` call-arg suffix.
        StructuralAction::PrepareCall { params } => {
            window.spill_except_top(driver, frame, params);
        }
        StructuralAction::ReturnScalar => {
            window.fill(driver, frame, 1);
            window.spill_except_top(driver, frame, 1);
        }
        StructuralAction::ElsePlannerFill => {}
    }
    window.ensure_capacity(driver, frame, resident, cell_types, budget, (0, 0))?;
    Ok(())
}

/// Result types the emit-side capacity reservation charges for an op's pushed
/// results: primitive result type(s), or the accessed local's type for
/// `local.get` / `local.tee`, or none otherwise.
fn emit_result_types(op: &SemanticOpKind, cell_types: &[ValueType]) -> collections::Vec<ValueType> {
    let local_ty = |idx: u16| {
        cell_types
            .get(idx as usize)
            .copied()
            .unwrap_or(ValueType::I64)
    };
    match op {
        SemanticOpKind::Primitive(kind) => {
            let (_pop, push) = primitive_op::stack_effect(kind);
            if push == 0 {
                collections::Vec::new()
            } else {
                let result_ty = primitive_op::result_type(kind).unwrap_or(ValueType::I64);
                collections::vec![result_ty; push]
            }
        }
        SemanticOpKind::LocalGet { idx } | SemanticOpKind::LocalTee { idx } => {
            collections::vec![local_ty(*idx)]
        }
        _ => collections::Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::cached_cell_get_can_source_alias;
    use crate::value_type::ValueType;

    #[test]
    fn gp32_i64_cached_get_needs_real_linear_pair() {
        assert!(!cached_cell_get_can_source_alias(ValueType::I64, 4));
    }

    #[test]
    fn gp64_and_non_i64_cached_gets_can_source_alias() {
        assert!(cached_cell_get_can_source_alias(ValueType::I64, 8));
        assert!(cached_cell_get_can_source_alias(ValueType::I32, 4));
        assert!(cached_cell_get_can_source_alias(ValueType::F64, 4));
    }
}
