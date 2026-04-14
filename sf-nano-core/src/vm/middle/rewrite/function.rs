//! Final SSA-IR materialization from the chosen joint plan.
//!
//! This file still holds most of the lowering mechanics. The important split is
//! already in place:
//! - `state.rs` owns mutable transient state and SSA value allocation
//! - `edge.rs` owns cached-local boundary repair block insertion
//! - this file coordinates one block-at-a-time lowering using planner decisions

use crate::collections;

use tracked_alloc::collections::BTreeSet;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        middle::{
            budget::{count_live_bank_budget_units, gp_value_budget_units},
            cfg::SemanticCfg,
            frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
            joint_plan::{
                init_locals::locals_reads_before_write, JointPlanner, LocalAccessDecision,
                LocalAccessQuery,
            },
            ssa_ir::{
                ir::{
                    entry_cache_requirement, LocalSlotInfo, SsaBinding, SsaBlock, SsaCallOp,
                    SsaEdge, SsaInst, SsaOp, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
                },
                target::SsaTarget,
            },
        },
        wasm::{
            common::{BrTableEntry, SemanticTarget},
            primitive_op::{self, PrimitiveOpKind},
            semantic_ir::{SemanticOpKind, SemanticProgram},
        },
    },
};

use super::{
    edge::insert_boundary_repair_blocks,
    state::{make_block_params, BlockState, ValueAlloc},
};

/// Scratch accumulator for the program-level pools during rewrite.
///
/// The rewriter only constructs `SsaProgram` at the very end, but the
/// per-op lowering needs to intern primitive ops and push call ops as it
/// goes. This builder holds those pools and is flushed into the final
/// `SsaProgram`. The `const_pool` is not maintained here because no rewrite
/// pass emits `SsaOperand::Const` — that happens later in `optimize`.
#[derive(Debug, Default)]
pub(super) struct ProgramBuilder {
    primitive_pool: collections::Vec<PrimitiveOpKind>,
    call_ops: collections::Vec<SsaCallOp>,
}

#[inline]
fn track_block_cfg_origins() -> bool {
    cfg!(any(debug_assertions, test))
}

impl ProgramBuilder {
    pub(super) fn intern_primitive(&mut self, kind: PrimitiveOpKind) -> Result<u32, WasmError> {
        if let Some(idx) = self.primitive_pool.iter().position(|k| k == &kind) {
            return Ok(idx as u32);
        }
        if self.primitive_pool.len() >= SsaOp::MAX_PRIMITIVE_POOL {
            return Err(WasmError::internal(
                "SSA-IR primitive op pool overflow: function has more distinct primitive ops than a u16 opcode can address",
            ));
        }
        let idx = self.primitive_pool.len() as u32;
        self.primitive_pool.push(kind);
        Ok(idx)
    }

    pub(super) fn push_call_op(&mut self, call: SsaCallOp) -> Result<u32, WasmError> {
        if self.call_ops.len() > u16::MAX as usize {
            return Err(WasmError::internal(
                "SSA-IR call_ops overflow: function has more calls than fit in SsaInst.meta (u16)",
            ));
        }
        let idx = self.call_ops.len() as u32;
        self.call_ops.push(call);
        Ok(idx)
    }
}

pub(crate) fn rewrite_function(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    planner: &JointPlanner,
    frame: FrameLayoutPlan,
) -> Result<SsaProgram, WasmError> {
    if semantic.ops.is_empty() {
        return Ok(SsaProgram {
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            local_slot_types: semantic.local_types.clone(),
            local_slot_info: collect_local_slot_info(semantic),
            block_entry_cached_slots: collections::Vec::new(),
            block_cfg_origins: collections::Vec::new(),
            value_types: collections::Vec::new(),
            value_sink_local: collections::Vec::new(),
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        });
    }

    let semantic_to_block = cfg
        .semantic_to_block
        .iter()
        .map(|id| SsaTarget(id.0))
        .collect::<collections::Vec<_>>();
    let setup = planner.function_setup();
    let mut values = ValueAlloc::default();
    let block_params = cfg
        .blocks
        .iter()
        .map(|block| {
            let block_open = planner.block_open(block.id);
            make_block_params(block_open.transient.live_types, &mut values)
        })
        .collect::<collections::Vec<_>>();

    let original_block_count = cfg.blocks.len();
    let track_cfg_origins = track_block_cfg_origins();
    let mut builder = ProgramBuilder::default();
    let mut blocks = collections::Vec::with_capacity(original_block_count);
    let mut block_entry_cached_slots = collections::Vec::with_capacity(original_block_count);
    let mut block_exit_cached_slots = collections::Vec::with_capacity(original_block_count);
    let mut block_cfg_origins = if track_cfg_origins {
        collections::Vec::with_capacity(original_block_count)
    } else {
        collections::Vec::new()
    };
    let mut extra_blocks = collections::Vec::new();
    let mut extra_block_cached_slots = collections::Vec::new();
    let mut extra_block_exit_cached_slots = collections::Vec::new();
    let mut extra_block_cfg_origins = collections::Vec::new();
    for (block_index, cfg_block) in cfg.blocks.iter().enumerate() {
        let block_entry = planner.block_open(cfg_block.id);
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(
            block_entry.transient,
            block_entry.stack_types,
            &params,
            setup.gp_unit_bytes,
            setup.gp_dynamic_budget,
            setup.fp_dynamic_budget,
        )?;
        let lowered = lower_block_range(
            cfg_block.range.clone(),
            state,
            semantic,
            planner,
            frame,
            &semantic_to_block,
            &block_params,
            block_entry.cached_locals,
            &mut values,
            &mut builder,
            original_block_count,
            extra_blocks.len(),
        )?;
        let final_entry = filter_block_entry_cached_slots(
            &planner.finalize_block_entry(cfg_block.id, &lowered.actual_exit_cached_slots),
            &lowered.actual_exit_cached_slots,
            &lowered.ops,
        );
        let actual_exit = simulate_materialized_cache_exit(&final_entry, &lowered.ops);
        block_entry_cached_slots.push(final_entry);
        block_exit_cached_slots.push(actual_exit);
        if track_cfg_origins {
            block_cfg_origins.push(collections::vec![block_index as u32]);
        }
        blocks.push(SsaBlock {
            id: SsaTarget(block_index as u32),
            params,
            ops: lowered.ops,
            extra_args: lowered.extra_args,
            terminator: lowered.terminator,
        });
        extra_block_cached_slots.extend(lowered.extra_block_cached_slots);
        extra_block_exit_cached_slots.extend(lowered.extra_block_exit_cached_slots);
        if track_cfg_origins {
            extra_block_cfg_origins.extend(lowered.extra_block_cfg_origins);
        }
        extra_blocks.extend(lowered.extra_blocks);
    }
    blocks.extend(extra_blocks);
    block_entry_cached_slots.extend(extra_block_cached_slots);
    block_exit_cached_slots.extend(extra_block_exit_cached_slots);
    if track_cfg_origins {
        block_cfg_origins.extend(extra_block_cfg_origins);
    }

    let mut program = SsaProgram {
        entry: SsaTarget(cfg.entry.0),
        local_slot_types: semantic.local_types.clone(),
        local_slot_info: collect_local_slot_info(semantic),
        block_entry_cached_slots,
        block_cfg_origins,
        blocks,
        value_types: values.take_types(),
        value_sink_local: collections::Vec::new(),
        const_pool: collections::Vec::new(),
        primitive_pool: builder.primitive_pool,
        call_ops: builder.call_ops,
    };

    if let Some(entry_block) = program
        .blocks
        .iter()
        .find(|block| block.id == program.entry)
    {
        if !entry_block.params.is_empty() {
            return Err(WasmError::internal(
                "entry block unexpectedly has SSA params after middle rewrite",
            ));
        }
    }

    insert_boundary_repair_blocks(&mut program, &block_exit_cached_slots);
    Ok(program)
}

fn filter_block_entry_cached_slots(
    entry_slots: &[FrameSlot],
    actual_exit_slots: &[FrameSlot],
    ops: &[SsaInst],
) -> collections::Vec<FrameSlot> {
    entry_slots
        .iter()
        .copied()
        .filter(|slot| {
            entry_cache_requirement(ops, *slot, actual_exit_slots.contains(slot)).is_some()
        })
        .collect()
}

fn simulate_materialized_cache_exit(
    entry_slots: &[FrameSlot],
    ops: &[SsaInst],
) -> collections::Vec<FrameSlot> {
    let mut materialized = entry_slots.iter().copied().collect::<BTreeSet<_>>();
    for inst in ops {
        match inst.op {
            SsaOp::LOCAL_GET_CACHE | SsaOp::LOCAL_SET_CACHE | SsaOp::LOCAL_ENSURE_CACHE => {
                materialized.insert(FrameSlot(inst.meta));
            }
            SsaOp::LOCAL_RESERVE_CACHE | SsaOp::LOCAL_DROP_CACHE => {
                materialized.remove(&FrameSlot(inst.meta));
            }
            SsaOp::CALL => materialized.clear(),
            _ => {}
        }
    }
    materialized.into_iter().collect()
}

fn collect_local_slot_info(semantic: &SemanticProgram) -> collections::Vec<LocalSlotInfo> {
    let reads_before_write = locals_reads_before_write(semantic);
    (0..semantic.local_count as usize)
        .map(|idx| LocalSlotInfo {
            is_param: (idx as u16) < semantic.params,
            reads_before_write: reads_before_write.get(idx).copied().unwrap_or(true),
        })
        .collect()
}

struct LoweredBlock {
    ops: collections::Vec<SsaInst>,
    extra_args: collections::Vec<SsaOperand>,
    terminator: SsaTerminator,
    actual_exit_cached_slots: collections::Vec<FrameSlot>,
    extra_blocks: collections::Vec<SsaBlock>,
    extra_block_cached_slots: collections::Vec<collections::Vec<FrameSlot>>,
    extra_block_exit_cached_slots: collections::Vec<collections::Vec<FrameSlot>>,
    extra_block_cfg_origins: collections::Vec<collections::Vec<u32>>,
}

/// Lower one CFG block exactly once from the planner-chosen entry state.
///
/// The intended long-term algorithm is:
/// 1. start from the tentative entry boundary
/// 2. consult the planner before each op
/// 3. observe the realized exit
/// 4. later derive the finalized entry and trivial cached-local repair
fn lower_block_range(
    semantic_range: core::ops::Range<usize>,
    mut state: BlockState,
    semantic: &SemanticProgram,
    planner: &JointPlanner,
    frame: FrameLayoutPlan,
    semantic_to_block: &[SsaTarget],
    block_params: &[collections::Vec<SsaValue>],
    entry_cached_locals: &[FrameSlot],
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    original_block_count: usize,
    extra_blocks_len: usize,
) -> Result<LoweredBlock, WasmError> {
    let mut resident_cache = entry_cached_locals.iter().copied().collect::<BTreeSet<_>>();
    let mut materialized_cache = resident_cache.clone();
    ensure_state_fits_with_cache(
        &state,
        &resident_cache,
        &semantic.local_types,
        "block entry boundary",
    )?;
    let last_index = semantic_range
        .end
        .checked_sub(1)
        .ok_or_else(|| WasmError::internal("SSA-IR block cannot be empty"))?;

    for semantic_index in semantic_range.start..last_index {
        if matches!(semantic.ops[semantic_index].kind, SemanticOpKind::End) {
            let target = fallthrough_target(semantic_index, semantic.ops.len())?;
            canonicalize_live_window_for_target(target, &mut state, frame, values, planner)?;
        }
        lower_block_body_op(
            semantic,
            semantic_index,
            planner,
            &mut state,
            frame,
            &semantic.local_types,
            &mut resident_cache,
            &mut materialized_cache,
            values,
            builder,
        )?;
    }

    let terminator = lower_block_terminator(
        semantic,
        last_index,
        planner,
        &mut state,
        frame,
        &semantic.local_types,
        &mut resident_cache,
        &mut materialized_cache,
        semantic_to_block,
        block_params,
        values,
        builder,
        original_block_count,
        extra_blocks_len,
    )?;

    Ok(LoweredBlock {
        ops: state.ops,
        extra_args: state.extra_args,
        terminator: terminator.terminator,
        actual_exit_cached_slots: materialized_cache.iter().copied().collect(),
        extra_blocks: terminator.extra_blocks,
        extra_block_cached_slots: terminator.extra_block_cached_slots,
        extra_block_exit_cached_slots: terminator.extra_block_exit_cached_slots,
        extra_block_cfg_origins: terminator.extra_block_cfg_origins,
    })
}

/// Check the central joint invariant:
/// live transient values plus resident cached locals must fit the dynamic bank
/// budgets at every rewrite point.
fn ensure_state_fits_with_cache(
    state: &BlockState,
    resident_cache: &BTreeSet<FrameSlot>,
    local_slot_types: &[ValueType],
    _context: &str,
) -> Result<(), WasmError> {
    let effective_live_types = state
        .live_types
        .iter()
        .zip(state.live_aliases().iter())
        .filter_map(|(ty, alias)| {
            alias
                .and_then(|slot| resident_cache.contains(&slot).then_some(()))
                .is_none()
                .then_some(*ty)
        })
        .collect::<collections::Vec<_>>();
    let (gp_live, fp_live) =
        count_live_bank_budget_units(&effective_live_types, state.gp_unit_bytes);
    let (gp_cache, fp_cache) =
        count_cached_local_budget_units(resident_cache, local_slot_types, state.gp_unit_bytes);
    if gp_live + gp_cache > state.gp_live_budget as usize
        || fp_live + fp_cache > state.fp_live_budget as usize
    {
        return Err(WasmError::internal(
            "planner exceeded dynamic bank budget during : gp > or fp >",
        ));
    }
    Ok(())
}

fn count_cached_local_budget_units(
    resident_cache: &BTreeSet<FrameSlot>,
    local_slot_types: &[ValueType],
    gp_unit_bytes: u8,
) -> (usize, usize) {
    let mut gp = 0usize;
    let mut fp = 0usize;
    for &slot in resident_cache {
        let ty = local_slot_types
            .get(slot.0 as usize)
            .copied()
            .unwrap_or(ValueType::I64);
        if ty.is_float() {
            fp += 1;
        } else {
            gp += gp_value_budget_units(ty, gp_unit_bytes);
        }
    }
    (gp, fp)
}

/// Inline prefix: fill/spill decisions made directly by the rewriter.
///
/// This replaces the old `lower_prefix_actions` which consulted
/// `before_op_decision` for a pre-computed `TransientContract` target.
/// The rewriter now makes fill/spill decisions locally based on the upcoming
/// op's needs and the current `BlockState`.
fn apply_inline_prefix(
    op: &SemanticOpKind,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    match op {
        SemanticOpKind::Primitive(kind) => {
            if matches!(kind, PrimitiveOpKind::Unreachable) {
                return Ok(());
            }
            let (pop, push) = primitive_op::stack_effect(kind);
            // Ensure capacity BEFORE filling operands. The fill must not be
            // undone by subsequent spills — operands are consumed immediately.
            // Reserve enough room for both:
            // - any currently spilled operands that will be filled into the
            //   live window
            // - any positive per-bank delta after the op replaces its
            //   consumed operands with results
            let result_ty = primitive_op::result_type(kind).unwrap_or(ValueType::I64);
            let result_types = if push == 0 {
                collections::Vec::new()
            } else {
                collections::vec![result_ty; push as usize]
            };
            let (extra_gp, extra_fp) = inline_required_capacity(state, pop as u16, &result_types);
            inline_ensure_capacity(
                state,
                frame,
                resident_cache,
                local_slot_types,
                extra_gp,
                extra_fp,
            )?;
            inline_fill_for_operands(state, frame, values, pop as u16)?;
        }
        SemanticOpKind::LocalGet { idx } => {
            // Reserve room for the push in the correct bank.
            let ty = local_slot_types
                .get(*idx as usize)
                .copied()
                .unwrap_or(ValueType::I64);
            let (extra_gp, extra_fp) = inline_required_capacity(state, 0, &[ty]);
            inline_ensure_capacity(
                state,
                frame,
                resident_cache,
                local_slot_types,
                extra_gp,
                extra_fp,
            )?;
        }
        SemanticOpKind::LocalSet { .. } | SemanticOpKind::LocalTee { .. } => {
            let result_types = match op {
                SemanticOpKind::LocalTee { idx } => collections::vec![local_slot_types
                    .get(*idx as usize)
                    .copied()
                    .unwrap_or(ValueType::I64),],
                _ => collections::Vec::new(),
            };
            let (extra_gp, extra_fp) = inline_required_capacity(state, 1, &result_types);
            inline_ensure_capacity(
                state,
                frame,
                resident_cache,
                local_slot_types,
                extra_gp,
                extra_fp,
            )?;
            inline_fill_for_operands(state, frame, values, 1)?;
        }
        SemanticOpKind::Block { .. } => {}
        SemanticOpKind::Loop { .. } => {
            inline_spill_all(state, frame)?;
        }
        SemanticOpKind::If { params, .. } => {
            let keep_live = params.saturating_add(1);
            inline_fill_for_operands(state, frame, values, keep_live)?;
            inline_spill_all_except_top(state, frame, keep_live)?;
        }
        SemanticOpKind::Else { .. } => {
            // Else restores to block start height + params. The structural
            // effect in apply_semantic_effect handles height/spill_depth changes.
            // No explicit fill needed here — the rewriter processes Else as a
            // control flow boundary that the semantic CFG already accounts for.
        }
        SemanticOpKind::End => {}
        SemanticOpKind::Br { arity, .. } => {
            inline_fill_for_operands(state, frame, values, *arity)?;
            inline_spill_all_except_top(state, frame, *arity)?;
        }
        SemanticOpKind::BrIf { arity, .. } => {
            let keep_live = arity.saturating_add(1);
            inline_fill_for_operands(state, frame, values, keep_live)?;
            inline_spill_all_except_top(state, frame, keep_live)?;
        }
        SemanticOpKind::BrTable { entries } => {
            let arity = entries.first().map(|entry| entry.arity).unwrap_or(0);
            let keep_live = arity.saturating_add(1);
            inline_fill_for_operands(state, frame, values, keep_live)?;
            inline_spill_all_except_top(state, frame, keep_live)?;
        }
        SemanticOpKind::CallDirect { .. }
        | SemanticOpKind::CallIndirect { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {
            inline_spill_all(state, frame)?;
        }
    }
    // Ensure capacity: if live transients + cached locals exceed the budget,
    // spill the bottom transient until it fits. This replaces the old planner's
    // ensure_capacity which was only needed because it ran with tentative cache.
    inline_ensure_capacity(state, frame, resident_cache, local_slot_types, 0, 0)?;
    Ok(())
}

fn inline_required_capacity(
    state: &BlockState,
    operand_count: u16,
    result_types: &[ValueType],
) -> (usize, usize) {
    let min_spill_depth = state.height().saturating_sub(operand_count);
    let fill_types = state.spilled_types_for_fill(min_spill_depth);
    let (fill_gp, fill_fp) = count_live_bank_budget_units(&fill_types, state.gp_unit_bytes);

    let already_live_operands = state.live().len().min(operand_count as usize);
    let existing_operand_types =
        &state.live_types[state.live_types.len().saturating_sub(already_live_operands)..];
    let (existing_gp, existing_fp) =
        count_live_bank_budget_units(existing_operand_types, state.gp_unit_bytes);
    let (result_gp, result_fp) = count_live_bank_budget_units(result_types, state.gp_unit_bytes);

    (
        fill_gp.max(result_gp.saturating_sub(existing_gp)),
        fill_fp.max(result_fp.saturating_sub(existing_fp)),
    )
}

/// Spill bottom transients until live + cache + extra_gp/extra_fp fits the budget.
///
/// `extra_gp` and `extra_fp` reserve room for values about to be pushed
/// (e.g., the result of a primitive op).
fn inline_ensure_capacity(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    resident_cache: &mut BTreeSet<FrameSlot>,
    local_slot_types: &[ValueType],
    extra_gp: usize,
    extra_fp: usize,
) -> Result<(), WasmError> {
    loop {
        let effective_live_types = state
            .live_types
            .iter()
            .zip(state.live_aliases().iter())
            .filter_map(|(ty, alias)| {
                alias
                    .and_then(|slot| resident_cache.contains(&slot).then_some(()))
                    .is_none()
                    .then_some(*ty)
            })
            .collect::<collections::Vec<_>>();
        let (gp_live, fp_live) =
            count_live_bank_budget_units(&effective_live_types, state.gp_unit_bytes);
        let (gp_cache, fp_cache) =
            count_cached_local_budget_units(resident_cache, local_slot_types, state.gp_unit_bytes);
        if gp_live + gp_cache + extra_gp <= state.gp_live_budget as usize
            && fp_live + fp_cache + extra_fp <= state.fp_live_budget as usize
        {
            return Ok(());
        }
        // Prefer spilling a transient over dropping a cache.
        if !state.live().is_empty() {
            let base_slot = frame.operand_slot(state.spill_depth());
            let spilled = state.spill_prefix(1)?;
            for (offset, src) in spilled.into_iter().enumerate() {
                state
                    .ops
                    .push(SsaInst::spill(base_slot.advance(offset as u16), src));
            }
            continue;
        }
        // No transients to spill — drop the weakest cached local.
        if resident_cache.is_empty() {
            return Err(WasmError::internal(
                "inline prefix: budget exceeded with nothing to evict",
            ));
        }
        // Drop the highest-numbered cached local. This is a simpler heuristic
        // than the old planner's `local_keep_key` scoring, but it's sufficient
        // because cache eviction within a block is rare — it only fires when
        // transient pressure + cache exceeds the budget with no transients left
        // to spill. The evicted local will be re-ensured at the next block
        // boundary by edge repair if needed.
        let victim = *resident_cache.iter().next_back().unwrap();
        resident_cache.remove(&victim);
        state.ops.push(SsaInst::local_drop_cache(victim));
    }
}

/// Fill spilled values to make at least `operand_count` values live.
fn inline_fill_for_operands(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    operand_count: u16,
) -> Result<(), WasmError> {
    let min_spill_depth = state.height().saturating_sub(operand_count);
    if state.spill_depth() <= min_spill_depth {
        return Ok(());
    }
    let fill_types = state.spilled_types_for_fill(min_spill_depth);
    let fill_count = fill_types.len();
    let base_slot = frame.operand_slot(min_spill_depth);
    let mut reloaded = collections::Vec::with_capacity(fill_count);
    for (offset, ty) in fill_types.iter().copied().enumerate() {
        let dst = values.fresh_typed(ty);
        state
            .ops
            .push(SsaInst::fill(base_slot.advance(offset as u16), dst));
        reloaded.push(dst);
    }
    state.fill_prefix(reloaded, fill_types)?;
    Ok(())
}

/// Spill all live transient values to frame slots.
fn inline_spill_all(state: &mut BlockState, frame: FrameLayoutPlan) -> Result<(), WasmError> {
    if state.live().is_empty() {
        return Ok(());
    }
    let count = state.live().len() as u16;
    let base_slot = frame.operand_slot(state.spill_depth());
    let spilled = state.spill_prefix(count)?;
    for (offset, src) in spilled.into_iter().enumerate() {
        state
            .ops
            .push(SsaInst::spill(base_slot.advance(offset as u16), src));
    }
    Ok(())
}

/// Spill all except the top `keep_top` live values.
fn inline_spill_all_except_top(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    keep_top: u16,
) -> Result<(), WasmError> {
    let live_count = state.live().len();
    let spill_count = live_count.saturating_sub(keep_top as usize);
    if spill_count == 0 {
        return Ok(());
    }
    let base_slot = frame.operand_slot(state.spill_depth());
    let spilled = state.spill_prefix(spill_count as u16)?;
    for (offset, src) in spilled.into_iter().enumerate() {
        state
            .ops
            .push(SsaInst::spill(base_slot.advance(offset as u16), src));
    }
    Ok(())
}

fn lower_block_body_op(
    semantic: &SemanticProgram,
    semantic_index: usize,
    planner: &JointPlanner,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
) -> Result<(), WasmError> {
    apply_inline_prefix(
        &semantic.ops[semantic_index].kind,
        state,
        frame,
        local_slot_types,
        resident_cache,
        values,
    )?;
    match &semantic.ops[semantic_index].kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => {
            Err(WasmError::internal("unreachable must end a block"))
        }
        SemanticOpKind::Primitive(kind) => lower_primitive(
            semantic,
            kind,
            semantic_index,
            state,
            planner,
            resident_cache,
            materialized_cache,
            values,
            builder,
        ),
        SemanticOpKind::LocalGet { idx } => lower_local_get(
            semantic,
            *idx,
            planner,
            semantic_index,
            state,
            frame,
            resident_cache,
            materialized_cache,
            values,
        ),
        SemanticOpKind::LocalSet { idx } => lower_local_set(
            *idx,
            planner,
            semantic_index,
            state,
            frame,
            local_slot_types,
            resident_cache,
            materialized_cache,
        ),
        SemanticOpKind::LocalTee { idx } => lower_local_tee(
            semantic,
            *idx,
            planner,
            semantic_index,
            state,
            frame,
            resident_cache,
            materialized_cache,
            values,
        ),
        SemanticOpKind::CallDirect {
            callee,
            params,
            results,
        } => {
            let rtypes = semantic
                .op_result_types
                .get(&semantic_index)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            lower_call_direct(
                *callee,
                *params,
                *results,
                rtypes,
                frame,
                state,
                resident_cache,
                materialized_cache,
                builder,
            )
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            let rtypes = semantic
                .op_result_types
                .get(&semantic_index)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            lower_call_indirect(
                *type_idx,
                *table_idx,
                *params,
                *results,
                rtypes,
                frame,
                state,
                resident_cache,
                materialized_cache,
                builder,
            )
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } | SemanticOpKind::End => Ok(()),
        SemanticOpKind::Else { .. } => Err(WasmError::internal("else must end a block")),
        SemanticOpKind::If { .. }
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => Err(WasmError::internal(
            "control/return op must be block terminator".into(),
        )),
    }
}

fn lower_primitive(
    semantic: &SemanticProgram,
    kind: &PrimitiveOpKind,
    semantic_index: usize,
    state: &mut BlockState,
    _planner: &JointPlanner,
    resident_cache: &mut BTreeSet<FrameSlot>,
    _materialized_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
) -> Result<(), WasmError> {
    let (pop, push) = primitive_op::stack_effect(kind);
    let args = state.top_values(pop as usize)?;
    let result_ty = if push == 0 {
        ValueType::I64
    } else if matches!(kind, PrimitiveOpKind::Select) {
        args.first()
            .map(|v| values.value_type(*v))
            .unwrap_or(ValueType::I64)
    } else if let Some(ty) = primitive_op::result_type(kind) {
        ty
    } else {
        semantic
            .op_result_types
            .get(&semantic_index)
            .and_then(|v| v.first().copied())
            .unwrap_or(ValueType::I64)
    };
    state.consume_top(pop as usize)?;
    if push > 1 {
        return Err(WasmError::internal(
            "primitive op produces >1 result; unsupported in flat SsaInst layout",
        ));
    }
    let result = if push == 0 {
        SsaValue::NONE
    } else {
        values.fresh_typed(result_ty)
    };
    let pool_idx = builder.intern_primitive(kind.clone())?;
    let (inline_args, extra_idx) = pack_primitive_args(&args, &mut state.extra_args)?;
    state
        .ops
        .push(SsaInst::primitive(pool_idx, result, inline_args, extra_idx));
    if push != 0 {
        state.push_results(
            collections::vec![result],
            collections::vec![result_ty; push as usize],
        )?;
    }
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "primitive op")
}

/// Package args for a primitive op into the 2-inline + optional extra_args
/// slot layout. Returns the inline `[SsaOperand; 2]` and the extra-arg index
/// (0 if no third operand). All rewrite-originated primitive args are
/// `SsaValue` operands.
fn pack_primitive_args(
    args: &[SsaValue],
    extra_args: &mut collections::Vec<SsaOperand>,
) -> Result<([SsaOperand; 2], u16), WasmError> {
    match args.len() {
        0 => Ok(([SsaOperand::NONE, SsaOperand::NONE], 0)),
        1 => Ok(([SsaOperand::value(args[0]), SsaOperand::NONE], 0)),
        2 => Ok(([SsaOperand::value(args[0]), SsaOperand::value(args[1])], 0)),
        3 => {
            if extra_args.len() >= u16::MAX as usize {
                return Err(WasmError::internal(
                    "block extra_args overflow while lowering 3-arg primitive",
                ));
            }
            let idx = extra_args.len() as u16;
            extra_args.push(SsaOperand::value(args[2]));
            Ok((
                [SsaOperand::value(args[0]), SsaOperand::value(args[1])],
                idx,
            ))
        }
        _ => Err(WasmError::internal(
            "primitive op has >3 args; unsupported in flat SsaInst layout",
        )),
    }
}

fn lower_local_get(
    semantic: &SemanticProgram,
    local_idx: u16,
    planner: &JointPlanner,
    semantic_index: usize,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    // `local.get` always needs one SSA result. The current planner decides
    // whether we also keep the local cached after that load.
    let ty = semantic
        .local_types
        .get(local_idx as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    let slot = frame.local_slot(local_idx);
    let access = planner.local_access(LocalAccessQuery {
        semantic_index,
        slot,
        resident_cache,
    });
    let dst = values.fresh_typed(ty);
    let inst = match access {
        LocalAccessDecision::Slot => SsaInst::local_get_slot(slot, dst),
        LocalAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::local_get_cache(slot, dst)
        }
    };
    state.ops.push(inst);
    let aliases = collections::vec![matches!(access, LocalAccessDecision::Cache)
        .then_some(slot)
        .filter(|_| cached_local_get_can_source_alias(ty, state.gp_unit_bytes))];
    state.push_results_with_aliases(collections::vec![dst], collections::vec![ty], aliases)?;
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "local.get")
}

fn lower_local_set(
    local_idx: u16,
    planner: &JointPlanner,
    semantic_index: usize,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
) -> Result<(), WasmError> {
    let src = state.pop_one()?;
    let slot = frame.local_slot(local_idx);
    let access = planner.local_access(LocalAccessQuery {
        semantic_index,
        slot,
        resident_cache,
    });
    let inst = match access {
        LocalAccessDecision::Slot => SsaInst::local_set_slot(slot, src),
        LocalAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::local_set_cache(slot, src)
        }
    };
    state.ops.push(inst);
    ensure_state_fits_with_cache(state, resident_cache, local_slot_types, "local.set")
}

fn lower_local_tee(
    semantic: &SemanticProgram,
    local_idx: u16,
    planner: &JointPlanner,
    semantic_index: usize,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    // `local.tee` is the same admission question as `local.get`, except stack
    // height stays flat: the source value stays logically available while we
    // decide whether the local also deserves cache residency.
    let src = state.pop_one()?;
    let slot = frame.local_slot(local_idx);
    let ty = semantic
        .local_types
        .get(local_idx as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    let access = planner.local_access(LocalAccessQuery {
        semantic_index,
        slot,
        resident_cache,
    });
    let set_inst = match access {
        LocalAccessDecision::Slot => SsaInst::local_set_slot(slot, src),
        LocalAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::local_set_cache(slot, src)
        }
    };
    state.ops.push(set_inst);
    let dst = values.fresh_typed(ty);
    let get_inst = match access {
        LocalAccessDecision::Slot => SsaInst::local_get_slot(slot, dst),
        LocalAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::local_get_cache(slot, dst)
        }
    };
    state.ops.push(get_inst);
    let aliases = collections::vec![matches!(access, LocalAccessDecision::Cache)
        .then_some(slot)
        .filter(|_| cached_local_get_can_source_alias(ty, state.gp_unit_bytes))];
    state.push_results_with_aliases(collections::vec![dst], collections::vec![ty], aliases)?;
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "local.tee")
}

fn lower_call_direct(
    callee: u32,
    params: u16,
    results: u16,
    result_types: &[ValueType],
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    builder: &mut ProgramBuilder,
) -> Result<(), WasmError> {
    let call_base = call_base_slot(frame, state.height(), params);
    let call_idx = builder.push_call_op(SsaCallOp::CallDirect {
        callee,
        args: FrameSpan::new(call_base, params),
        results: FrameSpan::new(call_base, results),
    })?;
    state.ops.push(SsaInst::call(call_idx));
    state.finish_call(params, results, result_types);
    resident_cache.clear();
    materialized_cache.clear();
    Ok(())
}

fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    result_types: &[ValueType],
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    builder: &mut ProgramBuilder,
) -> Result<(), WasmError> {
    let consumed = params.saturating_add(1);
    let call_base = call_base_slot(frame, state.height(), consumed);
    let call_idx = builder.push_call_op(SsaCallOp::CallIndirect {
        type_idx,
        table_idx,
        index_slot: call_base.advance(params),
        args: FrameSpan::new(call_base, params),
        results: FrameSpan::new(call_base, results),
    })?;
    state.ops.push(SsaInst::call(call_idx));
    state.finish_call(consumed, results, result_types);
    resident_cache.clear();
    materialized_cache.clear();
    Ok(())
}

fn call_base_slot(frame: FrameLayoutPlan, stack_height: u16, consumed: u16) -> FrameSlot {
    frame.operand_slot(stack_height.saturating_sub(consumed))
}

#[inline]
fn cached_local_get_can_source_alias(ty: ValueType, gp_unit_bytes: u8) -> bool {
    !(matches!(ty, ValueType::I64) && gp_unit_bytes == 4)
}

fn branch_payload(
    frame: FrameLayoutPlan,
    current_height: u16,
    stack_drop: u32,
    arity: u16,
) -> Option<FrameSpan> {
    if arity == 0 {
        None
    } else {
        let base = current_height
            .saturating_sub(stack_drop as u16)
            .saturating_sub(arity);
        Some(FrameSpan::new(frame.operand_slot(base), arity))
    }
}

fn return_results(frame: FrameLayoutPlan, results: u16) -> Option<FrameSpan> {
    (results != 0).then(|| FrameSpan::new(frame.operand_slot(0), results))
}

struct LoweredTerminator {
    terminator: SsaTerminator,
    extra_blocks: collections::Vec<SsaBlock>,
    extra_block_cached_slots: collections::Vec<collections::Vec<FrameSlot>>,
    extra_block_exit_cached_slots: collections::Vec<collections::Vec<FrameSlot>>,
    extra_block_cfg_origins: collections::Vec<collections::Vec<u32>>,
}

impl LoweredTerminator {
    fn new(terminator: SsaTerminator) -> Self {
        Self {
            terminator,
            extra_blocks: collections::Vec::new(),
            extra_block_cached_slots: collections::Vec::new(),
            extra_block_exit_cached_slots: collections::Vec::new(),
            extra_block_cfg_origins: collections::Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_block_terminator(
    semantic: &SemanticProgram,
    semantic_index: usize,
    planner: &JointPlanner,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    semantic_to_block: &[SsaTarget],
    block_params: &[collections::Vec<SsaValue>],
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    original_block_count: usize,
    extra_blocks_len: usize,
) -> Result<LoweredTerminator, WasmError> {
    apply_inline_prefix(
        &semantic.ops[semantic_index].kind,
        state,
        frame,
        local_slot_types,
        resident_cache,
        values,
    )?;
    match &semantic.ops[semantic_index].kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => {
            Ok(LoweredTerminator::new(SsaTerminator::TrapUnreachable))
        }
        SemanticOpKind::Primitive(kind) => {
            lower_primitive(
                semantic,
                kind,
                semantic_index,
                state,
                planner,
                resident_cache,
                materialized_cache,
                values,
                builder,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::LocalGet { idx } => {
            lower_local_get(
                semantic,
                *idx,
                planner,
                semantic_index,
                state,
                frame,
                resident_cache,
                materialized_cache,
                values,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::LocalSet { idx } => {
            lower_local_set(
                *idx,
                planner,
                semantic_index,
                state,
                frame,
                local_slot_types,
                resident_cache,
                materialized_cache,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::LocalTee { idx } => {
            lower_local_tee(
                semantic,
                *idx,
                planner,
                semantic_index,
                state,
                frame,
                resident_cache,
                materialized_cache,
                values,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } => {
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::If { else_target, .. } => {
            let cond = state.pop_one()?;
            maybe_publish_live_window_for_targets(
                &[
                    fallthrough_target(semantic_index, semantic.ops.len())?,
                    *else_target,
                ],
                state,
                frame,
                planner,
            );
            let then_edge = next_edge(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?;
            let else_edge = edge_to_target(
                *else_target,
                state,
                EdgeMapping::Identity,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?;
            Ok(LoweredTerminator::new(SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            }))
        }
        SemanticOpKind::Else { end_target } => {
            maybe_publish_live_window_for_targets(&[*end_target], state, frame, planner);
            Ok(LoweredTerminator::new(SsaTerminator::Goto(edge_to_target(
                *end_target,
                state,
                EdgeMapping::Identity,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?)))
        }
        SemanticOpKind::End => {
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::Br {
            stack_drop,
            arity,
            target,
        } => {
            if target_expects_canonical_payload(*target, *stack_drop, state, planner)? {
                publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
            }
            Ok(LoweredTerminator::new(SsaTerminator::Goto(edge_to_target(
                *target,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                },
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?)))
        }
        SemanticOpKind::BrIf {
            stack_drop,
            arity,
            target,
        } => {
            let cond = state.pop_one()?;
            let fallthrough = fallthrough_target(semantic_index, semantic.ops.len())?;
            let needs_then_bridge =
                target_expects_canonical_payload(*target, *stack_drop, state, planner)?
                    && *arity != 0;
            if needs_then_bridge {
                maybe_publish_live_window_for_targets(&[fallthrough], state, frame, planner);
                let payload = state.top_values(*arity as usize)?;
                let then_block_id = SsaTarget((original_block_count + extra_blocks_len) as u32);
                let payload_types = payload
                    .iter()
                    .map(|v| values.value_type(*v))
                    .collect::<collections::Vec<_>>();
                let then_params = values.many_typed(&payload_types);
                let payload_span = branch_payload(frame, state.height(), *stack_drop, *arity)
                    .ok_or_else(|| {
                        WasmError::internal(
                            "taken br_if with payload must have payload span".into(),
                        )
                    })?;
                let target_block = semantic_to_block[target.index().as_usize()];
                let target_params = &block_params[target_block.as_usize()];
                if !target_params.is_empty() {
                    return Err(WasmError::internal(
                        "synthetic br_if then bridge requires canonical-only branch target".into(),
                    ));
                }
                let mut then_ops = collections::Vec::with_capacity(*arity as usize);
                for (offset, param) in then_params.iter().copied().enumerate() {
                    then_ops.push(SsaInst::spill(
                        payload_span.start.advance(offset as u16),
                        param,
                    ));
                }
                let then_edge = SsaEdge {
                    target: then_block_id,
                    bindings: then_params
                        .iter()
                        .copied()
                        .zip(payload.into_iter())
                        .map(|(param, value)| SsaBinding { param, value })
                        .collect(),
                };
                let bridge_target = edge_to_target(
                    *target,
                    state,
                    EdgeMapping::TakenBranch {
                        stack_drop: *stack_drop,
                        payload: None,
                    },
                    frame,
                    values,
                    semantic_to_block,
                    block_params,
                    planner,
                )?;
                let else_edge = next_edge(
                    semantic_index,
                    semantic.ops.len(),
                    state,
                    frame,
                    values,
                    semantic_to_block,
                    block_params,
                    planner,
                )?;
                let bridge_block = SsaBlock {
                    id: then_block_id,
                    params: then_params,
                    ops: then_ops,
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(bridge_target),
                };
                Ok(LoweredTerminator {
                    terminator: SsaTerminator::Branch {
                        cond,
                        then_edge,
                        else_edge,
                    },
                    extra_blocks: collections::vec![bridge_block],
                    extra_block_cached_slots: collections::vec![resident_cache
                        .iter()
                        .copied()
                        .collect()],
                    extra_block_exit_cached_slots: collections::vec![materialized_cache
                        .iter()
                        .copied()
                        .collect()],
                    extra_block_cfg_origins: if track_block_cfg_origins() {
                        collections::vec![collections::Vec::new()]
                    } else {
                        collections::Vec::new()
                    },
                })
            } else {
                if target_expects_canonical_payload(*target, *stack_drop, state, planner)? {
                    publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
                }
                maybe_publish_live_window_for_targets(&[fallthrough], state, frame, planner);
                let then_edge = edge_to_target(
                    *target,
                    state,
                    EdgeMapping::TakenBranch {
                        stack_drop: *stack_drop,
                        payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                    },
                    frame,
                    values,
                    semantic_to_block,
                    block_params,
                    planner,
                )?;
                let else_edge = next_edge(
                    semantic_index,
                    semantic.ops.len(),
                    state,
                    frame,
                    values,
                    semantic_to_block,
                    block_params,
                    planner,
                )?;
                Ok(LoweredTerminator::new(SsaTerminator::Branch {
                    cond,
                    then_edge,
                    else_edge,
                }))
            }
        }
        SemanticOpKind::BrTable { entries } => {
            let index = state.pop_one()?;
            maybe_publish_taken_branch_payloads(entries, state, frame, planner)?;
            let entries = entries
                .iter()
                .map(|entry| {
                    br_table_edge(
                        entry,
                        branch_payload(frame, state.height(), entry.stack_drop, entry.arity),
                        state,
                        frame,
                        values,
                        semantic_to_block,
                        block_params,
                        planner,
                    )
                })
                .collect::<Result<collections::Vec<_>, _>>()?;
            Ok(LoweredTerminator::new(SsaTerminator::BrTable {
                index,
                entries,
            }))
        }
        SemanticOpKind::CallDirect {
            callee,
            params,
            results,
        } => {
            let rtypes = semantic
                .op_result_types
                .get(&semantic_index)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            lower_call_direct(
                *callee,
                *params,
                *results,
                rtypes,
                frame,
                state,
                resident_cache,
                materialized_cache,
                builder,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            let rtypes = semantic
                .op_result_types
                .get(&semantic_index)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            lower_call_indirect(
                *type_idx,
                *table_idx,
                *params,
                *results,
                rtypes,
                frame,
                state,
                resident_cache,
                materialized_cache,
                builder,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(goto_next(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                semantic_to_block,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::ReturnVoid => Ok(LoweredTerminator::new(SsaTerminator::Return {
            results: None,
        })),
        SemanticOpKind::ReturnOne => Ok(LoweredTerminator::new(SsaTerminator::Return {
            results: {
                canonicalize_return_results(state, frame, values, 1, &semantic.result_types);
                return_results(frame, 1)
            },
        })),
        SemanticOpKind::Return { arity } => Ok(LoweredTerminator::new(SsaTerminator::Return {
            results: {
                canonicalize_return_results(state, frame, values, *arity, &semantic.result_types);
                return_results(frame, *arity)
            },
        })),
    }
}

fn fallthrough_target(
    semantic_index: usize,
    semantic_len: usize,
) -> Result<SemanticTarget, WasmError> {
    let next = semantic_index
        .checked_add(1)
        .filter(|next| *next < semantic_len)
        .ok_or_else(|| WasmError::invalid("missing fallthrough target"))?;
    Ok(SemanticTarget::new(next))
}

fn maybe_publish_live_window_for_targets(
    targets: &[SemanticTarget],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    planner: &JointPlanner,
) {
    if state.live().is_empty() {
        return;
    }
    let max_target_spill_depth = targets
        .iter()
        .map(|target| planner.target_entry(target.index().as_usize()))
        .filter(|entry| entry.stack_height == state.height())
        .map(|entry| entry.spill_depth)
        .max()
        .unwrap_or(state.spill_depth());
    if max_target_spill_depth <= state.spill_depth() {
        return;
    }
    let publish_count = max_target_spill_depth.saturating_sub(state.spill_depth()) as usize;
    let base_slot = frame.operand_slot(state.spill_depth());
    let prefix_values = state.live()[..publish_count].to_vec();
    for (offset, value) in prefix_values.into_iter().enumerate() {
        state
            .ops
            .push(SsaInst::spill(base_slot.advance(offset as u16), value));
    }
}

fn canonicalize_live_window_for_target(
    target: SemanticTarget,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    planner: &JointPlanner,
) -> Result<(), WasmError> {
    let target_entry = planner.target_entry(target.index().as_usize());
    if target_entry.stack_height != state.height() {
        return Ok(());
    }
    if target_entry.spill_depth > state.spill_depth() {
        // Spill: target expects more values to be spilled.
        let publish_count = target_entry.spill_depth.saturating_sub(state.spill_depth());
        let base_slot = frame.operand_slot(state.spill_depth());
        let spilled = state.spill_prefix(publish_count)?;
        for (offset, value) in spilled.into_iter().enumerate() {
            state
                .ops
                .push(SsaInst::spill(base_slot.advance(offset as u16), value));
        }
    } else if target_entry.spill_depth < state.spill_depth() {
        // Fill: target expects more values to be live.
        inline_fill_for_operands(
            state,
            frame,
            values,
            state.height().saturating_sub(target_entry.spill_depth),
        )?;
    }
    Ok(())
}

fn goto_next(
    semantic_index: usize,
    semantic_len: usize,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    semantic_to_block: &[SsaTarget],
    block_params: &[collections::Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaTerminator, WasmError> {
    Ok(SsaTerminator::Goto(next_edge(
        semantic_index,
        semantic_len,
        state,
        frame,
        values,
        semantic_to_block,
        block_params,
        planner,
    )?))
}

fn next_edge(
    semantic_index: usize,
    semantic_len: usize,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    semantic_to_block: &[SsaTarget],
    block_params: &[collections::Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaEdge, WasmError> {
    let next = semantic_index
        .checked_add(1)
        .filter(|next| *next < semantic_len)
        .ok_or_else(|| WasmError::invalid("missing fallthrough target"))?;
    edge_to_target(
        SemanticTarget::new(next),
        state,
        EdgeMapping::Identity,
        frame,
        values,
        semantic_to_block,
        block_params,
        planner,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeMapping {
    Identity,
    TakenBranch {
        stack_drop: u32,
        payload: Option<FrameSpan>,
    },
}

fn edge_to_target(
    target: SemanticTarget,
    state: &mut BlockState,
    mapping: EdgeMapping,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    semantic_to_block: &[SsaTarget],
    block_params: &[collections::Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaEdge, WasmError> {
    let target_entry = planner.target_entry(target.index().as_usize());
    let target_block = semantic_to_block
        .get(target.index().as_usize())
        .copied()
        .ok_or_else(|| WasmError::invalid("edge target out of range"))?;
    let target_params = block_params
        .get(target_block.as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range"))?;

    let mapped_height = match mapping {
        EdgeMapping::Identity => state.height(),
        EdgeMapping::TakenBranch { stack_drop, .. } => {
            state.height().saturating_sub(stack_drop as u16)
        }
    };
    if mapped_height != target_entry.stack_height {
        return Err(WasmError::internal(
            "edge to semantic op computes stack height , but target expects",
        ));
    }

    // Ensure target's required live values are in the live window.
    let needed = target_entry.live_value_count();
    if needed as usize > state.live().len() {
        inline_fill_for_operands(state, frame, values, needed)?;
    }

    let bindings = match mapping {
        EdgeMapping::Identity => {
            let live_values = state.top_values(target_entry.live_value_count() as usize)?;
            bind_values(target_params, &live_values)?
        }
        EdgeMapping::TakenBranch { payload, .. } => {
            if target_entry.live_value_count() == 0 {
                if !target_params.is_empty() {
                    return Err(WasmError::internal(
                        "taken branch target expects no live values but still has params".into(),
                    ));
                }
                collections::Vec::new()
            } else {
                let live_needed = target_entry.live_value_count() as usize;
                let live_values = state.top_values(live_needed)?;
                if payload.map(|span| span.count).unwrap_or(0) != live_needed as u16 {
                    return Err(WasmError::internal(
                        "taken branch payload width mismatch".into(),
                    ));
                }
                bind_values(target_params, &live_values)?
            }
        }
    };

    Ok(SsaEdge {
        target: target_block,
        bindings,
    })
}

fn br_table_edge(
    entry: &BrTableEntry,
    payload: Option<FrameSpan>,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    semantic_to_block: &[SsaTarget],
    block_params: &[collections::Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaEdge, WasmError> {
    edge_to_target(
        entry.target,
        state,
        EdgeMapping::TakenBranch {
            stack_drop: entry.stack_drop,
            payload,
        },
        frame,
        values,
        semantic_to_block,
        block_params,
        planner,
    )
}

fn bind_values(
    target_params: &[SsaValue],
    values: &[SsaValue],
) -> Result<collections::Vec<SsaBinding>, WasmError> {
    if target_params.len() != values.len() {
        return Err(WasmError::internal("edge binding mismatch"));
    }
    Ok(target_params
        .iter()
        .zip(values.iter())
        .map(|(param, value)| SsaBinding {
            param: *param,
            value: *value,
        })
        .collect())
}

fn target_expects_canonical_payload(
    target: SemanticTarget,
    stack_drop: u32,
    state: &BlockState,
    planner: &JointPlanner,
) -> Result<bool, WasmError> {
    let entry = planner.target_entry(target.index().as_usize());
    Ok(entry.live_value_count() == 0
        && entry.stack_height == state.height().saturating_sub(stack_drop as u16))
}

fn publish_taken_branch_payload_at(
    stack_drop: u32,
    arity: u16,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
) -> Result<(), WasmError> {
    if arity == 0 {
        return Ok(());
    }
    let payload = state.top_values(arity as usize)?;
    let base_slot = frame.operand_slot(
        state
            .height()
            .saturating_sub(stack_drop as u16)
            .saturating_sub(arity),
    );
    for (offset, value) in payload.into_iter().enumerate() {
        state
            .ops
            .push(SsaInst::spill(base_slot.advance(offset as u16), value));
    }
    Ok(())
}

fn maybe_publish_taken_branch_payloads(
    entries: &[BrTableEntry],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    planner: &JointPlanner,
) -> Result<(), WasmError> {
    let mut published = collections::Vec::<u32>::new();
    for entry in entries {
        if !target_expects_canonical_payload(entry.target, entry.stack_drop, state, planner)? {
            continue;
        }
        if published.contains(&entry.stack_drop) {
            continue;
        }
        publish_taken_branch_payload_at(entry.stack_drop, entry.arity, state, frame)?;
        published.push(entry.stack_drop);
    }
    Ok(())
}

fn canonicalize_return_results(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    arity: u16,
    result_types: &[ValueType],
) {
    if arity == 0 {
        return;
    }
    let src = frame.operand_slot(state.height().saturating_sub(arity));
    let dst = frame.operand_slot(0);
    if src == dst {
        return;
    }
    for offset in 0..arity as usize {
        let value = values.fresh_typed(result_types.get(offset).copied().unwrap_or(ValueType::I64));
        state
            .ops
            .push(SsaInst::fill(src.advance(offset as u16), value));
        state
            .ops
            .push(SsaInst::spill(dst.advance(offset as u16), value));
    }
}

#[cfg(test)]
mod tests {
    use super::cached_local_get_can_source_alias;
    use crate::value_type::ValueType;

    #[test]
    fn gp32_i64_cached_get_needs_real_linear_pair() {
        assert!(!cached_local_get_can_source_alias(ValueType::I64, 4));
    }

    #[test]
    fn gp64_and_non_i64_cached_gets_can_source_alias() {
        assert!(cached_local_get_can_source_alias(ValueType::I64, 8));
        assert!(cached_local_get_can_source_alias(ValueType::I32, 4));
        assert!(cached_local_get_can_source_alias(ValueType::F64, 4));
    }
}
