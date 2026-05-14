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
            cfg::{CfgBlockId, SemanticCfg},
            frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
            joint_plan::{
                init_locals::locals_reads_before_write, JointPlanner, LocalAccessDecision,
                LocalAccessQuery,
            },
            ssa_ir::{
                ir::{
                    entry_cache_requirement, LocalSlotInfo, SsaBinding, SsaBlock, SsaCallArgs,
                    SsaCallLiveArg, SsaCallOp, SsaCallOperandLoc, SsaEdge, SsaInst, SsaOp,
                    SsaOperand, SsaProgram, SsaTerminator, SsaValue,
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

pub(crate) struct RewriteCfg {
    entry: CfgBlockId,
    blocks: collections::Vec<RewriteCfgBlock>,
}

struct RewriteCfgBlock {
    id: CfgBlockId,
    range: core::ops::Range<usize>,
}

impl RewriteCfg {
    pub(crate) fn from_semantic_cfg(cfg: &SemanticCfg) -> Self {
        let mut blocks = cfg
            .blocks
            .iter()
            .map(|block| RewriteCfgBlock {
                id: block.id,
                range: block.range.clone(),
            })
            .collect::<collections::Vec<_>>();
        blocks.shrink_to_fit();
        Self {
            entry: cfg.entry,
            blocks,
        }
    }

    fn block_for_semantic_index(&self, semantic_index: usize) -> Option<CfgBlockId> {
        let mut lo = 0usize;
        let mut hi = self.blocks.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.blocks[mid].range.end <= semantic_index {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        self.blocks
            .get(lo)
            .and_then(|block| block.range.contains(&semantic_index).then_some(block.id))
    }
}

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
    mut semantic: SemanticProgram,
    cfg: &RewriteCfg,
    planner: JointPlanner,
    frame: FrameLayoutPlan,
) -> Result<SsaProgram, WasmError> {
    if semantic.ops.is_empty() {
        let local_slot_info = collect_local_slot_info(&semantic);
        let local_slot_types = core::mem::take(&mut semantic.local_types);
        drop(semantic);
        return Ok(SsaProgram {
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            local_slot_types,
            local_slot_info,
            block_entry_cached_slots: collections::Vec::new(),
            block_cfg_origins: collections::Vec::new(),
            value_types: collections::Vec::new(),
            value_sink_local: collections::Vec::new(),
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        });
    }

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
            cfg_block.id,
            cfg_block.range.clone(),
            state,
            &semantic,
            &planner,
            frame,
            cfg,
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
        let mut final_entry = final_entry;
        final_entry.shrink_to_fit();
        let mut actual_exit = actual_exit;
        actual_exit.shrink_to_fit();
        block_entry_cached_slots.push(final_entry);
        block_exit_cached_slots.push(actual_exit);
        if track_cfg_origins {
            block_cfg_origins.push(collections::vec![block_index as u32]);
        }
        let mut block = SsaBlock {
            id: SsaTarget(block_index as u32),
            params,
            ops: lowered.ops,
            extra_args: lowered.extra_args,
            terminator: lowered.terminator,
        };
        shrink_ssa_block_storage(&mut block);
        blocks.push(block);
        extra_block_cached_slots.extend(lowered.extra_block_cached_slots);
        extra_block_exit_cached_slots.extend(lowered.extra_block_exit_cached_slots);
        if track_cfg_origins {
            extra_block_cfg_origins.extend(lowered.extra_block_cfg_origins);
        }
        for mut block in lowered.extra_blocks {
            shrink_ssa_block_storage(&mut block);
            extra_blocks.push(block);
        }
    }
    blocks.reserve_exact(extra_blocks.len());
    blocks.extend(extra_blocks);
    block_entry_cached_slots.reserve_exact(extra_block_cached_slots.len());
    block_entry_cached_slots.extend(extra_block_cached_slots);
    block_exit_cached_slots.reserve_exact(extra_block_exit_cached_slots.len());
    block_exit_cached_slots.extend(extra_block_exit_cached_slots);
    if track_cfg_origins {
        block_cfg_origins.reserve_exact(extra_block_cfg_origins.len());
        block_cfg_origins.extend(extra_block_cfg_origins);
    }

    drop(block_params);
    drop(planner);
    let local_slot_info = collect_local_slot_info(&semantic);
    let local_slot_types = core::mem::take(&mut semantic.local_types);
    drop(semantic);

    let mut program = SsaProgram {
        entry: SsaTarget(cfg.entry.0),
        local_slot_types,
        local_slot_info,
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
    drop(block_exit_cached_slots);
    Ok(program)
}

fn shrink_ssa_block_storage(block: &mut SsaBlock) {
    block.params.shrink_to_fit();
    block.ops.shrink_to_fit();
    block.extra_args.shrink_to_fit();
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
    block_id: CfgBlockId,
    semantic_range: core::ops::Range<usize>,
    mut state: BlockState,
    semantic: &SemanticProgram,
    planner: &JointPlanner,
    frame: FrameLayoutPlan,
    rewrite_cfg: &RewriteCfg,
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
            canonicalize_live_window_for_target(
                target,
                rewrite_cfg,
                &mut state,
                frame,
                values,
                planner,
            )?;
        }
        lower_block_body_op(
            semantic,
            block_id,
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
        block_id,
        last_index,
        planner,
        &mut state,
        frame,
        &semantic.local_types,
        &mut resident_cache,
        &mut materialized_cache,
        rewrite_cfg,
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
        SemanticOpKind::BrOnNull { arity, .. } => {
            let keep_live = arity.saturating_add(1);
            inline_fill_for_operands(state, frame, values, keep_live)?;
            inline_spill_all_except_top(state, frame, keep_live)?;
        }
        SemanticOpKind::BrOnNonNull { arity, .. } => {
            inline_fill_for_operands(state, frame, values, *arity)?;
            inline_spill_all_except_top(state, frame, *arity)?;
        }
        SemanticOpKind::BrOnCast { arity, .. } | SemanticOpKind::BrOnCastFail { arity, .. } => {
            inline_fill_for_operands(state, frame, values, *arity)?;
            inline_spill_all_except_top(state, frame, *arity)?;
        }
        SemanticOpKind::BrTable { entries } => {
            let arity = entries.first().map(|entry| entry.arity).unwrap_or(0);
            let keep_live = arity.saturating_add(1);
            inline_fill_for_operands(state, frame, values, keep_live)?;
            inline_spill_all_except_top(state, frame, keep_live)?;
        }
        SemanticOpKind::CallDirect { params, .. } => {
            inline_prepare_call_operands(state, frame, *params)?;
        }
        SemanticOpKind::CallIndirect { params, .. } | SemanticOpKind::CallRef { params, .. } => {
            inline_prepare_call_operands(state, frame, params.saturating_add(1))?;
        }
        SemanticOpKind::ReturnCallDirect { params, .. } => {
            inline_prepare_call_operands(state, frame, *params)?;
        }
        SemanticOpKind::ReturnCallIndirect { params, .. }
        | SemanticOpKind::ReturnCallRef { params, .. } => {
            inline_prepare_call_operands(state, frame, params.saturating_add(1))?;
        }
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => {
            inline_spill_all(state, frame)?;
        }
        SemanticOpKind::AllocExnRef { .. } => {
            inline_spill_all(state, frame)?;
        }
        SemanticOpKind::TryTable { .. } => {}
        SemanticOpKind::Throw { arity, .. } => {
            inline_fill_for_operands(state, frame, values, *arity)?;
            inline_spill_all_except_top(state, frame, *arity)?;
        }
        SemanticOpKind::ThrowRef => {
            inline_fill_for_operands(state, frame, values, 1)?;
            inline_spill_all_except_top(state, frame, 1)?;
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

/// Publish live values below the call operand suffix, while leaving the
/// current bounded operand suffix live for MachineIR call planning.
fn inline_prepare_call_operands(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    consumed: u16,
) -> Result<(), WasmError> {
    let base_slot = frame.operand_slot(state.spill_depth());
    let spilled = state.spill_live_below_top(consumed)?;
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
    block_id: CfgBlockId,
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
            block_id,
            state,
            frame,
            resident_cache,
            materialized_cache,
            values,
        ),
        SemanticOpKind::LocalSet { idx } => lower_local_set(
            *idx,
            planner,
            block_id,
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
            block_id,
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
        SemanticOpKind::CallRef {
            type_idx,
            params,
            results,
        } => {
            let rtypes = semantic
                .op_result_types
                .get(&semantic_index)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            lower_call_ref(
                *type_idx,
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
        SemanticOpKind::AllocExnRef { tag_idx } => lower_alloc_exn_ref(
            *tag_idx,
            semantic,
            semantic_index,
            state,
            planner,
            resident_cache,
            materialized_cache,
            values,
            builder,
        ),
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::End
        | SemanticOpKind::TryTable { .. } => Ok(()),
        SemanticOpKind::Else { .. } => Err(WasmError::internal("else must end a block")),
        SemanticOpKind::If { .. }
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrOnNull { .. }
        | SemanticOpKind::BrOnNonNull { .. }
        | SemanticOpKind::BrOnCast { .. }
        | SemanticOpKind::BrOnCastFail { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::ReturnCallDirect { .. }
        | SemanticOpKind::ReturnCallIndirect { .. }
        | SemanticOpKind::ReturnCallRef { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. }
        | SemanticOpKind::Throw { .. }
        | SemanticOpKind::ThrowRef => Err(WasmError::internal(
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

/// Package args for a primitive op into the 2-inline + overflow-extra_args
/// slot layout. Returns the inline `[SsaOperand; 2]` and the start index of
/// any overflow operands (0 if there are none). All rewrite-originated
/// primitive args are `SsaValue` operands.
fn pack_primitive_args(
    args: &[SsaValue],
    extra_args: &mut collections::Vec<SsaOperand>,
) -> Result<([SsaOperand; 2], u16), WasmError> {
    match args.len() {
        0 => Ok(([SsaOperand::NONE, SsaOperand::NONE], 0)),
        1 => Ok(([SsaOperand::value(args[0]), SsaOperand::NONE], 0)),
        2 => Ok(([SsaOperand::value(args[0]), SsaOperand::value(args[1])], 0)),
        _ => {
            let extra_count = args.len() - 2;
            let Some(new_len) = extra_args.len().checked_add(extra_count) else {
                return Err(WasmError::internal(
                    "block extra_args overflow while lowering primitive operands",
                ));
            };
            if new_len > u16::MAX as usize {
                return Err(WasmError::internal(
                    "block extra_args overflow while lowering primitive operands",
                ));
            }
            let idx = extra_args.len() as u16;
            extra_args.extend(args[2..].iter().copied().map(SsaOperand::value));
            Ok((
                [SsaOperand::value(args[0]), SsaOperand::value(args[1])],
                idx,
            ))
        }
    }
}

fn lower_local_get(
    semantic: &SemanticProgram,
    local_idx: u16,
    planner: &JointPlanner,
    block_id: CfgBlockId,
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
        block: block_id,
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
    block_id: CfgBlockId,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
) -> Result<(), WasmError> {
    let src = state.pop_one()?;
    let slot = frame.local_slot(local_idx);
    let access = planner.local_access(LocalAccessQuery {
        block: block_id,
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
    block_id: CfgBlockId,
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
        block: block_id,
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
    let stack_base = state.height().saturating_sub(params);
    let call_base = frame.operand_slot(stack_base);
    let args = capture_call_args(state, stack_base, call_base, params)?;
    let call_idx = builder.push_call_op(SsaCallOp::CallDirect {
        callee,
        args,
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
    let stack_base = state.height().saturating_sub(consumed);
    let call_base = frame.operand_slot(stack_base);
    let args = capture_call_args(state, stack_base, call_base, params)?;
    let index = capture_operand_loc(
        state,
        stack_base.saturating_add(params),
        call_base.advance(params),
    )?;
    let call_idx = builder.push_call_op(SsaCallOp::CallIndirect {
        type_idx,
        table_idx,
        index,
        args,
        results: FrameSpan::new(call_base, results),
    })?;
    state.ops.push(SsaInst::call(call_idx));
    state.finish_call(consumed, results, result_types);
    resident_cache.clear();
    materialized_cache.clear();
    Ok(())
}

fn lower_call_ref(
    type_idx: u32,
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
    let stack_base = state.height().saturating_sub(consumed);
    let call_base = frame.operand_slot(stack_base);
    let args = capture_call_args(state, stack_base, call_base, params)?;
    let callee_ref = capture_operand_loc(
        state,
        stack_base.saturating_add(params),
        call_base.advance(params),
    )?;
    let call_idx = builder.push_call_op(SsaCallOp::CallRef {
        type_idx,
        callee_ref,
        args,
        results: FrameSpan::new(call_base, results),
    })?;
    state.ops.push(SsaInst::call(call_idx));
    state.finish_call(consumed, results, result_types);
    resident_cache.clear();
    materialized_cache.clear();
    Ok(())
}

fn lower_alloc_exn_ref(
    tag_idx: u32,
    semantic: &SemanticProgram,
    semantic_index: usize,
    state: &mut BlockState,
    planner: &JointPlanner,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
) -> Result<(), WasmError> {
    lower_primitive(
        semantic,
        &PrimitiveOpKind::EhAllocExnRef { tag_idx },
        semantic_index,
        state,
        planner,
        resident_cache,
        materialized_cache,
        values,
        builder,
    )
}

fn lower_tail_call_direct(
    callee: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &BlockState,
) -> Result<SsaTerminator, WasmError> {
    let stack_base = state.height().saturating_sub(params);
    let call_base = frame.operand_slot(stack_base);
    let args = capture_call_args(state, stack_base, call_base, params)?;
    Ok(SsaTerminator::TailCallDirect {
        callee,
        args,
        return_results: return_results(frame, results),
    })
}

fn lower_tail_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &BlockState,
) -> Result<SsaTerminator, WasmError> {
    let consumed = params.saturating_add(1);
    let stack_base = state.height().saturating_sub(consumed);
    let call_base = frame.operand_slot(stack_base);
    let args = capture_call_args(state, stack_base, call_base, params)?;
    let index = capture_operand_loc(
        state,
        stack_base.saturating_add(params),
        call_base.advance(params),
    )?;
    Ok(SsaTerminator::TailCallIndirect {
        type_idx,
        table_idx,
        index,
        args,
        return_results: return_results(frame, results),
    })
}

fn lower_tail_call_ref(
    type_idx: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &BlockState,
) -> Result<SsaTerminator, WasmError> {
    let consumed = params.saturating_add(1);
    let stack_base = state.height().saturating_sub(consumed);
    let call_base = frame.operand_slot(stack_base);
    let args = capture_call_args(state, stack_base, call_base, params)?;
    let callee_ref = capture_operand_loc(
        state,
        stack_base.saturating_add(params),
        call_base.advance(params),
    )?;
    Ok(SsaTerminator::TailCallRef {
        type_idx,
        callee_ref,
        args,
        return_results: return_results(frame, results),
    })
}

fn call_base_slot(frame: FrameLayoutPlan, stack_height: u16, consumed: u16) -> FrameSlot {
    frame.operand_slot(stack_height.saturating_sub(consumed))
}

fn capture_call_args(
    state: &BlockState,
    stack_base: u16,
    frame_base: FrameSlot,
    params: u16,
) -> Result<SsaCallArgs, WasmError> {
    let mut live_suffix = collections::Vec::new();
    let mut param_types = collections::Vec::with_capacity(params as usize);
    let mut stack_prefix_count = params;
    for param_index in 0..params {
        let slot = frame_base.advance(param_index);
        let stack_index = stack_base.saturating_add(param_index);
        let ty = state
            .type_at_stack_index(stack_index)
            .unwrap_or(ValueType::I64);
        param_types.push(ty);
        if let Some(value) = state.value_at_stack_index(stack_index) {
            if stack_prefix_count == params {
                stack_prefix_count = param_index;
            }
            live_suffix.push(SsaCallLiveArg {
                param_index,
                value,
                ty,
                frame_slot: slot,
            });
        } else if !live_suffix.is_empty() {
            return Err(WasmError::internal(
                "SSA call live arguments must be a contiguous suffix",
            ));
        }
    }
    Ok(SsaCallArgs {
        frame_base,
        total_params: params,
        param_types,
        stack_prefix_count,
        live_suffix,
    })
}

fn capture_operand_loc(
    state: &BlockState,
    stack_index: u16,
    slot: FrameSlot,
) -> Result<SsaCallOperandLoc, WasmError> {
    Ok(
        if let Some(value) = state.value_at_stack_index(stack_index) {
            SsaCallOperandLoc::Live {
                value,
                ty: state
                    .type_at_stack_index(stack_index)
                    .unwrap_or(ValueType::I64),
                slot,
            }
        } else {
            SsaCallOperandLoc::Stack { slot }
        },
    )
}

#[inline]
fn cached_local_get_can_source_alias(ty: ValueType, gp_unit_bytes: u8) -> bool {
    !(matches!(ty, ValueType::I64) && gp_unit_bytes == 4)
}

fn emit_ref_is_null_condition(
    ref_value: SsaValue,
    state: &mut BlockState,
    builder: &mut ProgramBuilder,
    values: &mut ValueAlloc,
) -> Result<SsaValue, WasmError> {
    let cond = values.fresh_typed(ValueType::I32);
    let pool_idx = builder.intern_primitive(PrimitiveOpKind::RefIsNull)?;
    let (inline_args, extra_idx) = pack_primitive_args(&[ref_value], &mut state.extra_args)?;
    state
        .ops
        .push(SsaInst::primitive(pool_idx, cond, inline_args, extra_idx));
    Ok(cond)
}

fn emit_ref_test_condition(
    ref_value: SsaValue,
    target_type: ValueType,
    state: &mut BlockState,
    builder: &mut ProgramBuilder,
    values: &mut ValueAlloc,
) -> Result<SsaValue, WasmError> {
    let ValueType::Ref(ref_type) = target_type else {
        return Err(WasmError::internal(
            "br_on_cast target type must be a reference".into(),
        ));
    };
    let cond = values.fresh_typed(ValueType::I32);
    let pool_idx = builder.intern_primitive(PrimitiveOpKind::RefTest { ref_type })?;
    let (inline_args, extra_idx) = pack_primitive_args(&[ref_value], &mut state.extra_args)?;
    state
        .ops
        .push(SsaInst::primitive(pool_idx, cond, inline_args, extra_idx));
    Ok(cond)
}

fn split_ref_value_for_null_test(
    ref_value: SsaValue,
    cond_ty: ValueType,
    refined_ty: ValueType,
    slot: FrameSlot,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> (SsaValue, SsaValue) {
    let cond_ref = values.fresh_typed(cond_ty);
    let refined = values.fresh_typed(refined_ty);
    state.ops.push(SsaInst::spill(slot, ref_value));
    state.ops.push(SsaInst::fill(slot, cond_ref));
    state.ops.push(SsaInst::fill(slot, refined));
    (cond_ref, refined)
}

fn split_ref_value_for_cast_test(
    ref_value: SsaValue,
    cond_ty: ValueType,
    source_ty: ValueType,
    cast_ty: ValueType,
    slot: FrameSlot,
    state: &mut BlockState,
    values: &mut ValueAlloc,
) -> (SsaValue, SsaValue, SsaValue) {
    let cond_ref = values.fresh_typed(cond_ty);
    let source_ref = values.fresh_typed(source_ty);
    let cast_ref = values.fresh_typed(cast_ty);
    state.ops.push(SsaInst::spill(slot, ref_value));
    state.ops.push(SsaInst::fill(slot, cond_ref));
    state.ops.push(SsaInst::fill(slot, source_ref));
    state.ops.push(SsaInst::fill(slot, cast_ref));
    (cond_ref, source_ref, cast_ref)
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
    let _ = frame;
    (results != 0).then(|| FrameSpan::new(FrameSlot(0), results))
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
    block_id: CfgBlockId,
    semantic_index: usize,
    planner: &JointPlanner,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    rewrite_cfg: &RewriteCfg,
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
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::LocalGet { idx } => {
            lower_local_get(
                semantic,
                *idx,
                planner,
                block_id,
                state,
                frame,
                resident_cache,
                materialized_cache,
                values,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::LocalSet { idx } => {
            lower_local_set(
                *idx,
                planner,
                block_id,
                state,
                frame,
                local_slot_types,
                resident_cache,
                materialized_cache,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::LocalTee { idx } => {
            lower_local_tee(
                semantic,
                *idx,
                planner,
                block_id,
                state,
                frame,
                resident_cache,
                materialized_cache,
                values,
            )?;
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::AllocExnRef { tag_idx } => {
            lower_alloc_exn_ref(
                *tag_idx,
                semantic,
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
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::TryTable { .. } => {
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::Throw { tag_idx, arity } => {
            publish_taken_branch_payload_at(0, *arity, state, frame)?;
            let args = FrameSpan::new(call_base_slot(frame, state.height(), *arity), *arity);
            Ok(LoweredTerminator::new(SsaTerminator::EhThrow {
                tag_idx: *tag_idx,
                args,
            }))
        }
        SemanticOpKind::ThrowRef => {
            publish_taken_branch_payload_at(0, 1, state, frame)?;
            let exnref_slot = call_base_slot(frame, state.height(), 1);
            Ok(LoweredTerminator::new(SsaTerminator::EhThrowRef {
                exnref_slot,
            }))
        }
        SemanticOpKind::If { else_target, .. } => {
            let cond = state.pop_one()?;
            maybe_publish_live_window_for_targets(
                &[
                    fallthrough_target(semantic_index, semantic.ops.len())?,
                    *else_target,
                ],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?;
            let else_edge = edge_to_target(
                *else_target,
                state,
                EdgeMapping::Identity,
                frame,
                values,
                rewrite_cfg,
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
            maybe_publish_live_window_for_targets(
                &[*end_target],
                rewrite_cfg,
                state,
                frame,
                planner,
            );
            Ok(LoweredTerminator::new(SsaTerminator::Goto(edge_to_target(
                *end_target,
                state,
                EdgeMapping::Identity,
                frame,
                values,
                rewrite_cfg,
                block_params,
                planner,
            )?)))
        }
        SemanticOpKind::End => {
            maybe_publish_live_window_for_targets(
                &[fallthrough_target(semantic_index, semantic.ops.len())?],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::Br {
            stack_drop,
            arity,
            target,
        } => {
            if target_expects_canonical_payload(*target, *stack_drop, rewrite_cfg, state, planner)?
            {
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
                rewrite_cfg,
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
            let needs_then_bridge = target_expects_canonical_payload(
                *target,
                *stack_drop,
                rewrite_cfg,
                state,
                planner,
            )? && *arity != 0;
            if needs_then_bridge {
                maybe_publish_live_window_for_targets(
                    &[fallthrough],
                    rewrite_cfg,
                    state,
                    frame,
                    planner,
                );
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
                let target_block = rewrite_cfg
                    .block_for_semantic_index(target.index().as_usize())
                    .ok_or_else(|| WasmError::invalid("edge target out of range"))?;
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
                    rewrite_cfg,
                    block_params,
                    planner,
                )?;
                let else_edge = next_edge(
                    semantic_index,
                    semantic.ops.len(),
                    state,
                    frame,
                    values,
                    rewrite_cfg,
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
                if target_expects_canonical_payload(
                    *target,
                    *stack_drop,
                    rewrite_cfg,
                    state,
                    planner,
                )? {
                    publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
                }
                maybe_publish_live_window_for_targets(
                    &[fallthrough],
                    rewrite_cfg,
                    state,
                    frame,
                    planner,
                );
                let then_edge = edge_to_target(
                    *target,
                    state,
                    EdgeMapping::TakenBranch {
                        stack_drop: *stack_drop,
                        payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                    },
                    frame,
                    values,
                    rewrite_cfg,
                    block_params,
                    planner,
                )?;
                let else_edge = next_edge(
                    semantic_index,
                    semantic.ops.len(),
                    state,
                    frame,
                    values,
                    rewrite_cfg,
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
        SemanticOpKind::BrOnNull {
            stack_drop,
            arity,
            target,
            ref_type,
        } => {
            let ref_value = state.pop_one()?;
            let ref_slot = frame.operand_slot(state.height());
            let (cond_ref, refined) = split_ref_value_for_null_test(
                ref_value,
                values.value_type(ref_value),
                *ref_type,
                ref_slot,
                state,
                values,
            );
            let cond = emit_ref_is_null_condition(cond_ref, state, builder, values)?;
            if target_expects_canonical_payload(*target, *stack_drop, rewrite_cfg, state, planner)?
            {
                publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
            }
            let then_edge = edge_to_target(
                *target,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                },
                frame,
                values,
                rewrite_cfg,
                block_params,
                planner,
            )?;
            state.push_results(collections::vec![refined], collections::vec![*ref_type])?;
            let fallthrough = fallthrough_target(semantic_index, semantic.ops.len())?;
            maybe_publish_live_window_for_targets(
                &[fallthrough],
                rewrite_cfg,
                state,
                frame,
                planner,
            );
            let else_edge = next_edge(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                rewrite_cfg,
                block_params,
                planner,
            )?;
            Ok(LoweredTerminator::new(SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            }))
        }
        SemanticOpKind::BrOnNonNull {
            stack_drop,
            arity,
            target,
            ref_type,
        } => {
            let ref_value = state.pop_one()?;
            let ref_slot = frame.operand_slot(state.height());
            let (cond_ref, refined) = split_ref_value_for_null_test(
                ref_value,
                values.value_type(ref_value),
                *ref_type,
                ref_slot,
                state,
                values,
            );
            let cond = emit_ref_is_null_condition(cond_ref, state, builder, values)?;
            let fallthrough = fallthrough_target(semantic_index, semantic.ops.len())?;
            maybe_publish_live_window_for_targets(
                &[fallthrough],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?;
            state.push_results(collections::vec![refined], collections::vec![*ref_type])?;
            if target_expects_canonical_payload(*target, *stack_drop, rewrite_cfg, state, planner)?
            {
                publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
            }
            let else_edge = edge_to_target(
                *target,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                },
                frame,
                values,
                rewrite_cfg,
                block_params,
                planner,
            )?;
            Ok(LoweredTerminator::new(SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            }))
        }
        SemanticOpKind::BrOnCast {
            stack_drop,
            arity,
            target,
            fail_type,
            cast_type,
        } => {
            let ref_value = state.pop_one()?;
            let ref_slot = frame.operand_slot(state.height());
            let (cond_ref, source_ref, cast_ref) = split_ref_value_for_cast_test(
                ref_value,
                values.value_type(ref_value),
                *fail_type,
                *cast_type,
                ref_slot,
                state,
                values,
            );
            let cond = emit_ref_test_condition(cond_ref, *cast_type, state, builder, values)?;
            state.push_results(collections::vec![cast_ref], collections::vec![*cast_type])?;
            if target_expects_canonical_payload(*target, *stack_drop, rewrite_cfg, state, planner)?
            {
                publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
            }
            let then_edge = edge_to_target(
                *target,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                },
                frame,
                values,
                rewrite_cfg,
                block_params,
                planner,
            )?;
            state.consume_top(1)?;
            state.push_results(collections::vec![source_ref], collections::vec![*fail_type])?;
            let fallthrough = fallthrough_target(semantic_index, semantic.ops.len())?;
            maybe_publish_live_window_for_targets(
                &[fallthrough],
                rewrite_cfg,
                state,
                frame,
                planner,
            );
            let else_edge = next_edge(
                semantic_index,
                semantic.ops.len(),
                state,
                frame,
                values,
                rewrite_cfg,
                block_params,
                planner,
            )?;
            Ok(LoweredTerminator::new(SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            }))
        }
        SemanticOpKind::BrOnCastFail {
            stack_drop,
            arity,
            target,
            fail_type,
            cast_type,
        } => {
            let ref_value = state.pop_one()?;
            let ref_slot = frame.operand_slot(state.height());
            let (cond_ref, source_ref, cast_ref) = split_ref_value_for_cast_test(
                ref_value,
                values.value_type(ref_value),
                *fail_type,
                *cast_type,
                ref_slot,
                state,
                values,
            );
            let cond = emit_ref_test_condition(cond_ref, *cast_type, state, builder, values)?;
            state.push_results(collections::vec![cast_ref], collections::vec![*cast_type])?;
            let fallthrough = fallthrough_target(semantic_index, semantic.ops.len())?;
            maybe_publish_live_window_for_targets(
                &[fallthrough],
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?;
            state.consume_top(1)?;
            state.push_results(collections::vec![source_ref], collections::vec![*fail_type])?;
            if target_expects_canonical_payload(*target, *stack_drop, rewrite_cfg, state, planner)?
            {
                publish_taken_branch_payload_at(*stack_drop, *arity, state, frame)?;
            }
            let else_edge = edge_to_target(
                *target,
                state,
                EdgeMapping::TakenBranch {
                    stack_drop: *stack_drop,
                    payload: branch_payload(frame, state.height(), *stack_drop, *arity),
                },
                frame,
                values,
                rewrite_cfg,
                block_params,
                planner,
            )?;
            Ok(LoweredTerminator::new(SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            }))
        }
        SemanticOpKind::BrTable { entries } => {
            let index = state.pop_one()?;
            maybe_publish_taken_branch_payloads(entries, rewrite_cfg, state, frame, planner)?;
            let entries = entries
                .iter()
                .map(|entry| {
                    br_table_edge(
                        entry,
                        branch_payload(frame, state.height(), entry.stack_drop, entry.arity),
                        state,
                        frame,
                        values,
                        rewrite_cfg,
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
                rewrite_cfg,
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
                rewrite_cfg,
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
                rewrite_cfg,
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
                rewrite_cfg,
                block_params,
                planner,
            )?))
        }
        SemanticOpKind::CallRef {
            type_idx,
            params,
            results,
        } => {
            let rtypes = semantic
                .op_result_types
                .get(&semantic_index)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            lower_call_ref(
                *type_idx,
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
                rewrite_cfg,
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
                rewrite_cfg,
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
        SemanticOpKind::ReturnCallDirect {
            callee,
            params,
            results,
        } => Ok(LoweredTerminator::new(lower_tail_call_direct(
            *callee, *params, *results, frame, state,
        )?)),
        SemanticOpKind::ReturnCallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => Ok(LoweredTerminator::new(lower_tail_call_indirect(
            *type_idx, *table_idx, *params, *results, frame, state,
        )?)),
        SemanticOpKind::ReturnCallRef {
            type_idx,
            params,
            results,
        } => Ok(LoweredTerminator::new(lower_tail_call_ref(
            *type_idx, *params, *results, frame, state,
        )?)),
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
    rewrite_cfg: &RewriteCfg,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    planner: &JointPlanner,
) {
    if state.live().is_empty() {
        return;
    }
    let max_target_spill_depth = targets
        .iter()
        .filter_map(|target| {
            rewrite_cfg
                .block_for_semantic_index(target.index().as_usize())
                .map(|block| planner.target_entry(block))
        })
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
    rewrite_cfg: &RewriteCfg,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    planner: &JointPlanner,
) -> Result<(), WasmError> {
    let target_block = rewrite_cfg
        .block_for_semantic_index(target.index().as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range"))?;
    let target_entry = planner.target_entry(target_block);
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
    rewrite_cfg: &RewriteCfg,
    block_params: &[collections::Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaTerminator, WasmError> {
    Ok(SsaTerminator::Goto(next_edge(
        semantic_index,
        semantic_len,
        state,
        frame,
        values,
        rewrite_cfg,
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
    rewrite_cfg: &RewriteCfg,
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
        rewrite_cfg,
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
    rewrite_cfg: &RewriteCfg,
    block_params: &[collections::Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaEdge, WasmError> {
    let target_block = rewrite_cfg
        .block_for_semantic_index(target.index().as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range"))?;
    let target_entry = planner.target_entry(target_block);
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
            let live_values = edge_binding_values(target_params, state, frame, values)?;
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
                let live_values = edge_binding_values(target_params, state, frame, values)?;
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
        target: SsaTarget(target_block.0),
        bindings,
    })
}

fn edge_binding_values(
    target_params: &[SsaValue],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<collections::Vec<SsaValue>, WasmError> {
    let live_needed = target_params.len();
    if live_needed == 0 {
        return Ok(collections::Vec::new());
    }

    let live_start = state
        .live()
        .len()
        .checked_sub(live_needed)
        .ok_or_else(|| WasmError::internal("edge binding underflow".into()))?;
    let live_base = state.spill_depth().saturating_add(live_start as u16);
    let live_values = state.live()[live_start..].to_vec();
    let mut rebound = collections::Vec::with_capacity(live_needed);

    for (offset, (param, value)) in target_params.iter().zip(live_values.iter()).enumerate() {
        let param_ty = values.value_type(*param);
        let value_ty = values.value_type(*value);
        if param_ty == value_ty {
            rebound.push(*value);
            continue;
        }

        match (param_ty, value_ty) {
            (ValueType::Ref(_), ValueType::Ref(_)) => {
                let slot = frame.operand_slot(live_base.saturating_add(offset as u16));
                state.ops.push(SsaInst::spill(slot, *value));
                let rebound_value = values.fresh_typed(param_ty);
                state.ops.push(SsaInst::fill(slot, rebound_value));
                rebound.push(rebound_value);
            }
            _ => {
                return Err(WasmError::internal(
                    "edge binding type refinement requires ref-compatible value".into(),
                ));
            }
        }
    }

    Ok(rebound)
}

fn br_table_edge(
    entry: &BrTableEntry,
    payload: Option<FrameSpan>,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    rewrite_cfg: &RewriteCfg,
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
        rewrite_cfg,
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
    rewrite_cfg: &RewriteCfg,
    state: &BlockState,
    planner: &JointPlanner,
) -> Result<bool, WasmError> {
    let target_block = rewrite_cfg
        .block_for_semantic_index(target.index().as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range"))?;
    let entry = planner.target_entry(target_block);
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
    rewrite_cfg: &RewriteCfg,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    planner: &JointPlanner,
) -> Result<(), WasmError> {
    let mut published = collections::Vec::<u32>::new();
    for entry in entries {
        if !target_expects_canonical_payload(
            entry.target,
            entry.stack_drop,
            rewrite_cfg,
            state,
            planner,
        )? {
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
    let dst = FrameSlot(0);
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
