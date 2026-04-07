//! Final SSA-IR materialization from the chosen joint plan.
//!
//! This file still holds most of the lowering mechanics. The important split is
//! already in place:
//! - `state.rs` owns mutable transient state and SSA value allocation
//! - `edge.rs` owns cached-local boundary repair block insertion
//! - this file coordinates one block-at-a-time lowering using planner decisions

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        middle::{
            budget::{count_live_bank_budget_units, gp_value_budget_units},
            cfg::SemanticCfg,
            frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
            joint_plan::{
                init_locals::locals_reads_before_write, BeforeOpQuery, JointPlanner,
                LocalAccessDecision, LocalAccessQuery, PressureFallbackQuery, TransientContract,
            },
            slot_ssa::SlotSsaProgram,
            ssa_ir::{
                ir::{
                    entry_cache_requirement, LocalSlotInfo, SsaBinding, SsaBlock, SsaCallOp,
                    SsaEdge, SsaInst, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
                },
                leaf::SsaLeafOp,
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

pub(crate) fn rewrite_function(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    _slot_program: &SlotSsaProgram,
    planner: &JointPlanner,
    frame: FrameLayoutPlan,
) -> Result<SsaProgram, WasmError> {
    if semantic.ops.is_empty() {
        return Ok(SsaProgram {
            entry: SsaTarget(0),
            blocks: Vec::new(),
            local_slot_types: semantic.local_types.clone(),
            local_slot_info: collect_local_slot_info(semantic),
            block_entry_cached_slots: Vec::new(),
            block_cfg_origins: Vec::new(),
            value_types: Vec::new(),
            value_sink_local: Vec::new(),
        });
    }

    let semantic_to_block = cfg
        .semantic_to_block
        .iter()
        .map(|id| SsaTarget(id.0))
        .collect::<Vec<_>>();
    let setup = planner.function_setup();
    let mut values = ValueAlloc::default();
    let block_params = cfg
        .blocks
        .iter()
        .map(|block| {
            let block_open = planner.block_open(block.id);
            make_block_params(block_open.transient.live_types, &mut values)
        })
        .collect::<Vec<_>>();

    let original_block_count = cfg.blocks.len();
    let mut blocks = Vec::with_capacity(original_block_count);
    let mut block_entry_cached_slots = Vec::with_capacity(original_block_count);
    let mut block_exit_cached_slots = Vec::with_capacity(original_block_count);
    let mut block_cfg_origins = Vec::with_capacity(original_block_count);
    let mut extra_blocks = Vec::new();
    let mut extra_block_cached_slots = Vec::new();
    let mut extra_block_exit_cached_slots = Vec::new();
    let mut extra_block_cfg_origins = Vec::new();
    for (block_index, cfg_block) in cfg.blocks.iter().enumerate() {
        let block_entry = planner.block_open(cfg_block.id);
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(
            block_entry.transient,
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
        block_cfg_origins.push(alloc::vec![block_index as u32]);
        blocks.push(SsaBlock {
            id: SsaTarget(block_index as u32),
            params,
            ops: lowered.ops,
            terminator: lowered.terminator,
        });
        extra_block_cached_slots.extend(lowered.extra_block_cached_slots);
        extra_block_exit_cached_slots.extend(lowered.extra_block_exit_cached_slots);
        extra_block_cfg_origins.extend(lowered.extra_block_cfg_origins);
        extra_blocks.extend(lowered.extra_blocks);
    }
    blocks.extend(extra_blocks);
    block_entry_cached_slots.extend(extra_block_cached_slots);
    block_exit_cached_slots.extend(extra_block_exit_cached_slots);
    block_cfg_origins.extend(extra_block_cfg_origins);

    let mut program = SsaProgram {
        entry: SsaTarget(cfg.entry.0),
        local_slot_types: semantic.local_types.clone(),
        local_slot_info: collect_local_slot_info(semantic),
        block_entry_cached_slots,
        block_cfg_origins,
        blocks,
        value_types: values.take_types(),
        value_sink_local: Vec::new(),
    };

    if let Some(entry_block) = program
        .blocks
        .iter()
        .find(|block| block.id == program.entry)
    {
        if !entry_block.params.is_empty() {
            return Err(WasmError::internal(alloc::format!(
                "entry block {} unexpectedly has {} SSA params after middle rewrite",
                entry_block.id.0,
                entry_block.params.len()
            )));
        }
    }

    insert_boundary_repair_blocks(&mut program, &block_exit_cached_slots);
    Ok(program)
}

fn filter_block_entry_cached_slots(
    entry_slots: &[FrameSlot],
    actual_exit_slots: &[FrameSlot],
    ops: &[SsaInst],
) -> Vec<FrameSlot> {
    entry_slots
        .iter()
        .copied()
        .filter(|slot| {
            entry_cache_requirement(ops, *slot, actual_exit_slots.contains(slot)).is_some()
        })
        .collect()
}

fn simulate_materialized_cache_exit(entry_slots: &[FrameSlot], ops: &[SsaInst]) -> Vec<FrameSlot> {
    let mut materialized = entry_slots.iter().copied().collect::<BTreeSet<_>>();
    for inst in ops {
        match inst.kind {
            SsaInstKind::LocalGetCache { slot, .. }
            | SsaInstKind::LocalSetCache { slot, .. }
            | SsaInstKind::LocalEnsureCache { slot } => {
                materialized.insert(slot);
            }
            SsaInstKind::LocalReserveCache { slot } | SsaInstKind::LocalDropCache { slot } => {
                materialized.remove(&slot);
            }
            SsaInstKind::Call(_) => materialized.clear(),
            SsaInstKind::Value { .. }
            | SsaInstKind::Fill { .. }
            | SsaInstKind::Spill { .. }
            | SsaInstKind::LocalGetSlot { .. }
            | SsaInstKind::LocalSetSlot { .. } => {}
        }
    }
    materialized.into_iter().collect()
}

fn collect_local_slot_info(semantic: &SemanticProgram) -> Vec<LocalSlotInfo> {
    let reads_before_write = locals_reads_before_write(semantic);
    (0..semantic.local_count as usize)
        .map(|idx| LocalSlotInfo {
            is_param: (idx as u16) < semantic.params,
            reads_before_write: reads_before_write.get(idx).copied().unwrap_or(true),
        })
        .collect()
}

struct LoweredBlock {
    ops: Vec<SsaInst>,
    terminator: SsaTerminator,
    actual_exit_cached_slots: Vec<FrameSlot>,
    extra_blocks: Vec<SsaBlock>,
    extra_block_cached_slots: Vec<Vec<FrameSlot>>,
    extra_block_exit_cached_slots: Vec<Vec<FrameSlot>>,
    extra_block_cfg_origins: Vec<Vec<u32>>,
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
    block_params: &[Vec<SsaValue>],
    entry_cached_locals: &[FrameSlot],
    values: &mut ValueAlloc,
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
        .ok_or_else(|| WasmError::internal("SSA-IR block cannot be empty".into()))?;

    for semantic_index in semantic_range.start..last_index {
        if matches!(semantic.ops[semantic_index].kind, SemanticOpKind::End) {
            let target = fallthrough_target(semantic_index, semantic.ops.len())?;
            canonicalize_live_window_for_target(target, &mut state, frame, planner)?;
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
        original_block_count,
        extra_blocks_len,
    )?;

    Ok(LoweredBlock {
        ops: state.ops,
        terminator: terminator.terminator,
        actual_exit_cached_slots: materialized_cache.iter().copied().collect(),
        extra_blocks: terminator.extra_blocks,
        extra_block_cached_slots: terminator.extra_block_cached_slots,
        extra_block_exit_cached_slots: terminator.extra_block_exit_cached_slots,
        extra_block_cfg_origins: terminator.extra_block_cfg_origins,
    })
}

/// Apply the planner-owned pre-op transition.
///
/// This is where the boundary contract becomes executable:
/// - planner may request dropping some cached locals first
/// - planner then supplies the exact transient contract that must hold
/// - rewrite validates and materializes that contract, but does not invent one
fn lower_prefix_actions(
    semantic_index: usize,
    planner: &JointPlanner,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    let before_op = planner.before_op(BeforeOpQuery {
        semantic_index,
        resident_cache,
    });
    for slot in before_op.drop_cached_locals {
        if !resident_cache.remove(&slot) {
            return Err(WasmError::internal(alloc::format!(
                "planner dropped uncached local slot {} before semantic op {}",
                slot.0,
                semantic_index
            )));
        }
        materialized_cache.remove(&slot);
        state.ops.push(SsaInst {
            kind: SsaInstKind::LocalDropCache { slot },
        });
    }
    realize_transient_contract(before_op.transient, state, frame, values)?;
    for slot in planner.pressure_fallback_drops(PressureFallbackQuery {
        semantic_index,
        resident_cache,
        live_types: &state.live_types,
        live_aliases: state.live_aliases(),
    }) {
        if !resident_cache.remove(&slot) {
            return Err(WasmError::internal(alloc::format!(
                "planner fallback dropped uncached local slot {} before semantic op {}",
                slot.0,
                semantic_index
            )));
        }
        materialized_cache.remove(&slot);
        state.ops.push(SsaInst {
            kind: SsaInstKind::LocalDropCache { slot },
        });
    }
    ensure_state_fits_with_cache(
        state,
        resident_cache,
        local_slot_types,
        "planned pre-op boundary",
    )?;
    Ok(())
}

/// Realize the exact transient contract chosen by the planner.
///
/// The rewriter only performs two legal shape changes here:
/// - fill a spilled prefix back into the live window
/// - spill a live prefix out to frame slots
///
/// Everything else about the stack shape must already agree.
fn realize_transient_contract(
    target: TransientContract<'_>,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    if state.height() != target.stack_height {
        return Err(WasmError::internal(alloc::format!(
            "planner expected stack height {} before op, but rewrite has {}",
            target.stack_height,
            state.height()
        )));
    }

    let current_spill_depth = state.spill_depth();
    if current_spill_depth > target.spill_depth {
        let fill_count = (current_spill_depth - target.spill_depth) as usize;
        if target.live_types.len() < fill_count {
            return Err(WasmError::internal(
                "planner fill contract has fewer types than required".into(),
            ));
        }
        let fill_types = target.live_types[..fill_count].to_vec();
        let mut reloaded = Vec::with_capacity(fill_count);
        let base_slot = frame.operand_slot(target.spill_depth);
        for (offset, ty) in fill_types.iter().copied().enumerate() {
            let dst = values.fresh_typed(ty);
            state.ops.push(SsaInst {
                kind: SsaInstKind::Fill {
                    slot: base_slot.advance(offset as u16),
                    dst,
                },
            });
            reloaded.push(dst);
        }
        state.fill_prefix(reloaded, fill_types)?;
    } else if current_spill_depth < target.spill_depth {
        let spill_count = target.spill_depth - current_spill_depth;
        let base_slot = frame.operand_slot(current_spill_depth);
        let spilled = state.spill_prefix(spill_count)?;
        for (offset, src) in spilled.into_iter().enumerate() {
            state.ops.push(SsaInst {
                kind: SsaInstKind::Spill {
                    slot: base_slot.advance(offset as u16),
                    src,
                },
            });
        }
    }

    if state.spill_depth() != target.spill_depth || state.live_types != target.live_types {
        return Err(WasmError::internal(
            "planner transient contract does not match realized rewrite state".into(),
        ));
    }
    Ok(())
}

/// Check the central joint invariant:
/// live transient values plus resident cached locals must fit the dynamic bank
/// budgets at every rewrite point.
fn ensure_state_fits_with_cache(
    state: &BlockState,
    resident_cache: &BTreeSet<FrameSlot>,
    local_slot_types: &[ValueType],
    context: &str,
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
        .collect::<Vec<_>>();
    let (gp_live, fp_live) =
        count_live_bank_budget_units(&effective_live_types, state.gp_unit_bytes);
    let (gp_cache, fp_cache) =
        count_cached_local_budget_units(resident_cache, local_slot_types, state.gp_unit_bytes);
    if gp_live + gp_cache > state.gp_live_budget as usize
        || fp_live + fp_cache > state.fp_live_budget as usize
    {
        return Err(WasmError::internal(alloc::format!(
            "planner exceeded dynamic bank budget during {context}: gp {} > {} or fp {} > {}",
            gp_live + gp_cache,
            state.gp_live_budget,
            fp_live + fp_cache,
            state.fp_live_budget,
        )));
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

fn apply_pressure_fallback_after_op(
    semantic_index: usize,
    planner: &JointPlanner,
    state: &mut BlockState,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
) -> Result<(), WasmError> {
    for slot in planner.pressure_fallback_drops(PressureFallbackQuery {
        semantic_index,
        resident_cache,
        live_types: &state.live_types,
        live_aliases: state.live_aliases(),
    }) {
        if !resident_cache.remove(&slot) {
            return Err(WasmError::internal(alloc::format!(
                "planner fallback dropped uncached local slot {} after semantic op {}",
                slot.0,
                semantic_index
            )));
        }
        materialized_cache.remove(&slot);
        state.ops.push(SsaInst {
            kind: SsaInstKind::LocalDropCache { slot },
        });
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
) -> Result<(), WasmError> {
    lower_prefix_actions(
        semantic_index,
        planner,
        state,
        frame,
        local_slot_types,
        resident_cache,
        materialized_cache,
        values,
    )?;
    match &semantic.ops[semantic_index].kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => {
            Err(WasmError::internal("unreachable must end a block".into()))
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
            lower_call_direct(
                *callee,
                *params,
                *results,
                frame,
                state,
                resident_cache,
                materialized_cache,
            );
            Ok(())
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(
                *type_idx,
                *table_idx,
                *params,
                *results,
                frame,
                state,
                resident_cache,
                materialized_cache,
            );
            Ok(())
        }
        SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } | SemanticOpKind::End => Ok(()),
        SemanticOpKind::Else { .. } => Err(WasmError::internal("else must end a block".into())),
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
    planner: &JointPlanner,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
    values: &mut ValueAlloc,
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
    let results = (0..push)
        .map(|_| values.fresh_typed(result_ty))
        .collect::<Vec<_>>();
    state.ops.push(SsaInst {
        kind: SsaInstKind::Value {
            op: SsaLeafOp::from_primitive(kind.clone()).expect("primitive leaf"),
            args: args.into_iter().map(SsaOperand::Value).collect(),
            results: results.clone(),
        },
    });
    state.push_results(results, alloc::vec![result_ty; push as usize])?;
    apply_pressure_fallback_after_op(
        semantic_index,
        planner,
        state,
        resident_cache,
        materialized_cache,
    )?;
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "primitive op")
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
    state.ops.push(SsaInst {
        kind: match access {
            LocalAccessDecision::Slot => SsaInstKind::LocalGetSlot { slot, dst },
            LocalAccessDecision::Cache => {
                resident_cache.insert(slot);
                materialized_cache.insert(slot);
                SsaInstKind::LocalGetCache { slot, dst }
            }
        },
    });
    let aliases = alloc::vec![matches!(access, LocalAccessDecision::Cache).then_some(slot)];
    state.push_results_with_aliases(alloc::vec![dst], alloc::vec![ty], aliases)?;
    apply_pressure_fallback_after_op(
        semantic_index,
        planner,
        state,
        resident_cache,
        materialized_cache,
    )?;
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
    state.ops.push(SsaInst {
        kind: match access {
            LocalAccessDecision::Slot => SsaInstKind::LocalSetSlot { slot, src },
            LocalAccessDecision::Cache => {
                resident_cache.insert(slot);
                materialized_cache.insert(slot);
                SsaInstKind::LocalSetCache { slot, src }
            }
        },
    });
    apply_pressure_fallback_after_op(
        semantic_index,
        planner,
        state,
        resident_cache,
        materialized_cache,
    )?;
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
    state.ops.push(SsaInst {
        kind: match access {
            LocalAccessDecision::Slot => SsaInstKind::LocalSetSlot { slot, src },
            LocalAccessDecision::Cache => {
                resident_cache.insert(slot);
                materialized_cache.insert(slot);
                SsaInstKind::LocalSetCache { slot, src }
            }
        },
    });
    let dst = values.fresh_typed(ty);
    state.ops.push(SsaInst {
        kind: match access {
            LocalAccessDecision::Slot => SsaInstKind::LocalGetSlot { slot, dst },
            LocalAccessDecision::Cache => {
                resident_cache.insert(slot);
                materialized_cache.insert(slot);
                SsaInstKind::LocalGetCache { slot, dst }
            }
        },
    });
    let aliases = alloc::vec![matches!(access, LocalAccessDecision::Cache).then_some(slot)];
    state.push_results_with_aliases(alloc::vec![dst], alloc::vec![ty], aliases)?;
    apply_pressure_fallback_after_op(
        semantic_index,
        planner,
        state,
        resident_cache,
        materialized_cache,
    )?;
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "local.tee")
}

fn lower_call_direct(
    callee: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
) {
    let call_base = call_base_slot(frame, state.height(), params);
    state.ops.push(SsaInst {
        kind: SsaInstKind::Call(SsaCallOp::CallDirect {
            callee,
            args: FrameSpan::new(call_base, params),
            results: FrameSpan::new(call_base, results),
        }),
    });
    state.finish_call(params, results);
    resident_cache.clear();
    materialized_cache.clear();
}

fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    resident_cache: &mut BTreeSet<FrameSlot>,
    materialized_cache: &mut BTreeSet<FrameSlot>,
) {
    let consumed = params.saturating_add(1);
    let call_base = call_base_slot(frame, state.height(), consumed);
    state.ops.push(SsaInst {
        kind: SsaInstKind::Call(SsaCallOp::CallIndirect {
            type_idx,
            table_idx,
            index_slot: call_base.advance(params),
            args: FrameSpan::new(call_base, params),
            results: FrameSpan::new(call_base, results),
        }),
    });
    state.finish_call(consumed, results);
    resident_cache.clear();
    materialized_cache.clear();
}

fn call_base_slot(frame: FrameLayoutPlan, stack_height: u16, consumed: u16) -> FrameSlot {
    frame.operand_slot(stack_height.saturating_sub(consumed))
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
    extra_blocks: Vec<SsaBlock>,
    extra_block_cached_slots: Vec<Vec<FrameSlot>>,
    extra_block_exit_cached_slots: Vec<Vec<FrameSlot>>,
    extra_block_cfg_origins: Vec<Vec<u32>>,
}

impl LoweredTerminator {
    fn new(terminator: SsaTerminator) -> Self {
        Self {
            terminator,
            extra_blocks: Vec::new(),
            extra_block_cached_slots: Vec::new(),
            extra_block_exit_cached_slots: Vec::new(),
            extra_block_cfg_origins: Vec::new(),
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
    block_params: &[Vec<SsaValue>],
    values: &mut ValueAlloc,
    original_block_count: usize,
    extra_blocks_len: usize,
) -> Result<LoweredTerminator, WasmError> {
    lower_prefix_actions(
        semantic_index,
        planner,
        state,
        frame,
        local_slot_types,
        resident_cache,
        materialized_cache,
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
                semantic_to_block,
                block_params,
                planner,
            )?;
            let else_edge = edge_to_target(
                *else_target,
                state,
                EdgeMapping::Identity,
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
                    .collect::<Vec<_>>();
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
                let mut then_ops = Vec::with_capacity(*arity as usize);
                for (offset, param) in then_params.iter().copied().enumerate() {
                    then_ops.push(SsaInst {
                        kind: SsaInstKind::Spill {
                            slot: payload_span.start.advance(offset as u16),
                            src: param,
                        },
                    });
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
                    semantic_to_block,
                    block_params,
                    planner,
                )?;
                let else_edge = next_edge(
                    semantic_index,
                    semantic.ops.len(),
                    state,
                    semantic_to_block,
                    block_params,
                    planner,
                )?;
                let bridge_block = SsaBlock {
                    id: then_block_id,
                    params: then_params,
                    ops: then_ops,
                    terminator: SsaTerminator::Goto(bridge_target),
                };
                Ok(LoweredTerminator {
                    terminator: SsaTerminator::Branch {
                        cond,
                        then_edge,
                        else_edge,
                    },
                    extra_blocks: alloc::vec![bridge_block],
                    extra_block_cached_slots: alloc::vec![resident_cache.iter().copied().collect()],
                    extra_block_exit_cached_slots: alloc::vec![materialized_cache
                        .iter()
                        .copied()
                        .collect()],
                    extra_block_cfg_origins: alloc::vec![Vec::new()],
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
                    semantic_to_block,
                    block_params,
                    planner,
                )?;
                let else_edge = next_edge(
                    semantic_index,
                    semantic.ops.len(),
                    state,
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
                        semantic_to_block,
                        block_params,
                        planner,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
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
            lower_call_direct(
                *callee,
                *params,
                *results,
                frame,
                state,
                resident_cache,
                materialized_cache,
            );
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
            lower_call_indirect(
                *type_idx,
                *table_idx,
                *params,
                *results,
                frame,
                state,
                resident_cache,
                materialized_cache,
            );
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
        .ok_or_else(|| WasmError::invalid("missing fallthrough target".into()))?;
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
        .filter(|entry| entry.transient.stack_height == state.height())
        .map(|entry| entry.transient.spill_depth)
        .max()
        .unwrap_or(state.spill_depth());
    if max_target_spill_depth <= state.spill_depth() {
        return;
    }
    let publish_count = max_target_spill_depth.saturating_sub(state.spill_depth()) as usize;
    let base_slot = frame.operand_slot(state.spill_depth());
    let prefix_values = state.live()[..publish_count].to_vec();
    for (offset, value) in prefix_values.into_iter().enumerate() {
        state.ops.push(SsaInst {
            kind: SsaInstKind::Spill {
                slot: base_slot.advance(offset as u16),
                src: value,
            },
        });
    }
}

fn canonicalize_live_window_for_target(
    target: SemanticTarget,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    planner: &JointPlanner,
) -> Result<(), WasmError> {
    let target_entry = planner.target_entry(target.index().as_usize());
    if target_entry.transient.stack_height != state.height()
        || target_entry.transient.spill_depth <= state.spill_depth()
    {
        return Ok(());
    }
    let publish_count = target_entry
        .transient
        .spill_depth
        .saturating_sub(state.spill_depth());
    let base_slot = frame.operand_slot(state.spill_depth());
    let spilled = state.spill_prefix(publish_count)?;
    for (offset, value) in spilled.into_iter().enumerate() {
        state.ops.push(SsaInst {
            kind: SsaInstKind::Spill {
                slot: base_slot.advance(offset as u16),
                src: value,
            },
        });
    }
    Ok(())
}

fn goto_next(
    semantic_index: usize,
    semantic_len: usize,
    state: &BlockState,
    semantic_to_block: &[SsaTarget],
    block_params: &[Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaTerminator, WasmError> {
    Ok(SsaTerminator::Goto(next_edge(
        semantic_index,
        semantic_len,
        state,
        semantic_to_block,
        block_params,
        planner,
    )?))
}

fn next_edge(
    semantic_index: usize,
    semantic_len: usize,
    state: &BlockState,
    semantic_to_block: &[SsaTarget],
    block_params: &[Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaEdge, WasmError> {
    let next = semantic_index
        .checked_add(1)
        .filter(|next| *next < semantic_len)
        .ok_or_else(|| WasmError::invalid("missing fallthrough target".into()))?;
    edge_to_target(
        SemanticTarget::new(next),
        state,
        EdgeMapping::Identity,
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
    state: &BlockState,
    mapping: EdgeMapping,
    semantic_to_block: &[SsaTarget],
    block_params: &[Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaEdge, WasmError> {
    let target_entry = planner.target_entry(target.index().as_usize());
    let target_block = semantic_to_block
        .get(target.index().as_usize())
        .copied()
        .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;
    let target_params = block_params
        .get(target_block.as_usize())
        .ok_or_else(|| WasmError::invalid("edge target out of range".into()))?;

    let mapped_height = match mapping {
        EdgeMapping::Identity => state.height(),
        EdgeMapping::TakenBranch { stack_drop, .. } => {
            state.height().saturating_sub(stack_drop as u16)
        }
    };
    if mapped_height != target_entry.transient.stack_height {
        return Err(WasmError::internal(alloc::format!(
            "edge to semantic op {} computes stack height {}, but target expects {}",
            target.index().as_usize(),
            mapped_height,
            target_entry.transient.stack_height,
        )));
    }

    let bindings = match mapping {
        EdgeMapping::Identity => {
            let live_values =
                state.top_values(target_entry.transient.live_value_count() as usize)?;
            bind_values(target_params, &live_values)?
        }
        EdgeMapping::TakenBranch { payload, .. } => {
            if target_entry.transient.live_value_count() == 0 {
                if !target_params.is_empty() {
                    return Err(WasmError::internal(
                        "taken branch target expects no live values but still has params".into(),
                    ));
                }
                Vec::new()
            } else {
                let live_needed = target_entry.transient.live_value_count() as usize;
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
    state: &BlockState,
    semantic_to_block: &[SsaTarget],
    block_params: &[Vec<SsaValue>],
    planner: &JointPlanner,
) -> Result<SsaEdge, WasmError> {
    edge_to_target(
        entry.target,
        state,
        EdgeMapping::TakenBranch {
            stack_drop: entry.stack_drop,
            payload,
        },
        semantic_to_block,
        block_params,
        planner,
    )
}

fn bind_values(
    target_params: &[SsaValue],
    values: &[SsaValue],
) -> Result<Vec<SsaBinding>, WasmError> {
    if target_params.len() != values.len() {
        return Err(WasmError::internal("edge binding mismatch".into()));
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
    Ok(entry.transient.live_value_count() == 0
        && entry.transient.stack_height == state.height().saturating_sub(stack_drop as u16))
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
        state.ops.push(SsaInst {
            kind: SsaInstKind::Spill {
                slot: base_slot.advance(offset as u16),
                src: value,
            },
        });
    }
    Ok(())
}

fn maybe_publish_taken_branch_payloads(
    entries: &[BrTableEntry],
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    planner: &JointPlanner,
) -> Result<(), WasmError> {
    let mut published = Vec::<u32>::new();
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
        state.ops.push(SsaInst {
            kind: SsaInstKind::Fill {
                slot: src.advance(offset as u16),
                dst: value,
            },
        });
        state.ops.push(SsaInst {
            kind: SsaInstKind::Spill {
                slot: dst.advance(offset as u16),
                src: value,
            },
        });
    }
}
