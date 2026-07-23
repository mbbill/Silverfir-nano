//! Final SSA-IR materialization from the chosen joint plan.
//!
//! This file still holds most of the lowering mechanics. The important split is
//! already in place:
//! - `state.rs` owns mutable transient state and SSA value allocation
//! - `edge.rs` owns cached-local boundary repair block insertion
//! - this file coordinates one block-at-a-time lowering using planner decisions

use crate::collections;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{
            budget::gp_value_budget_units,
            cell::{CellId, CellSet},
            cfg::{CfgBlockId, SemanticCfg},
            discipline,
            frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
            joint_plan::{
                init_locals::locals_reads_before_write, CellAccessDecision, CellAccessQuery,
                JointPlanner,
            },
            ssa_ir::{
                ir::{
                    entry_cache_requirements, CellInfo, EntryCacheRequirement, SsaBinding,
                    SsaBlock, SsaCallArgs, SsaCallLiveArg, SsaCallOp, SsaCallOperandLoc, SsaEdge,
                    SsaInst, SsaOp, SsaOperand, SsaProgram, SsaScalarResultLoc, SsaTerminator,
                    SsaValue,
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
    edge::{insert_boundary_repair_blocks, PlanRepairs},
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
    /// Pool indices sorted by primitive value. The pool itself remains in
    /// insertion order because `SsaOp` encodes its indices directly.
    primitive_index: collections::Vec<u32>,
    call_ops: collections::Vec<SsaCallOp>,
}

impl ProgramBuilder {
    pub(super) fn intern_primitive(&mut self, kind: &PrimitiveOpKind) -> Result<u32, WasmError> {
        match self
            .primitive_index
            .binary_search_by(|pool_idx| self.primitive_pool[*pool_idx as usize].cmp(kind))
        {
            Ok(index_pos) => return Ok(self.primitive_index[index_pos]),
            Err(index_pos) => {
                if self.primitive_pool.len() >= SsaOp::MAX_PRIMITIVE_POOL {
                    return Err(WasmError::internal(
                        "SSA-IR primitive op pool overflow: function has more distinct primitive ops than a u16 opcode can address",
                    ));
                }
                let pool_idx = self.primitive_pool.len() as u32;
                self.primitive_pool.push(kind.clone());
                self.primitive_index.insert(index_pos, pool_idx);
                Ok(pool_idx)
            }
        }
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
    config: BackendConfig,
) -> Result<SsaProgram, WasmError> {
    if semantic.ops.is_empty() {
        let cell_info = collect_local_slot_info(&semantic);
        let cell_types = core::mem::take(&mut semantic.local_types);
        let result_types = core::mem::take(&mut semantic.result_types);
        drop(semantic);
        let cell_homes = (0..cell_types.len() as u16)
            .map(|i| frame.local_slot(i))
            .collect();
        return Ok(SsaProgram {
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            cell_types,
            result_types,
            cell_info,
            block_entry_cached_cells: collections::Vec::new(),
            block_entry_cache_requirements: collections::Vec::new(),
            preferred_preserved: collections::Vec::new(),
            value_types: collections::Vec::new(),
            value_sink_cell: collections::Vec::new(),
            cell_homes,
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
    let mut builder = ProgramBuilder::default();
    let mut blocks = collections::Vec::with_capacity(original_block_count);
    let mut block_entry_cached_cells = collections::Vec::with_capacity(original_block_count);
    let mut block_entry_cache_requirements = collections::Vec::with_capacity(original_block_count);
    let mut block_exit_cached_slots = collections::Vec::with_capacity(original_block_count);
    let mut extra_blocks = collections::Vec::new();
    let mut extra_block_cached_slots = collections::Vec::new();
    let mut extra_block_cache_requirements = collections::Vec::new();
    let mut extra_block_exit_cached_slots = collections::Vec::new();
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
            // Lower once from the solver's public resident set. After emission
            // we trim unused entry residents and replay only the compact cache
            // op stream to publish the realized entry/exit rows.
            planner.planned_residents(cfg_block.id),
            &mut values,
            &mut builder,
            original_block_count,
            extra_blocks.len(),
            config,
        )?;
        let entry_slots = filter_block_entry_cached_cells(
            planner.planned_residents(cfg_block.id),
            &lowered.hint_exit_cached_cells,
            &lowered.ops,
        );
        let exit_slots = lowered.actual_exit_cached_cells;
        // The no-output exact walker remains a debug oracle. Release builds do
        // not execute it; these rows are empty there and the branch folds away.
        if cfg!(debug_assertions) {
            debug_assert_eq!(
                entry_slots.as_slice(),
                planner.exact_entry(cfg_block.id),
                "block {block_index}: emit-derived entry cache set diverged from the exact walker",
            );
            debug_assert_eq!(
                exit_slots.as_slice(),
                planner.exact_exit(cfg_block.id),
                "block {block_index}: emit-derived exit cache set diverged from the exact walker",
            );
        }
        // The requirement row is a parallel placeholder here; it is finalized in
        // `prepare_function` by scanning the FINAL (post-cleanup) block ops, which
        // is the only faithful view once merges fold blocks together. We keep it
        // 1:1 with the entry set through cleanup so the parallel-row invariant
        // (cleanup's re-index, validate's length check) holds.
        block_entry_cache_requirements
            .push(collections::vec![EntryCacheRequirement::Ensure; entry_slots.len()]);
        block_entry_cached_cells.push(entry_slots);
        block_exit_cached_slots.push(exit_slots);
        let mut block = SsaBlock {
            id: SsaTarget(block_index as u32),
            params,
            ops: lowered.ops,
            extra_args: lowered.extra_args,
            terminator: lowered.terminator,
        };
        shrink_ssa_block_storage(&mut block);
        blocks.push(block);
        // Bridge blocks (br_if canonical payload) thread the origin block's live
        // cache; each threaded cache arrives materialized, so its requirement is
        // Ensure. The machine assigns the physical lane.
        for bridge_slots in &lowered.extra_block_cached_slots {
            extra_block_cache_requirements
                .push(collections::vec![EntryCacheRequirement::Ensure; bridge_slots.len()]);
        }
        extra_block_cached_slots.extend(lowered.extra_block_cached_slots);
        extra_block_exit_cached_slots.extend(lowered.extra_block_exit_cached_slots);
        for mut block in lowered.extra_blocks {
            shrink_ssa_block_storage(&mut block);
            extra_blocks.push(block);
        }
    }
    blocks.reserve_exact(extra_blocks.len());
    blocks.extend(extra_blocks);
    block_entry_cached_cells.reserve_exact(extra_block_cached_slots.len());
    block_entry_cached_cells.extend(extra_block_cached_slots);
    block_entry_cache_requirements.reserve_exact(extra_block_cache_requirements.len());
    block_entry_cache_requirements.extend(extra_block_cache_requirements);
    block_exit_cached_slots.reserve_exact(extra_block_exit_cached_slots.len());
    block_exit_cached_slots.extend(extra_block_exit_cached_slots);

    drop(block_params);
    let cell_info = collect_local_slot_info(&semantic);
    let cell_types = core::mem::take(&mut semantic.local_types);
    let result_types = core::mem::take(&mut semantic.result_types);
    drop(semantic);

    let cell_homes = (0..cell_types.len() as u16)
        .map(|i| frame.local_slot(i))
        .collect();
    let mut program = SsaProgram {
        entry: SsaTarget(cfg.entry.0),
        cell_types,
        result_types,
        cell_info,
        block_entry_cached_cells,
        block_entry_cache_requirements,
        // Finalized in `prepare_function` over the FINAL SSA (the literal 2c2e010
        // cross-count dataflow); empty here.
        preferred_preserved: collections::Vec::new(),
        blocks,
        value_types: values.take_types(),
        value_sink_cell: collections::Vec::new(),
        cell_homes,
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

    // Reconcile edges from the realized predecessor/successor rows. Passing a
    // semantic count of zero routes every edge through the emit-derived path;
    // the exact plan's repair pool exists only in debug builds as an oracle.
    insert_boundary_repair_blocks(
        &mut program,
        &block_exit_cached_slots,
        PlanRepairs {
            pool: &[],
            slot_arena: &[],
            index_arena: &[],
            spans: &[],
        },
        0,
    );
    drop(planner);
    drop(block_exit_cached_slots);
    // Structural standing guard: the block vector and its parallel entry-cache
    // row + requirement-row vectors stay in lockstep, and every block id is its
    // own index. A bridge or repair block pushed without its cache rows would
    // desync these.
    debug_assert_eq!(
        program.blocks.len(),
        program.block_entry_cached_cells.len(),
        "block vector and entry-cache-row vector desynced after repair insertion",
    );
    debug_assert_eq!(
        program.blocks.len(),
        program.block_entry_cache_requirements.len(),
        "block vector and entry-cache-requirement-row vector desynced after repair insertion",
    );
    debug_assert!(
        program
            .blocks
            .iter()
            .enumerate()
            .all(|(index, block)| block.id == SsaTarget(index as u32)),
        "block id layout is not identity after repair insertion",
    );
    Ok(program)
}

fn shrink_ssa_block_storage(block: &mut SsaBlock) {
    block.params.shrink_to_fit();
    block.ops.shrink_to_fit();
    block.extra_args.shrink_to_fit();
}

fn collect_local_slot_info(semantic: &SemanticProgram) -> collections::Vec<CellInfo> {
    let reads_before_write = locals_reads_before_write(semantic);
    (0..semantic.local_count as usize)
        .map(|idx| CellInfo {
            is_param: (idx as u16) < semantic.params,
            reads_before_write: reads_before_write.get(idx).copied().unwrap_or(true),
        })
        .collect()
}

/// Trim the solver's public entry set to cells the emitted block consumes or
/// carries through. This is intentionally based on emitted SSA, making rewrite
/// the release-build authority instead of predicting it with a second semantic
/// walk.
fn filter_block_entry_cached_cells(
    planned: &[CellId],
    hint_exit: &[CellId],
    ops: &[SsaInst],
) -> collections::Vec<CellId> {
    let requirements = entry_cache_requirements(ops, planned, hint_exit);
    planned
        .iter()
        .copied()
        .zip(requirements)
        .filter_map(|(cell, requirement)| requirement.map(|_| cell))
        .collect()
}

struct LoweredBlock {
    ops: collections::Vec<SsaInst>,
    extra_args: collections::Vec<SsaOperand>,
    terminator: SsaTerminator,
    actual_exit_cached_cells: collections::Vec<CellId>,
    hint_exit_cached_cells: collections::Vec<CellId>,
    extra_blocks: collections::Vec<SsaBlock>,
    extra_block_cached_slots: collections::Vec<collections::Vec<CellId>>,
    extra_block_exit_cached_slots: collections::Vec<collections::Vec<CellId>>,
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
    entry_cached_cells: &[CellId],
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    original_block_count: usize,
    extra_blocks_len: usize,
    config: BackendConfig,
) -> Result<LoweredBlock, WasmError> {
    let mut resident_cache = entry_cached_cells.iter().copied().collect::<CellSet>();
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
            config,
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
        config,
    )?;

    Ok(LoweredBlock {
        ops: state.ops,
        extra_args: state.extra_args,
        terminator: terminator.terminator,
        // This is the actual resident set maintained by the authoritative
        // lowering walk. Any planned entry cell trimmed above was necessarily
        // killed by a call or an emitted eviction here, so no second replay is
        // needed to obtain the final exit row.
        actual_exit_cached_cells: resident_cache.iter().copied().collect(),
        // Admission hint used to decide which untouched planned residents were
        // actually carried through. Unlike `resident_cache`, this intentionally
        // ignores evictions; the exact exit is replayed from emitted cache ops
        // after the entry set has been trimmed.
        hint_exit_cached_cells: materialized_cache.iter().copied().collect(),
        extra_blocks: terminator.extra_blocks,
        extra_block_cached_slots: terminator.extra_block_cached_slots,
        extra_block_exit_cached_slots: terminator.extra_block_exit_cached_slots,
    })
}

/// Check the central joint invariant:
/// live transient values plus resident cached locals must fit the dynamic bank
/// budgets at every rewrite point.
fn ensure_state_fits_with_cache(
    state: &BlockState,
    resident_cache: &CellSet,
    cell_types: &[ValueType],
    context: &str,
) -> Result<(), WasmError> {
    let mut gp_live = 0usize;
    let mut fp_live = 0usize;
    for (&ty, alias) in state.live_types().iter().zip(state.live_aliases()) {
        if alias.is_some_and(|slot| resident_cache.contains(&slot)) {
            continue;
        }
        if ty.is_fp() {
            fp_live += 1;
        } else {
            gp_live += gp_value_budget_units(ty, state.gp_unit_bytes);
        }
    }
    let (gp_cache, fp_cache) =
        discipline::count_cached_cell_budget_units(resident_cache, cell_types, state.gp_unit_bytes);
    if gp_live + gp_cache > state.gp_live_budget as usize
        || fp_live + fp_cache > state.fp_live_budget as usize
    {
        return Err(WasmError::internal(match context {
            "block entry boundary" => "planner exceeded dynamic bank budget at block entry",
            "local.get" => "planner exceeded dynamic bank budget after local.get",
            "local.set" => "planner exceeded dynamic bank budget after local.set",
            "local.tee" => "planner exceeded dynamic bank budget after local.tee",
            "primitive op" => "planner exceeded dynamic bank budget after primitive op",
            _ => "planner exceeded dynamic bank budget",
        }));
    }
    Ok(())
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
    cell_types: &[ValueType],
    resident_cache: &mut CellSet,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    // Route through the shared engine dispatch — the exact same path pass D's
    // measure walker uses, so the rewriter and the walker cannot drift.
    state.apply_op_discipline(op, frame, resident_cache, cell_types, values)
}

fn lower_block_body_op(
    semantic: &SemanticProgram,
    block_id: CfgBlockId,
    semantic_index: usize,
    planner: &JointPlanner,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    cell_types: &[ValueType],
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    config: BackendConfig,
) -> Result<(), WasmError> {
    apply_inline_prefix(
        &semantic.ops[semantic_index].kind,
        state,
        frame,
        cell_types,
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
            resident_cache,
            materialized_cache,
            values,
        ),
        SemanticOpKind::LocalSet { idx } => lower_local_set(
            *idx,
            planner,
            block_id,
            state,
            cell_types,
            resident_cache,
            materialized_cache,
        ),
        SemanticOpKind::LocalTee { idx } => lower_local_tee(
            semantic,
            *idx,
            planner,
            block_id,
            state,
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
                planner,
                resident_cache,
                materialized_cache,
                values,
                builder,
                config,
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
                values,
                builder,
                config,
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
                values,
                builder,
                config,
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
    resident_cache: &mut CellSet,
    _materialized_cache: &mut CellSet,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
) -> Result<(), WasmError> {
    let (pop, push) = primitive_op::stack_effect(kind);
    let (inline_args, extra_idx, first_arg) = state.pack_top_primitive_args(pop)?;
    let result_ty = if push == 0 {
        ValueType::I64
    } else if matches!(kind, PrimitiveOpKind::Select) {
        first_arg
            .map(|v| values.value_type(v))
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
    let pool_idx = builder.intern_primitive(kind)?;
    state
        .ops
        .push(SsaInst::primitive(pool_idx, result, inline_args, extra_idx));
    if push != 0 {
        state.push_result(result, result_ty)?;
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
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    // `local.get` always needs one SSA result. The current planner decides
    // whether we also keep the local cached after that load.
    let ty = semantic
        .local_types
        .get(local_idx as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    let slot = CellId(local_idx);
    let access = planner.cell_access(CellAccessQuery {
        block: block_id,
        slot,
        resident_cache,
    });
    let dst = values.fresh_typed(ty);
    let inst = match access {
        CellAccessDecision::Slot => SsaInst::cell_get_slot(slot, dst),
        CellAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::cell_get_cache(slot, dst)
        }
    };
    state.ops.push(inst);
    let alias = matches!(access, CellAccessDecision::Cache)
        .then_some(slot)
        .filter(|_| discipline::cached_cell_get_can_source_alias(ty, state.gp_unit_bytes));
    state.push_result_with_alias(dst, ty, alias)?;
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "local.get")
}

fn lower_local_set(
    local_idx: u16,
    planner: &JointPlanner,
    block_id: CfgBlockId,
    state: &mut BlockState,
    cell_types: &[ValueType],
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
) -> Result<(), WasmError> {
    let src = state.pop_one()?;
    let slot = CellId(local_idx);
    let access = planner.cell_access(CellAccessQuery {
        block: block_id,
        slot,
        resident_cache,
    });
    let inst = match access {
        CellAccessDecision::Slot => SsaInst::cell_set_slot(slot, src),
        CellAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::cell_set_cache(slot, src)
        }
    };
    state.ops.push(inst);
    ensure_state_fits_with_cache(state, resident_cache, cell_types, "local.set")
}

fn lower_local_tee(
    semantic: &SemanticProgram,
    local_idx: u16,
    planner: &JointPlanner,
    block_id: CfgBlockId,
    state: &mut BlockState,
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    // `local.tee` is the same admission question as `local.get`, except stack
    // height stays flat: the source value stays logically available while we
    // decide whether the local also deserves cache residency.
    let src = state.pop_one()?;
    let slot = CellId(local_idx);
    let ty = semantic
        .local_types
        .get(local_idx as usize)
        .copied()
        .unwrap_or(ValueType::I64);
    let access = planner.cell_access(CellAccessQuery {
        block: block_id,
        slot,
        resident_cache,
    });
    let set_inst = match access {
        CellAccessDecision::Slot => SsaInst::cell_set_slot(slot, src),
        CellAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::cell_set_cache(slot, src)
        }
    };
    state.ops.push(set_inst);
    let dst = values.fresh_typed(ty);
    let get_inst = match access {
        CellAccessDecision::Slot => SsaInst::cell_get_slot(slot, dst),
        CellAccessDecision::Cache => {
            resident_cache.insert(slot);
            materialized_cache.insert(slot);
            SsaInst::cell_get_cache(slot, dst)
        }
    };
    state.ops.push(get_inst);
    let alias = matches!(access, CellAccessDecision::Cache)
        .then_some(slot)
        .filter(|_| discipline::cached_cell_get_can_source_alias(ty, state.gp_unit_bytes));
    state.push_result_with_alias(dst, ty, alias)?;
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "local.tee")
}

fn lower_call_direct(
    callee: u32,
    params: u16,
    results: u16,
    result_types: &[ValueType],
    frame: FrameLayoutPlan,
    state: &mut BlockState,
    planner: &JointPlanner,
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    config: BackendConfig,
) -> Result<(), WasmError> {
    let stack_base = state.height().saturating_sub(params);
    let call_base = frame.operand_slot(stack_base);
    let args = capture_call_args(state, stack_base, call_base, params)?;
    let live_result = live_scalar_call_result(config, results, result_types, values);
    let call_idx = builder.push_call_op(SsaCallOp::CallDirect {
        callee,
        args,
        results: FrameSpan::new(call_base, results),
        result_types: result_types.to_vec().into(),
    })?;
    match live_result {
        Some((value, ty)) => {
            state.ops.push(SsaInst::call_result(call_idx, value));
            state.finish_call_with_live_scalar(values, params, value, ty)?;
        }
        None => {
            state.ops.push(SsaInst::call(call_idx));
            state.finish_call(values, params, results, result_types);
        }
    }
    // Mirror of the exact walker's `call_effect`: a survivable (direct
    // local-JIT) call keeps preserved-class nominated residents alive in their
    // callee-saved registers; everything else dies at the boundary.
    if planner.direct_call_survivable(callee) {
        resident_cache.retain(|slot| planner.is_preserved_nominated(*slot));
        materialized_cache.retain(|slot| planner.is_preserved_nominated(*slot));
    } else {
        resident_cache.clear();
        materialized_cache.clear();
    }
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
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    config: BackendConfig,
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
        result_types: result_types.to_vec().into(),
    })?;
    let live_result = live_scalar_call_result(config, results, result_types, values);
    match live_result {
        Some((value, ty)) => {
            state.ops.push(SsaInst::call_result(call_idx, value));
            state.finish_call_with_live_scalar(values, consumed, value, ty)?;
        }
        None => {
            state.ops.push(SsaInst::call(call_idx));
            state.finish_call(values, consumed, results, result_types);
        }
    }
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
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    config: BackendConfig,
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
        result_types: result_types.to_vec().into(),
    })?;
    let live_result = live_scalar_call_result(config, results, result_types, values);
    match live_result {
        Some((value, ty)) => {
            state.ops.push(SsaInst::call_result(call_idx, value));
            state.finish_call_with_live_scalar(values, consumed, value, ty)?;
        }
        None => {
            state.ops.push(SsaInst::call(call_idx));
            state.finish_call(values, consumed, results, result_types);
        }
    }
    resident_cache.clear();
    materialized_cache.clear();
    Ok(())
}

fn live_scalar_call_result(
    config: BackendConfig,
    results: u16,
    result_types: &[ValueType],
    values: &mut ValueAlloc,
) -> Option<(SsaValue, ValueType)> {
    if !config.scalar_return_lanes || results != 1 || result_types.len() != 1 {
        return None;
    }
    let ty = result_types[0];
    scalar_return_supported(ty).then(|| (values.fresh_typed(ty), ty))
}

fn lower_alloc_exn_ref(
    tag_idx: u32,
    semantic: &SemanticProgram,
    semantic_index: usize,
    state: &mut BlockState,
    planner: &JointPlanner,
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
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

fn emit_ref_is_null_condition(
    ref_value: SsaValue,
    state: &mut BlockState,
    builder: &mut ProgramBuilder,
    values: &mut ValueAlloc,
) -> Result<SsaValue, WasmError> {
    let cond = values.fresh_typed(ValueType::I32);
    let pool_idx = builder.intern_primitive(&PrimitiveOpKind::RefIsNull)?;
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
    let pool_idx = builder.intern_primitive(&PrimitiveOpKind::RefTest { ref_type })?;
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
    extra_block_cached_slots: collections::Vec<collections::Vec<CellId>>,
    extra_block_exit_cached_slots: collections::Vec<collections::Vec<CellId>>,
}

impl LoweredTerminator {
    fn new(terminator: SsaTerminator) -> Self {
        Self {
            terminator,
            extra_blocks: collections::Vec::new(),
            extra_block_cached_slots: collections::Vec::new(),
            extra_block_exit_cached_slots: collections::Vec::new(),
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
    cell_types: &[ValueType],
    resident_cache: &mut CellSet,
    materialized_cache: &mut CellSet,
    rewrite_cfg: &RewriteCfg,
    block_params: &[collections::Vec<SsaValue>],
    values: &mut ValueAlloc,
    builder: &mut ProgramBuilder,
    original_block_count: usize,
    extra_blocks_len: usize,
    config: BackendConfig,
) -> Result<LoweredTerminator, WasmError> {
    apply_inline_prefix(
        &semantic.ops[semantic_index].kind,
        state,
        frame,
        cell_types,
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
                cell_types,
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
            state.push_result(refined, *ref_type)?;
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
            state.push_result(refined, *ref_type)?;
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
            state.push_result(cast_ref, *cast_type)?;
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
            state.push_result(source_ref, *fail_type)?;
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
            state.push_result(cast_ref, *cast_type)?;
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
            state.push_result(source_ref, *fail_type)?;
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
                planner,
                resident_cache,
                materialized_cache,
                values,
                builder,
                config,
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
                values,
                builder,
                config,
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
                values,
                builder,
                config,
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
        SemanticOpKind::ReturnOne => Ok(LoweredTerminator::new(return_scalar_or_frame(
            state,
            frame,
            values,
            &semantic.result_types,
        )?)),
        SemanticOpKind::Return { arity } if *arity == 1 => Ok(LoweredTerminator::new(
            return_scalar_or_frame(state, frame, values, &semantic.result_types)?,
        )),
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
        state.spill_bottom(values, frame, publish_count);
    } else if target_entry.spill_depth < state.spill_depth() {
        // Fill: target expects more values to be live.
        let fill_count = state.height().saturating_sub(target_entry.spill_depth);
        state.fill_operands(values, frame, fill_count);
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
        state.fill_operands(values, frame, needed);
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

fn return_scalar_or_frame(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    result_types: &[ValueType],
) -> Result<SsaTerminator, WasmError> {
    let Some(results) = return_results(frame, 1) else {
        return Ok(SsaTerminator::Return { results: None });
    };
    let ty = result_types.first().copied().unwrap_or(ValueType::I64);
    if !scalar_return_supported(ty) {
        canonicalize_scalar_return_result(state, frame, values, ty);
        return Ok(SsaTerminator::Return {
            results: Some(results),
        });
    }
    let stack_index = state.height().saturating_sub(1);
    let slot = frame.operand_slot(stack_index);
    let result = if let Some(value) = state.value_at_stack_index(stack_index) {
        SsaScalarResultLoc::Live { value, ty, slot }
    } else {
        SsaScalarResultLoc::Stack { slot, ty }
    };
    Ok(SsaTerminator::ReturnScalar { result, results })
}

#[inline]
fn scalar_return_supported(ty: ValueType) -> bool {
    !matches!(ty, ValueType::V128)
}

fn canonicalize_scalar_return_result(
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    values: &mut ValueAlloc,
    ty: ValueType,
) {
    let stack_index = state.height().saturating_sub(1);
    if let Some(value) = state.value_at_stack_index(stack_index) {
        state.ops.push(SsaInst::spill(FrameSlot(0), value));
        return;
    }

    let src = frame.operand_slot(stack_index);
    if src == FrameSlot(0) {
        return;
    }
    let value = values.fresh_typed(ty);
    state.ops.push(SsaInst::fill(src, value));
    state.ops.push(SsaInst::spill(FrameSlot(0), value));
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
