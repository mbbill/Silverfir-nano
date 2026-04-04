//! Final SSA-IR materialization from the chosen joint plan.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        middle::{
            budget::{count_live_bank_budget_units, gp_value_budget_units},
            cfg::{CfgBlockId, SemanticCfg},
            frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
            joint_plan::{
                BeforeOpQuery, EdgeRepairQuery, JointPlanner, LocalAccessDecision,
                LocalAccessQuery, TransientContract,
            },
            slot_ssa::SlotSsaProgram,
            ssa_ir::{
                ir::{
                    LocalSlotInfo, SsaBinding, SsaBlock, SsaCallOp, SsaEdge, SsaInst, SsaInstKind,
                    SsaOperand, SsaProgram, SsaTerminator, SsaValue,
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
    let mut extra_blocks = Vec::new();
    let mut extra_block_cached_slots = Vec::new();
    let mut extra_block_exit_cached_slots = Vec::new();
    for (block_index, cfg_block) in cfg.blocks.iter().enumerate() {
        let block_entry = planner.block_open(cfg_block.id);
        let block_exit = planner.block_exit(cfg_block.id);
        let params = block_params[block_index].clone();
        let state = BlockState::from_entry(
            block_entry.transient,
            &params,
            setup.gp_unit_bytes,
            setup.gp_dynamic_budget,
            setup.fp_dynamic_budget,
        )?;
        let lowered = lower_block_range(
            cfg_block.id,
            cfg_block.range.clone(),
            state,
            semantic,
            planner,
            frame,
            &semantic_to_block,
            &block_params,
            block_entry.cached_locals,
            block_exit.cached_locals,
            &mut values,
            original_block_count,
            extra_blocks.len(),
        )?;
        block_entry_cached_slots.push(block_entry.cached_locals.to_vec());
        block_exit_cached_slots.push(block_exit.cached_locals.to_vec());
        blocks.push(SsaBlock {
            id: SsaTarget(block_index as u32),
            params,
            ops: lowered.ops,
            terminator: lowered.terminator,
        });
        extra_block_cached_slots.extend(lowered.extra_block_cached_slots);
        extra_block_exit_cached_slots.extend(lowered.extra_block_exit_cached_slots);
        extra_blocks.extend(lowered.extra_blocks);
    }
    blocks.extend(extra_blocks);
    block_entry_cached_slots.extend(extra_block_cached_slots);
    block_exit_cached_slots.extend(extra_block_exit_cached_slots);

    let mut program = SsaProgram {
        entry: SsaTarget(cfg.entry.0),
        local_slot_types: semantic.local_types.clone(),
        local_slot_info: collect_local_slot_info(semantic),
        block_entry_cached_slots,
        blocks,
        value_types: values.take_types(),
        value_sink_local: Vec::new(),
    };

    insert_boundary_repair_blocks(&mut program, &block_exit_cached_slots, planner);
    Ok(program)
}

fn collect_local_slot_info(semantic: &SemanticProgram) -> Vec<LocalSlotInfo> {
    let reads_before_write = entry_scope_reads_before_write(semantic);
    (0..semantic.local_count as usize)
        .map(|idx| LocalSlotInfo {
            is_param: (idx as u16) < semantic.params,
            reads_before_write: reads_before_write.get(idx).copied().unwrap_or(true),
        })
        .collect()
}

#[derive(Clone, Debug, Default)]
struct ValueAlloc {
    next: u32,
    types: Vec<ValueType>,
}

impl ValueAlloc {
    fn fresh_typed(&mut self, ty: ValueType) -> SsaValue {
        let value = SsaValue(self.next);
        self.next += 1;
        self.types.push(ty);
        value
    }

    fn many_typed(&mut self, types: &[ValueType]) -> Vec<SsaValue> {
        types.iter().map(|ty| self.fresh_typed(*ty)).collect()
    }

    fn value_type(&self, value: SsaValue) -> ValueType {
        self.types
            .get(value.0 as usize)
            .copied()
            .unwrap_or(ValueType::I64)
    }

    fn take_types(&mut self) -> Vec<ValueType> {
        core::mem::take(&mut self.types)
    }
}

#[derive(Clone, Debug)]
struct BlockState {
    gp_unit_bytes: u8,
    gp_live_budget: u8,
    fp_live_budget: u8,
    stack_height: u16,
    spill_depth: u16,
    live: Vec<SsaValue>,
    live_types: Vec<ValueType>,
    ops: Vec<SsaInst>,
}

impl BlockState {
    fn from_entry(
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

    fn height(&self) -> u16 {
        self.stack_height
    }

    fn spill_depth(&self) -> u16 {
        self.spill_depth
    }

    fn live(&self) -> &[SsaValue] {
        &self.live
    }

    fn top_values(&self, count: usize) -> Result<Vec<SsaValue>, WasmError> {
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

    fn pop_one(&mut self) -> Result<SsaValue, WasmError> {
        let value = self
            .live
            .pop()
            .ok_or_else(|| WasmError::internal("transient underflow".into()))?;
        self.live_types.pop();
        self.stack_height = self.stack_height.saturating_sub(1);
        self.spill_depth = self.spill_depth.min(self.stack_height);
        Ok(value)
    }

    fn consume_top(&mut self, count: usize) -> Result<(), WasmError> {
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

    fn push_results(
        &mut self,
        results: Vec<SsaValue>,
        result_types: Vec<ValueType>,
    ) -> Result<(), WasmError> {
        self.stack_height = self.stack_height.saturating_add(results.len() as u16);
        self.live.extend(results);
        self.live_types.extend(result_types);
        self.ensure_live_fit("value push")
    }

    fn spill_prefix(&mut self, count: u16) -> Result<Vec<SsaValue>, WasmError> {
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

    fn fill_prefix(
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

    fn finish_call(&mut self, consumed: u16, produced: u16) {
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

fn make_block_params(entry_live_types: &[ValueType], values: &mut ValueAlloc) -> Vec<SsaValue> {
    values.many_typed(entry_live_types)
}

struct LoweredBlock {
    ops: Vec<SsaInst>,
    terminator: SsaTerminator,
    extra_blocks: Vec<SsaBlock>,
    extra_block_cached_slots: Vec<Vec<FrameSlot>>,
    extra_block_exit_cached_slots: Vec<Vec<FrameSlot>>,
}

fn lower_block_range(
    block_id: CfgBlockId,
    semantic_range: core::ops::Range<usize>,
    mut state: BlockState,
    semantic: &SemanticProgram,
    planner: &JointPlanner,
    frame: FrameLayoutPlan,
    semantic_to_block: &[SsaTarget],
    block_params: &[Vec<SsaValue>],
    entry_cached_locals: &[FrameSlot],
    exit_cached_locals: &[FrameSlot],
    values: &mut ValueAlloc,
    original_block_count: usize,
    extra_blocks_len: usize,
) -> Result<LoweredBlock, WasmError> {
    let mut resident_cache = entry_cached_locals.iter().copied().collect::<BTreeSet<_>>();
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
        semantic_to_block,
        block_params,
        values,
        original_block_count,
        extra_blocks_len,
    )?;
    validate_planned_cached_locals(block_id, &resident_cache, exit_cached_locals)?;

    Ok(LoweredBlock {
        ops: state.ops,
        terminator: terminator.terminator,
        extra_blocks: terminator.extra_blocks,
        extra_block_cached_slots: terminator.extra_block_cached_slots,
        extra_block_exit_cached_slots: terminator.extra_block_exit_cached_slots,
    })
}

fn lower_prefix_actions(
    semantic_index: usize,
    planner: &JointPlanner,
    state: &mut BlockState,
    frame: FrameLayoutPlan,
    local_slot_types: &[ValueType],
    resident_cache: &mut BTreeSet<FrameSlot>,
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
        state.ops.push(SsaInst {
            kind: SsaInstKind::LocalDropCache { slot },
        });
    }
    realize_transient_contract(before_op.transient, state, frame, values)?;
    ensure_state_fits_with_cache(
        state,
        resident_cache,
        local_slot_types,
        "planned pre-op boundary",
    )?;
    Ok(())
}

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

fn ensure_state_fits_with_cache(
    state: &BlockState,
    resident_cache: &BTreeSet<FrameSlot>,
    local_slot_types: &[ValueType],
    context: &str,
) -> Result<(), WasmError> {
    let (gp_live, fp_live) = count_live_bank_budget_units(&state.live_types, state.gp_unit_bytes);
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

fn validate_planned_cached_locals(
    block: CfgBlockId,
    resident_cache: &BTreeSet<FrameSlot>,
    planned_cached_locals: &[FrameSlot],
) -> Result<(), WasmError> {
    if resident_cache.len() != planned_cached_locals.len()
        || planned_cached_locals
            .iter()
            .any(|slot| !resident_cache.contains(slot))
    {
        return Err(WasmError::internal(alloc::format!(
            "planner exit cache boundary mismatch for block {}",
            block.0
        )));
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
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
    lower_prefix_actions(
        semantic_index,
        planner,
        state,
        frame,
        local_slot_types,
        resident_cache,
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
            resident_cache,
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
        ),
        SemanticOpKind::LocalTee { idx } => lower_local_tee(
            semantic,
            *idx,
            planner,
            semantic_index,
            state,
            frame,
            resident_cache,
            values,
        ),
        SemanticOpKind::CallDirect {
            callee,
            params,
            results,
        } => {
            lower_call_direct(*callee, *params, *results, frame, state);
            Ok(())
        }
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => {
            lower_call_indirect(*type_idx, *table_idx, *params, *results, frame, state);
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
    resident_cache: &mut BTreeSet<FrameSlot>,
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
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
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
                SsaInstKind::LocalGetCache { slot, dst }
            }
        },
    });
    state.push_results(alloc::vec![dst], alloc::vec![ty])?;
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
                SsaInstKind::LocalSetCache { slot, src }
            }
        },
    });
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
    values: &mut ValueAlloc,
) -> Result<(), WasmError> {
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
                SsaInstKind::LocalGetCache { slot, dst }
            }
        },
    });
    state.push_results(alloc::vec![dst], alloc::vec![ty])?;
    ensure_state_fits_with_cache(state, resident_cache, &semantic.local_types, "local.tee")
}

fn lower_call_direct(
    callee: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &mut BlockState,
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
}

fn lower_call_indirect(
    type_idx: u32,
    table_idx: u32,
    params: u16,
    results: u16,
    frame: FrameLayoutPlan,
    state: &mut BlockState,
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
}

impl LoweredTerminator {
    fn new(terminator: SsaTerminator) -> Self {
        Self {
            terminator,
            extra_blocks: Vec::new(),
            extra_block_cached_slots: Vec::new(),
            extra_block_exit_cached_slots: Vec::new(),
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
                resident_cache,
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
                    extra_block_exit_cached_slots: alloc::vec![resident_cache
                        .iter()
                        .copied()
                        .collect()],
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
            lower_call_direct(*callee, *params, *results, frame, state);
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
            lower_call_indirect(*type_idx, *table_idx, *params, *results, frame, state);
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

fn insert_boundary_repair_blocks(
    program: &mut SsaProgram,
    exit_cached_slots: &[Vec<FrameSlot>],
    planner: &JointPlanner,
) {
    let original_len = program.blocks.len();
    let target_params = program
        .blocks
        .iter()
        .map(|block| block.params.clone())
        .collect::<Vec<_>>();
    let target_entries = program.block_entry_cached_slots.clone();
    let mut extra_blocks = Vec::new();

    for block_index in 0..original_len {
        let pred_exit = exit_cached_slots
            .get(block_index)
            .cloned()
            .unwrap_or_default();
        let terminator = &mut program.blocks[block_index].terminator;
        match terminator {
            SsaTerminator::Goto(edge) => maybe_repair_edge(
                edge,
                &pred_exit,
                &target_entries,
                &target_params,
                &mut extra_blocks,
                &mut program.block_entry_cached_slots,
                original_len,
                planner,
            ),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                maybe_repair_edge(
                    then_edge,
                    &pred_exit,
                    &target_entries,
                    &target_params,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    original_len,
                    planner,
                );
                maybe_repair_edge(
                    else_edge,
                    &pred_exit,
                    &target_entries,
                    &target_params,
                    &mut extra_blocks,
                    &mut program.block_entry_cached_slots,
                    original_len,
                    planner,
                );
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    maybe_repair_edge(
                        edge,
                        &pred_exit,
                        &target_entries,
                        &target_params,
                        &mut extra_blocks,
                        &mut program.block_entry_cached_slots,
                        original_len,
                        planner,
                    );
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }

    program.blocks.extend(extra_blocks);
}

fn maybe_repair_edge(
    edge: &mut SsaEdge,
    pred_exit: &[FrameSlot],
    target_entries: &[Vec<FrameSlot>],
    target_params: &[Vec<SsaValue>],
    extra_blocks: &mut Vec<SsaBlock>,
    block_entry_cached_slots: &mut Vec<Vec<FrameSlot>>,
    original_len: usize,
    planner: &JointPlanner,
) {
    let target_id = edge.target.as_usize();
    let target_entry = target_entries.get(target_id).cloned().unwrap_or_default();
    let repair = planner.edge_repair(EdgeRepairQuery {
        pred_exit,
        succ_entry: &target_entry,
    });
    if repair.ensure_cached_locals.is_empty() && repair.drop_cached_locals.is_empty() {
        return;
    }

    let repair_id = SsaTarget((original_len + extra_blocks.len()) as u32);
    let repair_params = target_params.get(target_id).cloned().unwrap_or_default();
    let mut ops = Vec::new();
    for slot in repair.drop_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalDropCache { slot },
        });
    }
    for slot in repair.ensure_cached_locals {
        ops.push(SsaInst {
            kind: SsaInstKind::LocalEnsureCache { slot },
        });
    }
    let repair_edge = SsaEdge {
        target: edge.target,
        bindings: repair_params
            .iter()
            .copied()
            .map(|param| SsaBinding {
                param,
                value: param,
            })
            .collect(),
    };
    extra_blocks.push(SsaBlock {
        id: repair_id,
        params: repair_params,
        ops,
        terminator: SsaTerminator::Goto(repair_edge),
    });
    block_entry_cached_slots.push(pred_exit.to_vec());
    edge.target = repair_id;
}

fn entry_scope_reads_before_write(semantic: &SemanticProgram) -> Vec<bool> {
    let n = semantic.local_count as usize;
    if n == 0 {
        return Vec::new();
    }

    let mut def_set = alloc::vec![false; n];
    let mut reads_init = alloc::vec![false; n];
    let mut reachable = true;

    #[derive(Clone, Copy, PartialEq)]
    enum FrameKind {
        Block,
        Loop,
        If,
    }

    struct Frame {
        kind: FrameKind,
        entry_def_set: Vec<bool>,
        branch_def_set: Option<Vec<bool>>,
        if_arm_def_set: Option<Vec<bool>>,
        if_arm_reachable: bool,
    }

    let mut stack: Vec<Frame> = Vec::new();
    stack.push(Frame {
        kind: FrameKind::Block,
        entry_def_set: def_set.clone(),
        branch_def_set: None,
        if_arm_def_set: None,
        if_arm_reachable: false,
    });

    for op in &semantic.ops {
        match op.kind {
            SemanticOpKind::Block { .. } => {
                stack.push(Frame {
                    kind: FrameKind::Block,
                    entry_def_set: def_set.clone(),
                    branch_def_set: None,
                    if_arm_def_set: None,
                    if_arm_reachable: false,
                });
            }
            SemanticOpKind::Loop { .. } => {
                stack.push(Frame {
                    kind: FrameKind::Loop,
                    entry_def_set: def_set.clone(),
                    branch_def_set: None,
                    if_arm_def_set: None,
                    if_arm_reachable: false,
                });
            }
            SemanticOpKind::If { .. } => {
                stack.push(Frame {
                    kind: FrameKind::If,
                    entry_def_set: def_set.clone(),
                    branch_def_set: None,
                    if_arm_def_set: None,
                    if_arm_reachable: false,
                });
            }
            SemanticOpKind::Else { .. } => {
                if let Some(frame) = stack.last_mut() {
                    frame.if_arm_def_set = Some(def_set.clone());
                    frame.if_arm_reachable = reachable;
                    def_set.copy_from_slice(&frame.entry_def_set);
                    reachable = true;
                }
            }
            SemanticOpKind::End => {
                if let Some(frame) = stack.pop() {
                    match frame.kind {
                        FrameKind::Block => {
                            def_set = merge_paths(
                                if reachable { Some(&def_set) } else { None },
                                frame.branch_def_set.as_deref(),
                                n,
                            );
                            reachable = reachable || frame.branch_def_set.is_some();
                        }
                        FrameKind::If => {
                            if let Some(if_arm) = &frame.if_arm_def_set {
                                let mut paths: Vec<&[bool]> = Vec::new();
                                if frame.if_arm_reachable {
                                    paths.push(if_arm);
                                }
                                if reachable {
                                    paths.push(&def_set);
                                }
                                if let Some(br) = &frame.branch_def_set {
                                    paths.push(br);
                                }
                                def_set = intersect_all(&paths, n);
                                reachable = reachable
                                    || frame.if_arm_reachable
                                    || frame.branch_def_set.is_some();
                            } else {
                                let mut paths: Vec<&[bool]> = Vec::new();
                                if reachable {
                                    paths.push(&def_set);
                                }
                                paths.push(&frame.entry_def_set);
                                if let Some(br) = &frame.branch_def_set {
                                    paths.push(br);
                                }
                                def_set = intersect_all(&paths, n);
                                reachable = true;
                            }
                        }
                        FrameKind::Loop => {}
                    }
                }
            }
            SemanticOpKind::Br { .. } => {
                if reachable {
                    for frame in stack.iter_mut() {
                        if frame.kind == FrameKind::Loop {
                            continue;
                        }
                        frame.branch_def_set = Some(match &frame.branch_def_set {
                            Some(existing) => intersect_vecs(existing, &def_set),
                            None => def_set.clone(),
                        });
                    }
                }
                reachable = false;
            }
            SemanticOpKind::BrIf { .. } => {
                if reachable {
                    for frame in stack.iter_mut() {
                        if frame.kind == FrameKind::Loop {
                            continue;
                        }
                        frame.branch_def_set = Some(match &frame.branch_def_set {
                            Some(existing) => intersect_vecs(existing, &def_set),
                            None => def_set.clone(),
                        });
                    }
                }
            }
            SemanticOpKind::BrTable { .. } => {
                if reachable {
                    for frame in stack.iter_mut() {
                        if frame.kind == FrameKind::Loop {
                            continue;
                        }
                        frame.branch_def_set = Some(match &frame.branch_def_set {
                            Some(existing) => intersect_vecs(existing, &def_set),
                            None => def_set.clone(),
                        });
                    }
                }
                reachable = false;
            }
            SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
            | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => {
                reachable = false;
            }
            SemanticOpKind::LocalGet { idx } => {
                if reachable {
                    let i = idx as usize;
                    if i < n && !def_set[i] {
                        reads_init[i] = true;
                    }
                }
            }
            SemanticOpKind::LocalSet { idx } | SemanticOpKind::LocalTee { idx } => {
                if reachable {
                    let i = idx as usize;
                    if i < n {
                        def_set[i] = true;
                    }
                }
            }
            _ => {}
        }
    }

    reads_init
}

fn intersect_vecs(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect()
}

fn intersect_all(paths: &[&[bool]], n: usize) -> Vec<bool> {
    if paths.is_empty() {
        return alloc::vec![false; n];
    }
    let mut out = alloc::vec![true; n];
    for path in paths {
        for (i, bit) in path.iter().enumerate() {
            out[i] &= *bit;
        }
    }
    out
}

fn merge_paths(a: Option<&[bool]>, b: Option<&[bool]>, n: usize) -> Vec<bool> {
    match (a, b) {
        (None, None) => alloc::vec![false; n],
        (Some(a), None) | (None, Some(a)) => a.to_vec(),
        (Some(a), Some(b)) => intersect_vecs(a, b),
    }
}
