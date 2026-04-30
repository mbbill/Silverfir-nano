use tracked_alloc::collections::BTreeMap;

use crate::collections;

use crate::value_type::ValueType;
use crate::vm::{
    backend::BackendConfig,
    middle::{
        cfg::{self, CfgBlockId, SemanticCfg},
        frame::{plan_frame_layout, FrameLayoutPlan, FrameSlot},
        joint_plan::JointPlanner,
        prepare_function,
        slot_ssa::{self},
        ssa_ir::ir::{SsaBlock, SsaInst, SsaOp, SsaProgram, SsaTerminator},
        PrepareInput, PreparedFunction,
    },
    wasm::{
        common::SemanticTarget,
        primitive_op::PrimitiveOpKind,
        semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
    },
};

fn total_gp_budget_for_allocatable(allocatable_gp_budget: u8, gp_unit_bytes: u8) -> u8 {
    if allocatable_gp_budget == 0 {
        return 0;
    }
    if gp_unit_bytes == 4 {
        allocatable_gp_budget.saturating_add(if allocatable_gp_budget == 1 { 1 } else { 2 })
    } else {
        allocatable_gp_budget.saturating_add(1)
    }
}

pub(super) struct PlannedPipeline {
    pub frame: FrameLayoutPlan,
    pub cfg: SemanticCfg,
    pub planner: JointPlanner,
}

#[inline]
pub(super) fn host_config(gp_dynamic_budget: u8, fp_dynamic_budget: u8) -> BackendConfig {
    // The middle-end tests only care about relative pressure behavior, so the
    // host-sized GP unit and normal 64-bit scratch layout are enough.
    BackendConfig::new(
        8,
        total_gp_budget_for_allocatable(gp_dynamic_budget, 8),
        fp_dynamic_budget,
        3,
    )
}

#[inline]
pub(super) fn op(kind: SemanticOpKind) -> SemanticOp {
    SemanticOp { kind }
}

#[inline]
pub(super) fn prim(kind: PrimitiveOpKind) -> SemanticOp {
    op(SemanticOpKind::Primitive(kind))
}

#[inline]
pub(super) fn target(index: usize) -> SemanticTarget {
    SemanticTarget::new(index)
}

pub(super) fn i32_program(
    local_count: u16,
    max_stack_height: u16,
    results: u16,
    ops: collections::Vec<SemanticOp>,
) -> SemanticProgram {
    typed_program(
        collections::vec![ValueType::I32; local_count as usize],
        collections::vec![ValueType::I32; results as usize],
        max_stack_height,
        ops,
    )
}

pub(super) fn typed_program(
    local_types: collections::Vec<ValueType>,
    result_types: collections::Vec<ValueType>,
    max_stack_height: u16,
    ops: collections::Vec<SemanticOp>,
) -> SemanticProgram {
    SemanticProgram {
        params: 0,
        results: result_types.len() as u16,
        local_count: local_types.len() as u16,
        max_stack_height,
        ops,
        local_types,
        result_types,
        op_result_types: BTreeMap::new(),
    }
}

pub(super) fn prepare_i32_program(
    semantic: &SemanticProgram,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> PreparedFunction {
    prepare_program(semantic, gp_dynamic_budget, fp_dynamic_budget)
}

pub(super) fn prepare_program(
    semantic: &SemanticProgram,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> PreparedFunction {
    prepare_function(
        PrepareInput {
            config: host_config(gp_dynamic_budget, fp_dynamic_budget),
            function_index: None,
            full_optimization: true,
        },
        semantic.clone(),
    )
    .unwrap_or_else(|err| {
        panic!(
            "middle prepare_function should succeed for test semantic program: {}",
            err.message()
        )
    })
}

pub(super) fn plan_i32_program(
    semantic: &SemanticProgram,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> PlannedPipeline {
    plan_program(semantic, gp_dynamic_budget, fp_dynamic_budget)
}

pub(super) fn plan_program(
    semantic: &SemanticProgram,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
) -> PlannedPipeline {
    let config = host_config(gp_dynamic_budget, fp_dynamic_budget);
    semantic
        .validate()
        .unwrap_or_else(|err| panic!("test semantic program must validate: {}", err.message()));
    let frame = plan_frame_layout(semantic.local_count, semantic.max_stack_height, 3);
    let cfg = cfg::build_semantic_cfg(semantic);
    let slot = slot_ssa::lower_slot_only_ssa(semantic, &cfg, frame).unwrap_or_else(|err| {
        panic!(
            "slot-only SSA lowering should succeed for test semantic program: {}",
            err.message()
        )
    });
    let planner = JointPlanner::build(semantic, &cfg, slot.blocks.len(), frame, config)
        .unwrap_or_else(|err| {
            panic!(
                "joint planner should build for test semantic program: {}",
                err.message()
            )
        });
    PlannedPipeline {
        frame,
        cfg,
        planner,
    }
}

pub(super) fn all_insts(program: &SsaProgram) -> collections::Vec<&SsaInst> {
    program
        .blocks
        .iter()
        .flat_map(|block| block.ops.iter())
        .collect()
}

/// Back-compat alias for tests that still import the old name.
pub(super) fn all_inst_kinds(program: &SsaProgram) -> collections::Vec<&SsaInst> {
    all_insts(program)
}

pub(super) fn first_local_get_for(program: &SsaProgram, slot: FrameSlot) -> Option<&SsaInst> {
    all_insts(program).into_iter().find(|inst| {
        matches!(inst.op, SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE)
            && FrameSlot(inst.meta) == slot
    })
}

pub(super) fn contains_ensure_cache(program: &SsaProgram, slot: FrameSlot) -> bool {
    all_insts(program)
        .into_iter()
        .any(|inst| inst.op == SsaOp::LOCAL_ENSURE_CACHE && FrameSlot(inst.meta) == slot)
}

pub(super) fn count_ensure_cache(program: &SsaProgram, slot: FrameSlot) -> usize {
    all_insts(program)
        .into_iter()
        .filter(|inst| inst.op == SsaOp::LOCAL_ENSURE_CACHE && FrameSlot(inst.meta) == slot)
        .count()
}

pub(super) fn incoming_cache_repair_blocks(
    program: &SsaProgram,
    target_block: usize,
) -> collections::Vec<&SsaBlock> {
    program
        .blocks
        .iter()
        .filter(|block| {
            matches!(
                &block.terminator,
                SsaTerminator::Goto(edge) if edge.target.as_usize() == target_block
            )
        })
        .filter(|block| {
            !block.ops.is_empty()
                && block.ops.iter().all(|inst| {
                    matches!(
                        inst.op,
                        SsaOp::LOCAL_DROP_CACHE
                            | SsaOp::LOCAL_ENSURE_CACHE
                            | SsaOp::LOCAL_RESERVE_CACHE
                    )
                })
        })
        .collect()
}

pub(super) fn block_for_semantic_index(cfg: &SemanticCfg, semantic_index: usize) -> CfgBlockId {
    cfg.block_for_semantic_index(semantic_index)
        .unwrap_or_else(|| panic!("semantic index {semantic_index} should map to one CFG block"))
}

pub(super) fn prepared_block_for_semantic_index(
    prepared: &PreparedFunction,
    cfg: &SemanticCfg,
    semantic_index: usize,
) -> usize {
    let cfg_block = block_for_semantic_index(cfg, semantic_index);
    prepared
        .ssa
        .final_block_for_cfg_block(cfg_block.0)
        .unwrap_or_else(|| {
            panic!(
                "final SSA program should preserve a block origin for CFG block {} (semantic index {semantic_index})",
                cfg_block.0
            )
        })
        .as_usize()
}
