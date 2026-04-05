use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::value_type::ValueType;
use crate::vm::{
    backend::BackendConfig,
    middle::{
        cfg::{self, CfgBlockId, SemanticCfg},
        frame::{plan_frame_layout, FrameLayoutPlan, FrameSlot},
        joint_plan::JointPlanner,
        prepare_function,
        slot_ssa::{self},
        ssa_ir::ir::{SsaBlock, SsaInstKind, SsaProgram, SsaTerminator},
        PrepareInput, PreparedFunction,
    },
    wasm::{
        common::SemanticTarget,
        primitive_op::PrimitiveOpKind,
        semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
    },
};

pub(super) struct PlannedPipeline {
    pub frame: FrameLayoutPlan,
    pub cfg: SemanticCfg,
    pub planner: JointPlanner,
}

#[inline]
pub(super) fn host_config(gp_dynamic_budget: u8, fp_dynamic_budget: u8) -> BackendConfig {
    // The middle-end tests only care about relative pressure behavior, so the
    // host-sized GP unit and normal 64-bit scratch layout are enough.
    BackendConfig::new(gp_dynamic_budget, fp_dynamic_budget, 8, 3)
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
    ops: Vec<SemanticOp>,
) -> SemanticProgram {
    typed_program(
        alloc::vec![ValueType::I32; local_count as usize],
        alloc::vec![ValueType::I32; results as usize],
        max_stack_height,
        ops,
    )
}

pub(super) fn typed_program(
    local_types: Vec<ValueType>,
    result_types: Vec<ValueType>,
    max_stack_height: u16,
    ops: Vec<SemanticOp>,
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
        },
        semantic,
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
    let planner = JointPlanner::build(semantic, &cfg, &slot, frame, config).unwrap_or_else(|err| {
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

pub(super) fn all_inst_kinds(program: &SsaProgram) -> Vec<&SsaInstKind> {
    program
        .blocks
        .iter()
        .flat_map(|block| block.ops.iter().map(|inst| &inst.kind))
        .collect()
}

pub(super) fn first_local_get_for(program: &SsaProgram, slot: FrameSlot) -> Option<&SsaInstKind> {
    all_inst_kinds(program).into_iter().find(|kind| {
        matches!(
            kind,
            SsaInstKind::LocalGetSlot { slot: got, .. }
                | SsaInstKind::LocalGetCache { slot: got, .. }
                if *got == slot
        )
    })
}

pub(super) fn local_set_kinds_for<'a>(
    program: &'a SsaProgram,
    slot: FrameSlot,
) -> Vec<&'a SsaInstKind> {
    all_inst_kinds(program)
        .into_iter()
        .filter(|kind| {
            matches!(
                kind,
                SsaInstKind::LocalSetSlot { slot: got, .. }
                    | SsaInstKind::LocalSetCache { slot: got, .. }
                    if *got == slot
            )
        })
        .collect()
}

pub(super) fn contains_drop_cache(program: &SsaProgram, slot: FrameSlot) -> bool {
    all_inst_kinds(program)
        .into_iter()
        .any(|kind| matches!(kind, SsaInstKind::LocalDropCache { slot: got } if *got == slot))
}

pub(super) fn contains_ensure_cache(program: &SsaProgram, slot: FrameSlot) -> bool {
    all_inst_kinds(program)
        .into_iter()
        .any(|kind| matches!(kind, SsaInstKind::LocalEnsureCache { slot: got } if *got == slot))
}

pub(super) fn count_ensure_cache(program: &SsaProgram, slot: FrameSlot) -> usize {
    all_inst_kinds(program)
        .into_iter()
        .filter(|kind| matches!(kind, SsaInstKind::LocalEnsureCache { slot: got } if *got == slot))
        .count()
}

pub(super) fn incoming_cache_repair_blocks(
    program: &SsaProgram,
    target_block: usize,
) -> Vec<&SsaBlock> {
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
                        inst.kind,
                        SsaInstKind::LocalDropCache { .. }
                            | SsaInstKind::LocalEnsureCache { .. }
                            | SsaInstKind::LocalReserveCache { .. }
                    )
                })
        })
        .collect()
}

pub(super) fn block_for_semantic_index(cfg: &SemanticCfg, semantic_index: usize) -> CfgBlockId {
    *cfg.semantic_to_block
        .get(semantic_index)
        .unwrap_or_else(|| panic!("semantic index {semantic_index} should map to one CFG block"))
}
