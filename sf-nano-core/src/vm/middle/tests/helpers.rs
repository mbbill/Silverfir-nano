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
    let planner = JointPlanner::build(semantic, &cfg, frame, config).unwrap_or_else(|err| {
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

/// Control-flow successor block indices of `block`.
fn ssa_successor_blocks(block: &SsaBlock) -> collections::Vec<usize> {
    let mut succs = collections::Vec::new();
    match &block.terminator {
        SsaTerminator::Goto(edge) => succs.push(edge.target.as_usize()),
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            succs.push(then_edge.target.as_usize());
            succs.push(else_edge.target.as_usize());
        }
        SsaTerminator::BrTable { entries, .. } => {
            for edge in entries {
                succs.push(edge.target.as_usize());
            }
        }
        _ => {}
    }
    succs
}

/// The (then, else) successor blocks of the program's single `Branch` block.
///
/// The branch-boundary tests each build exactly one conditional, so the sole
/// `Branch` terminator names the then/else blocks directly.
pub(super) fn branch_edge_targets(program: &SsaProgram) -> (usize, usize) {
    let mut found = None;
    for block in &program.blocks {
        if let SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } = &block.terminator
        {
            assert!(
                found.is_none(),
                "expected exactly one Branch block in the test program"
            );
            found = Some((then_edge.target.as_usize(), else_edge.target.as_usize()));
        }
    }
    found.expect("test program should contain one Branch block")
}

/// The natural-loop backedges `(latch, header)` reachable from the entry,
/// found by a depth-first search that flags edges back to an on-stack ancestor.
fn loop_backedges(program: &SsaProgram) -> collections::Vec<(usize, usize)> {
    let n = program.blocks.len();
    let succs: collections::Vec<collections::Vec<usize>> =
        program.blocks.iter().map(ssa_successor_blocks).collect();
    // 0 = unvisited, 1 = on DFS stack, 2 = finished.
    let mut state = collections::vec![0u8; n];
    let mut next = collections::vec![0usize; n];
    let mut stack = collections::vec![program.entry.as_usize()];
    let mut backedges = collections::Vec::new();
    state[program.entry.as_usize()] = 1;
    while let Some(&node) = stack.last() {
        if next[node] < succs[node].len() {
            let succ = succs[node][next[node]];
            next[node] += 1;
            match state[succ] {
                0 => {
                    state[succ] = 1;
                    stack.push(succ);
                }
                1 => backedges.push((node, succ)),
                _ => {}
            }
        } else {
            state[node] = 2;
            stack.pop();
        }
    }
    backedges
}

/// The single loop header (backedge target) in the test program.
pub(super) fn loop_header_block(program: &SsaProgram) -> usize {
    let backedges = loop_backedges(program);
    let header = backedges
        .first()
        .expect("test program should contain one loop")
        .1;
    assert!(
        backedges.iter().all(|(_, h)| *h == header),
        "expected all backedges to share one loop header; backedges={backedges:?}"
    );
    header
}

/// The loop body block: the loop header's successor that stays inside the loop
/// (i.e. can reach the header again). Distinct from the header itself unless the
/// loop is a single self-looping block.
pub(super) fn loop_body_block(program: &SsaProgram) -> usize {
    let header = loop_header_block(program);
    let mut body = None;
    for succ in ssa_successor_blocks(&program.blocks[header]) {
        if succ != header && can_reach(program, succ, header) {
            assert!(
                body.is_none_or(|existing| existing == succ),
                "expected one in-loop successor of the loop header"
            );
            body = Some(succ);
        }
    }
    body.expect("loop header should have one in-loop successor block")
}

/// The natural loop of the (unique) loop header: every block the header reaches
/// that can also reach the header again. Reachability-based so it does not
/// assume the final SSA keeps blocks in execution order (it does not — rewrite
/// appends repair/preheader blocks at higher indices than the blocks they feed).
pub(super) fn natural_loop_blocks(program: &SsaProgram) -> collections::Vec<usize> {
    let header = loop_header_block(program);
    (0..program.blocks.len())
        .filter(|&block| can_reach(program, header, block) && can_reach(program, block, header))
        .collect()
}

/// Whether `target` is reachable from `from` following control-flow edges.
fn can_reach(program: &SsaProgram, from: usize, target: usize) -> bool {
    let mut seen = collections::vec![false; program.blocks.len()];
    let mut stack = collections::vec![from];
    seen[from] = true;
    while let Some(node) = stack.pop() {
        if node == target {
            return true;
        }
        for succ in ssa_successor_blocks(&program.blocks[node]) {
            if !seen[succ] {
                seen[succ] = true;
                stack.push(succ);
            }
        }
    }
    false
}
