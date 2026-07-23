//! Precompute invariant linear-memory address bases at loop entry.
//!
//! An indexed Wasm access such as `[mem0 + object + field_offset]` needs a
//! scratch address add on targets that cannot encode both the dynamic Wasm
//! offset and the field displacement in one memory operand. When `object` is
//! loop invariant, compute the native `mem0 + zext32(object)` base once and
//! carry it around the loop as an ordinary block parameter. The individual
//! accesses then become plain base-plus-immediate loads/stores.
//!
//! This deliberately hoists only address arithmetic, never a memory access.
//! Wasm trap ordering is therefore unchanged.

use crate::collections;
use crate::vm::{
    backend::BackendConfig,
    machine::machine_ir::{
        MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineConvertOp,
        MachineIndexExtend, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntWidth,
        MachineMemWidth, MachineProgram, MachineReg, MachineRegOwner, MachineStorageType,
        MachineTerminator, MachineValue, MACHINE_FIXED_REG_COUNT, MACHINE_MEM0_BASE_REG,
    },
};

use super::helpers::{inst_defines, inst_uses_value, terminator_uses_reg};

pub(super) struct LoopGraph {
    pub predecessors: collections::Vec<collections::Vec<usize>>,
    pub latches_by_header: collections::Vec<collections::Vec<usize>>,
}

pub(super) fn hoist_loop_address_bases(
    program: &mut MachineProgram,
    config: BackendConfig,
    loop_graph: &LoopGraph,
) {
    let entry = program.entry;
    let blocks = &mut program.blocks;
    let entry_index = block_index_for_id(blocks, entry);
    if config.gp_unit_bytes != 8 || blocks.len() < 2 {
        return;
    }

    let latches_by_header = &loop_graph.latches_by_header;
    let predecessors = &loop_graph.predecessors;

    // Inner/later loops first. A surrounding loop can still use another free
    // dynamic register after seeing the parameter introduced here.
    for header in (0..blocks.len()).rev() {
        if latches_by_header[header].is_empty() {
            continue;
        }
        let loop_nodes = natural_loop_nodes(header, &latches_by_header[header], predecessors);
        if blocks[header].id == entry
            || entry_index.is_some_and(|entry| loop_nodes.contains(&entry))
        {
            continue;
        }
        try_hoist_one_base(blocks, header, &loop_nodes, config);
    }
}

pub(super) fn analyze_loop_graph(blocks: &[MachineBlock], entry: MachineBlockId) -> LoopGraph {
    let mut predecessors: collections::Vec<collections::Vec<usize>> =
        (0..blocks.len()).map(|_| collections::Vec::new()).collect();
    let mut successors: collections::Vec<collections::Vec<usize>> =
        (0..blocks.len()).map(|_| collections::Vec::new()).collect();
    for source in 0..blocks.len() {
        visit_edges(&blocks[source].terminator, |edge| {
            let Some(target) = block_index_for_id(blocks, edge.target) else {
                return;
            };
            if !successors[source].contains(&target) {
                successors[source].push(target);
            }
            if !predecessors[target].contains(&source) {
                predecessors[target].push(source);
            }
        });
    }

    let mut latches_by_header: collections::Vec<collections::Vec<usize>> =
        (0..blocks.len()).map(|_| collections::Vec::new()).collect();
    let has_backward_edge = successors.iter().enumerate().any(|(source, targets)| {
        targets
            .iter()
            .any(|&target| blocks[target].id.0 <= blocks[source].id.0)
    });
    if !has_backward_edge {
        return LoopGraph {
            predecessors,
            latches_by_header,
        };
    }

    // A natural-loop backedge must target a block that dominates its source.
    // Merely sharing an SCC proves that the edge participates in some cycle,
    // not that its target is that cycle's header. Treating every numerically
    // backward SCC edge as a latch manufactures many overlapping pseudo-loops
    // in large irreducible components.
    let entry_index = block_index_for_id(blocks, entry);
    let immediate_dominators = entry_index
        .map(|entry| compute_immediate_dominators(&successors, &predecessors, entry))
        .unwrap_or_else(|| collections::vec![None; blocks.len()]);
    for (source, targets) in successors.iter().enumerate() {
        for &target in targets {
            if blocks[target].id.0 <= blocks[source].id.0
                && dominates(&immediate_dominators, target, source)
                && !latches_by_header[target].contains(&source)
            {
                latches_by_header[target].push(source);
            }
        }
    }

    LoopGraph {
        predecessors,
        latches_by_header,
    }
}

fn compute_immediate_dominators(
    successors: &[collections::Vec<usize>],
    predecessors: &[collections::Vec<usize>],
    entry: usize,
) -> collections::Vec<Option<usize>> {
    let mut visited = collections::vec![false; successors.len()];
    let mut reverse_postorder = collections::Vec::with_capacity(successors.len());
    let mut dfs = collections::vec![(entry, 0usize)];
    visited[entry] = true;
    while let Some((block, next_successor)) = dfs.last_mut() {
        if let Some(&successor) = successors[*block].get(*next_successor) {
            *next_successor += 1;
            if !visited[successor] {
                visited[successor] = true;
                dfs.push((successor, 0));
            }
        } else {
            let (block, _) = dfs.pop().unwrap();
            reverse_postorder.push(block);
        }
    }
    reverse_postorder.reverse();

    let mut rpo_index = collections::vec![usize::MAX; successors.len()];
    for (index, &block) in reverse_postorder.iter().enumerate() {
        rpo_index[block] = index;
    }
    let mut idom = collections::vec![None; successors.len()];
    idom[entry] = Some(entry);
    let mut changed = true;
    while changed {
        changed = false;
        for &block in reverse_postorder.iter().skip(1) {
            let mut reachable_predecessors = predecessors[block]
                .iter()
                .copied()
                .filter(|predecessor| idom[*predecessor].is_some());
            let Some(mut new_idom) = reachable_predecessors.next() else {
                continue;
            };
            for predecessor in reachable_predecessors {
                new_idom = intersect_dominators(&idom, &rpo_index, predecessor, new_idom);
            }
            if idom[block] != Some(new_idom) {
                idom[block] = Some(new_idom);
                changed = true;
            }
        }
    }

    idom
}

fn intersect_dominators(
    idom: &[Option<usize>],
    rpo_index: &[usize],
    mut lhs: usize,
    mut rhs: usize,
) -> usize {
    while lhs != rhs {
        while rpo_index[lhs] > rpo_index[rhs] {
            lhs = idom[lhs].expect("reachable block must have an immediate dominator");
        }
        while rpo_index[rhs] > rpo_index[lhs] {
            rhs = idom[rhs].expect("reachable block must have an immediate dominator");
        }
    }
    lhs
}

fn dominates(idom: &[Option<usize>], dominator: usize, mut block: usize) -> bool {
    if idom.get(block).copied().flatten().is_none() {
        return false;
    }
    loop {
        if block == dominator {
            return true;
        }
        let Some(parent) = idom[block] else {
            return false;
        };
        if parent == block {
            return false;
        }
        block = parent;
    }
}

pub(super) fn natural_loop_nodes(
    header: usize,
    latches: &[usize],
    predecessors: &[collections::Vec<usize>],
) -> collections::Vec<usize> {
    let mut included = collections::vec![false; predecessors.len()];
    included[header] = true;
    let mut pending = collections::Vec::new();
    for &latch in latches {
        if !included[latch] {
            included[latch] = true;
            pending.push(latch);
        }
    }
    while let Some(block) = pending.pop() {
        if block == header {
            continue;
        }
        for &predecessor in &predecessors[block] {
            if !included[predecessor] {
                included[predecessor] = true;
                pending.push(predecessor);
            }
        }
    }
    included
        .into_iter()
        .enumerate()
        .filter_map(|(index, included)| included.then_some(index))
        .collect()
}

fn try_hoist_one_base(
    blocks: &mut [MachineBlock],
    header: usize,
    loop_nodes: &[usize],
    config: BackendConfig,
) {
    if loop_nodes.is_empty() || loop_can_change_memory_base(blocks, loop_nodes) {
        return;
    }

    // Prefer the invariant Wasm address used by the most accesses. Two uses
    // are enough to amortize the entry conversion/add on the first trip.
    let mut candidates: collections::Vec<(MachineReg, usize)> = collections::Vec::new();
    for &block_index in loop_nodes {
        let block = &blocks[block_index];
        for inst in &block.ops {
            let Some(index) = mem0_index(&inst.kind) else {
                continue;
            };
            if loop_nodes.iter().any(|&block_index| {
                blocks[block_index]
                    .ops
                    .iter()
                    .any(|inst| inst_defines(&inst.kind, index))
            }) {
                continue;
            }
            if let Some((_, count)) = candidates.iter_mut().find(|(reg, _)| *reg == index) {
                *count += 1;
            } else {
                candidates.push((index, 1));
            }
        }
    }
    let Some((index, use_count)) =
        candidates
            .into_iter()
            .max_by(|(reg_a, count_a), (reg_b, count_b)| {
                count_a.cmp(count_b).then_with(|| reg_b.0.cmp(&reg_a.0))
            })
    else {
        return;
    };
    if use_count < 2 {
        return;
    }

    let Some(index_param) = blocks[header]
        .params
        .iter()
        .position(|param| param.reg == index)
    else {
        return;
    };

    // A MachineReg number is only a physical lane; a block edge can bind a
    // different logical value into that same lane. Prove that every loop
    // block which explicitly rebinds the lane carries this particular value
    // through as an identity argument before treating it as loop invariant.
    // Blocks without such a parameter leave the physical lane untouched.
    for &target in loop_nodes {
        let Some(target_param) = blocks[target]
            .params
            .iter()
            .position(|param| param.reg == index)
        else {
            continue;
        };
        let mut has_predecessor = false;
        let mut preserves_index = true;
        for source in 0..blocks.len() {
            visit_edges(&blocks[source].terminator, |edge| {
                if edge.target == blocks[target].id {
                    has_predecessor = true;
                    preserves_index &= matches!(
                        edge.args.get(target_param),
                        Some(MachineValue::Reg(reg)) if *reg == index
                    );
                }
            });
        }
        if !has_predecessor || !preserves_index {
            return;
        }
    }

    let mut entry_predecessors = collections::Vec::new();
    for source in 0..blocks.len() {
        let source_inside = loop_nodes.contains(&source);
        let mut valid = true;
        let mut enters = false;
        visit_edges(&blocks[source].terminator, |edge| {
            let Some(target) = block_index_for_id(blocks, edge.target) else {
                return;
            };
            if !source_inside && loop_nodes.contains(&target) {
                enters = true;
                valid &= target == header
                    && matches!(edge.args.get(index_param), Some(MachineValue::Reg(reg)) if *reg == index);
            }
        });
        if enters {
            if !valid {
                return;
            }
            entry_predecessors.push(source);
        }
    }
    if entry_predecessors.is_empty() {
        return;
    }

    let gp_end = MACHINE_FIXED_REG_COUNT
        + u16::from(config.gp_volatile_dynamic)
        + u16::from(config.gp_preserved_dynamic);
    let spare = (MACHINE_FIXED_REG_COUNT..gp_end)
        .map(MachineReg)
        .find(|&reg| {
            loop_nodes
                .iter()
                .all(|&block_index| !block_mentions_reg(&blocks[block_index], reg))
                && entry_predecessors
                    .iter()
                    .all(|&source| !block_mentions_reg(&blocks[source], reg))
        });
    let Some(spare) = spare else {
        return;
    };

    // Seed the native base on every legal entry. The conversion preserves the
    // exact `uxtw` semantics of the indexed access before adding mem0.
    for &source in &entry_predecessors {
        blocks[source].ops.push(MachineInst {
            kind: MachineInstKind::Convert {
                op: MachineConvertOp::I64ExtendI32U,
                dst: spare,
                src: MachineValue::Reg(index),
            },
        });
        blocks[source].ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op: MachineIntBinaryOp::Add,
                dst: spare,
                lhs: MachineValue::Reg(MACHINE_MEM0_BASE_REG),
                rhs: MachineValue::Reg(spare),
            },
        });
    }

    for &block_index in loop_nodes {
        let block = &mut blocks[block_index];
        block.params.push(MachineBlockParam {
            reg: spare,
            ty: MachineStorageType::GpI64,
            owner: MachineRegOwner::LinearValue,
        });
        for inst in &mut block.ops {
            rewrite_access(&mut inst.kind, index, spare, &block.params);
        }
    }

    let loop_ids: collections::Vec<MachineBlockId> = loop_nodes
        .iter()
        .map(|&block_index| blocks[block_index].id)
        .collect();
    for block in blocks.iter_mut() {
        visit_edges_mut(&mut block.terminator, |edge| {
            if loop_ids.contains(&edge.target) {
                edge.args.push(MachineValue::Reg(spare));
            }
        });
    }
}

fn mem0_index(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::IndexedLoad {
            base,
            index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            ..
        }
        | MachineInstKind::IndexedStore {
            base,
            index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            ..
        } if *base == MACHINE_MEM0_BASE_REG => Some(*index),
        _ => None,
    }
}

fn rewrite_access(
    kind: &mut MachineInstKind,
    index: MachineReg,
    native_base: MachineReg,
    params: &[MachineBlockParam],
) {
    let replacement = match *kind {
        MachineInstKind::IndexedLoad {
            dst,
            base: MACHINE_MEM0_BASE_REG,
            index: access_index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            offset,
            width,
            extension,
        } if access_index == index => {
            let owner = params
                .iter()
                .find(|param| param.reg == dst)
                .map(|param| param.owner)
                .unwrap_or(MachineRegOwner::LinearValue);
            Some(MachineInstKind::Load {
                owner,
                ty: storage_type(width),
                dst,
                addr: MachineAddr {
                    base: native_base,
                    offset,
                },
                width,
                extension,
            })
        }
        MachineInstKind::IndexedStore {
            base: MACHINE_MEM0_BASE_REG,
            index: access_index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            offset,
            width,
            src,
        } if access_index == index => Some(MachineInstKind::Store {
            ty: storage_type(width),
            addr: MachineAddr {
                base: native_base,
                offset,
            },
            width,
            src,
        }),
        _ => None,
    };
    if let Some(replacement) = replacement {
        *kind = replacement;
    }
}

fn storage_type(width: MachineMemWidth) -> MachineStorageType {
    match width {
        MachineMemWidth::U64 => MachineStorageType::GpI64,
        MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32 => {
            MachineStorageType::GpWord
        }
    }
}

fn loop_can_change_memory_base(blocks: &[MachineBlock], loop_nodes: &[usize]) -> bool {
    loop_nodes.iter().any(|&block_index| {
        let block = &blocks[block_index];
        block.ops.iter().any(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::MemoryGrow { .. } | MachineInstKind::CallRuntime(_)
            )
        }) || matches!(
            block.terminator,
            MachineTerminator::Call { .. } | MachineTerminator::TailCall { .. }
        )
    })
}

pub(super) fn block_mentions_reg(block: &MachineBlock, reg: MachineReg) -> bool {
    block.params.iter().any(|param| param.reg == reg)
        || block
            .ops
            .iter()
            .any(|inst| inst_defines(&inst.kind, reg) || inst_uses_value(&inst.kind, reg))
        || terminator_uses_reg(&block.terminator, reg)
}

pub(super) fn block_index_for_id(blocks: &[MachineBlock], id: MachineBlockId) -> Option<usize> {
    let index = id.as_usize();
    if blocks.get(index).is_some_and(|block| block.id == id) {
        Some(index)
    } else {
        blocks.iter().position(|block| block.id == id)
    }
}

pub(super) fn visit_edges(
    terminator: &MachineTerminator,
    mut visit: impl FnMut(&crate::vm::machine::machine_ir::MachineEdge),
) {
    match terminator {
        MachineTerminator::Jump(edge) => visit(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            visit(then_edge);
            visit(else_edge);
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries {
                visit(edge);
            }
        }
        MachineTerminator::Call { success, .. } => visit(success),
        MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => {}
    }
}

pub(super) fn visit_edges_mut(
    terminator: &mut MachineTerminator,
    mut visit: impl FnMut(&mut crate::vm::machine::machine_ir::MachineEdge),
) {
    match terminator {
        MachineTerminator::Jump(edge) => visit(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            visit(then_edge);
            visit(else_edge);
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries {
                visit(edge);
            }
        }
        MachineTerminator::Call { success, .. } => visit(success),
        MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::machine::machine_ir::{MachineBranchCond, MachineEdge, MachineLoadExtension};

    fn edge(target: u32, args: &[MachineReg]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args.iter().copied().map(MachineValue::Reg).collect(),
        }
    }

    fn indexed_load(dst: u16, index: MachineReg, offset: i32) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: MachineReg(dst),
                base: MACHINE_MEM0_BASE_REG,
                index,
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        }
    }

    fn loop_program() -> MachineProgram {
        let index = MachineReg(5);
        let cond = MachineReg(6);
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::vec![
                        MachineBlockParam::gp_word(index),
                        MachineBlockParam::gp_word(cond),
                    ],
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Jump(edge(1, &[index, cond])),
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::vec![
                        MachineBlockParam::gp_word(index),
                        MachineBlockParam::gp_word(cond),
                    ],
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Branch {
                        cond: MachineBranchCond::Value(MachineValue::Reg(cond)),
                        then_edge: edge(2, &[index, cond]),
                        else_edge: edge(3, &[]),
                    },
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::vec![
                        MachineBlockParam::gp_word(index),
                        MachineBlockParam::gp_word(cond),
                    ],
                    ops: collections::vec![indexed_load(7, index, 4), indexed_load(8, index, 328),],
                    terminator: MachineTerminator::Jump(edge(1, &[index, cond])),
                },
                MachineBlock {
                    id: MachineBlockId(3),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Return,
                },
            ],
        }
    }

    fn config() -> BackendConfig {
        BackendConfig::with_volatility(8, 6, 0, 2, 0, 0, 4, 0, false, 3)
    }

    #[test]
    fn acyclic_backward_edge_is_not_a_loop_latch() {
        let blocks = collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(edge(0, &[])),
            },
        ];

        let loop_graph = analyze_loop_graph(&blocks, MachineBlockId(0));

        assert!(loop_graph
            .latches_by_header
            .iter()
            .all(|latches| latches.is_empty()));
        assert_eq!(loop_graph.predecessors[0].as_slice(), &[1]);
    }

    #[test]
    fn irreducible_scc_edges_are_not_natural_loop_latches() {
        let blocks = collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Imm64(1)),
                    then_edge: edge(1, &[]),
                    else_edge: edge(2, &[]),
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(edge(3, &[])),
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(edge(3, &[])),
            },
            MachineBlock {
                id: MachineBlockId(3),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Imm64(1)),
                    then_edge: edge(1, &[]),
                    else_edge: edge(2, &[]),
                },
            },
        ];

        let loop_graph = analyze_loop_graph(&blocks, MachineBlockId(0));

        assert!(loop_graph
            .latches_by_header
            .iter()
            .all(|latches| latches.is_empty()));
    }

    #[test]
    fn hoists_reused_invariant_mem0_index_into_loop_parameter() {
        let mut program = loop_program();
        let loop_graph = analyze_loop_graph(&program.blocks, program.entry);

        hoist_loop_address_bases(&mut program, config(), &loop_graph);

        let spare = MachineReg(4);
        assert!(matches!(
            program.blocks[0].ops.as_slice(),
            [
                MachineInst {
                    kind: MachineInstKind::Convert {
                        op: MachineConvertOp::I64ExtendI32U,
                        dst,
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: add_dst,
                        lhs: MachineValue::Reg(MACHINE_MEM0_BASE_REG),
                        rhs: MachineValue::Reg(rhs),
                    },
                },
            ] if *dst == spare && *add_dst == spare && *rhs == spare
        ));
        for block in &program.blocks[1..=2] {
            assert_eq!(block.params.last().map(|param| param.reg), Some(spare));
        }
        assert!(program.blocks[2].ops.iter().all(|inst| matches!(
            inst.kind,
            MachineInstKind::Load {
                addr: MachineAddr { base, .. },
                ..
            } if base == spare
        )));
        let MachineTerminator::Jump(backedge) = &program.blocks[2].terminator else {
            panic!("expected loop backedge");
        };
        assert_eq!(backedge.args.last(), Some(&MachineValue::Reg(spare)));
    }

    #[test]
    fn does_not_hoist_across_memory_grow() {
        let mut program = loop_program();
        program.blocks[2].ops.push(MachineInst {
            kind: MachineInstKind::MemoryGrow {
                mem_idx: 0,
                dst: MachineReg(9),
                delta: MachineValue::Imm64(1),
            },
        });
        let original = program.clone();
        let loop_graph = analyze_loop_graph(&program.blocks, program.entry);

        hoist_loop_address_bases(&mut program, config(), &loop_graph);

        assert_eq!(program, original);
    }

    #[test]
    fn does_not_hoist_when_backedge_rebinds_the_index_lane() {
        let mut program = loop_program();
        let MachineTerminator::Jump(backedge) = &mut program.blocks[2].terminator else {
            panic!("expected loop backedge");
        };
        backedge.args[0] = MachineValue::Reg(MachineReg(7));
        let original = program.clone();
        let loop_graph = analyze_loop_graph(&program.blocks, program.entry);

        hoist_loop_address_bases(&mut program, config(), &loop_graph);

        assert_eq!(program, original);
    }
}
