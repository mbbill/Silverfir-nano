//! Carry a recently published frame word across a single-predecessor edge.
//!
//! The store stays at its original trap-visible position. A normal block
//! parameter carries its full native word to the successor; only a redundant
//! frame load becomes a move. Guest loads, stores and traps never move.

use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineConvertOp, MachineEdge,
    MachineInst, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineRegOwner, MachineStorageType, MachineTerminator, MachineValue, MACHINE_FIXED_REG_COUNT,
    MACHINE_FP_REG, MACHINE_MEM0_BASE_REG,
};

use super::helpers::{inst_defines, inst_uses_value, store_may_alias, terminator_uses_reg};
use super::hoist_loop_address_bases::{
    block_index_for_id, block_mentions_reg, visit_edges, visit_edges_mut, LoopGraph,
};
use super::{copy_propagate, forward_stored_values, fuse_isel, BlockOptCtx};

// Bound work per edge, including in very large generated basic blocks. A miss
// leaves the load intact; this is an analysis budget, not a semantic limit.
const SCAN_LIMIT: usize = 64;

fn transparent(kind: &MachineInstKind) -> bool {
    matches!(
        kind,
        MachineInstKind::Move { .. }
            | MachineInstKind::Load { .. }
            | MachineInstKind::Store { .. }
            | MachineInstKind::IndexedLoad { .. }
            | MachineInstKind::IndexedStore { .. }
            | MachineInstKind::IntUnary { .. }
            | MachineInstKind::IntBinary { .. }
            | MachineInstKind::IntCompare { .. }
            | MachineInstKind::IntBinaryShifted { .. }
            | MachineInstKind::BitfieldExtractU { .. }
            | MachineInstKind::TestBits { .. }
            | MachineInstKind::Select { .. }
            | MachineInstKind::TrapIf { .. }
            | MachineInstKind::Convert {
                op: MachineConvertOp::I64ExtendI32U
                    | MachineConvertOp::I64ExtendI32S
                    | MachineConvertOp::I32WrapI64,
                ..
            }
    )
}

fn changes_frame(kind: &MachineInstKind, addr: MachineAddr, width: MachineMemWidth) -> bool {
    if inst_defines(kind, MACHINE_FP_REG) || !transparent(kind) {
        return true;
    }
    match *kind {
        MachineInstKind::Store {
            addr: other,
            width: other_width,
            ..
        } => store_may_alias(addr, width, other, other_width),
        // Only an explicit guest-memory base proves a variable-index store
        // cannot address the native frame.
        MachineInstKind::IndexedStore { base, .. } => base != MACHINE_MEM0_BASE_REG,
        _ => false,
    }
}

fn available_store(
    block: &MachineBlock,
    addr: MachineAddr,
    width: MachineMemWidth,
    gp_end: u16,
) -> Option<MachineValue> {
    for (index, inst) in block.ops.iter().enumerate().rev().take(SCAN_LIMIT) {
        if let MachineInstKind::Store {
            ty: MachineStorageType::GpWord,
            addr: stored_addr,
            width: stored_width,
            src,
        } = inst.kind
        {
            if stored_addr == addr && stored_width == width {
                return match src {
                    MachineValue::Reg(reg)
                        if (MACHINE_FIXED_REG_COUNT..gp_end).contains(&reg.0)
                            && block.ops[index + 1..]
                                .iter()
                                .all(|next| !inst_defines(&next.kind, reg)) =>
                    {
                        Some(src)
                    }
                    _ => None,
                };
            }
        }
        if changes_frame(&inst.kind, addr, width) {
            return None;
        }
    }
    None
}

fn existing_parameter(
    source: &MachineBlock,
    target: &MachineBlock,
    value: MachineValue,
    load_index: usize,
) -> Option<MachineReg> {
    target.params.iter().enumerate().find_map(|(index, param)| {
        if param.ty != MachineStorageType::GpWord
            || target.ops[..load_index]
                .iter()
                .any(|inst| inst_defines(&inst.kind, param.reg))
        {
            return None;
        }
        let mut found = false;
        let mut all_match = true;
        visit_edges(&source.terminator, |edge| {
            if edge.target == target.id {
                found = true;
                all_match &= edge.args.get(index) == Some(&value);
            }
        });
        (found && all_match).then_some(param.reg)
    })
}

/// Prove that an incoming register word dies on a straight-line path. Edge
/// arguments are parallel reads, so check all of them before treating a target
/// parameter as a replacement definition. A reservation does not write a word.
fn dead_on_edge<'a>(
    blocks: &'a [MachineBlock],
    mut edge: &'a MachineEdge,
    reg: MachineReg,
) -> bool {
    let mut remaining = SCAN_LIMIT;
    loop {
        if remaining == 0 {
            return false;
        }
        remaining -= 1;
        if edge.args.iter().any(|arg| *arg == MachineValue::Reg(reg)) {
            return false;
        }
        let Some(index) = block_index_for_id(blocks, edge.target) else {
            return false;
        };
        let block = &blocks[index];
        if block.params.len() != edge.args.len() {
            return false;
        }
        if block
            .params
            .iter()
            .zip(&edge.args)
            .any(|(param, arg)| param.reg == reg && !matches!(arg, MachineValue::ReservedReg(_)))
        {
            return true;
        }
        for inst in &block.ops {
            if remaining == 0 || !transparent(&inst.kind) || inst_uses_value(&inst.kind, reg) {
                return false;
            }
            remaining -= 1;
            if inst_defines(&inst.kind, reg) {
                return true;
            }
        }
        if terminator_uses_reg(&block.terminator, reg) {
            return false;
        }
        match &block.terminator {
            MachineTerminator::Return | MachineTerminator::ReturnScalar { .. } => return true,
            MachineTerminator::Jump(next) => edge = next,
            // Calls, branches and loops beyond the analysis budget retain
            // their original register state. This is only a local proof.
            _ => return false,
        }
    }
}

fn can_transport_without_new_stub(
    blocks: &[MachineBlock],
    source: usize,
    target: MachineBlockId,
    value: MachineValue,
    reg: MachineReg,
) -> bool {
    if value == MachineValue::Reg(reg)
        || matches!(blocks[source].terminator, MachineTerminator::Jump(_))
    {
        return true;
    }
    // A new conditional edge word must be copied before the condition, with
    // no changed input to either the condition or any parallel edge binding.
    if terminator_uses_reg(&blocks[source].terminator, reg) {
        return false;
    }
    let mut safe = true;
    visit_edges(&blocks[source].terminator, |edge| {
        if edge.target != target {
            safe &= dead_on_edge(blocks, edge, reg);
        }
    });
    safe
}

pub(super) fn forward_frame_values_across_edges(
    blocks: &mut [MachineBlock],
    graph: &LoopGraph,
    entry: MachineBlockId,
    ctx: &mut BlockOptCtx,
) {
    let gp_end = MACHINE_FIXED_REG_COUNT + u16::from(ctx.config.allocatable_gp_dynamic_budget());
    for target in 0..blocks.len() {
        let [source] = graph.predecessors[target].as_slice() else {
            continue;
        };
        let source = *source;
        if source == target
            || blocks[target].id == entry
            || !matches!(
                blocks[source].terminator,
                MachineTerminator::Jump(_) | MachineTerminator::Branch { .. }
            )
        {
            continue;
        }
        // At most one new carried word per edge. Other words remain published
        // and loaded normally without expanding register pressure further.
        for index in 0..blocks[target].ops.len().min(SCAN_LIMIT) {
            let MachineInstKind::Load {
                owner,
                ty: MachineStorageType::GpWord,
                dst,
                addr,
                width,
                extension: MachineLoadExtension::None,
            } = blocks[target].ops[index].kind
            else {
                continue;
            };
            if addr.base != MACHINE_FP_REG
                || addr.offset < 0
                || addr.offset % i32::from(ctx.config.gp_unit_bytes) != 0
                || width.bytes() != u32::from(ctx.config.gp_unit_bytes)
                || !(MACHINE_FIXED_REG_COUNT..gp_end).contains(&dst.0)
            {
                continue;
            }
            let Some(value) = available_store(&blocks[source], addr, width, gp_end) else {
                continue;
            };
            // A target prefix only needs checking when its predecessor can
            // supply this frame word. Most loads have no matching store.
            if blocks[target].ops[..index]
                .iter()
                .any(|inst| changes_frame(&inst.kind, addr, width))
            {
                continue;
            }
            let existing = existing_parameter(&blocks[source], &blocks[target], value, index);
            let carry = existing.or_else(|| {
                // Prefer the load's destination when its prior contents are
                // dead. Otherwise use a lane entirely absent from this block.
                if !blocks[target].params.iter().any(|param| param.reg == dst)
                    && blocks[target].ops[..index].iter().all(|inst| {
                        !inst_defines(&inst.kind, dst) && !inst_uses_value(&inst.kind, dst)
                    })
                    && can_transport_without_new_stub(blocks, source, blocks[target].id, value, dst)
                {
                    Some(dst)
                } else {
                    (MACHINE_FIXED_REG_COUNT..gp_end)
                        .map(MachineReg)
                        .find(|&reg| {
                            !block_mentions_reg(&blocks[target], reg)
                                && can_transport_without_new_stub(
                                    blocks,
                                    source,
                                    blocks[target].id,
                                    value,
                                    reg,
                                )
                        })
                }
            });
            let Some(carry) = carry else { continue };
            if existing.is_none() {
                let target_id = blocks[target].id;
                let edge_value =
                    if matches!(blocks[source].terminator, MachineTerminator::Branch { .. }) {
                        if value != MachineValue::Reg(carry) {
                            blocks[source].ops.push(MachineInst {
                                kind: MachineInstKind::Move {
                                    owner: MachineRegOwner::LinearValue,
                                    ty: MachineStorageType::GpWord,
                                    dst: carry,
                                    src: value,
                                },
                            });
                        }
                        MachineValue::Reg(carry)
                    } else {
                        value
                    };
                blocks[target].params.push(MachineBlockParam {
                    reg: carry,
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                });
                visit_edges_mut(&mut blocks[source].terminator, |edge| {
                    if edge.target == target_id {
                        edge.args.push(edge_value);
                    }
                });
            }
            blocks[target].ops[index].kind = MachineInstKind::Move {
                owner,
                ty: MachineStorageType::GpWord,
                dst,
                src: MachineValue::Reg(carry),
            };
            // Resolve the introduced register copy before forwarding local
            // frame words: the copy can hide their matching store operands.
            // Copy propagation can also expose instruction-selection pairs.
            // Keep this cleanup limited to those consequences of the rewrite.
            copy_propagate::copy_propagate(&mut blocks[target], ctx.config, &mut ctx.cp_scratch);
            forward_stored_values::forward_stored_values(
                &mut blocks[target],
                ctx.config,
                &mut ctx.tracked_stores,
            );
            fuse_isel::fuse_isel(&mut blocks[target], ctx.config);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hoist_loop_address_bases::analyze_loop_graph;
    use super::*;
    use crate::collections;
    use crate::vm::jit::backend::BackendConfig;
    use crate::vm::jit::machine::machine_ir::{
        MachineBranchCond, MachineEdge, MachineInst, MachineIntBinaryOp, MachineIntWidth,
        MachineResultSrc, MachineReturnValue,
    };

    fn reg(reg: u16) -> MachineValue {
        MachineValue::Reg(MachineReg(reg))
    }
    fn inst(kind: MachineInstKind) -> MachineInst {
        MachineInst { kind }
    }
    fn mov(dst: u16, src: u16) -> MachineInst {
        inst(MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(dst),
            src: reg(src),
        })
    }
    fn load() -> MachineInst {
        inst(MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(6),
            addr: MachineAddr {
                base: MACHINE_FP_REG,
                offset: 16,
            },
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        })
    }
    fn store(src: u16, offset: i32, width: MachineMemWidth) -> MachineInst {
        inst(MachineInstKind::Store {
            ty: MachineStorageType::GpWord,
            addr: MachineAddr {
                base: MACHINE_FP_REG,
                offset,
            },
            width,
            src: reg(src),
        })
    }
    fn edge(target: u32, args: &[u16]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args.iter().map(|&r| reg(r)).collect(),
        }
    }
    fn returned(r: u16) -> MachineTerminator {
        MachineTerminator::ReturnScalar {
            value: MachineReturnValue::ScalarGp {
                src: MachineResultSrc::Reg(MachineReg(r)),
                ty: MachineStorageType::GpWord,
            },
        }
    }
    fn fixture() -> collections::Vec<MachineBlock> {
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![],
                ops: collections::vec![store(4, 16, MachineMemWidth::U64)],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(reg(9)),
                    then_edge: edge(1, &[5]),
                    else_edge: edge(2, &[]),
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(5))],
                ops: collections::vec![
                    mov(6, 5),
                    mov(7, 6),
                    load(),
                    inst(MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Xor,
                        dst: MachineReg(8),
                        lhs: reg(6),
                        rhs: reg(7),
                    })
                ],
                terminator: returned(8),
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: returned(4)
            },
        ]
    }
    fn optimize(blocks: &mut [MachineBlock], entry: u32) {
        let graph = analyze_loop_graph(blocks, MachineBlockId(entry));
        // Seven allocatable lanes plus the backend-owned scratch lane.
        let mut ctx = BlockOptCtx::new(BackendConfig::new(8, 8, 0, 0));
        forward_frame_values_across_edges(blocks, &graph, MachineBlockId(entry), &mut ctx);
    }
    fn run(blocks: &[MachineBlock], a: u64, b: u64, branch: u64) -> (u64, [u8; 32]) {
        let mut regs = [0u64; 16];
        regs[4] = a;
        regs[5] = b;
        regs[9] = branch;
        let mut frame = [0u8; 32];
        let value = |v: MachineValue, regs: &[u64; 16]| match v {
            MachineValue::Reg(r) => regs[r.0 as usize],
            MachineValue::Imm64(v) => v,
            _ => panic!("unexpected value"),
        };
        let mut block = &blocks[0];
        for _ in 0..8 {
            for i in &block.ops {
                match i.kind {
                    MachineInstKind::Store {
                        addr, width, src, ..
                    } => {
                        let start = addr.offset as usize;
                        frame[start..start + width.bytes() as usize].copy_from_slice(
                            &value(src, &regs).to_le_bytes()[..width.bytes() as usize],
                        );
                    }
                    MachineInstKind::Load {
                        dst, addr, width, ..
                    } => {
                        let mut bytes = [0; 8];
                        bytes[..width.bytes() as usize].copy_from_slice(
                            &frame[addr.offset as usize
                                ..addr.offset as usize + width.bytes() as usize],
                        );
                        regs[dst.0 as usize] = u64::from_le_bytes(bytes);
                    }
                    MachineInstKind::Move { dst, src, .. } => {
                        regs[dst.0 as usize] = value(src, &regs)
                    }
                    MachineInstKind::IntBinary {
                        dst,
                        op: MachineIntBinaryOp::Xor,
                        lhs,
                        rhs,
                        ..
                    } => regs[dst.0 as usize] = value(lhs, &regs) ^ value(rhs, &regs),
                    _ => panic!("unexpected instruction {:?}", i.kind),
                }
            }
            let e = match &block.terminator {
                MachineTerminator::Jump(e) => e,
                MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(c),
                    then_edge,
                    else_edge,
                } => {
                    if value(*c, &regs) as u32 != 0 {
                        then_edge
                    } else {
                        else_edge
                    }
                }
                MachineTerminator::ReturnScalar {
                    value:
                        MachineReturnValue::ScalarGp {
                            src: MachineResultSrc::Reg(r),
                            ..
                        },
                } => return (regs[r.0 as usize], frame),
                _ => panic!("unexpected terminator"),
            };
            let args: collections::Vec<_> = e.args.iter().map(|&a| value(a, &regs)).collect();
            block = &blocks[e.target.0 as usize];
            assert_eq!(args.len(), block.params.len());
            for (p, v) in block.params.iter().zip(args) {
                regs[p.reg.0 as usize] = v;
            }
        }
        panic!("nonterminating fixture")
    }

    #[test]
    fn forwards_full_words_without_changing_other_edge_or_published_frame() {
        let before = fixture();
        let mut after = before.clone();
        optimize(&mut after, 0);
        assert_eq!(after[0].ops, before[0].ops);
        assert_eq!(after[1].params.len(), 2);
        assert!(!after[1]
            .ops
            .iter()
            .any(|i| matches!(i.kind, MachineInstKind::Load { .. })));
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..256 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for branch in [0, 1, 1 << 32, u64::MAX] {
                assert_eq!(
                    run(&before, state, !state.rotate_left(17), branch),
                    run(&after, state, !state.rotate_left(17), branch)
                );
            }
        }
    }

    #[test]
    fn reuses_an_existing_edge_parameter_without_allocating_a_lane() {
        let mut blocks = fixture();
        if let MachineTerminator::Branch { then_edge, .. } = &mut blocks[0].terminator {
            then_edge.args[0] = reg(4);
        }
        let before = blocks.clone();
        optimize(&mut blocks, 0);
        assert_eq!(blocks[1].params.len(), 1);
        for a in [0, 1, u64::MAX, 1 << 63, 0xffff_ffff_0000_0000] {
            assert_eq!(run(&before, a, 17, 1), run(&blocks, a, 17, 1));
        }
    }

    #[test]
    fn rejects_clobbers_overlaps_and_frame_changes_on_either_side() {
        for side in [0, 1] {
            for mutation in [mov(4, 5), mov(1, 5), store(5, 20, MachineMemWidth::U32)] {
                let mut blocks = fixture();
                if side == 0 {
                    blocks[0].ops.push(mutation.clone());
                } else {
                    blocks[1].ops.insert(0, mutation.clone());
                }
                // Clobbering the predecessor's source in the successor is
                // harmless when a separate edge parameter carries the word.
                if side == 1 && mutation == mov(4, 5) {
                    continue;
                }
                let before = blocks.clone();
                optimize(&mut blocks, 0);
                assert_eq!(blocks, before);
            }
        }
    }

    #[test]
    fn rejects_multiple_predecessors_and_implicit_entry_edges() {
        let mut blocks = fixture();
        blocks[2].terminator = MachineTerminator::Jump(edge(1, &[5]));
        let before = blocks.clone();
        optimize(&mut blocks, 0);
        assert_eq!(blocks, before);
        let mut blocks = fixture();
        let before = blocks.clone();
        optimize(&mut blocks, 1);
        assert_eq!(blocks, before);
    }

    #[test]
    fn cannot_reuse_a_parameter_overwritten_before_the_load() {
        let mut blocks = fixture();
        if let MachineTerminator::Branch { then_edge, .. } = &mut blocks[0].terminator {
            then_edge.args[0] = reg(4);
        }
        blocks[1].ops.insert(0, mov(5, 9));
        let before = blocks.clone();
        optimize(&mut blocks, 0);
        assert_eq!(blocks[1].params.len(), 2);
        assert_eq!(run(&before, u64::MAX, 21, 1), run(&blocks, u64::MAX, 21, 1));
    }

    #[test]
    fn both_edges_to_one_target_receive_the_carried_word() {
        let mut blocks = fixture();
        if let MachineTerminator::Branch { else_edge, .. } = &mut blocks[0].terminator {
            *else_edge = edge(1, &[4]);
        }
        let before = blocks.clone();
        optimize(&mut blocks, 0);
        assert_eq!(blocks[1].params.len(), 2);
        for branch in [0, 1] {
            assert_eq!(
                run(&before, 0xffff_0000_1234_5678, 99, branch),
                run(&blocks, 0xffff_0000_1234_5678, 99, branch)
            );
        }
    }

    #[test]
    fn partial_or_extended_loads_keep_their_memory_operation() {
        for mode in 0..3 {
            let mut blocks = fixture();
            if let MachineInstKind::Load {
                width,
                extension,
                addr,
                ..
            } = &mut blocks[1].ops[2].kind
            {
                match mode {
                    0 => *width = MachineMemWidth::U32,
                    1 => *extension = MachineLoadExtension::SignExtend,
                    _ => addr.offset = 20,
                }
            }
            let before = blocks.clone();
            optimize(&mut blocks, 0);
            assert_eq!(blocks, before);
        }
    }

    #[test]
    fn opaque_runtime_operations_and_variable_frame_stores_are_barriers() {
        use crate::vm::jit::machine::machine_ir::{
            MachineCallRuntime, MachineConstId, MachineIndexExtend,
        };
        let barriers = [
            inst(MachineInstKind::CallRuntime(MachineCallRuntime {
                metadata: MachineConstId(0),
            })),
            inst(MachineInstKind::IndexedStore {
                base: MACHINE_FP_REG,
                index: MachineReg(5),
                index_extend: MachineIndexExtend::None,
                offset: 0,
                width: MachineMemWidth::U64,
                src: reg(5),
            }),
        ];
        for barrier in barriers {
            for side in [0, 1] {
                let mut blocks = fixture();
                if side == 0 {
                    blocks[0].ops.push(barrier.clone());
                } else {
                    blocks[1].ops.insert(0, barrier.clone());
                }
                let before = blocks.clone();
                optimize(&mut blocks, 0);
                assert_eq!(blocks, before);
            }
        }
    }

    #[test]
    fn declines_when_every_lane_carries_an_existing_value() {
        let mut blocks = fixture();
        blocks[1].params = (4..11)
            .map(|r| MachineBlockParam::gp_word(MachineReg(r)))
            .collect();
        if let MachineTerminator::Branch { then_edge, .. } = &mut blocks[0].terminator {
            then_edge.args = (4..11).map(|_| reg(5)).collect();
        }
        let before = blocks.clone();
        optimize(&mut blocks, 0);
        assert_eq!(blocks, before);
    }

    #[test]
    fn forwards_native_32_bit_words_without_synthesizing_a_wider_move() {
        let mut blocks = fixture();
        if let MachineInstKind::Store { width, .. } = &mut blocks[0].ops[0].kind {
            *width = MachineMemWidth::U32;
        }
        if let MachineInstKind::Load { width, .. } = &mut blocks[1].ops[2].kind {
            *width = MachineMemWidth::U32;
        }
        let graph = analyze_loop_graph(&blocks, MachineBlockId(0));
        let mut ctx = BlockOptCtx::new(BackendConfig::new(4, 7, 0, 0));
        forward_frame_values_across_edges(&mut blocks, &graph, MachineBlockId(0), &mut ctx);
        assert_eq!(blocks[1].params.len(), 2);
        assert!(!blocks[1]
            .ops
            .iter()
            .any(|i| matches!(i.kind, MachineInstKind::Load { .. })));
        assert!(blocks[1]
            .params
            .iter()
            .all(|p| p.ty == MachineStorageType::GpWord));
    }

    fn copied_fixture() -> collections::Vec<MachineBlock> {
        let mut blocks = fixture();
        // The stored source is reused in the destination before its load;
        // r6..r8 are occupied and r9 is the branch condition. Only r10 can
        // carry the published word without changing another live value.
        blocks[1].ops.insert(0, mov(4, 5));
        blocks
    }

    #[test]
    fn prebranch_copy_removes_only_the_new_edge_move() {
        let before = copied_fixture();
        let mut after = before.clone();
        optimize(&mut after, 0);
        assert_eq!(after[0].ops.last(), Some(&mov(10, 4)));
        assert_eq!(after[1].params.last().unwrap().reg, MachineReg(10));
        let MachineTerminator::Branch { then_edge, .. } = &after[0].terminator else {
            panic!()
        };
        assert!(after[1]
            .params
            .iter()
            .zip(&then_edge.args)
            .all(|(param, arg)| *arg == MachineValue::Reg(param.reg)));
        let mut state = 0xb137_5ace_842d_f609u64;
        for _ in 0..256 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for branch in [0, 1, 1 << 32, u64::MAX] {
                assert_eq!(
                    run(&before, state, !state.rotate_left(9), branch),
                    run(&after, state, !state.rotate_left(9), branch)
                );
            }
        }
    }

    #[test]
    fn follows_straight_paths_and_distinguishes_later_reads_from_overwrites() {
        for overwrite in [false, true] {
            let mut blocks = copied_fixture();
            blocks[2].terminator = MachineTerminator::Jump(edge(3, &[]));
            blocks.push(MachineBlock {
                id: MachineBlockId(3),
                params: collections::vec![],
                ops: if overwrite {
                    collections::vec![mov(10, 4)]
                } else {
                    collections::vec![]
                },
                terminator: returned(10),
            });
            let before = blocks.clone();
            optimize(&mut blocks, 0);
            if overwrite {
                assert_eq!(blocks[0].ops.last(), Some(&mov(10, 4)));
                for branch in [0, 1] {
                    assert_eq!(
                        run(&before, u64::MAX, 19, branch),
                        run(&blocks, u64::MAX, 19, branch)
                    );
                }
            } else {
                assert_eq!(blocks, before);
            }
        }
    }

    #[test]
    fn edge_arguments_are_read_before_parallel_parameter_definitions() {
        for reads_old_word in [false, true] {
            let mut blocks = copied_fixture();
            blocks[2].terminator =
                MachineTerminator::Jump(edge(3, if reads_old_word { &[4, 10] } else { &[4, 5] }));
            blocks.push(MachineBlock {
                id: MachineBlockId(3),
                params: collections::vec![
                    MachineBlockParam::gp_word(MachineReg(10)),
                    MachineBlockParam::gp_word(MachineReg(4))
                ],
                ops: collections::vec![],
                terminator: returned(4),
            });
            let before = blocks.clone();
            optimize(&mut blocks, 0);
            if reads_old_word {
                assert_eq!(blocks, before);
            } else {
                assert_eq!(blocks[0].ops.last(), Some(&mov(10, 4)));
                for branch in [0, 1] {
                    assert_eq!(
                        run(&before, 0x8000_0000_0000_0001, 43, branch),
                        run(&blocks, 0x8000_0000_0000_0001, 43, branch)
                    );
                }
            }
        }
    }

    #[test]
    fn conditions_reservations_calls_and_unbounded_paths_do_not_prove_a_dead_word() {
        use crate::vm::jit::machine::machine_ir::{MachineCallRuntime, MachineConstId};
        for case in 0..6 {
            let mut blocks = copied_fixture();
            match case {
                0 => blocks[1].ops.insert(0, mov(10, 5)), // Only free r9 is the condition.
                1 => {
                    let MachineTerminator::Branch { else_edge, .. } = &mut blocks[0].terminator
                    else {
                        panic!()
                    };
                    else_edge
                        .args
                        .push(MachineValue::ReservedReg(MachineReg(10)));
                    blocks[2]
                        .params
                        .push(MachineBlockParam::gp_word(MachineReg(10)));
                    blocks[2].terminator = returned(10);
                }
                2 => blocks[2]
                    .ops
                    .push(inst(MachineInstKind::CallRuntime(MachineCallRuntime {
                        metadata: MachineConstId(0),
                    }))),
                3 => blocks[2].terminator = MachineTerminator::Jump(edge(2, &[])),
                4 => blocks[2].ops.push(inst(MachineInstKind::IntBinary {
                    width: MachineIntWidth::I64,
                    op: MachineIntBinaryOp::Xor,
                    dst: MachineReg(10),
                    lhs: reg(10),
                    rhs: reg(4),
                })),
                _ => blocks[2].ops.push(inst(MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(10),
                    addr: MachineAddr {
                        base: MachineReg(10),
                        offset: 0,
                    },
                    width: MachineMemWidth::U64,
                    extension: MachineLoadExtension::None,
                })),
            }
            let before = blocks.clone();
            optimize(&mut blocks, 0);
            assert_eq!(blocks, before, "case={case}");
        }
    }
}
