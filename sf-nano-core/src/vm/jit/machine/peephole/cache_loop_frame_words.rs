//! Reuse a native frame word in a GP lane left free throughout a natural loop.
//!
//! Stores remain in their original positions and update the carried copy as
//! well. Consequently exits and traps see the same published frame contents.
//! Only the loop's repeated native-frame reads disappear; guest memory reads
//! never move. Calls and opaque runtime operations are barriers.

use crate::collections;
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineConvertOp, MachineInst,
    MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner,
    MachineStorageType, MachineTerminator, MachineValue, MACHINE_FP_REG,
};

use super::helpers::{
    inst_defines, store_may_alias, terminator_uses_reg, visit_source_values,
    visit_terminator_source_regs,
};
use super::hoist_loop_address_bases::{
    block_mentions_reg, natural_loop_nodes, visit_edges_mut, LoopGraph,
};
use super::{optimize_block, BlockOptCtx};

#[derive(Clone, Copy, PartialEq, Eq)]
struct FrameWord {
    addr: MachineAddr,
    width: MachineMemWidth,
}

fn loaded_word(kind: &MachineInstKind, gp_bytes: u8) -> Option<FrameWord> {
    let MachineInstKind::Load {
        ty: MachineStorageType::GpWord,
        addr,
        width,
        extension: MachineLoadExtension::None,
        ..
    } = *kind
    else {
        return None;
    };
    (addr.base == MACHINE_FP_REG
        && addr.offset >= 0
        && addr.offset % i32::from(gp_bytes) == 0
        && width.bytes() == u32::from(gp_bytes))
    .then_some(FrameWord { addr, width })
}

pub(super) fn cache_loop_frame_words(
    blocks: &mut [MachineBlock],
    graph: &LoopGraph,
    entry: MachineBlockId,
    ctx: &mut BlockOptCtx,
) {
    if graph
        .latches_by_header
        .iter()
        .all(|latches| latches.is_empty())
    {
        return;
    }
    // Summarize loop blocks on demand instead of re-walking every instruction for
    // every physical lane and every enclosing loop. These are exact masks
    // for the low registers; unusually high synthetic/future lanes retain
    // the original membership scan below.
    let mut masks = collections::vec![None; blocks.len()];
    let gp_end = crate::vm::jit::backend::BackendConfig::FIXED
        + u16::from(ctx.config.allocatable_gp_dynamic_budget());
    let available_mask = (crate::vm::jit::backend::BackendConfig::FIXED..gp_end)
        .fold(0, |mask, reg| mask | reg_bit(MachineReg(reg)));
    for header in (0..blocks.len()).rev() {
        if graph.latches_by_header[header].is_empty() {
            continue;
        }
        let header_mask =
            *masks[header].get_or_insert_with(|| block_register_mask(&blocks[header]));
        if gp_end <= 64 && header_mask & available_mask == available_mask {
            continue;
        }
        let nodes = natural_loop_nodes(
            header,
            &graph.latches_by_header[header],
            &graph.predecessors,
        );
        if nodes.iter().any(|&index| blocks[index].id == entry) {
            continue;
        }
        let mut loop_mask = 0;
        for &index in &nodes {
            loop_mask |= *masks[index].get_or_insert_with(|| block_register_mask(&blocks[index]));
        }
        // Register pressure often rules out a loop before any frame-slot or
        // entry-edge analysis is useful.
        if gp_end <= 64 && loop_mask & available_mask == available_mask {
            continue;
        }
        try_cache_word(blocks, header, &nodes, graph, ctx, &mut masks, loop_mask);
    }
}

fn reg_bit(reg: MachineReg) -> u64 {
    1u64.checked_shl(u32::from(reg.0)).unwrap_or(0)
}

fn block_register_mask(block: &MachineBlock) -> u64 {
    let mut mask = 0;
    let mut note = |reg| mask |= reg_bit(reg);
    for param in &block.params {
        note(param.reg);
    }
    for inst in &block.ops {
        inst.kind.for_each_defined_reg(&mut note);
        visit_source_values(&inst.kind, |value| {
            if let MachineValue::Reg(reg) = value {
                note(*reg);
            }
        });
    }
    visit_terminator_source_regs(&block.terminator, &mut note);
    mask
}

fn try_cache_word(
    blocks: &mut [MachineBlock],
    header: usize,
    nodes: &[usize],
    graph: &LoopGraph,
    ctx: &mut BlockOptCtx,
    masks: &mut [Option<u64>],
    loop_mask: u64,
) {
    let config = ctx.config;
    // Reject loops without a reusable native-frame word before allocating
    // whole-function membership and predecessor scratch.
    let mut candidates: collections::Vec<(FrameWord, usize)> = collections::Vec::new();
    for &index in nodes {
        if matches!(
            blocks[index].terminator,
            MachineTerminator::Call { .. } | MachineTerminator::TailCall { .. }
        ) {
            return;
        }
        for inst in &blocks[index].ops {
            if inst_defines(&inst.kind, MACHINE_FP_REG) || !transparent(&inst.kind) {
                return;
            }
            if let Some(word) = loaded_word(&inst.kind, config.gp_unit_bytes) {
                if let Some((_, count)) = candidates.iter_mut().find(|(found, _)| *found == word) {
                    *count += 1;
                } else {
                    candidates.push((word, 1));
                }
            }
        }
    }
    // Repeated reads or a read/write recurrence can amortize the entry load.
    // A lone read-only reload still does not justify taking another lane.
    candidates.retain(|(word, count)| {
        let mut updated = false;
        let unaliased = nodes.iter().all(|&index| {
            blocks[index].ops.iter().all(|inst| match inst.kind {
                MachineInstKind::Store {
                    ty, addr, width, ..
                } => {
                    let exact = addr == word.addr
                        && width == word.width
                        && ty == MachineStorageType::GpWord;
                    updated |= exact;
                    !store_may_alias(word.addr, word.width, addr, width) || exact
                }
                // Indexed stores must be in guest memory, not a variable
                // offset within the frame or another runtime address space.
                MachineInstKind::IndexedStore { base, .. } => {
                    super::helpers::unknown_store_may_alias(base)
                }
                _ => true,
            })
        });
        unaliased && (*count >= 2 || updated)
    });
    let Some((word, _)) = candidates
        .into_iter()
        .max_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| b.addr.offset.cmp(&a.addr.offset)))
    else {
        return;
    };

    let mut in_loop = collections::vec![false; blocks.len()];
    for &index in nodes {
        in_loop[index] = true;
    }
    let mut preheaders = collections::Vec::new();
    for &target in nodes {
        for &source in &graph.predecessors[target] {
            if in_loop[source] {
                continue;
            }
            // A plain entry jump also makes the new register definition
            // unobservable along any path which does not enter the loop.
            if target != header
                || !matches!(&blocks[source].terminator,
                    MachineTerminator::Jump(edge) if edge.target == blocks[header].id)
            {
                return;
            }
            if !preheaders.contains(&source) {
                preheaders.push(source);
            }
        }
    }
    if preheaders.is_empty() {
        return;
    }

    let gp_end = crate::vm::jit::backend::BackendConfig::FIXED
        + u16::from(config.allocatable_gp_dynamic_budget());
    let Some(carry) = (crate::vm::jit::backend::BackendConfig::FIXED..gp_end)
        .map(MachineReg)
        .find(|&reg| {
            (if reg.0 < 64 {
                loop_mask & reg_bit(reg) == 0
            } else {
                nodes
                    .iter()
                    .all(|&index| !block_mentions_reg(&blocks[index], reg))
            }) && preheaders
                .iter()
                .all(|&index| !terminator_uses_reg(&blocks[index].terminator, reg))
        })
    else {
        return;
    };

    for &index in &preheaders {
        blocks[index].ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::CachedCell,
                ty: MachineStorageType::GpWord,
                dst: carry,
                addr: word.addr,
                width: word.width,
                extension: MachineLoadExtension::None,
            },
        });
    }
    let mut rewritten = collections::Vec::new();
    for &index in nodes {
        blocks[index].params.push(MachineBlockParam {
            reg: carry,
            ty: MachineStorageType::GpWord,
            owner: MachineRegOwner::CachedCell,
        });
        // Carry-only blocks gain an edge binding but have no new local value
        // flow to simplify. Leave their instruction storage intact and avoid
        // running the complete block-local pipeline again.
        if !blocks[index].ops.iter().any(|inst| {
            loaded_word(&inst.kind, config.gp_unit_bytes) == Some(word)
                || matches!(inst.kind, MachineInstKind::Store { addr, width, .. }
                    if addr == word.addr && width == word.width)
        }) {
            continue;
        }
        rewritten.push(index);
        let old = core::mem::take(&mut blocks[index].ops);
        let mut ops = collections::Vec::with_capacity(old.len());
        for mut inst in old {
            if loaded_word(&inst.kind, config.gp_unit_bytes) == Some(word) {
                let MachineInstKind::Load { owner, dst, .. } = inst.kind else {
                    unreachable!()
                };
                inst.kind = MachineInstKind::Move {
                    owner,
                    ty: MachineStorageType::GpWord,
                    dst,
                    src: MachineValue::Reg(carry),
                };
            }
            let copy = match inst.kind {
                MachineInstKind::Store {
                    addr, width, src, ..
                } if addr == word.addr && width == word.width => Some(src),
                _ => None,
            };
            ops.push(inst);
            if let Some(src) = copy {
                ops.push(MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::CachedCell,
                        ty: MachineStorageType::GpWord,
                        dst: carry,
                        src,
                    },
                });
            }
        }
        blocks[index].ops = ops;
    }
    let ids: collections::Vec<_> = nodes.iter().map(|&index| blocks[index].id).collect();
    for &index in nodes.iter().chain(preheaders.iter()) {
        visit_edges_mut(&mut blocks[index].terminator, |edge| {
            if ids.contains(&edge.target) {
                edge.args.push(MachineValue::Reg(carry));
            }
        });
        masks[index] = None;
    }
    for &index in rewritten.iter().chain(preheaders.iter()) {
        optimize_block(ctx, &mut blocks[index]);
    }
}

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

#[cfg(test)]
mod tests {
    use super::super::hoist_loop_address_bases::analyze_loop_graph;
    use super::*;
    use crate::vm::jit::backend::BackendConfig;
    use crate::vm::jit::machine::machine_ir::{
        MachineBranchCond, MachineCallRuntime, MachineConstId, MachineEdge, MachineIntBinaryOp,
        MachineIntWidth,
    };

    fn edge(target: u32, args: &[u16]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args
                .iter()
                .map(|&reg| MachineValue::Reg(MachineReg(reg)))
                .collect(),
        }
    }

    fn load(dst: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(dst),
                addr: MachineAddr {
                    base: MACHINE_FP_REG,
                    offset: 16,
                },
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        }
    }

    fn store(src: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: MachineAddr {
                    base: MACHINE_FP_REG,
                    offset: 16,
                },
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(MachineReg(src)),
            },
        }
    }

    fn add(dst: u16, lhs: u16, rhs: u64) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst: MachineReg(dst),
                lhs: MachineValue::Reg(MachineReg(lhs)),
                rhs: MachineValue::Imm64(rhs),
            },
        }
    }

    fn loop_blocks() -> collections::Vec<MachineBlock> {
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(4))],
                ops: collections::vec![store(4)],
                terminator: MachineTerminator::Jump(edge(1, &[4])),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(4))],
                ops: collections::vec![load(5), add(5, 5, 1)],
                terminator: MachineTerminator::Jump(edge(2, &[4, 5])),
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::vec![
                    MachineBlockParam::gp_word(MachineReg(4)),
                    MachineBlockParam::gp_word(MachineReg(5))
                ],
                ops: collections::vec![load(6), add(6, 6, 2), store(5), add(4, 4, u64::MAX)],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(4))),
                    then_edge: edge(1, &[4]),
                    else_edge: edge(3, &[5]),
                },
            },
            MachineBlock {
                id: MachineBlockId(3),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(5))],
                ops: collections::vec![load(4)],
                terminator: MachineTerminator::Return,
            },
        ]
    }

    fn run(blocks: &mut [MachineBlock], entry: MachineBlockId, gp_budget: u8) {
        let graph = analyze_loop_graph(blocks, entry);
        // The synthetic config reserves one lowering-only scratch lane.
        // A budget of five leaves r4..r7 allocatable and keeps r8 out.
        let config = BackendConfig::new(8, gp_budget, 0, 0);
        let mut ctx = BlockOptCtx::new(config);
        cache_loop_frame_words(blocks, &graph, entry, &mut ctx);
    }

    #[test]
    fn register_summary_matches_exact_mentions_before_and_after_rewriting() {
        let mut blocks = loop_blocks();
        for rewrite in [false, true] {
            if rewrite {
                run(&mut blocks, MachineBlockId(0), 6);
            }
            for block in &blocks {
                for reg in (0..64).map(MachineReg) {
                    assert_eq!(
                        block_register_mask(block) & reg_bit(reg) != 0,
                        block_mentions_reg(block, reg),
                    );
                }
            }
        }
    }

    #[test]
    fn carries_mutable_frame_word_without_delaying_publication_or_exit_reload() {
        let mut blocks = loop_blocks();
        let exit = blocks[3].clone();
        let published = blocks[2].ops[2].clone();
        run(&mut blocks, MachineBlockId(0), 5);
        let carry = blocks[1].params.last().unwrap().reg;
        assert_eq!(carry, MachineReg(7));
        for block in &blocks[1..3] {
            assert_eq!(block.params.last().unwrap().reg, carry);
            assert!(!block
                .ops
                .iter()
                .any(|inst| loaded_word(&inst.kind, 8).is_some()));
        }
        assert!(
            blocks[2].ops.contains(&published),
            "store stays in the loop"
        );
        assert_eq!(
            blocks[3], exit,
            "outside reads still consume published memory"
        );
        let MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } = &blocks[2].terminator
        else {
            unreachable!()
        };
        assert_eq!(then_edge.args.last(), Some(&MachineValue::Reg(carry)));
        assert_eq!(
            else_edge.args.len(),
            1,
            "no carried state leaks onto exit edges"
        );
    }

    // Execute the small loop fixtures independently of the optimizer and
    // compare every published store, including values observed by narrow loads.
    fn published_values(
        blocks: &[MachineBlock],
        iterations: u64,
        seed: u64,
    ) -> collections::Vec<(i32, u64)> {
        let mut regs = [0u64; 16];
        regs[4] = iterations;
        regs[5] = seed;
        let mut memory = [0u8; 32];
        let mut published = collections::Vec::new();
        let value = |value: MachineValue, regs: &[u64; 16]| match value {
            MachineValue::Reg(reg) => regs[usize::from(reg.0)],
            MachineValue::Imm64(value) => value,
            MachineValue::ReservedReg(reg) => panic!("fixture reads a reserved lane: {reg:?}"),
        };
        let mut current = 0;
        for _ in 0..1024 {
            let block = &blocks[current];
            for inst in &block.ops {
                match inst.kind {
                    MachineInstKind::Move { dst, src, .. } => {
                        regs[usize::from(dst.0)] = value(src, &regs);
                    }
                    MachineInstKind::Load {
                        dst,
                        addr,
                        width,
                        extension: MachineLoadExtension::None,
                        ..
                    } => {
                        assert_eq!(addr.base, MACHINE_FP_REG);
                        let start = usize::try_from(addr.offset).unwrap();
                        let end = start + width.bytes() as usize;
                        let mut bytes = [0u8; 8];
                        bytes[..end - start].copy_from_slice(&memory[start..end]);
                        regs[usize::from(dst.0)] = u64::from_le_bytes(bytes);
                    }
                    MachineInstKind::Store {
                        addr, width, src, ..
                    } => {
                        assert_eq!(addr.base, MACHINE_FP_REG);
                        let start = usize::try_from(addr.offset).unwrap();
                        let count = width.bytes() as usize;
                        let stored = value(src, &regs);
                        memory[start..start + count]
                            .copy_from_slice(&stored.to_le_bytes()[..count]);
                        published.push((addr.offset, stored));
                    }
                    MachineInstKind::IntBinary {
                        width,
                        op: MachineIntBinaryOp::Add,
                        dst,
                        lhs,
                        rhs,
                    } => {
                        let sum = value(lhs, &regs).wrapping_add(value(rhs, &regs));
                        regs[usize::from(dst.0)] = match width {
                            MachineIntWidth::I32 => u64::from(sum as u32),
                            MachineIntWidth::I64 => sum,
                        };
                    }
                    _ => panic!("unsupported fixture instruction: {:?}", inst.kind),
                }
            }
            let edge = match &block.terminator {
                MachineTerminator::Jump(edge) => edge,
                MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(cond),
                    then_edge,
                    else_edge,
                } => {
                    if value(*cond, &regs) != 0 {
                        then_edge
                    } else {
                        else_edge
                    }
                }
                MachineTerminator::Return => return published,
                other => panic!("unsupported fixture terminator: {other:?}"),
            };
            let next = blocks
                .iter()
                .position(|block| block.id == edge.target)
                .unwrap();
            let args: collections::Vec<_> =
                edge.args.iter().map(|&arg| value(arg, &regs)).collect();
            assert_eq!(args.len(), blocks[next].params.len());
            for (param, arg) in blocks[next].params.iter().zip(args) {
                regs[usize::from(param.reg.0)] = arg;
            }
            current = next;
        }
        panic!("fixture did not terminate");
    }

    #[test]
    fn single_read_updated_word_preserves_publication_and_narrow_observers() {
        for (width, offset) in [
            (MachineMemWidth::U8, 16),
            (MachineMemWidth::U16, 18),
            (MachineMemWidth::U32, 16),
            (MachineMemWidth::U32, 20),
        ] {
            for observe_after_store in [false, true] {
                let mut blocks = loop_blocks();
                blocks[0]
                    .params
                    .push(MachineBlockParam::gp_word(MachineReg(5)));
                blocks[0].ops[0] = store(5);
                if let MachineInstKind::IntBinary { width, .. } = &mut blocks[1].ops[1].kind {
                    *width = MachineIntWidth::I64;
                }
                let mut observe = load(6);
                if let MachineInstKind::Load {
                    width: load_width,
                    addr,
                    ..
                } = &mut observe.kind
                {
                    *load_width = width;
                    addr.offset = offset;
                }
                let mut publish_observation = store(6);
                if let MachineInstKind::Store { addr, .. } = &mut publish_observation.kind {
                    addr.offset = 24;
                }
                blocks[2].ops = if observe_after_store {
                    collections::vec![store(5), observe, publish_observation, add(4, 4, u64::MAX)]
                } else {
                    collections::vec![observe, publish_observation, store(5), add(4, 4, u64::MAX)]
                };
                let before = blocks.clone();
                run(&mut blocks, MachineBlockId(0), 5);
                assert_eq!(blocks[1].params.last().unwrap().reg, MachineReg(7));
                assert!(!blocks[1]
                    .ops
                    .iter()
                    .any(|inst| loaded_word(&inst.kind, 8).is_some()));
                for iterations in [1, 2, 17] {
                    for seed in [0, u32::MAX as u64, 0x1234_5678_ffff_ffff, u64::MAX] {
                        assert_eq!(
                            published_values(&blocks, iterations, seed),
                            published_values(&before, iterations, seed)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn single_read_readonly_word_does_not_claim_a_lane() {
        let mut blocks = loop_blocks();
        blocks[2].ops = collections::vec![add(4, 4, u64::MAX)];
        let before = blocks.clone();
        run(&mut blocks, MachineBlockId(0), 5);
        assert_eq!(blocks, before);
    }

    #[test]
    fn updated_word_takes_priority_without_losing_exit_only_value() {
        let mut original = loop_blocks();
        let mut seed_exit = store(4);
        if let MachineInstKind::Store { addr, .. } = &mut seed_exit.kind {
            addr.offset = 0;
        }
        original[0].ops.push(seed_exit);
        let mut exit_load = load(7);
        if let MachineInstKind::Load { addr, .. } = &mut exit_load.kind {
            addr.offset = 0;
        }
        original[3].ops.insert(0, exit_load.clone());
        let mut publish_exit = store(7);
        if let MachineInstKind::Store { addr, .. } = &mut publish_exit.kind {
            addr.offset = 24;
        }
        original[3].ops.push(publish_exit);
        let graph = analyze_loop_graph(&original, MachineBlockId(0));

        let mut exit_first = original.clone();
        super::super::reuse_loop_frame_values::reuse_loop_frame_values(
            &mut exit_first,
            &graph,
            MachineBlockId(0),
        );
        run(&mut exit_first, MachineBlockId(0), 5);
        assert!(exit_first[1]
            .ops
            .iter()
            .any(|inst| loaded_word(&inst.kind, 8).is_some()));
        assert!(!exit_first[3].ops.contains(&exit_load));

        let mut updated_first = original.clone();
        run(&mut updated_first, MachineBlockId(0), 5);
        super::super::reuse_loop_frame_values::reuse_loop_frame_values(
            &mut updated_first,
            &graph,
            MachineBlockId(0),
        );
        assert!(!updated_first[1]
            .ops
            .iter()
            .any(|inst| loaded_word(&inst.kind, 8).is_some()));
        assert!(updated_first[3].ops.contains(&exit_load));
        for iterations in [1, 2, 17] {
            let expected = published_values(&original, iterations, 0);
            assert_eq!(published_values(&exit_first, iterations, 0), expected);
            assert_eq!(published_values(&updated_first, iterations, 0), expected);
        }
    }

    #[test]
    fn rejects_aliases_calls_live_lanes_and_ambiguous_entries() {
        for case in 0..7 {
            let mut blocks = loop_blocks();
            let mut entry = MachineBlockId(0);
            let mut budget = 5;
            match case {
                0 => budget = 4,
                1 => {
                    blocks[0].terminator = MachineTerminator::Branch {
                        cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(4))),
                        then_edge: edge(1, &[4]),
                        else_edge: edge(3, &[4]),
                    }
                }
                2 => {
                    if let MachineInstKind::Store { width, .. } = &mut blocks[2].ops[2].kind {
                        *width = MachineMemWidth::U32;
                    }
                }
                3 => blocks[2].ops.insert(
                    0,
                    MachineInst {
                        kind: MachineInstKind::CallRuntime(MachineCallRuntime {
                            metadata: MachineConstId(0),
                        }),
                    },
                ),
                4 => blocks[2]
                    .ops
                    .insert(0, add(MACHINE_FP_REG.0, MACHINE_FP_REG.0, 8)),
                5 => entry = MachineBlockId(1),
                6 => {
                    // The only spare loop lane holds an existing entry argument.
                    blocks[0].terminator = MachineTerminator::Jump(edge(1, &[7]));
                }
                _ => unreachable!(),
            }
            let before = blocks.clone();
            run(&mut blocks, entry, budget);
            assert_eq!(blocks, before, "unsafe case {case}");
        }
    }
}
