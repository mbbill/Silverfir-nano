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

#[derive(Clone, Copy)]
enum CacheBlockFacts {
    Barrier,
    Registers(u64),
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
    'headers: for header in (0..blocks.len()).rev() {
        if graph.latches_by_header[header].is_empty() {
            continue;
        }
        let CacheBlockFacts::Registers(header_mask) =
            *masks[header].get_or_insert_with(|| block_cache_facts(&blocks[header]))
        else {
            continue;
        };
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
            let CacheBlockFacts::Registers(mask) =
                *masks[index].get_or_insert_with(|| block_cache_facts(&blocks[index]))
            else {
                continue 'headers;
            };
            loop_mask |= mask;
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

fn block_cache_facts(block: &MachineBlock) -> CacheBlockFacts {
    if matches!(
        block.terminator,
        MachineTerminator::Call { .. } | MachineTerminator::TailCall { .. }
    ) {
        return CacheBlockFacts::Barrier;
    }
    let mut mask = 0;
    let mut note = |reg| mask |= reg_bit(reg);
    for param in &block.params {
        note(param.reg);
    }
    for inst in &block.ops {
        // Calls and frame-base changes rule out every frame word at once.
        // Remember that fact before allocating or scanning entry-edge state.
        if inst_defines(&inst.kind, MACHINE_FP_REG) || !transparent(&inst.kind) {
            return CacheBlockFacts::Barrier;
        }
        inst.kind.for_each_defined_reg(&mut note);
        visit_source_values(&inst.kind, |value| {
            if let MachineValue::Reg(reg) = value {
                note(*reg);
            }
        });
    }
    visit_terminator_source_regs(&block.terminator, &mut note);
    CacheBlockFacts::Registers(mask)
}

fn try_cache_word(
    blocks: &mut [MachineBlock],
    header: usize,
    nodes: &[usize],
    graph: &LoopGraph,
    ctx: &mut BlockOptCtx,
    masks: &mut [Option<CacheBlockFacts>],
    loop_mask: u64,
) {
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

    let config = ctx.config;
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

    let mut candidates: collections::Vec<(FrameWord, usize)> = collections::Vec::new();
    for &index in nodes {
        for inst in &blocks[index].ops {
            if let Some(word) = loaded_word(&inst.kind, config.gp_unit_bytes) {
                if let Some((_, count)) = candidates.iter_mut().find(|(found, _)| *found == word) {
                    *count += 1;
                } else {
                    candidates.push((word, 1));
                }
            }
        }
    }
    // Require repeated static reads as well as a loop. This avoids paying
    // entry setup and write-through copies for a lone cheap reload.
    candidates.retain(|(word, count)| {
        *count >= 2
            && nodes.iter().all(|&index| {
                blocks[index].ops.iter().all(|inst| match inst.kind {
                    MachineInstKind::Store {
                        ty, addr, width, ..
                    } => {
                        !store_may_alias(word.addr, word.width, addr, width)
                            || (addr == word.addr
                                && width == word.width
                                && ty == MachineStorageType::GpWord)
                    }
                    // Indexed stores must be in guest memory, not a variable
                    // offset within the frame or another runtime address space.
                    MachineInstKind::IndexedStore { base, .. } => {
                        super::helpers::unknown_store_may_alias(base)
                    }
                    _ => true,
                })
            })
    });
    let Some((word, _)) = candidates
        .into_iter()
        .max_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| b.addr.offset.cmp(&a.addr.offset)))
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
                let CacheBlockFacts::Registers(mask) = block_cache_facts(block) else {
                    panic!("the fixture has no cache barrier");
                };
                for reg in (0..64).map(MachineReg) {
                    assert_eq!(mask & reg_bit(reg) != 0, block_mentions_reg(block, reg),);
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

    #[test]
    fn rejects_aliases_calls_live_lanes_and_ambiguous_entries() {
        for case in 0..8 {
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
                    if let MachineInstKind::Load { width, .. } = &mut blocks[2].ops[0].kind {
                        *width = MachineMemWidth::U32;
                    }
                }
                7 => {
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
