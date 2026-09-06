//! Forward a published frame word across one edge, or recompute its cheap
//! integer expression from still-live edge inputs. The store stays in place.
//!
//! Only unique-predecessor blocks participate. Both instruction windows are
//! bounded, and every input must have an explicit, unchanged edge binding.

use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineEdge, MachineInstKind, MachineIntBinaryOp,
    MachineIntWidth, MachineLoadExtension, MachineRegOwner, MachineStorageType, MachineTerminator,
    MachineValue, MACHINE_FP_REG,
};

use super::helpers::inst_defines;
use super::hoist_loop_address_bases::LoopGraph;

const WINDOW: usize = 8;

pub(super) fn rematerialize_frame_reloads(
    blocks: &mut [MachineBlock],
    graph: &LoopGraph,
    entry: MachineBlockId,
    gp_bytes: u8,
) {
    for target in 0..blocks.len() {
        let [source] = graph.predecessors[target].as_slice() else {
            continue;
        };
        if *source == target || blocks[target].id == entry {
            continue;
        }
        for index in 0..blocks[target].ops.len().min(WINDOW) {
            let Some(edge) = unique_edge(&blocks[*source].terminator, blocks[target].id) else {
                break;
            };
            if !preserves_frame(&blocks[target].ops[index].kind) {
                break;
            }
            if let Some(kind) =
                replacement(&blocks[*source], &blocks[target], edge, index, gp_bytes)
            {
                blocks[target].ops[index].kind = kind;
            }
        }
    }
}

fn unique_edge(term: &MachineTerminator, target: MachineBlockId) -> Option<&MachineEdge> {
    match term {
        MachineTerminator::Jump(edge) if edge.target == target => Some(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => match (then_edge.target == target, else_edge.target == target) {
            (true, false) => Some(then_edge),
            (false, true) => Some(else_edge),
            (true, true) if then_edge == else_edge => Some(then_edge),
            _ => None,
        },
        _ => None,
    }
}

fn preserves_frame(kind: &MachineInstKind) -> bool {
    !inst_defines(kind, MACHINE_FP_REG)
        && matches!(
            kind,
            MachineInstKind::Move { .. }
                | MachineInstKind::Load { .. }
                | MachineInstKind::IndexedLoad { .. }
                | MachineInstKind::IntUnary { .. }
                | MachineInstKind::IntBinary { .. }
                | MachineInstKind::IntBinaryShifted { .. }
                | MachineInstKind::IntCompare { .. }
                | MachineInstKind::TestBits { .. }
                | MachineInstKind::BitfieldExtractU { .. }
                | MachineInstKind::Select { .. }
                | MachineInstKind::TrapIf { .. }
        )
}

fn replacement(
    source: &MachineBlock,
    target: &MachineBlock,
    edge: &MachineEdge,
    load_index: usize,
    gp_bytes: u8,
) -> Option<MachineInstKind> {
    let MachineInstKind::Load {
        owner: MachineRegOwner::LinearValue,
        ty: MachineStorageType::GpWord,
        dst,
        addr,
        width,
        extension: MachineLoadExtension::None,
    } = target.ops[load_index].kind
    else {
        return None;
    };
    if addr.base != MACHINE_FP_REG
        || addr.offset < 0
        || addr.offset % i32::from(gp_bytes) != 0
        || width.bytes() != u32::from(gp_bytes)
    {
        return None;
    }
    let (store_index, stored) =
        source
            .ops
            .iter()
            .enumerate()
            .rev()
            .take(WINDOW)
            .find_map(|(index, inst)| match inst.kind {
                MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr: stored_addr,
                    width: stored_width,
                    src,
                } if addr == stored_addr && width == stored_width => Some((index, src)),
                _ => None,
            })?;
    if !source.ops[store_index + 1..]
        .iter()
        .all(|inst| preserves_frame(&inst.kind))
    {
        return None;
    }
    let map_input = |value: MachineValue, defined_at: usize| match value {
        MachineValue::Imm64(_) => Some(value),
        MachineValue::Reg(reg)
            if !source.ops[defined_at + 1..]
                .iter()
                .any(|inst| inst_defines(&inst.kind, reg)) =>
        {
            edge.args
                .iter()
                .zip(&target.params)
                .find_map(|(arg, param)| {
                    (*arg == value
                        && param.ty == MachineStorageType::GpWord
                        && !target.ops[..load_index]
                            .iter()
                            .any(|inst| inst_defines(&inst.kind, param.reg)))
                    .then_some(MachineValue::Reg(param.reg))
                })
        }
        _ => None,
    };
    if let Some(src) = map_input(stored, store_index) {
        return Some(MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst,
            src,
        });
    }
    let MachineValue::Reg(stored_reg) = stored else {
        return None;
    };
    let (producer_index, producer) = source.ops[..store_index]
        .iter()
        .enumerate()
        .rev()
        .take(WINDOW)
        .find(|(_, inst)| inst_defines(&inst.kind, stored_reg))?;
    let MachineInstKind::IntBinary {
        width: int_width,
        op,
        dst: produced,
        lhs,
        rhs,
    } = producer.kind
    else {
        return None;
    };
    if !source.ops[producer_index + 1..store_index]
        .iter()
        .all(|inst| preserves_frame(&inst.kind))
        || (int_width == MachineIntWidth::I64 && gp_bytes != 8)
        || !matches!(
            op,
            MachineIntBinaryOp::Add
                | MachineIntBinaryOp::Sub
                | MachineIntBinaryOp::And
                | MachineIntBinaryOp::Or
                | MachineIntBinaryOp::Xor
        )
    {
        return None;
    }
    let operand = |value| {
        let value = if value == MachineValue::Reg(produced) {
            // A two-address operation commonly follows an explicit copy.
            // Its pre-update input remains available in that copy's source.
            let MachineInstKind::Move {
                ty: MachineStorageType::GpWord,
                dst: copied,
                src,
                ..
            } = source.ops.get(producer_index.checked_sub(1)?)?.kind
            else {
                return None;
            };
            if copied != produced || src == MachineValue::Reg(produced) {
                return None;
            }
            src
        } else {
            value
        };
        map_input(value, producer_index)
    };
    Some(MachineInstKind::IntBinary {
        width: int_width,
        op,
        dst,
        lhs: operand(lhs)?,
        rhs: operand(rhs)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::hoist_loop_address_bases::analyze_loop_graph;
    use super::*;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlockParam, MachineBranchCond, MachineCallRuntime, MachineConstId,
        MachineInst, MachineMemWidth, MachineReg,
    };

    fn copy(dst: u16, src: MachineValue) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(dst),
                src,
            },
        }
    }

    fn fixture(bytes: u8, renamed: bool) -> collections::Vec<MachineBlock> {
        let addr = MachineAddr {
            base: MACHINE_FP_REG,
            offset: 24,
        };
        let width = if bytes == 8 {
            MachineMemWidth::U64
        } else {
            MachineMemWidth::U32
        };
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(4))],
                ops: collections::vec![
                    copy(5, MachineValue::Reg(MachineReg(4))),
                    MachineInst {
                        kind: MachineInstKind::IntBinary {
                            width: MachineIntWidth::I32,
                            op: MachineIntBinaryOp::Add,
                            dst: MachineReg(5),
                            lhs: MachineValue::Reg(MachineReg(5)),
                            rhs: MachineValue::Imm64(1),
                        }
                    },
                    MachineInst {
                        kind: MachineInstKind::Store {
                            ty: MachineStorageType::GpWord,
                            addr,
                            width,
                            src: MachineValue::Reg(MachineReg(5)),
                        }
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::Reg(MachineReg(4))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(if renamed {
                    7
                } else {
                    4
                }))],
                ops: collections::vec![
                    copy(5, MachineValue::Imm64(123)),
                    MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(5),
                            addr,
                            width,
                            extension: MachineLoadExtension::None,
                        }
                    },
                ],
                terminator: MachineTerminator::Return,
            },
        ]
    }

    fn run(blocks: &mut [MachineBlock], bytes: u8) {
        let graph = analyze_loop_graph(blocks, MachineBlockId(0));
        rematerialize_frame_reloads(blocks, &graph, MachineBlockId(0), bytes);
    }

    #[test]
    fn reconstructs_wrapping_arithmetic_through_identity_and_renamed_edges() {
        for bytes in [4, 8] {
            for renamed in [false, true] {
                for op in [
                    MachineIntBinaryOp::Add,
                    MachineIntBinaryOp::Sub,
                    MachineIntBinaryOp::And,
                    MachineIntBinaryOp::Or,
                    MachineIntBinaryOp::Xor,
                ] {
                    let mut blocks = fixture(bytes, renamed);
                    let MachineInstKind::IntBinary { op: operation, .. } =
                        &mut blocks[0].ops[1].kind
                    else {
                        unreachable!()
                    };
                    *operation = op;
                    let source = blocks[0].clone();
                    run(&mut blocks, bytes);
                    assert_eq!(
                        blocks[0], source,
                        "the original computation and store stay published"
                    );
                    assert_eq!(
                        blocks[1].ops[1].kind,
                        MachineInstKind::IntBinary {
                            width: MachineIntWidth::I32,
                            op,
                            dst: MachineReg(5),
                            lhs: MachineValue::Reg(MachineReg(if renamed { 7 } else { 4 })),
                            rhs: MachineValue::Imm64(1),
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn forwards_an_available_stored_value_without_recomputing_it() {
        let mut blocks = fixture(8, true);
        let MachineTerminator::Jump(edge) = &mut blocks[0].terminator else {
            unreachable!()
        };
        edge.args[0] = MachineValue::Reg(MachineReg(5));
        run(&mut blocks, 8);
        assert_eq!(blocks[1].ops[1], copy(5, MachineValue::Reg(MachineReg(7))));
    }

    #[test]
    fn preserves_two_distinct_inputs_and_the_full_integer_width() {
        for width in [MachineIntWidth::I32, MachineIntWidth::I64] {
            let mut blocks = fixture(8, true);
            blocks[0]
                .params
                .push(MachineBlockParam::gp_word(MachineReg(6)));
            blocks[0].ops.remove(0);
            blocks[0].ops[0].kind = MachineInstKind::IntBinary {
                width,
                op: MachineIntBinaryOp::Sub,
                dst: MachineReg(5),
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Reg(MachineReg(6)),
            };
            let MachineTerminator::Jump(edge) = &mut blocks[0].terminator else {
                unreachable!()
            };
            edge.args.push(MachineValue::Reg(MachineReg(6)));
            blocks[1]
                .params
                .push(MachineBlockParam::gp_word(MachineReg(4)));
            let before = blocks.clone();
            run(&mut blocks, 8);
            assert_eq!(blocks[0], before[0]);
            assert_eq!(
                blocks[1].ops[1].kind,
                MachineInstKind::IntBinary {
                    width,
                    op: MachineIntBinaryOp::Sub,
                    dst: MachineReg(5),
                    lhs: MachineValue::Reg(MachineReg(7)),
                    rhs: MachineValue::Reg(MachineReg(4)),
                }
            );
            let mut clobbered = before;
            clobbered[1].ops[0] = copy(4, MachineValue::Imm64(0));
            let before = clobbered.clone();
            run(&mut clobbered, 8);
            assert_eq!(clobbered, before, "the second input must also remain live");
        }
    }

    #[test]
    fn rejects_clobbers_opaque_operations_partial_slots_and_ambiguous_edges() {
        for case in 0..11 {
            let mut blocks = fixture(8, false);
            match case {
                0 => blocks[0].ops.push(copy(4, MachineValue::Imm64(0))),
                1 => blocks[1].ops[0] = copy(4, MachineValue::Imm64(0)),
                2 => blocks[1].ops[0] = copy(MACHINE_FP_REG.0, MachineValue::Imm64(0)),
                3 | 4 => {
                    let call = MachineInst {
                        kind: MachineInstKind::CallRuntime(MachineCallRuntime {
                            metadata: MachineConstId(0),
                        }),
                    };
                    if case == 3 {
                        blocks[0].ops.insert(2, call);
                    } else {
                        blocks[1].ops[0] = call;
                    }
                }
                5 => {
                    let mut store = blocks[0].ops[2].clone();
                    let MachineInstKind::Store { width, .. } = &mut store.kind else {
                        unreachable!()
                    };
                    *width = MachineMemWidth::U32;
                    blocks[0].ops.push(store);
                }
                6 => {
                    let MachineInstKind::IntBinary { op, .. } = &mut blocks[0].ops[1].kind else {
                        unreachable!()
                    };
                    *op = MachineIntBinaryOp::DivU;
                }
                7 => {
                    blocks[0].ops.remove(0);
                }
                8 => {
                    let MachineTerminator::Jump(edge) = &blocks[0].terminator else {
                        unreachable!()
                    };
                    let then_edge = edge.clone();
                    let mut else_edge = edge.clone();
                    else_edge.args[0] = MachineValue::Imm64(77);
                    blocks[0].terminator = MachineTerminator::Branch {
                        cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(4))),
                        then_edge,
                        else_edge,
                    };
                }
                9 => {
                    let MachineInstKind::Load { extension, .. } = &mut blocks[1].ops[1].kind else {
                        unreachable!()
                    };
                    *extension = MachineLoadExtension::ZeroExtend;
                }
                10 => {
                    let mut other = blocks[0].clone();
                    other.id = MachineBlockId(2);
                    blocks.push(other);
                }
                _ => unreachable!(),
            };
            let before = blocks.clone();
            run(&mut blocks, 8);
            assert_eq!(blocks, before, "invalid proof {case}");
        }
    }
}
