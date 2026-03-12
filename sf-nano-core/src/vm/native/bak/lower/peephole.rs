//! Target-independent native peepholes.
//!
//! This pass should stay small and mechanical:
//! - remove redundant moves/copies
//! - normalize parallel-copy shapes
//! - fuse obvious compare+branch tails
//! - clean up after placement
//!
//! It must not become another semantic optimizer layer.

use crate::vm::native::ir::NativeProgram;

#[inline]
pub(super) fn run_native_peepholes(program: &mut NativeProgram) {
    for block in &mut program.blocks {
        let mut cleaned = alloc::vec::Vec::with_capacity(block.ops.len());
        for inst in block.ops.drain(..) {
            match &inst.kind {
                crate::vm::native::ir::NativeInstKind::Move(mov) if matches!(mov.src, crate::vm::native::abi::NativeValue::Place(src) if src == mov.dst) =>
                {
                    continue;
                }
                crate::vm::native::ir::NativeInstKind::Move(mov) => {
                    if let Some(crate::vm::native::ir::NativeInst {
                        kind: crate::vm::native::ir::NativeInstKind::Move(prev),
                    }) = cleaned.last_mut()
                    {
                        if prev.dst == mov.dst {
                            *prev = *mov;
                            continue;
                        }
                    }
                }
                _ => {}
            }
            cleaned.push(inst);
        }
        block.ops = cleaned;

        match &mut block.terminator {
            crate::vm::native::ir::NativeTerminator::Goto(edge) => prune_edge(edge),
            crate::vm::native::ir::NativeTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                prune_edge(then_edge);
                prune_edge(else_edge);
            }
            crate::vm::native::ir::NativeTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    prune_edge(edge);
                }
            }
            crate::vm::native::ir::NativeTerminator::Return { .. }
            | crate::vm::native::ir::NativeTerminator::TrapUnreachable => {}
        }
    }
}

fn prune_edge(edge: &mut crate::vm::native::ir::NativeEdge) {
    let mut cleaned: alloc::vec::Vec<crate::vm::native::ir::NativeMove> =
        alloc::vec::Vec::with_capacity(edge.copies.len());
    for mov in edge.copies.drain(..) {
        if matches!(mov.src, crate::vm::native::abi::NativeValue::Place(src) if src == mov.dst) {
            continue;
        }
        if let Some(prev) = cleaned.last_mut() {
            if prev.dst == mov.dst {
                *prev = mov;
                continue;
            }
        }
        cleaned.push(mov);
    }
    edge.copies = cleaned;
}

#[cfg(test)]
mod tests {
    use super::run_native_peepholes;
    use crate::vm::native::{
        abi::{BlockAbi, NativeAbi, NativeLocation, NativePlace, NativeValue},
        ir::{
            NativeBlock, NativeBlockId, NativeEdge, NativeInst, NativeInstKind, NativeMove,
            NativeProgram, NativeTerminator,
        },
    };

    #[test]
    fn drops_redundant_move_and_last_write_wins() {
        let mut program = NativeProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![NativeBlock {
                id: NativeBlockId(0),
                entry_abi: BlockAbi::default(),
                ops: alloc::vec![
                    NativeInst {
                        kind: NativeInstKind::Move(NativeMove {
                            dst: NativePlace::Location(NativeLocation::Tmp(0)),
                            src: NativeValue::Place(NativePlace::Location(NativeLocation::Tmp(0))),
                        }),
                    },
                    NativeInst {
                        kind: NativeInstKind::Move(NativeMove {
                            dst: NativePlace::Location(NativeLocation::Tmp(0)),
                            src: NativeValue::Imm64(1),
                        }),
                    },
                    NativeInst {
                        kind: NativeInstKind::Move(NativeMove {
                            dst: NativePlace::Location(NativeLocation::Tmp(0)),
                            src: NativeValue::Imm64(2),
                        }),
                    },
                ],
                terminator: NativeTerminator::Goto(NativeEdge {
                    target: NativeBlockId(0),
                    copies: alloc::vec![
                        NativeMove {
                            dst: NativePlace::Location(NativeLocation::Tos(0)),
                            src: NativeValue::Imm64(3),
                        },
                        NativeMove {
                            dst: NativePlace::Location(NativeLocation::Tos(0)),
                            src: NativeValue::Imm64(4),
                        },
                    ],
                }),
            }],
            abi: NativeAbi {
                ctx_register_count: 1,
                fp_register_count: 1,
                tmp_register_count: 1,
                hot_local_count: 0,
                tos_register_count: 1,
            },
            frame: crate::vm::plan::frame::FrameLayoutPlan::default(),
            spill_slots: 0,
            groups: crate::vm::plan::group::GroupPlan::default(),
            hot_locals: None,
        };

        run_native_peepholes(&mut program);

        assert_eq!(program.blocks[0].ops.len(), 1);
        assert_eq!(
            program.blocks[0].ops[0].kind,
            NativeInstKind::Move(NativeMove {
                dst: NativePlace::Location(NativeLocation::Tmp(0)),
                src: NativeValue::Imm64(2),
            })
        );
        assert_eq!(
            program.blocks[0].terminator,
            NativeTerminator::Goto(NativeEdge {
                target: NativeBlockId(0),
                copies: alloc::vec![NativeMove {
                    dst: NativePlace::Location(NativeLocation::Tos(0)),
                    src: NativeValue::Imm64(4),
                }],
            })
        );
    }
}
