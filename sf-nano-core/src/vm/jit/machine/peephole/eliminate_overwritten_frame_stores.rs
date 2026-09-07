//! Remove a frame store overwritten before any operation can observe it.
//!
//! Only exact native-word stores in one block participate. Every memory
//! access other than the next candidate store, possible trap, call and opaque
//! instruction ends the proof. A final store can also die when a register-only
//! scalar return discards the frame without any intervening observer.

use crate::collections;
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineInstKind, MachineIntBinaryOp, MachineMemWidth,
    MachineResultSrc, MachineReturnValue, MachineStorageType, MachineTerminator, MACHINE_FP_REG,
};

use super::helpers::inst_defines;

pub(super) fn has_gp_register_return(block: &MachineBlock) -> bool {
    matches!(
        block.terminator,
        MachineTerminator::ReturnScalar {
            value: MachineReturnValue::ScalarGp {
                src: MachineResultSrc::Reg(_),
                ..
            }
        }
    )
}

pub(super) fn eliminate_overwritten_frame_stores(block: &mut MachineBlock, gp_bytes: u8) {
    let mut pending: Option<(MachineAddr, MachineMemWidth, usize)> = None;
    let mut removed = collections::Vec::new();
    for (index, inst) in block.ops.iter().enumerate() {
        if inst_defines(&inst.kind, MACHINE_FP_REG) {
            pending = None;
            continue;
        }
        match inst.kind {
            MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr,
                width,
                ..
            } if addr.base == MACHINE_FP_REG
                && addr.offset >= 0
                && addr.offset % i32::from(gp_bytes) == 0
                && width.bytes() == u32::from(gp_bytes) =>
            {
                if let Some((old_addr, old_width, old_index)) = pending {
                    if old_addr == addr && old_width == width {
                        removed.push(old_index);
                    }
                }
                pending = Some((addr, width, index));
            }
            MachineInstKind::Move { .. }
            | MachineInstKind::IntUnary { .. }
            | MachineInstKind::IntCompare { .. }
            | MachineInstKind::TestBits { .. }
            | MachineInstKind::BitfieldExtractU { .. }
            | MachineInstKind::Select {
                ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
                ..
            } => {}
            MachineInstKind::IntBinary { op, .. }
            | MachineInstKind::IntBinaryShifted { op, .. }
                if matches!(
                    op,
                    MachineIntBinaryOp::Add
                        | MachineIntBinaryOp::Sub
                        | MachineIntBinaryOp::Mul
                        | MachineIntBinaryOp::And
                        | MachineIntBinaryOp::Or
                        | MachineIntBinaryOp::Xor
                        | MachineIntBinaryOp::Shl
                        | MachineIntBinaryOp::ShrS
                        | MachineIntBinaryOp::ShrU
                        | MachineIntBinaryOp::Rotl
                        | MachineIntBinaryOp::Rotr
                ) => {}
            _ => pending = None,
        }
    }
    // Scalar GP results travel in return lanes. The caller does not read
    // this callee's local frame after a successful register-only return.
    // Keep frame-fallback returns and every possibly observing suffix intact.
    if has_gp_register_return(block) {
        if let Some((_, _, index)) = pending {
            removed.push(index);
        }
    }
    if removed.is_empty() {
        return;
    }
    // Indices are strictly increasing: a candidate replaces its predecessor.
    let mut removed = removed.into_iter().peekable();
    let mut index = 0;
    block.ops.retain(|_| {
        let keep = removed.peek() != Some(&index);
        if !keep {
            removed.next();
        }
        index += 1;
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineCallRuntime, MachineConstId, MachineInst,
        MachineIntWidth, MachineLoadExtension, MachineReg, MachineRegOwner, MachineShiftOp,
        MachineTrapKind, MachineValue, MACHINE_MEM0_BASE_REG,
    };

    fn stores(bytes: u8) -> MachineBlock {
        let store = |value| MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: MachineAddr {
                    base: MACHINE_FP_REG,
                    offset: 24,
                },
                width: if bytes == 8 {
                    MachineMemWidth::U64
                } else {
                    MachineMemWidth::U32
                },
                src: MachineValue::Imm64(value),
            },
        };
        MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                store(11),
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::Xor,
                        dst: MachineReg(4),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(7),
                    }
                },
                store(22),
                store(33)
            ],
            terminator: MachineTerminator::Return,
        }
    }

    #[test]
    fn drops_overwritten_native_words_and_keeps_the_final_published_value() {
        for bytes in [4, 8] {
            for shifted in [false, true] {
                let mut block = stores(bytes);
                if shifted {
                    block.ops[1].kind = MachineInstKind::IntBinaryShifted {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::Xor,
                        dst: MachineReg(4),
                        lhs: MachineReg(4),
                        rhs: MachineReg(5),
                        shift: MachineShiftOp::Lsr,
                        amount: 2,
                    };
                }
                let arithmetic = block.ops[1].clone();
                let final_store = block.ops[3].clone();
                eliminate_overwritten_frame_stores(&mut block, bytes);
                assert_eq!(block.ops, collections::vec![arithmetic, final_store]);
            }
        }
    }

    #[test]
    fn retains_stores_across_reads_traps_calls_and_frame_base_changes() {
        for barrier in [
            MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(5),
                addr: MachineAddr {
                    base: MACHINE_FP_REG,
                    offset: 24,
                },
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
            MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::DivS,
                dst: MachineReg(4),
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Reg(MachineReg(5)),
            },
            MachineInstKind::TrapIf {
                cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(5))),
                kind: MachineTrapKind::Unreachable,
            },
            MachineInstKind::CallRuntime(MachineCallRuntime {
                metadata: MachineConstId(0),
            }),
            MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MACHINE_FP_REG,
                src: MachineValue::Reg(MachineReg(5)),
            },
        ] {
            let mut block = stores(8);
            block.ops.truncate(3);
            block.ops[1].kind = barrier;
            let expected = block.clone();
            eliminate_overwritten_frame_stores(&mut block, 8);
            assert_eq!(block, expected);
        }
    }

    #[test]
    fn requires_exact_aligned_native_frame_stores() {
        for case in 0..5 {
            let mut block = stores(8);
            block.ops.truncate(3);
            if let MachineInstKind::Store { addr, width, .. } = &mut block.ops[2].kind {
                match case {
                    0 => addr.offset += 8,
                    1 => addr.offset += 1,
                    2 => *width = MachineMemWidth::U32,
                    3 => addr.base = MACHINE_MEM0_BASE_REG,
                    4 => addr.offset = -8,
                    _ => unreachable!(),
                }
            }
            let expected = block.clone();
            eliminate_overwritten_frame_stores(&mut block, 8);
            assert_eq!(block, expected);
        }
        let mut guest = stores(8);
        for inst in &mut guest.ops {
            if let MachineInstKind::Store { addr, .. } = &mut inst.kind {
                addr.base = MACHINE_MEM0_BASE_REG;
            }
        }
        let expected = guest.clone();
        eliminate_overwritten_frame_stores(&mut guest, 8);
        assert_eq!(guest, expected);
    }

    #[test]
    fn drops_the_unobserved_final_store_only_for_gp_register_returns() {
        use crate::vm::jit::middle::frame::FrameSlot;

        for bytes in [4, 8] {
            for src in [
                MachineResultSrc::Reg(MachineReg(4)),
                MachineResultSrc::FrameSlot(FrameSlot(3)),
                MachineResultSrc::FrameSlotOffset {
                    slot: FrameSlot(3),
                    byte_offset: 0,
                },
            ] {
                let mut block = stores(bytes);
                block.ops.truncate(2);
                block.terminator = MachineTerminator::ReturnScalar {
                    value: MachineReturnValue::ScalarGp {
                        src,
                        ty: MachineStorageType::GpWord,
                    },
                };
                let expected = if matches!(src, MachineResultSrc::Reg(_)) {
                    collections::vec![block.ops[1].clone()]
                } else {
                    block.ops.clone()
                };
                eliminate_overwritten_frame_stores(&mut block, bytes);
                assert_eq!(block.ops, expected);
            }
        }
    }

    #[test]
    fn retains_final_frame_stores_before_observers_even_with_a_register_return() {
        for barrier in [
            MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(4),
                addr: MachineAddr {
                    base: MACHINE_FP_REG,
                    offset: 24,
                },
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
            MachineInstKind::CallRuntime(MachineCallRuntime {
                metadata: MachineConstId(0),
            }),
            MachineInstKind::TrapIf {
                cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(5))),
                kind: MachineTrapKind::Unreachable,
            },
            MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::DivU,
                dst: MachineReg(4),
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Reg(MachineReg(5)),
            },
            MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MACHINE_FP_REG,
                src: MachineValue::Reg(MachineReg(5)),
            },
        ] {
            let mut block = stores(8);
            block.ops.truncate(2);
            block.ops[1].kind = barrier;
            block.terminator = MachineTerminator::ReturnScalar {
                value: MachineReturnValue::ScalarGp {
                    src: MachineResultSrc::Reg(MachineReg(4)),
                    ty: MachineStorageType::GpWord,
                },
            };
            let expected = block.clone();
            eliminate_overwritten_frame_stores(&mut block, 8);
            assert_eq!(block, expected);
        }
    }

    #[test]
    fn final_store_proof_does_not_remove_guest_partial_or_misaligned_stores() {
        for case in 0..4 {
            let mut block = stores(8);
            block.ops.truncate(2);
            if let MachineInstKind::Store { addr, width, .. } = &mut block.ops[0].kind {
                match case {
                    0 => addr.base = MACHINE_MEM0_BASE_REG,
                    1 => addr.offset = -8,
                    2 => addr.offset = 25,
                    3 => *width = MachineMemWidth::U32,
                    _ => unreachable!(),
                }
            }
            block.terminator = MachineTerminator::ReturnScalar {
                value: MachineReturnValue::ScalarGp {
                    src: MachineResultSrc::Reg(MachineReg(4)),
                    ty: MachineStorageType::GpWord,
                },
            };
            let expected = block.clone();
            eliminate_overwritten_frame_stores(&mut block, 8);
            assert_eq!(block, expected);
        }
    }
}
