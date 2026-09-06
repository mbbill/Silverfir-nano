//! Prove when a body does not need alignment for a nested native call.
//!
//! Keep an explicit set of inline lowerings. Unknown/new operations, runtime
//! helpers, compiled calls, and explicit trap paths retain the ordinary shim.
//! Guard-page load faults return through the body's error tail and do not
//! issue a native call from the generated body.

use crate::vm::jit::machine::machine_ir::{
    MachineConvertOp, MachineFunction, MachineInstKind, MachineIntBinaryOp, MachineStorageType,
    MachineTerminator,
};

fn is_inline_inst(kind: &MachineInstKind) -> bool {
    match kind {
        MachineInstKind::Move { ty, .. } | MachineInstKind::Select { ty, .. } => {
            *ty != MachineStorageType::V128
        }
        MachineInstKind::Load { .. }
        | MachineInstKind::Store { .. }
        | MachineInstKind::IndexedLoad { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::IntUnary { .. }
        | MachineInstKind::IntCompare { .. }
        | MachineInstKind::BitfieldExtractU { .. }
        | MachineInstKind::TestBits { .. }
        | MachineInstKind::FloatConst { .. } => true,
        MachineInstKind::IntBinary { op, .. } | MachineInstKind::IntBinaryShifted { op, .. } => {
            matches!(
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
            )
        }
        MachineInstKind::Convert { op, .. } => matches!(
            op,
            MachineConvertOp::I32WrapI64
                | MachineConvertOp::I64ExtendI32S
                | MachineConvertOp::I64ExtendI32U
        ),
        _ => false,
    }
}

pub(super) fn is_inline_leaf(function: &MachineFunction) -> bool {
    function.program.blocks.iter().all(|block| {
        matches!(
            block.terminator,
            MachineTerminator::Jump(_)
                | MachineTerminator::Branch { .. }
                | MachineTerminator::JumpTable { .. }
                | MachineTerminator::Return
                | MachineTerminator::ReturnScalar { .. }
        ) && block.ops.iter().all(|inst| is_inline_inst(&inst.kind))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        collections,
        vm::jit::machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBranchCond, MachineCallArgs, MachineCallResults,
            MachineCallTarget, MachineEdge, MachineFuncId, MachineInst, MachineIntWidth,
            MachineProgram, MachineReg, MachineTrapKind, MachineValue,
        },
    };

    fn function(ops: &[MachineInstKind]) -> MachineFunction {
        MachineFunction {
            program: MachineProgram {
                blocks: collections::vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::vec![],
                    ops: ops
                        .iter()
                        .cloned()
                        .map(|kind| MachineInst { kind })
                        .collect(),
                    terminator: MachineTerminator::Return,
                }],
                ..MachineProgram::default()
            },
            ..MachineFunction::default()
        }
    }

    #[test]
    fn integer_operations_are_inline_but_division_keeps_trap_alignment() {
        for op in [
            MachineIntBinaryOp::Add,
            MachineIntBinaryOp::ShrU,
            MachineIntBinaryOp::DivS,
            MachineIntBinaryOp::RemU,
        ] {
            let f = function(&[MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op,
                dst: MachineReg(4),
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Reg(MachineReg(5)),
            }]);
            assert_eq!(
                is_inline_leaf(&f),
                matches!(op, MachineIntBinaryOp::Add | MachineIntBinaryOp::ShrU)
            );
        }
    }

    #[test]
    fn explicit_traps_and_helper_conversions_keep_alignment() {
        let mut f = function(&[]);
        assert!(is_inline_leaf(&f));
        f.program.blocks[0].terminator = MachineTerminator::Trap {
            kind: MachineTrapKind::Unreachable,
        };
        assert!(!is_inline_leaf(&f));
        for op in [
            MachineConvertOp::I64ExtendI32U,
            MachineConvertOp::I32TruncF64S,
            MachineConvertOp::I64TruncSatF32U,
        ] {
            let f = function(&[MachineInstKind::Convert {
                op,
                dst: MachineReg(4),
                src: MachineValue::Reg(MachineReg(5)),
            }]);
            assert_eq!(is_inline_leaf(&f), op == MachineConvertOp::I64ExtendI32U);
        }
        let f = function(&[MachineInstKind::TrapIf {
            kind: MachineTrapKind::MemoryOutOfBounds,
            cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(4))),
        }]);
        assert!(!is_inline_leaf(&f));
    }

    #[test]
    fn compiled_calls_keep_the_alignment_shim() {
        let mut f = function(&[]);
        f.program.blocks[0].terminator = MachineTerminator::Call {
            target: MachineCallTarget::Direct(MachineFuncId(0)),
            frame_delta: 32,
            args: MachineCallArgs::default(),
            results: MachineCallResults::None,
            success: MachineEdge::default(),
        };
        assert!(!is_inline_leaf(&f));
        f.program.blocks[0].terminator = MachineTerminator::TailCall {
            target: MachineCallTarget::Direct(MachineFuncId(0)),
            args: MachineCallArgs::default(),
        };
        assert!(!is_inline_leaf(&f));
    }
}
