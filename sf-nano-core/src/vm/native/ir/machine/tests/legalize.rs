use alloc::format;
use alloc::vec;

use crate::vm::native::ir::machine::{
    MachineBlock, MachineBlockId, MachineBlockParam, MachineFunction, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineIntWidth, MachineModule, MachineProgram, MachineReg,
    MachineStorageType, MachineTerminator, MachineValue,
};

#[test]
fn legalize_scaffold_infers_gp_i64_chain_on_32_bit() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 7,
        reg_count: 7,
        fp_transient_count: 0,
        fp_reg_init_widths: vec![],
        blocks: vec![MachineBlock {
            id: MachineBlockId(0),
            params: vec![MachineBlockParam::gp_i64(MachineReg(4))],
            ops: vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpI64,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(6),
                        lhs: MachineValue::Reg(MachineReg(5)),
                        rhs: MachineValue::Imm64(1),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    let types = program
        .infer_32bit_reg_storage_types()
        .expect("32-bit legalizer prep should infer i64 GP registers");

    assert_eq!(types[4], Some(MachineStorageType::GpI64));
    assert_eq!(types[5], Some(MachineStorageType::GpI64));
    assert_eq!(types[6], Some(MachineStorageType::GpI64));
}

#[test]
fn legalize_scaffold_keeps_word_sized_gp_ops_as_gp_word() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 6,
        reg_count: 6,
        fp_transient_count: 0,
        fp_reg_init_widths: vec![],
        blocks: vec![MachineBlock {
            id: MachineBlockId(0),
            params: vec![MachineBlockParam::gp_word(MachineReg(4))],
            ops: vec![MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: MachineIntWidth::I32,
                    op: MachineIntBinaryOp::Add,
                    dst: MachineReg(5),
                    lhs: MachineValue::Reg(MachineReg(4)),
                    rhs: MachineValue::Imm64(4),
                },
            }],
            terminator: MachineTerminator::Return,
        }],
    };

    let types = program
        .infer_32bit_reg_storage_types()
        .expect("word-sized GP ops should remain GP-word");

    assert_eq!(types[4], Some(MachineStorageType::GpWord));
    assert_eq!(types[5], Some(MachineStorageType::GpWord));
}

#[test]
fn legalize_scaffold_rejects_conflicting_gp_storage() {
    let mut module = MachineModule {
        functions: vec![MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 6,
                reg_count: 6,
                fp_transient_count: 0,
                fp_reg_init_widths: vec![],
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![MachineBlockParam::gp_word(MachineReg(4))],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::IntBinary {
                            width: MachineIntWidth::I64,
                            op: MachineIntBinaryOp::Add,
                            dst: MachineReg(4),
                            lhs: MachineValue::Reg(MachineReg(4)),
                            rhs: MachineValue::Imm64(1),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    let err = module
        .legalize(4)
        .expect_err("conflicting GP storage should fail legalizer prep");
    let message = format!("{err}");
    assert!(
        message.contains("conflicting 32-bit storage types"),
        "unexpected error: {message}"
    );
}

#[test]
fn legalize_scaffold_infers_cross_bank_f64_moves() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 7,
        reg_count: 8,
        fp_transient_count: 1,
        fp_reg_init_widths: vec![None],
        blocks: vec![MachineBlock {
            id: MachineBlockId(0),
            params: vec![],
            ops: vec![MachineInst {
                kind: MachineInstKind::Move {
                    ty: MachineStorageType::Fp64,
                    dst: MachineReg(7),
                    src: MachineValue::Reg(MachineReg(4)),
                },
            }],
            terminator: MachineTerminator::Return,
        }],
    };

    let types = program
        .infer_32bit_reg_storage_types()
        .expect("cross-bank f64 moves should infer matching GP and FP storage");

    assert_eq!(types[4], Some(MachineStorageType::GpI64));
    assert_eq!(types[7], Some(MachineStorageType::Fp64));
}
