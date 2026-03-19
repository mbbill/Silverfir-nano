use alloc::format;
use alloc::vec;

use crate::vm::native::ir::machine::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineCompareKind,
    MachineConvertOp, MachineFloatUnaryOp, MachineFloatWidth, MachineFunction, MachineInst,
    MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension,
    MachineMemWidth, MachineModule, MachineProgram, MachineReg, MachineSign, MachineStorageFact,
    MachineStorageType, MachineTerminator, MachineValue,
};

#[test]
fn legalize_scaffold_tracks_gp_i64_flow_on_32_bit() {
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

    let flow = program
        .analyze_32bit_storage_flow()
        .expect("32-bit legalizer prep should track i64 GP flow");
    let exit = flow.block_exit[0]
        .as_ref()
        .expect("block 0 should have an exit state");

    assert_eq!(
        exit[4],
        MachineStorageFact::Known(MachineStorageType::GpI64)
    );
    assert_eq!(
        exit[5],
        MachineStorageFact::Known(MachineStorageType::GpI64)
    );
    assert_eq!(
        exit[6],
        MachineStorageFact::Known(MachineStorageType::GpI64)
    );
}

#[test]
fn legalize_scaffold_keeps_word_sized_gp_flow_as_gp_word() {
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

    let flow = program
        .analyze_32bit_storage_flow()
        .expect("word-sized GP ops should remain GP-word");
    let exit = flow.block_exit[0]
        .as_ref()
        .expect("block 0 should have an exit state");

    assert_eq!(
        exit[4],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
    assert_eq!(
        exit[5],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
}

#[test]
fn legalize_scaffold_rejects_use_that_requires_i64_from_gp_word() {
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
        .expect_err("mismatched GP use should fail legalizer prep");
    let message = format!("{err}");
    assert!(
        message.contains("requires GpI64"),
        "unexpected error: {message}"
    );
}

#[test]
fn legalize_scaffold_tracks_cross_bank_f64_moves() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 7,
        reg_count: 8,
        fp_transient_count: 1,
        fp_reg_init_widths: vec![None],
        blocks: vec![MachineBlock {
            id: MachineBlockId(0),
            params: vec![MachineBlockParam::gp_i64(MachineReg(4))],
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

    let flow = program
        .analyze_32bit_storage_flow()
        .expect("cross-bank f64 moves should infer matching GP and FP storage");
    let exit = flow.block_exit[0]
        .as_ref()
        .expect("block 0 should have an exit state");

    assert_eq!(
        exit[4],
        MachineStorageFact::Known(MachineStorageType::GpI64)
    );
    assert_eq!(exit[7], MachineStorageFact::Known(MachineStorageType::Fp64));
}

#[test]
fn legalize_scaffold_allows_gp_transient_reuse_across_storage_widths() {
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
                    kind: MachineInstKind::Convert {
                        op: MachineConvertOp::I32WrapI64,
                        dst: MachineReg(6),
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(6)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    let flow = program
        .analyze_32bit_storage_flow()
        .expect("transient register reuse across widths should be allowed");
    let exit = flow.block_exit[0]
        .as_ref()
        .expect("block 0 should have an exit state");

    assert_eq!(
        exit[5],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
    assert_eq!(
        exit[6],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
}

#[test]
fn legalize_scaffold_treats_i64_eqz_result_as_gp_word() {
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
                    kind: MachineInstKind::IntUnary {
                        width: MachineIntWidth::I64,
                        op: MachineIntUnaryOp::Eqz,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntUnary {
                        width: MachineIntWidth::I32,
                        op: MachineIntUnaryOp::Eqz,
                        dst: MachineReg(6),
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    let flow = program
        .analyze_32bit_storage_flow()
        .expect("i64.eqz should produce a gp-word result on 32-bit");
    let exit = flow.block_exit[0]
        .as_ref()
        .expect("block 0 should have an exit state");

    assert_eq!(
        exit[5],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
    assert_eq!(
        exit[6],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
}

#[test]
fn gp_expansion_finalization_shifts_fp_bank_and_compacts_new_gp_regs() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 4,
        reg_count: 7,
        fp_transient_count: 1,
        fp_reg_init_widths: vec![Some(MachineFloatWidth::F32), None],
        blocks: vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: vec![],
                ops: vec![MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        src: MachineValue::Imm64(1),
                    },
                }],
                terminator: MachineTerminator::Jump(crate::vm::native::ir::machine::MachineEdge {
                    target: MachineBlockId(1),
                    args: vec![
                        MachineValue::Reg(MachineReg(4)),
                        MachineValue::Reg(MachineReg(6)),
                    ],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: vec![
                    MachineBlockParam::fp(MachineReg(4), MachineFloatWidth::F32),
                    MachineBlockParam::gp_word(MachineReg(6)),
                ],
                ops: vec![
                    MachineInst {
                        kind: MachineInstKind::FloatUnary {
                            width: MachineFloatWidth::F32,
                            op: MachineFloatUnaryOp::Neg,
                            dst: MachineReg(5),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::Store {
                            ty: MachineStorageType::GpWord,
                            addr: MachineAddr {
                                base: MachineReg(6),
                                offset: 0,
                            },
                            width: MachineMemWidth::U32,
                            src: MachineValue::Reg(MachineReg(6)),
                        },
                    },
                ],
                terminator: MachineTerminator::Branch {
                    cond: crate::vm::native::ir::machine::MachineBranchCond::FloatCompare {
                        width: MachineFloatWidth::F32,
                        kind: MachineCompareKind::Eq,
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Reg(MachineReg(5)),
                    },
                    then_edge: crate::vm::native::ir::machine::MachineEdge {
                        target: MachineBlockId(1),
                        args: vec![
                            MachineValue::Reg(MachineReg(5)),
                            MachineValue::Reg(MachineReg(6)),
                        ],
                    },
                    else_edge: crate::vm::native::ir::machine::MachineEdge {
                        target: MachineBlockId(1),
                        args: vec![
                            MachineValue::Reg(MachineReg(4)),
                            MachineValue::Reg(MachineReg(6)),
                        ],
                    },
                },
            },
        ],
    };

    program
        .finalize_32bit_gp_expansion(6, 1)
        .expect("gp expansion finalization should succeed");

    assert_eq!(program.first_fp_reg, 5);
    assert_eq!(program.reg_count, 7);

    let block0 = &program.blocks[0];
    let MachineInstKind::Move { dst, .. } = &block0.ops[0].kind else {
        panic!("expected block0 move");
    };
    assert_eq!(*dst, MachineReg(4));
    let MachineTerminator::Jump(edge) = &block0.terminator else {
        panic!("expected block0 jump");
    };
    assert_eq!(
        edge.args,
        vec![
            MachineValue::Reg(MachineReg(5)),
            MachineValue::Reg(MachineReg(4))
        ]
    );

    let block1 = &program.blocks[1];
    assert_eq!(
        block1.params,
        vec![
            MachineBlockParam::fp(MachineReg(5), MachineFloatWidth::F32),
            MachineBlockParam::gp_word(MachineReg(4)),
        ]
    );
    let MachineInstKind::FloatUnary { dst, src, .. } = &block1.ops[0].kind else {
        panic!("expected block1 float unary");
    };
    assert_eq!(*dst, MachineReg(6));
    assert_eq!(*src, MachineValue::Reg(MachineReg(5)));
    let MachineInstKind::Store { addr, src, .. } = &block1.ops[1].kind else {
        panic!("expected block1 store");
    };
    assert_eq!(addr.base, MachineReg(4));
    assert_eq!(*src, MachineValue::Reg(MachineReg(4)));

    let MachineTerminator::Branch {
        cond,
        then_edge,
        else_edge,
    } = &block1.terminator
    else {
        panic!("expected block1 branch");
    };
    let crate::vm::native::ir::machine::MachineBranchCond::FloatCompare { lhs, rhs, .. } = cond
    else {
        panic!("expected float compare branch");
    };
    assert_eq!(*lhs, MachineValue::Reg(MachineReg(5)));
    assert_eq!(*rhs, MachineValue::Reg(MachineReg(6)));
    assert_eq!(
        then_edge.args,
        vec![
            MachineValue::Reg(MachineReg(6)),
            MachineValue::Reg(MachineReg(4))
        ]
    );
    assert_eq!(
        else_edge.args,
        vec![
            MachineValue::Reg(MachineReg(5)),
            MachineValue::Reg(MachineReg(4))
        ]
    );

    program
        .validate()
        .expect("finalized program should validate");
}

#[test]
fn legalize_scaffold_allows_gpword_moves_from_gpi64_sources() {
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
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: MachineIntWidth::I32,
                        kind: crate::vm::native::ir::machine::MachineCompareKind::Eq,
                        sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                        dst: MachineReg(6),
                        lhs: MachineValue::Reg(MachineReg(5)),
                        rhs: MachineValue::Imm64(0),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    let flow = program
        .analyze_32bit_storage_flow()
        .expect("gpword moves should be allowed to consume the low word of gpi64 sources");
    let exit = flow.block_exit[0]
        .as_ref()
        .expect("block 0 should have an exit state");

    assert_eq!(
        exit[5],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
    assert_eq!(
        exit[6],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
}

#[test]
fn legalize_scaffold_allows_gpi64_moves_from_gpword_sources() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 7,
        reg_count: 7,
        fp_transient_count: 0,
        fp_reg_init_widths: vec![],
        blocks: vec![MachineBlock {
            id: MachineBlockId(0),
            params: vec![MachineBlockParam::gp_word(MachineReg(4))],
            ops: vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpI64,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpI64,
                        addr: crate::vm::native::ir::machine::MachineAddr {
                            base: crate::vm::native::ir::machine::MACHINE_FP_REG,
                            offset: 0,
                        },
                        width: crate::vm::native::ir::machine::MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    let flow = program
        .analyze_32bit_storage_flow()
        .expect("gpi64 moves should allow zero-extending gpword sources");
    let exit = flow.block_exit[0]
        .as_ref()
        .expect("block 0 should have an exit state");

    assert_eq!(
        exit[4],
        MachineStorageFact::Known(MachineStorageType::GpWord)
    );
    assert_eq!(
        exit[5],
        MachineStorageFact::Known(MachineStorageType::GpI64)
    );
}

#[test]
fn legalize_rewrites_i64_add_into_word_pairs() {
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
                    params: vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::IntBinary {
                            width: MachineIntWidth::I64,
                            op: MachineIntBinaryOp::Add,
                            dst: MachineReg(5),
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

    module
        .legalize(4)
        .expect("32-bit legalizer should rewrite i64 add");

    let program = &module.functions[0].program;
    assert_eq!(
        program.blocks[0].params,
        vec![
            MachineBlockParam::gp_word(MachineReg(4)),
            MachineBlockParam::gp_word(MachineReg(6)),
        ]
    );
    let ops = &program.blocks[0].ops;
    assert_eq!(ops.len(), 6);
    assert_eq!(
        ops[0].kind,
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            dst: MachineReg(8),
            lhs: MachineValue::Reg(MachineReg(4)),
            rhs: MachineValue::Imm64(1),
        }
    );
    assert_eq!(
        ops[1].kind,
        MachineInstKind::IntCompare {
            width: MachineIntWidth::I32,
            kind: MachineCompareKind::Lt,
            sign: MachineSign::Unsigned,
            dst: MachineReg(9),
            lhs: MachineValue::Reg(MachineReg(8)),
            rhs: MachineValue::Reg(MachineReg(4)),
        }
    );
    assert_eq!(
        ops[2].kind,
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            dst: MachineReg(10),
            lhs: MachineValue::Reg(MachineReg(6)),
            rhs: MachineValue::Imm64(0),
        }
    );
    assert_eq!(
        ops[3].kind,
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            dst: MachineReg(10),
            lhs: MachineValue::Reg(MachineReg(10)),
            rhs: MachineValue::Reg(MachineReg(9)),
        }
    );
    assert_eq!(
        ops[4].kind,
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord,
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(8)),
        }
    );
    assert_eq!(
        ops[5].kind,
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord,
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(10)),
        }
    );
}

#[test]
fn legalize_zero_extends_gpword_sources_even_if_reg_has_a_shadow_pair() {
    let mut module = MachineModule {
        functions: vec![MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: MachineProgram {
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
                            kind: MachineInstKind::Convert {
                                op: MachineConvertOp::I32WrapI64,
                                dst: MachineReg(5),
                                src: MachineValue::Reg(MachineReg(4)),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpI64,
                                dst: MachineReg(5),
                                src: MachineValue::Reg(MachineReg(5)),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    module
        .legalize(4)
        .expect("32-bit legalizer should use flow-sensitive move-like source types");

    let ops = &module.functions[0].program.blocks[0].ops;
    assert_eq!(
        ops[0].kind,
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord,
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(4)),
        }
    );
    assert_eq!(
        ops[1].kind,
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord,
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(5)),
        }
    );
    assert_eq!(
        ops[2].kind,
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord,
            dst: MachineReg(8),
            src: MachineValue::Imm64(0),
        }
    );
}

#[test]
fn legalize_rewrites_i64_div_into_pair_divrem_op() {
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
                    params: vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::IntBinary {
                            width: MachineIntWidth::I64,
                            op: MachineIntBinaryOp::DivS,
                            dst: MachineReg(5),
                            lhs: MachineValue::Reg(MachineReg(4)),
                            rhs: MachineValue::Imm64(3),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    module
        .legalize(4)
        .expect("32-bit legalizer should rewrite i64 signed division");

    assert_eq!(
        module.functions[0].program.blocks[0].ops[0].kind,
        MachineInstKind::Int64PairDivRem {
            sign: MachineSign::Signed,
            rem: false,
            dst_lo: MachineReg(5),
            dst_hi: MachineReg(7),
            lhs_lo: MachineValue::Reg(MachineReg(4)),
            lhs_hi: MachineValue::Reg(MachineReg(6)),
            rhs_lo: MachineValue::Imm64(3),
            rhs_hi: MachineValue::Imm64(0),
        }
    );
}

#[test]
fn legalize_rewrites_i64_to_f64_convert_into_pair_convert_op() {
    let mut module = MachineModule {
        functions: vec![MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 6,
                reg_count: 7,
                fp_transient_count: 1,
                fp_reg_init_widths: vec![None],
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::Convert {
                            op: MachineConvertOp::F64ConvertI64S,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    module
        .legalize(4)
        .expect("32-bit legalizer should rewrite i64-to-f64 conversion");

    assert_eq!(
        module.functions[0].program.blocks[0].ops[0].kind,
        MachineInstKind::ConvertI64PairToFloat {
            width: MachineFloatWidth::F64,
            sign: MachineSign::Signed,
            dst: MachineReg(12),
            src_lo: MachineValue::Reg(MachineReg(4)),
            src_hi: MachineValue::Reg(MachineReg(6)),
        }
    );
}

#[test]
fn legalize_rewrites_i64_shift_into_pair_shift_op_with_low_word_count() {
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
                    params: vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::IntBinary {
                            width: MachineIntWidth::I64,
                            op: MachineIntBinaryOp::ShrU,
                            dst: MachineReg(5),
                            lhs: MachineValue::Reg(MachineReg(4)),
                            rhs: MachineValue::Imm64((1_u64 << 32) | 65),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    module
        .legalize(4)
        .expect("32-bit legalizer should rewrite i64 shifts");

    assert_eq!(
        module.functions[0].program.blocks[0].ops[0].kind,
        MachineInstKind::Int64PairShift {
            op: MachineIntBinaryOp::ShrU,
            dst_lo: MachineReg(5),
            dst_hi: MachineReg(7),
            lhs_lo: MachineValue::Reg(MachineReg(4)),
            lhs_hi: MachineValue::Reg(MachineReg(6)),
            rhs: MachineValue::Imm64(65),
        }
    );
}

#[test]
fn legalize_rewrites_i64_popcnt_into_pair_unary_op() {
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
                    params: vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: MachineIntWidth::I64,
                            op: MachineIntUnaryOp::Popcnt,
                            dst: MachineReg(5),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    module
        .legalize(4)
        .expect("32-bit legalizer should rewrite i64 popcnt");

    assert_eq!(
        module.functions[0].program.blocks[0].ops[0].kind,
        MachineInstKind::Int64PairUnary {
            op: MachineIntUnaryOp::Popcnt,
            dst_lo: MachineReg(5),
            dst_hi: MachineReg(7),
            src_lo: MachineValue::Reg(MachineReg(4)),
            src_hi: MachineValue::Reg(MachineReg(6)),
        }
    );
}

#[test]
fn legalize_rewrites_i64_reinterpret_f64_into_pair_reinterpret_op() {
    let mut module = MachineModule {
        functions: vec![MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 6,
                reg_count: 7,
                fp_transient_count: 1,
                fp_reg_init_widths: vec![Some(MachineFloatWidth::F64)],
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![MachineBlockParam::fp(MachineReg(6), MachineFloatWidth::F64)],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::Convert {
                            op: MachineConvertOp::I64ReinterpretF64,
                            dst: MachineReg(4),
                            src: MachineValue::Reg(MachineReg(6)),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    module
        .legalize(4)
        .expect("32-bit legalizer should rewrite i64 reinterpret f64");

    assert_eq!(
        module.functions[0].program.blocks[0].ops[0].kind,
        MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo: MachineReg(4),
            dst_hi: MachineReg(6),
            src: MachineValue::Reg(MachineReg(12)),
        }
    );
}

#[test]
fn legalize_rewrites_i64_trunc_f32_into_pair_convert_op() {
    let mut module = MachineModule {
        functions: vec![MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 6,
                reg_count: 7,
                fp_transient_count: 1,
                fp_reg_init_widths: vec![Some(MachineFloatWidth::F32)],
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![MachineBlockParam::fp(MachineReg(6), MachineFloatWidth::F32)],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::Convert {
                            op: MachineConvertOp::I64TruncF32S,
                            dst: MachineReg(4),
                            src: MachineValue::Reg(MachineReg(6)),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: vec![],
        externs: vec![],
    };

    module
        .legalize(4)
        .expect("32-bit legalizer should rewrite i64 trunc from f32");

    assert_eq!(
        module.functions[0].program.blocks[0].ops[0].kind,
        MachineInstKind::ConvertFloatToI64Pair {
            op: MachineConvertOp::I64TruncF32S,
            dst_lo: MachineReg(4),
            dst_hi: MachineReg(6),
            src: MachineValue::Reg(MachineReg(12)),
        }
    );
}
