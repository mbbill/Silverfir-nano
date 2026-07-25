use crate::collections;
use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
    MachineCallRuntime, MachineCompareKind, MachineConstId, MachineConvertOp, MachineEdge,
    MachineFloatBinaryOp, MachineFloatWidth, MachineIndexExtend, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
    MachineProgram, MachineReg, MachineRegOwner, MachineShiftOp, MachineSign, MachineStorageType,
    MachineTerminator, MachineValue, MACHINE_CTX_REG,
};
use crate::vm::jit::machine::peephole::optimize;

/// Build a BackendConfig matching the historical machine-test register layout.
///
/// `first_dynamic_gp` — old first GP dynamic register ID.
/// `gp_reg_width` — GP register width in bytes (4 or 8).
/// `first_fp_reg` — old first FP register ID.
/// `reg_count` — old total register count.
/// `_fp_dynamic_count` — old FP dynamic count.
fn test_config(
    first_dynamic_gp: u16,
    gp_reg_width: u8,
    first_fp_reg: u16,
    reg_count: u16,
    _fp_dynamic_count: u16,
) -> BackendConfig {
    let _ = first_dynamic_gp;
    let gp_dynamic = (first_fp_reg - BackendConfig::FIXED) as u8;
    let fp_dynamic = reg_count.saturating_sub(first_fp_reg) as u8;
    BackendConfig::new(
        gp_reg_width,
        gp_dynamic,
        fp_dynamic,
        if gp_reg_width == 4 { 8 } else { 3 },
    )
}

fn clz_use(src: MachineReg) -> MachineInst {
    MachineInst {
        kind: MachineInstKind::IntUnary {
            width: MachineIntWidth::I32,
            op: MachineIntUnaryOp::Clz,
            dst: MachineReg(6),
            src: MachineValue::Reg(src),
        },
    }
}

#[test]
fn copy_propagates_linear_value_moves_into_ops_and_edges() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(8))],
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(7),
                            src: MachineValue::Reg(MachineReg(8)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: MachineIntWidth::I32,
                            op: MachineIntUnaryOp::Clz,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::Reg(MachineReg(7))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(7))],
                ops: collections::vec![clz_use(MachineReg(7))],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(
        block.ops.len(),
        1,
        "copy propagation should remove the redundant move and leave only the rewritten unary op; ops={:?}",
        block.ops
    );
    let unary_uses_reg8 = matches!(
        block
            .ops
            .iter()
            .find(|inst| matches!(inst.kind, MachineInstKind::IntUnary { .. }))
            .map(|inst| &inst.kind),
        Some(MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(8)),
            ..
        })
    );
    assert!(
        unary_uses_reg8,
        "optimized block did not rewrite the unary source as expected: ops={:?}, term={:?}",
        block.ops, block.terminator
    );
    assert!(
        block.ops.iter().all(|inst| {
            !matches!(
                inst.kind,
                MachineInstKind::Move {
                    dst: MachineReg(7),
                    src: MachineValue::Reg(MachineReg(8)),
                    ..
                }
            )
        }),
        "copy propagation should remove the redundant linear-value move"
    );
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![MachineValue::Reg(MachineReg(8))]
    );
}

#[test]
fn copy_propagate_drops_self_move_and_advances() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::vec![MachineBlockParam::gp_word(MachineReg(7))],
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntUnary {
                        width: MachineIntWidth::I32,
                        op: MachineIntUnaryOp::Clz,
                        dst: MachineReg(8),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 1);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::IntUnary {
            dst: MachineReg(8),
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn does_not_copy_propagate_move_from_cached_cell_block_param() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                // This uses a dynamic GP register that historically would have
                // been treated as a plain "transient" number. The explicit
                // owner says otherwise, and the peephole must obey the owner.
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(7))
                    .with_owner(MachineRegOwner::CachedCell,)],
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(8),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: MachineIntWidth::I32,
                            op: MachineIntUnaryOp::Clz,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(8)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::Reg(MachineReg(8))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(8))],
                ops: collections::vec![clz_use(MachineReg(8))],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 12, 12, 0));

    let block = &program.blocks[0];
    assert_eq!(
        block.ops.len(),
        2,
        "cached-local block params are not linear values, so the move must stay explicit; ops={:?}",
        block.ops
    );
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(8),
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(8)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![MachineValue::Reg(MachineReg(8))]
    );
}

#[test]
fn copy_propagates_linear_value_load_defs_even_in_high_dynamic_regs() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(11),
                            addr: MachineAddr {
                                base: MachineReg(1),
                                offset: 16,
                            },
                            width: MachineMemWidth::U64,
                            extension: MachineLoadExtension::ZeroExtend,
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(8),
                            src: MachineValue::Reg(MachineReg(11)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: MachineIntWidth::I32,
                            op: MachineIntUnaryOp::Clz,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(8)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::Reg(MachineReg(8))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(8))],
                ops: collections::vec![clz_use(MachineReg(8))],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 12, 14, 0));

    let block = &program.blocks[0];
    assert_eq!(
        block.ops.len(),
        2,
        "a load tagged as LinearValue should seed aliasing even when it defines a high dynamic register; ops={:?}",
        block.ops
    );
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(11),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(11)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![MachineValue::Reg(MachineReg(11))]
    );
}

#[test]
fn does_not_copy_propagate_cached_cell_load_defs() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::CachedCell,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(7),
                            addr: MachineAddr {
                                base: MachineReg(1),
                                offset: 24,
                            },
                            width: MachineMemWidth::U64,
                            extension: MachineLoadExtension::ZeroExtend,
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(8),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: MachineIntWidth::I32,
                            op: MachineIntUnaryOp::Clz,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(8)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::Reg(MachineReg(8))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(8))],
                ops: collections::vec![clz_use(MachineReg(8))],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 12, 12, 0));

    let block = &program.blocks[0];
    assert_eq!(
        block.ops.len(),
        3,
        "a load tagged as CachedCell must not be treated as a linear alias source; ops={:?}",
        block.ops
    );
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(7),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(8),
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(8)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![MachineValue::Reg(MachineReg(8))]
    );
}

#[test]
fn constant_folding_keeps_live_constant_when_later_select_reads_and_writes_same_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(5),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Select {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        on_true: MachineValue::Reg(MachineReg(7)),
                        on_false: MachineValue::Reg(MachineReg(5)),
                        cond: MachineValue::Reg(MachineReg(4)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 8, 8, 0));

    let block = &program.blocks[0];
    let const_idx = block
        .ops
        .iter()
        .position(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    dst: MachineReg(7),
                    src: MachineValue::Imm64(5),
                    ..
                }
            )
        })
        .expect(
            "the tracked constant in reg7 must stay materialized because a later op still reads it",
        );
    let later_use_idx = block
        .ops
        .iter()
        .enumerate()
        .skip(const_idx + 1)
        .find_map(|(idx, inst)| {
            if matches!(
                inst.kind,
                MachineInstKind::Select {
                    dst: MachineReg(7),
                    on_true: MachineValue::Reg(MachineReg(7)),
                    ..
                }
            ) {
                Some(idx)
            } else {
                None
            }
        })
        .expect("a later select must still read reg7 after the constant move");
    assert!(const_idx < later_use_idx);
    assert!(block.ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Move {
                dst: MachineReg(7),
                src: MachineValue::Imm64(5),
                ..
            }
        )
    }));
}

#[test]
fn deduplicate_constants_kills_tracked_constant_when_i64_pair_instruction_redefines_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(5),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Add,
                        dst_lo: MachineReg(7),
                        dst_hi: MachineReg(8),
                        lhs_lo: MachineValue::Reg(MachineReg(2)),
                        lhs_hi: MachineValue::Reg(MachineReg(3)),
                        rhs_lo: MachineValue::Imm64(1),
                        rhs_hi: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(9),
                        src: MachineValue::Imm64(5),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 12, 12, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Move {
            dst: MachineReg(9),
            src: MachineValue::Imm64(5),
            ..
        }
    ));
}

#[test]
fn forwards_non_adjacent_u64_store_load_pairs() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Imm64(80),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(8),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 5);
    // Store-load forwarding turns the Load into a Move from original source
    assert!(matches!(
        block.ops[3].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        block.ops[4].kind,
        MachineInstKind::Store {
            addr: MachineAddr {
                base: MachineReg(8),
                offset: 0,
            },
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn removes_u64_self_reload_after_store() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(5),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        addr: MachineAddr {
                            base: MachineReg(5),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(4, 8, 8, 8, 0));

    assert_eq!(program.blocks[0].ops.len(), 1);
    assert!(matches!(
        program.blocks[0].ops[0].kind,
        MachineInstKind::Store { .. }
    ));
}

#[test]
fn forwards_store_reload_exposed_by_copy_propagation() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::vec![MachineBlockParam::gp_word(MachineReg(6))],
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(6)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(5),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        addr: MachineAddr {
                            base: MachineReg(6),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(4, 8, 8, 8, 0));

    assert_eq!(program.blocks[0].ops.len(), 1);
    assert!(matches!(
        program.blocks[0].ops[0].kind,
        MachineInstKind::Store {
            addr: MachineAddr {
                base: MachineReg(6),
                offset: 0
            },
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
}

#[test]
fn carries_invariant_context_load_across_loop_backedge() {
    let pointer_load = || MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(5),
            addr: MachineAddr {
                base: MACHINE_CTX_REG,
                offset: 64,
            },
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        },
    };
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![pointer_load()],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::Vec::new(),
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::vec![
                    pointer_load(),
                    MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(4),
                            addr: MachineAddr {
                                base: MachineReg(5),
                                offset: 0,
                            },
                            width: MachineMemWidth::U64,
                            extension: MachineLoadExtension::None,
                        },
                    },
                ],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(4))),
                    then_edge: MachineEdge {
                        target: MachineBlockId(1),
                        args: collections::Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(4, 8, 8, 8, 0));

    assert_eq!(program.blocks[1].ops.len(), 1);
    assert_eq!(
        program.blocks[1].params,
        collections::vec![MachineBlockParam::gp_word(MachineReg(5))]
    );
    let MachineTerminator::Jump(entry_edge) = &program.blocks[0].terminator else {
        panic!("expected entry jump")
    };
    assert_eq!(
        entry_edge.args,
        collections::vec![MachineValue::Reg(MachineReg(5))]
    );
    let MachineTerminator::Branch { then_edge, .. } = &program.blocks[1].terminator else {
        panic!("expected loop branch")
    };
    assert_eq!(
        then_edge.args,
        collections::vec![MachineValue::Reg(MachineReg(5))]
    );
}

#[test]
fn keeps_loop_context_load_when_the_context_slot_is_written() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        addr: MachineAddr {
                            base: MACHINE_CTX_REG,
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                }],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::Vec::new(),
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Load {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(5),
                            addr: MachineAddr {
                                base: MACHINE_CTX_REG,
                                offset: 64,
                            },
                            width: MachineMemWidth::U64,
                            extension: MachineLoadExtension::None,
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::Store {
                            ty: MachineStorageType::GpWord,
                            addr: MachineAddr {
                                base: MACHINE_CTX_REG,
                                offset: 64,
                            },
                            width: MachineMemWidth::U64,
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::Vec::new(),
                }),
            },
        ],
    };

    optimize(&mut program, test_config(4, 8, 8, 8, 0));

    assert_eq!(program.blocks[1].ops.len(), 2);
    assert!(program.blocks[1].params.is_empty());
}

#[test]
fn removes_dead_block_param_carried_only_around_a_loop() {
    let params = || {
        collections::vec![
            MachineBlockParam::gp_word(MachineReg(4)),
            MachineBlockParam::gp_word(MachineReg(5)),
        ]
    };
    let args = || {
        collections::vec![
            MachineValue::Reg(MachineReg(4)),
            MachineValue::Reg(MachineReg(5)),
        ]
    };
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: args(),
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: params(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(2),
                    args: args(),
                }),
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: params(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(5))),
                    then_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: args(),
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(3),
                        args: collections::Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(3),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(4, 8, 8, 8, 0));

    for block_index in [1usize, 2] {
        assert_eq!(
            program.blocks[block_index].params,
            collections::vec![MachineBlockParam::gp_word(MachineReg(5))]
        );
    }
    let MachineTerminator::Jump(entry) = &program.blocks[0].terminator else {
        panic!("expected entry jump")
    };
    assert_eq!(
        entry.args,
        collections::vec![MachineValue::Reg(MachineReg(5))]
    );
    let MachineTerminator::Branch { then_edge, .. } = &program.blocks[2].terminator else {
        panic!("expected loop branch")
    };
    assert_eq!(
        then_edge.args,
        collections::vec![MachineValue::Reg(MachineReg(5))]
    );
}

/// The P1 gp32-i64-pair shape from Mandelbrot's inner loop:
///
/// ```text
///     store.gp.u32 [fp+72] <- r4         ; local.set_slot fp[9] lo
///     store.gp.u32 [fp+76] <- r6         ; local.set_slot fp[9] hi
///     load.linear.gp.u32 r4 <- [fp+72]   ; local.get_slot fp[9] lo — dead
///     load.linear.gp.u32 r6 <- [fp+76]   ; local.get_slot fp[9] hi — dead
///     <consume r4, r6>
/// ```
///
/// Both loads are value-preserving no-ops for their destination registers
/// (the stores wrote the exact same register values to those slots with no
/// intervening writes). Forward-stored-values must delete them.
#[test]
fn forwards_u32_pair_self_store_reload_from_gp32_i64_slot() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 76,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(6)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 76,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(4, 4, 8, 8, 0));

    let block = &program.blocks[0];
    // Both stores remain (they're needed — the wasm local still has to be
    // written). Both self-loads disappear.
    assert_eq!(block.ops.len(), 2);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Store {
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 72
            },
            width: MachineMemWidth::U32,
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Store {
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 76
            },
            width: MachineMemWidth::U32,
            src: MachineValue::Reg(MachineReg(6)),
            ..
        }
    ));
}

/// When the stored register gets clobbered between the store and the matching
/// load, the load must NOT be forwarded even though addr + width match.
#[test]
fn does_not_forward_u32_load_when_stored_reg_was_clobbered() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                // Clobber r4 before the load.
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(4),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(1),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(4, 4, 8, 8, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 3);
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Load {
            dst: MachineReg(5),
            width: MachineMemWidth::U32,
            ..
        }
    ));
}

/// Width-mismatch safety: a `U32` store at [base+K] must NOT forward into a
/// `U64` load at [base+K] even though the addresses are equal. We do not
/// synthesize wider loads from narrower stores.
#[test]
fn does_not_forward_u32_store_into_u64_load_same_address() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 8, 8, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 2);
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Load {
            dst: MachineReg(7),
            width: MachineMemWidth::U64,
            ..
        }
    ));
}

/// Mixed-width overlap invalidation: a later `U64` store at [base+64] must
/// kill a tracked `U32` store at [base+68] (they overlap bytes 68..72). A
/// subsequent `U32` load at [base+68] must not be forwarded from the stale
/// tracked U32 entry.
#[test]
fn u64_store_invalidates_tracked_u32_store_on_overlap() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 68,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 68,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(4, 4, 8, 8, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 3);
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 68
            },
            width: MachineMemWidth::U32,
            ..
        }
    ));
}

#[test]
fn forwards_fp_spill_reload_into_gp_move() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::Fp64,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(11)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 11, 12, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(11)),
            ..
        }
    ));
}

#[test]
fn does_not_forward_when_i64_pair_instruction_redefines_stored_source_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Add,
                        dst_lo: MachineReg(7),
                        dst_hi: MachineReg(8),
                        lhs_lo: MachineValue::Reg(MachineReg(2)),
                        lhs_hi: MachineValue::Reg(MachineReg(3)),
                        rhs_lo: MachineValue::Imm64(1),
                        rhs_hi: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(9),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 12, 12, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Load {
            dst: MachineReg(9),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 64,
            },
            ..
        }
    ));
}

#[test]
fn does_not_forward_when_stored_source_reg_is_redefined() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 8, 8, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Load {
            dst: MachineReg(7),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 64,
            },
            ..
        }
    ));
}

#[test]
fn does_not_forward_across_overlapping_store() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 68,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 64,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 8, 8, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Load {
            dst: MachineReg(7),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 64,
            },
            ..
        }
    ));
}

#[test]
fn reuses_identical_loads_when_memory_stays_unchanged() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Imm64(16),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(9),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(8),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(9)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 10, 10, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 3);
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Store {
            addr: MachineAddr {
                base: MachineReg(8),
                offset: 0,
            },
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn reuses_frame_loads_after_indexed_memory_fusion() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 8,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Convert {
                        op: MachineConvertOp::I64ExtendI32U,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(7),
                            offset: 0,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(8)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 8,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Convert {
                        op: MachineConvertOp::I64ExtendI32U,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(7),
                            offset: 0,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(9)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 10, 10, 0));

    let block = &program.blocks[0];
    let load_count = block
        .ops
        .iter()
        .filter(|inst| matches!(inst.kind, MachineInstKind::Load { .. }))
        .count();
    let indexed_store_count = block
        .ops
        .iter()
        .filter(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::IndexedStore {
                    base: MachineReg(2),
                    index: MachineReg(7),
                    index_extend: MachineIndexExtend::ZeroExtend32,
                    ..
                }
            )
        })
        .count();

    assert_eq!(
        load_count, 1,
        "the second frame load should be removed after indexed-memory fusion; ops={:?}",
        block.ops
    );
    assert_eq!(
        indexed_store_count, 2,
        "both address-building sequences should fuse to indexed stores; ops={:?}",
        block.ops
    );
}

#[test]
fn does_not_reuse_identical_loads_across_distinct_storage_types() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::Fp64,
                        dst: MachineReg(11),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 11, 12, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Load {
            dst: MachineReg(7),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 80,
            },
            ..
        }
    ));
}

#[test]
fn does_not_reuse_load_after_loaded_reg_is_redefined() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(8),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Load {
            dst: MachineReg(8),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 80,
            },
            ..
        }
    ));
}

#[test]
fn does_not_reuse_load_after_i64_pair_instruction_redefines_loaded_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Add,
                        dst_lo: MachineReg(7),
                        dst_hi: MachineReg(8),
                        lhs_lo: MachineValue::Reg(MachineReg(2)),
                        lhs_hi: MachineValue::Reg(MachineReg(3)),
                        rhs_lo: MachineValue::Imm64(1),
                        rhs_hi: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(9),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 80,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 12, 12, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Load {
            dst: MachineReg(9),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 80,
            },
            ..
        }
    ));
}

#[test]
fn copy_propagate_kills_alias_when_i64_pair_instruction_redefines_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Add,
                        dst_lo: MachineReg(7),
                        dst_hi: MachineReg(8),
                        lhs_lo: MachineValue::Reg(MachineReg(2)),
                        lhs_hi: MachineValue::Reg(MachineReg(3)),
                        rhs_lo: MachineValue::Imm64(1),
                        rhs_hi: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntUnary {
                        width: MachineIntWidth::I32,
                        op: MachineIntUnaryOp::Clz,
                        dst: MachineReg(9),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 12, 12, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn preserves_linear_value_move_when_linear_source_reg_is_redefined_before_terminator_use() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(8)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(8),
                        src: MachineValue::Imm64(0),
                    },
                },
            ],
            terminator: MachineTerminator::Jump(MachineEdge {
                target: MachineBlockId(1),
                args: collections::vec![MachineValue::Reg(MachineReg(7))],
            }),
        }],
    };

    optimize(&mut program, test_config(7, 8, 10, 10, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(8)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![MachineValue::Reg(MachineReg(7))]
    );
}

#[test]
fn copy_propagates_linear_copies_of_cached_cell_snapshots() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(7),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(8),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: MachineIntWidth::I32,
                            op: MachineIntUnaryOp::Clz,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(8)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::Reg(MachineReg(8))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(8))],
                ops: collections::vec![clz_use(MachineReg(8))],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 2);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![MachineValue::Reg(MachineReg(7))]
    );
}

#[test]
fn preserves_linear_value_move_live_across_helper_barrier() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(8)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::CallRuntime(MachineCallRuntime {
                        metadata: MachineConstId(0),
                    },),
                },
                MachineInst {
                    kind: MachineInstKind::IntUnary {
                        width: MachineIntWidth::I32,
                        op: MachineIntUnaryOp::Clz,
                        dst: MachineReg(6),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 3);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(8)),
            ..
        }
    ));
    assert!(matches!(block.ops[1].kind, MachineInstKind::CallRuntime(_)));
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn does_not_copy_propagate_cached_cell_snapshots_into_integer_uses_or_edges() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(7),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: MachineIntWidth::I32,
                            op: MachineIntUnaryOp::Clz,
                            dst: MachineReg(8),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::Reg(MachineReg(7))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(7))],
                ops: collections::vec![clz_use(MachineReg(7))],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 2);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![MachineValue::Reg(MachineReg(7))]
    );
}

#[test]
fn rewrites_float_uses_of_gp_aliases_back_to_fp_regs() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpI64,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(10)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::FloatBinary {
                        width: MachineFloatWidth::F64,
                        op: MachineFloatBinaryOp::Add,
                        dst: MachineReg(11),
                        lhs: MachineValue::Reg(MachineReg(10)),
                        rhs: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 10, 12, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::FloatBinary {
            rhs: MachineValue::Reg(MachineReg(10)),
            ..
        }
    ));
}

#[test]
fn rewrites_u64_store_of_gp_float_alias_back_to_fp_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpI64,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(10)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpI64,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 32,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 10, 11, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(10)),
            ..
        }
    ));
}

#[test]
fn preserves_moves_into_fp_cached_cells() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![
            None,
            None,
            Some(MachineFloatWidth::F32),
            Some(MachineFloatWidth::F32),
        ],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::CachedCell,
                        ty: MachineStorageType::Fp32,
                        dst: MachineReg(13),
                        src: MachineValue::Reg(MachineReg(11)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::Fp32,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 48,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(13)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 11, 15, 2));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 2);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(13),
            src: MachineValue::Reg(MachineReg(11)),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(13)),
            ..
        }
    ));
}

#[test]
fn does_not_fuse_i64_compare_branch_on_32_bit_targets() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: MachineIntWidth::I64,
                        kind: MachineCompareKind::Eq,
                        sign: MachineSign::Unsigned,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(0),
                    },
                }],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(7)),),
                    then_edge: MachineEdge {
                        target: MachineBlockId(1),
                        args: collections::Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 4, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 1);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::IntCompare {
            width: MachineIntWidth::I64,
            dst: MachineReg(7),
            ..
        }
    ));
    assert!(matches!(
        block.terminator,
        MachineTerminator::Branch {
            cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(7))),
            ..
        }
    ));
}

#[test]
fn still_fuses_i32_compare_branch_on_32_bit_targets() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: MachineIntWidth::I32,
                        kind: MachineCompareKind::Eq,
                        sign: MachineSign::Unsigned,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(0),
                    },
                }],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(7)),),
                    then_edge: MachineEdge {
                        target: MachineBlockId(1),
                        args: collections::Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 4, 9, 9, 0));

    let block = &program.blocks[0];
    assert!(block.ops.is_empty());
    assert!(matches!(
        block.terminator,
        MachineTerminator::Branch {
            cond: MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq,
                ..
            },
            ..
        }
    ));
}

#[test]
fn fuses_compare_branch_for_high_dynamic_result_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: MachineIntWidth::I32,
                        kind: MachineCompareKind::Eq,
                        sign: MachineSign::Unsigned,
                        dst: MachineReg(11),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(0),
                    },
                }],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(11)),),
                    then_edge: MachineEdge {
                        target: MachineBlockId(1),
                        args: collections::Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 12, 14, 0));

    let block = &program.blocks[0];
    assert!(
        block.ops.is_empty(),
        "compare-branch fusion must not depend on old transient/cache register-number folklore"
    );
    assert!(matches!(
        block.terminator,
        MachineTerminator::Branch {
            cond: MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq,
                ..
            },
            ..
        }
    ));
}

#[test]
fn fuses_test_bits_for_high_dynamic_result_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::And,
                        dst: MachineReg(11),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(0xff),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: MachineIntWidth::I32,
                        kind: MachineCompareKind::Eq,
                        sign: MachineSign::Unsigned,
                        dst: MachineReg(13),
                        lhs: MachineValue::Reg(MachineReg(11)),
                        rhs: MachineValue::Imm64(0),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 12, 16, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 1);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::TestBits {
            dst: MachineReg(13),
            src: MachineReg(4),
            mask: MachineValue::Imm64(0xff),
            ..
        }
    ));
}

#[test]
fn does_not_fold_constant_past_non_adjacent_instruction() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![None],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::FloatConst {
                        width: MachineFloatWidth::F32,
                        dst: MachineReg(9),
                        bits: 0,
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 24,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 9, 10, 1));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 3);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Imm64(0),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::FloatConst { .. }
    ));
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn does_not_fold_constant_used_as_non_replaceable_address_base() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(64),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        addr: MachineAddr {
                            base: MachineReg(7),
                            offset: 0,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 8, 8, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 2);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Imm64(64),
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Load {
            addr: MachineAddr {
                base: MachineReg(7),
                ..
            },
            ..
        }
    ));
}

#[test]
fn fuses_shru_and_into_bitfield_extract() {
    // ShrU(r4, #1) + And(result, #32767) → BitfieldExtractU(r4, lsb=1, bits=15)
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::ShrU,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(1),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::And,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(7)),
                        rhs: MachineValue::Imm64(32767),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(
        block.ops.len(),
        1,
        "should fuse ShrU+And into 1 instruction, got: {:?}",
        block.ops
    );
    assert!(
        matches!(
            block.ops[0].kind,
            MachineInstKind::BitfieldExtractU {
                width: MachineIntWidth::I32,
                dst: MachineReg(8),
                src: MachineReg(4),
                lsb: 1,
                bits: 15,
            }
        ),
        "expected BitfieldExtractU, got: {:?}",
        block.ops[0].kind
    );
}

#[test]
fn fuses_rotate_then_xor_into_rotated_register_operand() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Rotl,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(1),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Xor,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(5)),
                        rhs: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 9, 9, 0));

    assert_eq!(program.blocks[0].ops.len(), 1);
    assert!(matches!(
        program.blocks[0].ops[0].kind,
        MachineInstKind::IntBinaryShifted {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Xor,
            dst: MachineReg(8),
            lhs: MachineReg(5),
            rhs: MachineReg(4),
            shift: MachineShiftOp::Ror,
            amount: 63,
        }
    ));
}

#[test]
fn skips_i64_pair_sign_ext_analysis_on_64_bit_targets() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Int64PairUnary {
                        op: MachineIntUnaryOp::Extend32S,
                        dst_lo: MachineReg(5),
                        dst_hi: MachineReg(6),
                        src_lo: MachineValue::Reg(MachineReg(4)),
                        src_hi: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Mul,
                        dst_lo: MachineReg(8),
                        dst_hi: MachineReg(9),
                        lhs_lo: MachineValue::Reg(MachineReg(5)),
                        lhs_hi: MachineValue::Reg(MachineReg(6)),
                        rhs_lo: MachineValue::Reg(MachineReg(5)),
                        rhs_hi: MachineValue::Reg(MachineReg(6)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 8, 10, 12, 0));

    assert!(matches!(
        program.blocks[0].ops.last().map(|inst| &inst.kind),
        Some(MachineInstKind::Int64PairBinary {
            op: MachineIntBinaryOp::Mul,
            ..
        })
    ));
}

/// Reproduces the MIR shape for `(a as i64) * (b as i64)` where the
/// `gp32` lowering inserts a `Move` to alias one operand to another reg
/// before the sign-extend (e.g., `Move r6 <- Reg(r4); ShrS r7 <- Reg(r4)`).
/// Without value-alias tracking through the `Move`, the lhs_lo (r6) and
/// the sign-ext source (r4) appear as different registers and the fusion
/// would miss this case. This is the second `i64.mul` in the actual
/// Mandelbrot kernel (`zx*zy`).
#[test]
fn fuses_i64_mul_with_move_alias_then_sign_ext() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                // r6 := r4   (the alias-creating Move)
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                // r7 := sign(r4)   (sign-extend on the original reg)
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::ShrS,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(31),
                    },
                },
                // spill r6 (= r4)
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 96,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(6)),
                    },
                },
                // spill r7
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 100,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                // reload r6
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 96,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
                // reload r7
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 100,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
                // i64.mul (r6:r7) * (r6:r7)  — should fuse since r6 aliases r4 and r7 = sign(r4)
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Mul,
                        dst_lo: MachineReg(8),
                        dst_hi: MachineReg(9),
                        lhs_lo: MachineValue::Reg(MachineReg(6)),
                        lhs_hi: MachineValue::Reg(MachineReg(7)),
                        rhs_lo: MachineValue::Reg(MachineReg(6)),
                        rhs_hi: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 10, 16, 0));

    let block = &program.blocks[0];
    let mul_inst = block.ops.last().expect("block must have a final inst");
    assert!(
        matches!(
            mul_inst.kind,
            MachineInstKind::Int64MulFromSignExt32 {
                dst_lo: MachineReg(8),
                dst_hi: MachineReg(9),
                // operand should be the canonical alias (r4), not the moved-into r6
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Reg(MachineReg(4)),
            }
        ),
        "expected Int64MulFromSignExt32 with operand=Reg(r4) (canonical alias), got: {:?}",
        mul_inst.kind
    );
}

/// Reproduces the MIR shape produced by gp32 lowering for
/// `(a as i64) * (b as i64)` after copy_propagate. Both operands of the
/// `i64.mul` are sign-extended-from-i32, with the typical
/// spill-to-frame + reload-into-the-same-register interlude that the
/// `gp32` pipeline inserts before pair operations. The fuse_smull_sign_ext
/// pass should rewrite the `Int64PairBinary{Mul}` into
/// `Int64MulFromSignExt32` despite the spill/reload roundtrip.
#[test]
fn fuses_i64_mul_with_self_spill_reload_sign_ext() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                // r6 = sign(r4)  (i32.shr_s r4, 31)
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::ShrS,
                        dst: MachineReg(6),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(31),
                    },
                },
                // spill r4 (lo) to slot
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                // spill r6 (hi) to slot
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 76,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(MachineReg(6)),
                    },
                },
                // reload r4 (lo) — same register, no-op spill
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 72,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
                // reload r6 (hi) — same register, no-op spill
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 76,
                        },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                },
                // i64.mul (r4:r6) * (r4:r6)  →  the squaring case
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Mul,
                        dst_lo: MachineReg(7),
                        dst_hi: MachineReg(8),
                        lhs_lo: MachineValue::Reg(MachineReg(4)),
                        lhs_hi: MachineValue::Reg(MachineReg(6)),
                        rhs_lo: MachineValue::Reg(MachineReg(4)),
                        rhs_hi: MachineValue::Reg(MachineReg(6)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 8, 16, 0));

    let block = &program.blocks[0];
    let mul_inst = block.ops.last().expect("block must have a final inst");
    assert!(
        matches!(
            mul_inst.kind,
            MachineInstKind::Int64MulFromSignExt32 {
                dst_lo: MachineReg(7),
                dst_hi: MachineReg(8),
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Reg(MachineReg(4)),
            }
        ),
        "expected Int64MulFromSignExt32 after spill/reload + sign-ext + i64.mul, got: {:?}",
        mul_inst.kind
    );
}

#[test]
fn fuses_i64_mul_after_pair_extend32s_with_copied_halves() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Int64PairUnary {
                        op: MachineIntUnaryOp::Extend32S,
                        dst_lo: MachineReg(13),
                        dst_hi: MachineReg(14),
                        src_lo: MachineValue::Reg(MachineReg(4)),
                        src_hi: MachineValue::Reg(MachineReg(18)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        src: MachineValue::Reg(MachineReg(13)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(18),
                        src: MachineValue::Reg(MachineReg(14)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(19),
                        src: MachineValue::Reg(MachineReg(13)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(20),
                        src: MachineValue::Reg(MachineReg(14)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Mul,
                        dst_lo: MachineReg(4),
                        dst_hi: MachineReg(18),
                        lhs_lo: MachineValue::Reg(MachineReg(4)),
                        lhs_hi: MachineValue::Reg(MachineReg(18)),
                        rhs_lo: MachineValue::Reg(MachineReg(19)),
                        rhs_hi: MachineValue::Reg(MachineReg(20)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    optimize(&mut program, test_config(7, 4, 22, 24, 0));

    let block = &program.blocks[0];
    let mul_inst = block.ops.last().expect("block must have a final inst");
    assert!(
        matches!(
            mul_inst.kind,
            MachineInstKind::Int64MulFromSignExt32 {
                dst_lo: MachineReg(4),
                dst_hi: MachineReg(18),
                lhs: MachineValue::Reg(MachineReg(4)),
                rhs: MachineValue::Reg(MachineReg(4)),
            }
        ),
        "expected Int64MulFromSignExt32 after i64.extend32_s pair lowering, got: {:?}",
        mul_inst.kind
    );
}

#[test]
fn fuses_i64_mul_from_sign_ext_pair_across_edge() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::Int64PairUnary {
                        op: MachineIntUnaryOp::Extend32S,
                        dst_lo: MachineReg(8),
                        dst_hi: MachineReg(9),
                        src_lo: MachineValue::Reg(MachineReg(4)),
                        src_hi: MachineValue::Reg(MachineReg(5)),
                    },
                }],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![
                        MachineValue::Reg(MachineReg(8)),
                        MachineValue::Reg(MachineReg(9)),
                    ],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![
                    MachineBlockParam::gp_word(MachineReg(10)),
                    MachineBlockParam::gp_word(MachineReg(11)),
                ],
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Mul,
                        dst_lo: MachineReg(12),
                        dst_hi: MachineReg(13),
                        lhs_lo: MachineValue::Reg(MachineReg(10)),
                        lhs_hi: MachineValue::Reg(MachineReg(11)),
                        rhs_lo: MachineValue::Reg(MachineReg(10)),
                        rhs_hi: MachineValue::Reg(MachineReg(11)),
                    },
                }],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 4, 24, 24, 0));

    let block = &program.blocks[1];
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Int64MulFromSignExt32 {
            dst_lo: MachineReg(12),
            dst_hi: MachineReg(13),
            lhs: MachineValue::Reg(MachineReg(10)),
            rhs: MachineValue::Reg(MachineReg(10)),
        }
    ));
}

#[test]
fn does_not_fuse_cross_block_sign_ext_fact_unless_all_predecessors_agree() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::Int64PairUnary {
                        op: MachineIntUnaryOp::Extend32S,
                        dst_lo: MachineReg(8),
                        dst_hi: MachineReg(9),
                        src_lo: MachineValue::Reg(MachineReg(4)),
                        src_hi: MachineValue::Reg(MachineReg(5)),
                    },
                }],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(4))),
                    then_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::vec![
                            MachineValue::Reg(MachineReg(8)),
                            MachineValue::Reg(MachineReg(9)),
                        ],
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(1),
                        args: collections::Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(2),
                    args: collections::vec![
                        MachineValue::Reg(MachineReg(6)),
                        MachineValue::Reg(MachineReg(7)),
                    ],
                }),
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::vec![
                    MachineBlockParam::gp_word(MachineReg(10)),
                    MachineBlockParam::gp_word(MachineReg(11)),
                ],
                ops: collections::vec![MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op: MachineIntBinaryOp::Mul,
                        dst_lo: MachineReg(12),
                        dst_hi: MachineReg(13),
                        lhs_lo: MachineValue::Reg(MachineReg(10)),
                        lhs_hi: MachineValue::Reg(MachineReg(11)),
                        rhs_lo: MachineValue::Reg(MachineReg(10)),
                        rhs_hi: MachineValue::Reg(MachineReg(11)),
                    },
                }],
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 4, 24, 24, 0));

    let block = &program.blocks[2];
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Int64PairBinary {
            op: MachineIntBinaryOp::Mul,
            ..
        }
    ));
}

#[test]
fn does_not_seed_cross_block_sign_ext_fact_from_self_loop_only() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![
                        MachineValue::Reg(MachineReg(6)),
                        MachineValue::Reg(MachineReg(7)),
                    ],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![
                    MachineBlockParam::gp_word(MachineReg(10)),
                    MachineBlockParam::gp_word(MachineReg(11)),
                ],
                ops: collections::vec![
                    MachineInst {
                        kind: MachineInstKind::Int64PairBinary {
                            op: MachineIntBinaryOp::Mul,
                            dst_lo: MachineReg(12),
                            dst_hi: MachineReg(13),
                            lhs_lo: MachineValue::Reg(MachineReg(10)),
                            lhs_hi: MachineValue::Reg(MachineReg(11)),
                            rhs_lo: MachineValue::Reg(MachineReg(10)),
                            rhs_hi: MachineValue::Reg(MachineReg(11)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::Int64PairUnary {
                            op: MachineIntUnaryOp::Extend32S,
                            dst_lo: MachineReg(10),
                            dst_hi: MachineReg(11),
                            src_lo: MachineValue::Reg(MachineReg(4)),
                            src_hi: MachineValue::Reg(MachineReg(5)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![
                        MachineValue::Reg(MachineReg(10)),
                        MachineValue::Reg(MachineReg(11)),
                    ],
                }),
            },
        ],
    };

    optimize(&mut program, test_config(7, 4, 24, 24, 0));

    let block = &program.blocks[1];
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Int64PairBinary {
            op: MachineIntBinaryOp::Mul,
            ..
        }
    ));
}

#[test]
fn optimize_preserves_block_ops_allocation() {
    let mut ops = collections::Vec::with_capacity(32);
    ops.push(MachineInst {
        kind: MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(8)),
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::IntUnary {
            width: MachineIntWidth::I32,
            op: MachineIntUnaryOp::Clz,
            dst: MachineReg(9),
            src: MachineValue::Reg(MachineReg(7)),
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::Store {
            ty: MachineStorageType::GpWord,
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 16,
            },
            width: MachineMemWidth::U32,
            src: MachineValue::Reg(MachineReg(9)),
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(10),
            addr: MachineAddr {
                base: MachineReg(1),
                offset: 16,
            },
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::None,
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::ShrU,
            dst: MachineReg(10),
            lhs: MachineValue::Reg(MachineReg(10)),
            rhs: MachineValue::Imm64(3),
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::And,
            dst: MachineReg(11),
            lhs: MachineValue::Reg(MachineReg(10)),
            rhs: MachineValue::Imm64(0x1f),
        },
    });

    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::vec![MachineBlockParam::gp_word(MachineReg(8))],
            ops,
            terminator: MachineTerminator::Return,
        }],
    };

    let ops_ptr = program.blocks[0].ops.as_ptr();
    let ops_capacity = program.blocks[0].ops.capacity();

    optimize(&mut program, test_config(7, 8, 12, 12, 0));

    assert_eq!(program.blocks[0].ops.as_ptr(), ops_ptr);
    assert_eq!(program.blocks[0].ops.capacity(), ops_capacity);
    assert!(
        program.blocks[0].ops.len() < 6,
        "test should exercise at least one in-place removal/fusion"
    );
}
