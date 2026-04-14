use crate::collections;
use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
    MachineCallExternal, MachineCompareKind, MachineConstId, MachineEdge, MachineFloatBinaryOp,
    MachineFloatWidth, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp,
    MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineProgram, MachineReg,
    MachineRegOwner, MachineSign, MachineStorageType, MachineTerminator, MachineValue,
};
use crate::vm::machine::peephole::optimize;

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
        gp_dynamic,
        fp_dynamic,
        gp_reg_width,
        if gp_reg_width == 4 { 8 } else { 3 },
    )
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
                ops: collections::Vec::new(),
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
fn does_not_copy_propagate_move_from_cached_local_block_param() {
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
                    .with_owner(MachineRegOwner::CachedLocal,)],
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
                ops: collections::Vec::new(),
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
                ops: collections::Vec::new(),
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
fn does_not_copy_propagate_cached_local_load_defs() {
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
                            owner: MachineRegOwner::CachedLocal,
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
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    optimize(&mut program, test_config(7, 8, 12, 12, 0));

    let block = &program.blocks[0];
    assert_eq!(
        block.ops.len(),
        3,
        "a load tagged as CachedLocal must not be treated as a linear alias source; ops={:?}",
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
fn copy_propagates_linear_copies_of_cached_local_snapshots() {
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
                ops: collections::Vec::new(),
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
                    kind: MachineInstKind::CallExternal(MachineCallExternal {
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
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::CallExternal(_)
    ));
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn does_not_copy_propagate_cached_local_snapshots_into_integer_uses_or_edges() {
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
                ops: collections::Vec::new(),
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
fn preserves_moves_into_fp_cached_locals() {
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
                        owner: MachineRegOwner::CachedLocal,
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
