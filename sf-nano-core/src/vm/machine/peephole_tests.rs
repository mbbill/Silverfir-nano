use alloc::{vec, vec::Vec};

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineEdge, MachineInst,
    MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineProgram, MachineReg,
    MachineStorageType, MachineTerminator, MachineValue,
};

/// Build a BackendConfig matching old test parameters.
///
/// `first_transient` — old first GP transient register ID.
/// `gp_reg_width` — GP register width in bytes (4 or 8).
/// `first_fp_reg` — old first FP register ID.
/// `reg_count` — old total register count.
/// `fp_transient_count` — old FP transient count.
fn test_config(
    first_transient: u16,
    gp_reg_width: u8,
    first_fp_reg: u16,
    reg_count: u16,
    fp_transient_count: u16,
) -> BackendConfig {
    let gp_cache = (first_transient - BackendConfig::FIXED) as u8;
    let gp_trans = (first_fp_reg - first_transient) as u8;
    let fp_total = (reg_count - first_fp_reg) as u8;
    let fp_trans = fp_transient_count as u8;
    let fp_cache = fp_total - fp_trans;
    BackendConfig::new_with_gp_unit_bytes(gp_cache, gp_trans, fp_cache, fp_trans, gp_reg_width)
}

#[test]
fn copy_propagates_transient_to_transient_moves_into_ops_and_edges() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: alloc::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(7),
                            src: MachineValue::Reg(MachineReg(8)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                            op: crate::vm::machine::machine_ir::MachineIntUnaryOp::Eqz,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: alloc::vec![MachineValue::Reg(MachineReg(7))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: alloc::vec![MachineBlockParam::gp_word(MachineReg(7))],
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 1);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(8)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(edge.args, alloc::vec![MachineValue::Reg(MachineReg(8))]);
}

#[test]
fn constant_folding_keeps_live_constant_when_later_select_reads_and_writes_same_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(5),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 8, 8, 0));

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Imm64(5),
            ..
        }
    ));
    assert!(matches!(
        block.ops[3].kind,
        MachineInstKind::Select {
            dst: MachineReg(7),
            on_true: MachineValue::Reg(MachineReg(7)),
            ..
        }
    ));
}

#[test]
fn forwards_non_adjacent_u64_store_load_pairs() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
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
                        width: crate::vm::machine::machine_ir::MachineIntWidth::I64,
                        op: crate::vm::machine::machine_ir::MachineIntBinaryOp::Add,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Imm64(80),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 4);
    assert!(matches!(
        block.ops[3].kind,
        MachineInstKind::Store {
            addr: MachineAddr {
                base: MachineReg(8),
                offset: 0,
            },
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
}

#[test]
fn forwards_fp_spill_reload_into_gp_move() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 11, 12, 0));

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
fn does_not_forward_when_stored_source_reg_is_redefined() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
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
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(4),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 8, 8, 0));

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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 8, 8, 0));

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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
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
                        width: crate::vm::machine::machine_ir::MachineIntWidth::I64,
                        op: crate::vm::machine::machine_ir::MachineIntBinaryOp::Add,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Imm64(16),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 10, 10, 0));

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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 11, 12, 0));

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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
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
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 9, 9, 0));

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
fn preserves_transient_move_when_transient_source_reg_is_redefined_before_terminator_use() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(8)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(8),
                        src: MachineValue::Imm64(0),
                    },
                },
            ],
            terminator: MachineTerminator::Jump(MachineEdge {
                target: MachineBlockId(1),
                args: alloc::vec![MachineValue::Reg(MachineReg(7))],
            }),
        }],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 10, 10, 0));

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
    assert_eq!(edge.args, alloc::vec![MachineValue::Reg(MachineReg(7))]);
}

#[test]
fn copy_propagates_transient_copies_of_cached_local_snapshots() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: alloc::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(7),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::Move {
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(8),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                            op: crate::vm::machine::machine_ir::MachineIntUnaryOp::Eqz,
                            dst: MachineReg(6),
                            src: MachineValue::Reg(MachineReg(8)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: alloc::vec![MachineValue::Reg(MachineReg(8))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: alloc::vec![MachineBlockParam::gp_word(MachineReg(8))],
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 9, 9, 0));

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
    assert_eq!(edge.args, alloc::vec![MachineValue::Reg(MachineReg(7))]);
}

#[test]
fn preserves_transient_move_live_across_helper_barrier() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(8)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::CallHelper(
                        crate::vm::machine::machine_ir::MachineHelperCall {
                            target: crate::vm::machine::machine_ir::MachineExternId(0),
                            metadata: crate::vm::machine::machine_ir::MachineConstId(0),
                        },
                    ),
                },
                MachineInst {
                    kind: MachineInstKind::IntUnary {
                        width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                        op: crate::vm::machine::machine_ir::MachineIntUnaryOp::Eqz,
                        dst: MachineReg(6),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 9, 9, 0));

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
    assert!(matches!(block.ops[1].kind, MachineInstKind::CallHelper(_)));
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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: alloc::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(7),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                            op: crate::vm::machine::machine_ir::MachineIntUnaryOp::Eqz,
                            dst: MachineReg(8),
                            src: MachineValue::Reg(MachineReg(7)),
                        },
                    },
                ],
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: alloc::vec![MachineValue::Reg(MachineReg(7))],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: alloc::vec![MachineBlockParam::gp_word(MachineReg(7))],
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 9, 9, 0));

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
    assert_eq!(edge.args, alloc::vec![MachineValue::Reg(MachineReg(7))]);
}

#[test]
fn rewrites_float_uses_of_gp_aliases_back_to_fp_regs() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpI64,
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(10)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::FloatBinary {
                        width: crate::vm::machine::machine_ir::MachineFloatWidth::F64,
                        op: crate::vm::machine::machine_ir::MachineFloatBinaryOp::Add,
                        dst: MachineReg(11),
                        lhs: MachineValue::Reg(MachineReg(10)),
                        rhs: MachineValue::Reg(MachineReg(7)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 10, 12, 0));

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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 10, 11, 0));

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
        fp_reg_init_widths: vec![
            None,
            None,
            Some(crate::vm::machine::machine_ir::MachineFloatWidth::F32),
            Some(crate::vm::machine::machine_ir::MachineFloatWidth::F32),
        ],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 8, 11, 15, 2));

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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: alloc::vec![MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: crate::vm::machine::machine_ir::MachineIntWidth::I64,
                        kind: crate::vm::machine::machine_ir::MachineCompareKind::Eq,
                        sign: crate::vm::machine::machine_ir::MachineSign::Unsigned,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(0),
                    },
                }],
                terminator: MachineTerminator::Branch {
                    cond: crate::vm::machine::machine_ir::MachineBranchCond::Value(
                        MachineValue::Reg(MachineReg(7)),
                    ),
                    then_edge: MachineEdge {
                        target: MachineBlockId(1),
                        args: Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 4, 9, 9, 0));

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 1);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::IntCompare {
            width: crate::vm::machine::machine_ir::MachineIntWidth::I64,
            dst: MachineReg(7),
            ..
        }
    ));
    assert!(matches!(
        block.terminator,
        MachineTerminator::Branch {
            cond: crate::vm::machine::machine_ir::MachineBranchCond::Value(MachineValue::Reg(
                MachineReg(7)
            )),
            ..
        }
    ));
}

#[test]
fn still_fuses_i32_compare_branch_on_32_bit_targets() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: alloc::vec![MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                        kind: crate::vm::machine::machine_ir::MachineCompareKind::Eq,
                        sign: crate::vm::machine::machine_ir::MachineSign::Unsigned,
                        dst: MachineReg(7),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(0),
                    },
                }],
                terminator: MachineTerminator::Branch {
                    cond: crate::vm::machine::machine_ir::MachineBranchCond::Value(
                        MachineValue::Reg(MachineReg(7)),
                    ),
                    then_edge: MachineEdge {
                        target: MachineBlockId(1),
                        args: Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: MachineBlockId(2),
                        args: Vec::new(),
                    },
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 4, 9, 9, 0));

    let block = &program.blocks[0];
    assert!(block.ops.is_empty());
    assert!(matches!(
        block.terminator,
        MachineTerminator::Branch {
            cond: crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                kind: crate::vm::machine::machine_ir::MachineCompareKind::Eq,
                ..
            },
            ..
        }
    ));
}

#[test]
fn does_not_fold_constant_past_non_adjacent_instruction() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![None],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::FloatConst {
                        width: crate::vm::machine::machine_ir::MachineFloatWidth::F32,
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 4, 9, 10, 1));

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
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(64),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::machine::peephole::optimize(&mut program, test_config(7, 4, 8, 8, 0));

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
