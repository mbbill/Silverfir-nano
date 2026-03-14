use alloc::vec::Vec;

use crate::vm::native::ir::machine::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineEdge, MachineInst,
    MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineProgram, MachineReg,
    MachineTerminator, MachineValue,
};

#[test]
fn copy_propagates_transient_moves_into_ops_and_edges() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 9,
        reg_count: 9,
        blocks: alloc::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: alloc::vec![
                    MachineInst {
                        kind: MachineInstKind::Move {
                            dst: MachineReg(7),
                            src: MachineValue::Reg(MachineReg(4)),
                        },
                    },
                    MachineInst {
                        kind: MachineInstKind::IntUnary {
                            width: crate::vm::native::ir::machine::MachineIntWidth::I32,
                            op: crate::vm::native::ir::machine::MachineIntUnaryOp::Eqz,
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
                params: alloc::vec![MachineBlockParam::gp(MachineReg(7))],
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 1);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::IntUnary {
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(edge.args, alloc::vec![MachineValue::Reg(MachineReg(4))]);
}

#[test]
fn keeps_cached_local_writes_but_rewrites_their_sources() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 8,
        reg_count: 8,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(4)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        addr: MachineAddr {
                            base: MachineReg(1),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        }],
    };

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

    let block = &program.blocks[0];
    assert_eq!(block.ops.len(), 2);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(4)),
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(5)),
            ..
        }
    ));
}

#[test]
fn constant_folding_keeps_live_constant_when_later_select_reads_and_writes_same_reg() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 8,
        reg_count: 8,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(4),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(5),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(5),
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Select {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Imm64(5),
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
        first_fp_reg: 9,
        reg_count: 9,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
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
                        width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                        op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Imm64(80),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

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
        first_fp_reg: 11,
        reg_count: 12,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(11)),
        }
    ));
}

#[test]
fn does_not_forward_when_stored_source_reg_is_redefined() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 8,
        reg_count: 8,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
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
                        dst: MachineReg(4),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

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
        first_fp_reg: 8,
        reg_count: 8,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

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
        first_fp_reg: 10,
        reg_count: 10,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
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
                        width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                        op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                        dst: MachineReg(8),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Imm64(16),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

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
fn reuses_identical_loads_from_fp_into_gp_move() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 11,
        reg_count: 12,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(11)),
        }
    ));
}

#[test]
fn does_not_reuse_load_after_loaded_reg_is_redefined() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 9,
        reg_count: 9,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Load {
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
                        dst: MachineReg(7),
                        src: MachineValue::Imm64(0),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

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
fn preserves_transient_move_when_source_reg_is_redefined_before_terminator_use() {
    let mut program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 8,
        reg_count: 8,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![
                MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(7),
                        src: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(5),
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

    crate::vm::native::ir::machine::peephole::optimize(&mut program, 7);

    let block = &program.blocks[0];
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(7),
            src: MachineValue::Reg(MachineReg(5)),
        }
    ));
    let MachineTerminator::Jump(edge) = &block.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(edge.args, alloc::vec![MachineValue::Reg(MachineReg(7))]);
}
