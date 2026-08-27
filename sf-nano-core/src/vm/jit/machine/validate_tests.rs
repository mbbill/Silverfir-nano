use crate::collections;
use crate::vm::jit::backend::BackendConfig;

use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineArgSrc, MachineBlock, MachineBlockId, MachineBlockParam, MachineCallArgs,
    MachineCallLaneArg, MachineCallRuntime, MachineCallTarget, MachineCompareKind,
    MachineConstData, MachineConstId, MachineEdge, MachineFuncId, MachineFunction, MachineInst,
    MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
    MachineModule, MachineProgram, MachineReg, MachineRegOwner, MachineResultSrc,
    MachineReturnValue, MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind,
    MachineValue, MACHINE_FP_REG,
};
use crate::vm::jit::middle::frame::FrameSlot;

/// Minimal config for validate tests: no extra dynamic budget beyond the minimum.
fn minimal_config() -> BackendConfig {
    let gp_unit_bytes = core::mem::size_of::<usize>() as u8;
    BackendConfig::new(
        gp_unit_bytes,
        if gp_unit_bytes == 4 { 5 } else { 3 },
        0,
        if gp_unit_bytes == 4 { 8 } else { 3 },
    )
}

fn program_with_inst(kind: MachineInstKind) -> MachineProgram {
    MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![MachineInst { kind }],
            terminator: MachineTerminator::Return,
        }],
    }
}

#[test]
fn rejects_edge_arity_mismatch() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::Vec::new(),
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(0))],
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("wrong number of args"));
}

#[test]
fn rejects_out_of_range_register() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(99),
                    src: MachineValue::Imm64(0),
                },
            }],
            terminator: MachineTerminator::Return,
        }],
    };

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("exceeds declared register count"));
}

#[test]
fn rejects_negative_frame_address() {
    let width = if core::mem::size_of::<usize>() == 4 {
        MachineMemWidth::U32
    } else {
        MachineMemWidth::U64
    };
    let program = program_with_inst(MachineInstKind::Load {
        owner: MachineRegOwner::LinearValue,
        ty: MachineStorageType::GpWord,
        dst: MachineReg(4),
        addr: MachineAddr {
            base: MACHINE_FP_REG,
            offset: -8,
        },
        width,
        extension: MachineLoadExtension::None,
    });

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("negative offset"));
}

#[test]
fn rejects_frame_pointer_as_an_ordinary_value() {
    let program = program_with_inst(MachineInstKind::Move {
        owner: MachineRegOwner::LinearValue,
        ty: MachineStorageType::GpWord,
        dst: MachineReg(4),
        src: MachineValue::Reg(MACHINE_FP_REG),
    });

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("native-stack guard lhs"));
}

#[test]
fn rejects_frame_pointer_redefinition() {
    let program = program_with_inst(MachineInstKind::Move {
        owner: MachineRegOwner::LinearValue,
        ty: MachineStorageType::GpWord,
        dst: MACHINE_FP_REG,
        src: MachineValue::Imm64(0),
    });

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("cannot redefine the frame pointer"));
}

#[test]
fn accepts_frame_pointer_as_the_exact_stack_guard_lhs() {
    let width = if core::mem::size_of::<usize>() == 4 {
        MachineIntWidth::I32
    } else {
        MachineIntWidth::I64
    };
    let program = program_with_inst(MachineInstKind::TrapIf {
        kind: MachineTrapKind::StackOverflow,
        cond: crate::vm::jit::machine::machine_ir::MachineBranchCond::IntCompare {
            width,
            kind: MachineCompareKind::Gt,
            sign: MachineSign::Unsigned,
            lhs: MachineValue::Reg(MACHINE_FP_REG),
            rhs: MachineValue::Reg(MachineReg(4)),
        },
    });

    program.validate(minimal_config()).unwrap();
}

#[test]
fn rejects_deriving_a_dynamic_value_from_the_frame_pointer() {
    let width = if core::mem::size_of::<usize>() == 4 {
        MachineIntWidth::I32
    } else {
        MachineIntWidth::I64
    };
    let program = program_with_inst(MachineInstKind::IntBinary {
        width,
        op: MachineIntBinaryOp::Add,
        dst: MachineReg(4),
        lhs: MachineValue::Reg(MACHINE_FP_REG),
        rhs: MachineValue::Imm64(8),
    });

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("native-stack guard lhs"));
}

#[test]
fn rejects_an_inexact_frame_pointer_stack_guard() {
    let width = if core::mem::size_of::<usize>() == 4 {
        MachineIntWidth::I32
    } else {
        MachineIntWidth::I64
    };
    let program = program_with_inst(MachineInstKind::TrapIf {
        kind: MachineTrapKind::StackOverflow,
        cond: crate::vm::jit::machine::machine_ir::MachineBranchCond::IntCompare {
            width,
            kind: MachineCompareKind::Ge,
            sign: MachineSign::Unsigned,
            lhs: MachineValue::Reg(MACHINE_FP_REG),
            rhs: MachineValue::Reg(MachineReg(4)),
        },
    });

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("native-stack guard lhs"));
}

#[test]
fn rejects_the_frame_pointer_reserved_on_an_edge() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: collections::vec![MachineValue::ReservedReg(MACHINE_FP_REG)],
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![MachineBlockParam::gp_word(MachineReg(4))
                    .with_owner(MachineRegOwner::CachedCell),],
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("cannot be a reserved edge value"));
}

#[test]
fn rejects_return_source_before_the_current_frame() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::Vec::new(),
            terminator: MachineTerminator::ReturnScalar {
                value: MachineReturnValue::ScalarGp {
                    src: MachineResultSrc::FrameSlotOffset {
                        slot: FrameSlot(0),
                        byte_offset: -1,
                    },
                    ty: MachineStorageType::GpWord,
                },
            },
        }],
    };

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("negative byte offset"));
}

#[test]
fn rejects_call_argument_source_before_the_current_frame() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: collections::vec![],
        blocks: collections::vec![MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::Vec::new(),
            terminator: MachineTerminator::TailCall {
                target: MachineCallTarget::Direct(MachineFuncId(0)),
                args: MachineCallArgs {
                    frame_params: Default::default(),
                    lane_args: collections::vec![MachineCallLaneArg::Gp {
                        param_index: 0,
                        lane: 0,
                        src: MachineArgSrc::FrameSlotOffset {
                            slot: FrameSlot(0),
                            byte_offset: -1,
                        },
                        ty: MachineStorageType::GpWord,
                    }],
                },
            },
        }],
    };

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(err.message().contains("negative byte offset"));
}

#[test]
fn rejects_out_of_range_helper_metadata() {
    let module = MachineModule {
        config: minimal_config(),
        functions: collections::vec![MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                fp_reg_init_widths: collections::vec![],
                blocks: collections::vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::Vec::new(),
                    ops: collections::vec![MachineInst {
                        kind: MachineInstKind::CallRuntime(MachineCallRuntime {
                            metadata: MachineConstId(1),
                        }),
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
            preserved_clobbers: collections::Vec::new(),
        }],
        consts: collections::vec![MachineConstData {
            id: MachineConstId(0),
            align: 8,
            bytes: collections::vec![0; 8],
        }],
    };

    let err = module.validate().unwrap_err();
    assert!(err.message().contains("out-of-range const"));
}
