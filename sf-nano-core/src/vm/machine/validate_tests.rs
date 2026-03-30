use alloc::{vec, vec::Vec};

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBlockParam, MachineConstData, MachineConstId, MachineEdge,
    MachineExternId, MachineFunction, MachineInst, MachineInstKind, MachineModule, MachineProgram,
    MachineReg, MachineStorageType, MachineTerminator, MachineValue,
};
use crate::vm::machine::machine_ir::{MachineExternBinding, MachineHelperSymbol};

/// Minimal config for validate tests: no cache/transient budget beyond the minimum.
fn minimal_config() -> BackendConfig {
    BackendConfig::new(
        0,
        1,
        0,
        0,
        core::mem::size_of::<usize>() as u8,
        if core::mem::size_of::<usize>() == 4 {
            8
        } else {
            3
        },
    )
}

#[test]
fn rejects_edge_arity_mismatch() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: MachineTerminator::Jump(MachineEdge {
                    target: MachineBlockId(1),
                    args: Vec::new(),
                }),
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: alloc::vec![MachineBlockParam::gp_word(MachineReg(0))],
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(alloc::format!("{err}").contains("supplies 0 args"));
}

#[test]
fn rejects_out_of_range_register() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        fp_reg_init_widths: vec![],
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![MachineInst {
                kind: MachineInstKind::Move {
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(99),
                    src: MachineValue::Imm64(0),
                },
            }],
            terminator: MachineTerminator::Return,
        }],
    };

    let err = program.validate(minimal_config()).unwrap_err();
    assert!(alloc::format!("{err}").contains("exceeds declared register count"));
}

#[test]
fn rejects_out_of_range_helper_metadata() {
    let module = MachineModule {
        config: minimal_config(),
        functions: alloc::vec![MachineFunction {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                fp_reg_init_widths: vec![],
                blocks: alloc::vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: Vec::new(),
                    ops: alloc::vec![MachineInst {
                        kind: MachineInstKind::CallHelper(
                            crate::vm::machine::machine_ir::MachineHelperCall {
                                target: MachineExternId(0),
                                metadata: MachineConstId(1),
                            },
                        ),
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        }],
        consts: alloc::vec![MachineConstData {
            id: MachineConstId(0),
            align: 8,
            bytes: alloc::vec![0; 8],
        }],
        externs: alloc::vec![MachineExternBinding {
            id: MachineExternId(0),
            symbol: MachineHelperSymbol::MemoryGrow,
        }],
    };

    let err = module.validate().unwrap_err();
    assert!(alloc::format!("{err}").contains("out-of-range const 1"));
}
