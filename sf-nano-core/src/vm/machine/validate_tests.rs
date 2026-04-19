use crate::collections;
use crate::vm::backend::BackendConfig;

use crate::vm::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBlockParam, MachineCallRuntime, MachineConstData,
    MachineConstId, MachineEdge, MachineFuncId, MachineFunction, MachineInst, MachineInstKind,
    MachineModule, MachineProgram, MachineReg, MachineRegOwner, MachineStorageType,
    MachineTerminator, MachineValue,
};

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
