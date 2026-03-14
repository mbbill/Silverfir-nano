use alloc::vec::Vec;

use crate::vm::native::ir::machine::{
    MachineBlock, MachineBlockId, MachineBlockParam, MachineConstData, MachineConstId,
    MachineEdge, MachineExternId, MachineFunction, MachineInst, MachineInstKind, MachineModule,
    MachineProgram, MachineReg, MachineTerminator, MachineValue,
};
use crate::vm::native::ir::runtime::{MachineExternBinding, MachineHelperSymbol};

#[test]
fn rejects_edge_arity_mismatch() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 2,
        reg_count: 2,
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
                params: alloc::vec![MachineBlockParam::gp(MachineReg(0))],
                ops: Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ],
    };

    let err = program.validate().unwrap_err();
    assert!(alloc::format!("{err}").contains("supplies 0 args"));
}

#[test]
fn rejects_out_of_range_register() {
    let program = MachineProgram {
        entry: MachineBlockId(0),
        first_fp_reg: 1,
        reg_count: 1,
        blocks: alloc::vec![MachineBlock {
            id: MachineBlockId(0),
            params: Vec::new(),
            ops: alloc::vec![MachineInst {
                kind: MachineInstKind::Move {
                    dst: MachineReg(1),
                    src: MachineValue::Imm64(0),
                },
            }],
            terminator: MachineTerminator::Return,
        }],
    };

    let err = program.validate().unwrap_err();
    assert!(alloc::format!("{err}").contains("exceeds declared register count"));
}

#[test]
fn rejects_out_of_range_helper_metadata() {
    let module = MachineModule {
        functions: alloc::vec![MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 2,
                reg_count: 2,
                blocks: alloc::vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: Vec::new(),
                    ops: alloc::vec![MachineInst {
                        kind: MachineInstKind::CallHelper(
                            crate::vm::native::ir::machine::MachineHelperCall {
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
