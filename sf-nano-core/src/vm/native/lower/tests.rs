use alloc::vec::Vec;

use crate::value_type::ValueType;
use crate::vm::{
    backend::BackendConfig,
    lir::{
        ir::{
            CachedLocalInfo, LirBinding, LirBlock, LirBoundaryOp, LirEdge, LirInst, LirInstKind,
            LirLocalCachePrefs, LirProgram, LirTerminator, LirValue,
        },
        leaf::LirLeafOp,
        target::LirTarget,
    },
    native::{
        ir::machine::{
            MachineBlockId, MachineCompareKind, MachineFloatWidth, MachineFunction,
            MachineInstKind, MachineIntBinaryOp, MachineMemWidth, MachineModule, MachineReg,
            MachineStorageType, MachineTerminator, MachineValue, MACHINE_FIXED_REG_COUNT,
        },
        ir::runtime::MachineHelperSymbol,
        lower::{lower_module, LowerFunctionInput, LowerModuleInput},
    },
    plan::frame::plan_frame_layout,
    wasm::primitive_op::PrimitiveOpKind,
};

fn assert_valid_32bit_gp_target(module: &MachineModule, backend: BackendConfig) {
    assert!(backend.is_32bit_gp_target());
    let max_gp_regs = MACHINE_FIXED_REG_COUNT
        + backend.gp_local_cache_budget as u16
        + backend.gp_transient_budget as u16;
    module
        .validate_32bit_gp_target(max_gp_regs)
        .unwrap_or_else(|err| panic!("32-bit lowered module must already validate: {err}"));
}

#[test]
fn lowers_simple_slot_and_add_block() {
    let frame = plan_frame_layout(1, 4, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: frame.local_slot(0),
                        dst: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                        args: alloc::vec![LirValue(0), LirValue(1)],
                        results: alloc::vec![LirValue(2)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(2),
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let MachineModule { functions, .. } = lowered.module;
    let MachineFunction { program, .. } = &functions[0];
    assert_eq!(lowered.runtime.call_link.slot_count, 3);
    assert_eq!(lowered.runtime.functions.len(), 1);
    assert_eq!(lowered.runtime.functions[0].frame_prefix_slots, 1);
    assert_eq!(
        lowered.runtime.functions[0].total_frame_slots,
        frame.total_slots()
    );
    assert_eq!(
        lowered.runtime.functions[0].call_scratch,
        Some(crate::vm::native::ir::runtime::MachineFrameRegion {
            base_slot: frame.call_scratch.unwrap().start.0,
            slots: frame.call_scratch.unwrap().count,
        })
    );
    assert_eq!(lowered.runtime.functions[0].helper_scratch, None);
    assert_eq!(lowered.runtime.functions[0].return_results, None);
    assert_eq!(program.entry, MachineBlockId(0));
    assert_eq!(program.blocks.len(), 1);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Return
    ));
    assert_eq!(program.blocks[0].ops.len(), 4);
    assert!(matches!(
        program.blocks[0].ops[0].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        program.blocks[0].ops[1].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(1),
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].ops[2].kind,
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].ops[3].kind,
        MachineInstKind::Store { .. }
    ));
}

#[test]
fn lowers_select_with_wasm_operand_order() {
    let frame = plan_frame_layout(0, 3, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 11 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 22 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(2)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::Select).unwrap(),
                        args: alloc::vec![LirValue(0), LirValue(1), LirValue(2)],
                        results: alloc::vec![LirValue(3)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 3, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("select lowering should succeed");

    let block = &lowered.module.functions[0].program.blocks[0];
    let first_reg = match &block.ops[0].kind {
        MachineInstKind::Move { dst, .. } => *dst,
        other => panic!("expected first const move, got {other:?}"),
    };
    let second_reg = match &block.ops[1].kind {
        MachineInstKind::Move { dst, .. } => *dst,
        other => panic!("expected second const move, got {other:?}"),
    };
    let select = match &block.ops[3].kind {
        MachineInstKind::Select {
            on_true, on_false, ..
        } => (*on_true, *on_false),
        other => panic!("expected machine select, got {other:?}"),
    };
    assert_eq!(select.0, MachineValue::Reg(first_reg));
    assert_eq!(select.1, MachineValue::Reg(second_reg));
}

#[test]
fn native_backend_requires_at_least_one_gp_transient_register() {
    let frame = plan_frame_layout(0, 0, 0);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let err = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 0, 0, 0),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect_err("zero-budget native backend should be rejected");

    assert!(alloc::format!("{err}").contains("at least one GP transient register"));
}

#[test]
fn projects_return_results_and_helper_scratch_from_frame_plan() {
    let frame = plan_frame_layout(2, 6, 6);
    let result_span = crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![],
            terminator: LirTerminator::Return {
                results: Some(result_span),
            },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let runtime = &lowered.runtime;
    assert_eq!(runtime.call_link.caller_result_base_offset, 16);
    assert_eq!(
        runtime.functions[0].helper_scratch,
        Some(crate::vm::native::ir::runtime::MachineFrameRegion {
            base_slot: frame.call_scratch.unwrap().start.0 + runtime.call_link.slot_count,
            slots: frame.call_scratch.unwrap().count - runtime.call_link.slot_count,
        })
    );
    assert_eq!(
        runtime.functions[0].return_results,
        Some(crate::vm::native::ir::runtime::MachineFrameRegion {
            base_slot: result_span.start.0,
            slots: result_span.count,
        })
    );
}

#[test]
fn rejects_inconsistent_return_result_spans() {
    let frame = plan_frame_layout(0, 4, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![
            LirBlock {
                id: LirTarget(0),
                params: alloc::vec![],
                ops: alloc::vec![],
                terminator: LirTerminator::Return {
                    results: Some(crate::vm::plan::frame::FrameSpan::new(
                        frame.operand_slot(0),
                        1
                    )),
                },
            },
            LirBlock {
                id: LirTarget(1),
                params: alloc::vec![],
                ops: alloc::vec![],
                terminator: LirTerminator::Return {
                    results: Some(crate::vm::plan::frame::FrameSpan::new(
                        frame.operand_slot(1),
                        1
                    )),
                },
            },
        ],
        value_types: alloc::vec![],
    };

    let err = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect_err("inconsistent return spans should be rejected");

    assert!(err.message().contains("inconsistent return result spans"));
}

#[test]
fn rejects_mixed_void_and_value_returns() {
    let frame = plan_frame_layout(0, 4, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![
            LirBlock {
                id: LirTarget(0),
                params: alloc::vec![],
                ops: alloc::vec![],
                terminator: LirTerminator::Return { results: None },
            },
            LirBlock {
                id: LirTarget(1),
                params: alloc::vec![],
                ops: alloc::vec![],
                terminator: LirTerminator::Return {
                    results: Some(crate::vm::plan::frame::FrameSpan::new(
                        frame.operand_slot(0),
                        1,
                    )),
                },
            },
        ],
        value_types: alloc::vec![],
    };

    let err = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect_err("mixed void and value returns should be rejected");

    assert!(err.message().contains("inconsistent return result spans"));
}

#[test]
fn lowers_branch_edge_bindings_into_machine_edge_args() {
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![
            LirBlock {
                id: LirTarget(0),
                params: alloc::vec![LirValue(0), LirValue(1)],
                ops: alloc::vec![],
                terminator: LirTerminator::Goto(LirEdge {
                    target: LirTarget(1),
                    bindings: alloc::vec![
                        LirBinding {
                            param: LirValue(3),
                            value: LirValue(1),
                        },
                        LirBinding {
                            param: LirValue(2),
                            value: LirValue(0),
                        },
                    ],
                }),
            },
            LirBlock {
                id: LirTarget(1),
                params: alloc::vec![LirValue(2), LirValue(3)],
                ops: alloc::vec![],
                terminator: LirTerminator::Return { results: None },
            },
        ],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame: plan_frame_layout(0, 2, 2),
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let term = &lowered.module.functions[0].program.blocks[0].terminator;
    let MachineTerminator::Jump(edge) = term else {
        panic!("expected jump terminator");
    };
    assert_eq!(edge.target, MachineBlockId(1));
    assert_eq!(edge.args.len(), 2);
    assert_eq!(edge.args[0], MachineValue::Reg(MachineReg(4)));
    assert_eq!(edge.args[1], MachineValue::Reg(MachineReg(5)));
}

#[test]
fn lowers_i64_branch_params_and_edge_args_as_gp_word_pairs_on_32bit_targets() {
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![
            LirBlock {
                id: LirTarget(0),
                params: alloc::vec![LirValue(0)],
                ops: alloc::vec![],
                terminator: LirTerminator::Goto(LirEdge {
                    target: LirTarget(1),
                    bindings: alloc::vec![LirBinding {
                        param: LirValue(1),
                        value: LirValue(0),
                    }],
                }),
            },
            LirBlock {
                id: LirTarget(1),
                params: alloc::vec![LirValue(1)],
                ops: alloc::vec![],
                terminator: LirTerminator::Return { results: None },
            },
        ],
        value_types: alloc::vec![ValueType::I64, ValueType::I64],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame: plan_frame_layout(0, 2, 2),
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 param lowering should succeed");

    let block0 = &lowered.module.functions[0].program.blocks[0];
    assert_eq!(block0.params.len(), 2);
    assert!(matches!(
        block0.params[0],
        crate::vm::native::ir::machine::MachineBlockParam {
            reg: MachineReg(4),
            ty: MachineStorageType::GpWord,
        }
    ));
    assert!(matches!(
        block0.params[1],
        crate::vm::native::ir::machine::MachineBlockParam {
            reg: MachineReg(5),
            ty: MachineStorageType::GpWord,
        }
    ));

    let MachineTerminator::Jump(edge) = &block0.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        alloc::vec![
            MachineValue::Reg(MachineReg(4)),
            MachineValue::Reg(MachineReg(5))
        ]
    );

    let block1 = &lowered.module.functions[0].program.blocks[1];
    assert_eq!(block1.params.len(), 2);
    assert!(matches!(block1.params[0].ty, MachineStorageType::GpWord));
    assert!(matches!(block1.params[1].ty, MachineStorageType::GpWord));
}

#[test]
fn lowers_i64_slot_and_pair_arithmetic_directly_to_legal_32bit_machineir() {
    let backend = BackendConfig::new_with_gp_unit_bytes(0, 6, 0, 2, 4);
    let frame = plan_frame_layout(1, 4, 4);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x0123_4567_89ab_cdef,
                        })
                        .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: frame.local_slot(0),
                        dst: LirValue(1),
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x1111_2222_3333_4444,
                        })
                        .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(2)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Add).unwrap(),
                        args: alloc::vec![LirValue(1), LirValue(2)],
                        results: alloc::vec![LirValue(3)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![
            ValueType::I64,
            ValueType::I64,
            ValueType::I64,
            ValueType::I64
        ],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 slot and add lowering should succeed");

    assert_valid_32bit_gp_target(&lowered.module, backend);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Int64PairBinary {
                op: MachineIntBinaryOp::Add,
                ..
            }
        )
    }));
}

#[test]
fn lowers_i64_global_get_set_directly_to_legal_32bit_machineir() {
    let backend = BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4);
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::GlobalSet { idx: 3 })
                            .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::GlobalGet { idx: 3 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![ValueType::I64, ValueType::I64],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit global get/set lowering should succeed");

    assert_valid_32bit_gp_target(&lowered.module, backend);
}

#[test]
fn lowers_i64_memory_load_store_directly_to_legal_32bit_machineir() {
    let backend = BackendConfig::new_with_gp_unit_bytes(0, 5, 0, 2, 4);
    let frame = plan_frame_layout(0, 3, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x8877_6655_4433_2211,
                        })
                        .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Store {
                            offset: 0,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(0), LirValue(1)],
                        results: alloc::vec![],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(2)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Load {
                            offset: 0,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(2)],
                        results: alloc::vec![LirValue(3)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![
            ValueType::I32,
            ValueType::I64,
            ValueType::I32,
            ValueType::I64
        ],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 memory load/store lowering should succeed");

    assert_valid_32bit_gp_target(&lowered.module, backend);
}

#[test]
fn lowers_direct_local_call_to_legal_32bit_machineir() {
    let backend = BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4);
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::CallInternal {
                    callee: 1,
                    args: crate::vm::plan::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::plan::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1,
                    ),
                    skip_reload: alloc::vec![],
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };
    let callee = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(0),
                frame: caller_frame,
                lir: &caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(1),
                frame: callee_frame,
                lir: &callee,
                result_count: 0,
            },
        ],
    })
    .expect("32-bit direct local call lowering should succeed");

    assert_valid_32bit_gp_target(&lowered.module, backend);
}

#[test]
fn lowers_cached_local_reads_and_writes_through_cache_regs() {
    let frame = plan_frame_layout(1, 2, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![frame.local_slot(0)],
            gp_preferred_types: alloc::vec![ValueType::I32],
            fp_preferred_slots: alloc::vec![],
            fp_preferred_types: alloc::vec![],
            gp_local_info: alloc::vec![CachedLocalInfo {
                is_param: true,
                reads_before_write: true
            }],
            fp_local_info: alloc::vec![],
        },
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: frame.local_slot(0),
                        dst: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 7 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(1),
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    // ops[0]: entry cache init — load param from frame into cache reg
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    // ops[1]: LoadSlot reads cached local → move from cache reg
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    // ops[2]: I32Const(7) coalesced directly into cache reg via StoreSlot
    assert!(matches!(
        ops[2].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            src: MachineValue::Imm64(7),
            ..
        }
    ));
}

#[test]
fn does_not_zero_unread_cached_locals_at_entry_on_32bit_targets() {
    let backend = BackendConfig::new_with_gp_unit_bytes(1, 4, 0, 2, 4);
    let frame = plan_frame_layout(1, 2, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![frame.local_slot(0)],
            gp_preferred_types: alloc::vec![ValueType::I32],
            fp_preferred_slots: alloc::vec![],
            fp_preferred_types: alloc::vec![],
            gp_local_info: alloc::vec![CachedLocalInfo {
                is_param: false,
                reads_before_write: false,
            }],
            fp_local_info: alloc::vec![],
        },
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 9 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![ValueType::I32],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit lowering should not zero unread cached locals");

    assert_valid_32bit_gp_target(&lowered.module, backend);

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 1);
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            src: MachineValue::Imm64(9),
            ..
        }
    ));
}

#[test]
fn lowers_runtime_memory_grow_through_frame_metadata() {
    let frame = plan_frame_layout(0, 1, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::MemoryGrow {
                    mem_idx: 0,
                    io: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 1),
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("runtime helper lowering should succeed");

    assert_eq!(lowered.module.externs.len(), 1);
    assert_eq!(
        lowered.module.externs[0].symbol,
        MachineHelperSymbol::MemoryGrow
    );
    assert_eq!(lowered.module.consts.len(), 1);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0].kind, MachineInstKind::CallHelper(_)));
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_memory_copy_through_frame_metadata() {
    let frame = plan_frame_layout(0, 3, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::MemoryCopy {
                    dst_mem_idx: 0,
                    src_mem_idx: 1,
                    args: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 3),
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("memory.copy helper lowering should succeed");

    assert_eq!(lowered.module.externs.len(), 1);
    assert_eq!(
        lowered.module.externs[0].symbol,
        MachineHelperSymbol::MemoryCopy
    );
    assert_eq!(lowered.module.consts.len(), 1);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0].kind, MachineInstKind::CallHelper(_)));
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_table_fill_through_frame_metadata() {
    let frame = plan_frame_layout(0, 3, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::TableFill {
                    table_idx: 2,
                    args: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 3),
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("table.fill helper lowering should succeed");

    assert_eq!(lowered.module.externs.len(), 1);
    assert_eq!(
        lowered.module.externs[0].symbol,
        MachineHelperSymbol::TableFill
    );
    assert_eq!(lowered.module.consts.len(), 1);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0].kind, MachineInstKind::CallHelper(_)));
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_call_external_through_frame_metadata_without_helper_scratch() {
    let frame = plan_frame_layout(0, 2, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::CallExternal {
                    func_idx: 7,
                    args: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 2),
                    results: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 1),
                    skip_reload: alloc::vec![],
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("external helper lowering should succeed");

    assert_eq!(lowered.module.externs.len(), 1);
    assert_eq!(
        lowered.module.externs[0].symbol,
        MachineHelperSymbol::CallExternal
    );
    assert_eq!(lowered.module.consts.len(), 1);
    assert!(lowered.runtime.functions[0].helper_scratch.is_none());
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0].kind, MachineInstKind::CallHelper(_)));
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Load { .. }));
}

#[test]
fn coalesces_dead_i64_const_directly_into_uncached_store_slot() {
    let frame = plan_frame_layout(0, 1, 1);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x0123_4567_89ab_cdef,
                        })
                        .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
            ],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![ValueType::I64],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("uncached const store lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 1);
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Store {
            ty: MachineStorageType::GpI64,
            src: MachineValue::Imm64(0x0123_4567_89ab_cdef),
            ..
        }
    ));
}

#[test]
fn flushes_and_reloads_cached_locals_around_call_external() {
    let frame = plan_frame_layout(1, 2, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![frame.local_slot(0)],
            gp_preferred_types: alloc::vec![ValueType::I32],
            fp_preferred_slots: alloc::vec![],
            fp_preferred_types: alloc::vec![],
            gp_local_info: alloc::vec![CachedLocalInfo {
                is_param: true,
                reads_before_write: true
            }],
            fp_local_info: alloc::vec![],
        },
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::Boundary(LirBoundaryOp::CallExternal {
                        func_idx: 7,
                        args: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 1),
                        results: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 0),
                        skip_reload: alloc::vec![],
                    }),
                },
            ],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("external helper lowering should succeed with cached locals");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 7);
    // ops[0]: entry cache init — load param from frame
    assert!(matches!(ops[0].kind, MachineInstKind::Load { .. }));
    // ops[1]: I64Const(9) coalesced directly into cache reg via StoreSlot
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(9),
            ..
        }
    ));
    // ops[2]: flush cache to frame before external call
    assert!(matches!(ops[2].kind, MachineInstKind::Store { .. }));
    // ops[3]: external call
    assert!(matches!(ops[3].kind, MachineInstKind::CallHelper(_)));
    // ops[4-5]: reload mem0 cache regs
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[5].kind, MachineInstKind::Load { .. }));
    // ops[6]: reload cached local after call
    assert!(matches!(ops[6].kind, MachineInstKind::Load { .. }));
}

#[test]
fn flushes_and_reloads_cached_locals_around_runtime_helpers() {
    let frame = plan_frame_layout(1, 2, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![frame.local_slot(0)],
            gp_preferred_types: alloc::vec![ValueType::I32],
            fp_preferred_slots: alloc::vec![],
            fp_preferred_types: alloc::vec![],
            gp_local_info: alloc::vec![CachedLocalInfo {
                is_param: true,
                reads_before_write: true
            }],
            fp_local_info: alloc::vec![],
        },
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 5 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::Boundary(LirBoundaryOp::MemoryGrow {
                        mem_idx: 0,
                        io: crate::vm::plan::frame::FrameSpan::new(frame.operand_slot(0), 1),
                    }),
                },
            ],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("runtime helper lowering should succeed with cached locals");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 7);
    // ops[0]: entry cache init — load param from frame
    assert!(matches!(ops[0].kind, MachineInstKind::Load { .. }));
    // ops[1]: I64Const(5) coalesced directly into cache reg via StoreSlot
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(5),
            ..
        }
    ));
    // ops[2]: flush cache to frame before runtime helper
    assert!(matches!(ops[2].kind, MachineInstKind::Store { .. }));
    // ops[3]: runtime helper call (memory grow)
    assert!(matches!(ops[3].kind, MachineInstKind::CallHelper(_)));
    // ops[4-5]: reload mem0 cache regs
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[5].kind, MachineInstKind::Load { .. }));
    // ops[6]: reload cached local after call
    assert!(matches!(ops[6].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_direct_local_call_with_continuation_block() {
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::CallInternal {
                    callee: 1,
                    args: crate::vm::plan::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::plan::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1
                    ),
                    skip_reload: alloc::vec![],
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };
    let callee = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(0),
                frame: caller_frame,
                lir: &caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(1),
                frame: callee_frame,
                lir: &callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call lowering should succeed");

    let caller_program = &lowered.module.functions[0].program;
    assert_eq!(caller_program.blocks.len(), 2);
    let call_block = &caller_program.blocks[0];
    let callee_frame_base = match call_block.terminator {
        MachineTerminator::CallDirect {
            callee,
            callee_frame_base,
            continuation,
        } => {
            assert_eq!(callee, crate::vm::native::ir::machine::MachineFuncId(1));
            assert_eq!(continuation, MachineBlockId(1));
            callee_frame_base
        }
        ref other => panic!("expected direct call terminator, got {other:?}"),
    };
    assert!(matches!(
        call_block.ops[0].kind,
        MachineInstKind::IntBinary {
            dst,
            lhs: MachineValue::Reg(MachineReg(1)),
            rhs: MachineValue::Imm64(offset),
            ..
        } if dst == callee_frame_base && offset == u64::from(caller_frame.operand_slot(1).0) * 8
    ));
    assert_eq!(call_block.ops.len(), 8);
    assert!(matches!(
        call_block.ops[1].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        call_block.ops[2].kind,
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Sub,
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[3].kind,
        MachineInstKind::TrapIf {
            cond:
                crate::vm::native::ir::machine::MachineBranchCond::IntCompare {
                    width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                    kind: MachineCompareKind::Gt,
                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                    lhs: MachineValue::Reg(lhs),
                    ..
                },
            kind: crate::vm::native::ir::machine::MachineTrapKind::StackOverflow,
        } if lhs == callee_frame_base
    ));
    assert!(matches!(
        call_block.ops[4].kind,
        MachineInstKind::Store {
            src: MachineValue::Imm64(0),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[5].kind,
        MachineInstKind::Store {
            src: MachineValue::Imm64(1),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[6].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(1)),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[7].kind,
        MachineInstKind::Store {
            src: MachineValue::Imm64(40),
            ..
        }
    ));

    let continuation = &caller_program.blocks[1];
    assert!(continuation.params.is_empty());
    assert!(continuation.ops.is_empty());
    assert!(matches!(
        continuation.terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::Unreachable
        }
    ));
}

#[test]
fn flushes_cached_local_before_second_direct_call() {
    let caller_frame = plan_frame_layout(2, 1, 4);
    let callee_frame = plan_frame_layout(0, 1, 4);

    let caller = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![caller_frame.local_slot(0)],
            gp_preferred_types: alloc::vec![ValueType::I32],
            fp_preferred_slots: alloc::vec![],
            fp_preferred_types: alloc::vec![],
            gp_local_info: alloc::vec![CachedLocalInfo {
                is_param: true,
                reads_before_write: true
            }],
            fp_local_info: alloc::vec![],
        },
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Boundary(LirBoundaryOp::CallInternal {
                        callee: 1,
                        args: crate::vm::plan::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            0,
                        ),
                        results: crate::vm::plan::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            1,
                        ),
                        skip_reload: alloc::vec![],
                    }),
                },
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: caller_frame.operand_slot(0),
                        dst: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: caller_frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::Boundary(LirBoundaryOp::CallInternal {
                        callee: 1,
                        args: crate::vm::plan::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            0,
                        ),
                        results: crate::vm::plan::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            1,
                        ),
                        skip_reload: alloc::vec![],
                    }),
                },
            ],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };
    let callee = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(0),
                frame: caller_frame,
                lir: &caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(1),
                frame: callee_frame,
                lir: &callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call lowering should succeed with cached locals");

    let caller_program = &lowered.module.functions[0].program;
    assert_eq!(caller_program.blocks.len(), 3);

    let second_call_block = &caller_program.blocks[1];
    // ops[0]: reload cached local after first call
    assert!(matches!(
        second_call_block.ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    // ops[1]: LoadSlot coalesced with StoreSlot directly into cache reg
    assert!(matches!(
        second_call_block.ops[1].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    // ops[2]: flush cache to frame before second call
    assert!(matches!(
        second_call_block.ops[2].kind,
        MachineInstKind::Store {
            addr: crate::vm::native::ir::machine::MachineAddr {
                base: MachineReg(1),
                offset: 0,
            },
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        second_call_block.terminator,
        MachineTerminator::CallDirect {
            callee: crate::vm::native::ir::machine::MachineFuncId(1),
            continuation: MachineBlockId(2),
            ..
        }
    ));
}

#[test]
fn preserves_cached_locals_across_block_edges() {
    let frame = plan_frame_layout(1, 1, 4);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![frame.local_slot(0)],
            gp_preferred_types: alloc::vec![ValueType::I32],
            fp_preferred_slots: alloc::vec![],
            fp_preferred_types: alloc::vec![],
            gp_local_info: alloc::vec![CachedLocalInfo {
                is_param: true,
                reads_before_write: true
            }],
            fp_local_info: alloc::vec![],
        },
        blocks: alloc::vec![
            LirBlock {
                id: LirTarget(0),
                params: alloc::vec![],
                ops: alloc::vec![
                    LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                                .unwrap(),
                            args: alloc::vec![],
                            results: alloc::vec![LirValue(0)],
                        },
                    },
                    LirInst {
                        kind: LirInstKind::StoreSlot {
                            slot: frame.local_slot(0),
                            src: LirValue(0),
                        },
                    },
                ],
                terminator: LirTerminator::Goto(LirEdge {
                    target: LirTarget(1),
                    bindings: alloc::vec![],
                }),
            },
            LirBlock {
                id: LirTarget(1),
                params: alloc::vec![],
                ops: alloc::vec![
                    LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: frame.local_slot(0),
                            dst: LirValue(1),
                        },
                    },
                    LirInst {
                        kind: LirInstKind::StoreSlot {
                            slot: frame.local_slot(0),
                            src: LirValue(1),
                        },
                    },
                ],
                terminator: LirTerminator::Return { results: None },
            },
        ],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("block-edge cache preservation lowering should succeed");

    let program = &lowered.module.functions[0].program;
    // Block 0: entry init + I64Const coalesced with StoreSlot
    assert!(matches!(
        program.blocks[0].ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    // I64Const(9) coalesced directly into cache reg via StoreSlot
    assert!(matches!(
        program.blocks[0].ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            src: MachineValue::Imm64(9),
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Jump(_)
    ));
    // Block 1: cached local preserved across edge — no reload needed
    assert!(matches!(
        program.blocks[1].ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        program.blocks[1].ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            src: MachineValue::Reg(MachineReg(5)),
            ..
        }
    ));
}

#[test]
fn rejects_cache_store_with_incompatible_gp_storage_types() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(1, 2, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![frame.local_slot(0)],
            gp_preferred_types: alloc::vec![ValueType::I32],
            fp_preferred_slots: alloc::vec![],
            fp_preferred_types: alloc::vec![],
            gp_local_info: alloc::vec![CachedLocalInfo {
                is_param: false,
                reads_before_write: false,
            }],
            fp_local_info: alloc::vec![],
        },
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![ValueType::I64],
    };

    let err = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(1, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect_err("typed i64 store into cached i32 local must be rejected");

    let message = alloc::format!("{err}");
    assert!(
        message.contains("StoreSlot src for cached local slot FrameSlot(0)")
            || message.contains("typed LIR store to cached local slot"),
        "unexpected error: {message}"
    );
}

#[test]
fn lowers_direct_local_call_with_sparse_machine_function_ids() {
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::CallInternal {
                    callee: 2,
                    args: crate::vm::plan::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::plan::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1,
                    ),
                    skip_reload: alloc::vec![],
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };
    let callee = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(0),
                frame: caller_frame,
                lir: &caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(2),
                frame: callee_frame,
                lir: &callee,
                result_count: 0,
            },
        ],
    })
    .expect("sparse-id local call lowering should succeed");

    assert_eq!(lowered.module.functions.len(), 3);
    assert_eq!(lowered.runtime.functions.len(), 3);
    assert!(matches!(
        lowered.module.functions[1].program.blocks[0].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::Unreachable
        }
    ));
    assert!(matches!(
        lowered.module.functions[0].program.blocks[0].terminator,
        MachineTerminator::CallDirect {
            callee: crate::vm::native::ir::machine::MachineFuncId(2),
            ..
        }
    ));
}

#[test]
fn lowers_memory_size_without_helper_boundary() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Value {
                    op: LirLeafOp::from_primitive(PrimitiveOpKind::MemorySize { mem_idx: 1 })
                        .unwrap(),
                    args: alloc::vec![],
                    results: alloc::vec![LirValue(0)],
                },
            }],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("memory.size should lower directly");

    let block = &lowered.module.functions[0].program.blocks[0];
    assert_eq!(block.ops.len(), 3);
    assert!(matches!(block.ops[0].kind, MachineInstKind::Load { .. }));
    assert!(matches!(block.ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::ShrU,
            rhs: MachineValue::Imm64(16),
            ..
        }
    ));
}

#[test]
fn lowers_memory_size_with_gp_word_width_on_32_bit_target() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Value {
                    op: LirLeafOp::from_primitive(PrimitiveOpKind::MemorySize { mem_idx: 1 })
                        .unwrap(),
                    args: alloc::vec![],
                    results: alloc::vec![LirValue(0)],
                },
            }],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("memory.size should lower directly on 32-bit targets");

    let block = &lowered.module.functions[0].program.blocks[0];
    assert_eq!(block.ops.len(), 3);
    assert!(matches!(
        block.ops[0].kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U32,
            ..
        }
    ));
    assert!(matches!(
        block.ops[1].kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U32,
            ..
        }
    ));
    assert!(matches!(
        block.ops[2].kind,
        MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::MachineIntWidth::I32,
            op: MachineIntBinaryOp::ShrU,
            rhs: MachineValue::Imm64(16),
            ..
        }
    ));
}

#[test]
fn lowers_call_indirect_with_local_and_external_dispatch_paths() {
    let frame = plan_frame_layout(0, 6, 4);
    let call_base = frame.operand_slot(0);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::CallIndirect {
                    type_idx: 3,
                    table_idx: 0,
                    index_slot: call_base.advance(2),
                    args: crate::vm::plan::frame::FrameSpan::new(call_base, 2),
                    results: crate::vm::plan::frame::FrameSpan::new(call_base, 1),
                    skip_reload: alloc::vec![],
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("call_indirect lowering should succeed");

    assert_eq!(lowered.module.externs.len(), 1);
    assert_eq!(
        lowered.module.externs[0].symbol,
        MachineHelperSymbol::CallIndirectExternal
    );

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 10);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[2].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::TableOutOfBounds
        }
    ));
    assert!(matches!(
        program.blocks[4].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::InvalidFunctionReference
        }
    ));
    assert!(matches!(
        program.blocks[6].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::IndirectCallTypeMismatch
        }
    ));
    assert!(matches!(
        program.blocks[7].terminator,
        MachineTerminator::CallIndirect {
            continuation: MachineBlockId(9),
            ..
        }
    ));
    assert_eq!(program.blocks[7].params.len(), 1);
    assert!(matches!(
        program.blocks[7].ops[0].kind,
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Add,
            ..
        }
    ));
    let type_check_ops = &program.blocks[3].ops;
    let type_check_scaled = match &type_check_ops[1].kind {
        MachineInstKind::Move { dst, .. } => *dst,
        other => panic!("expected scaled-index move in type-check block, got {other:?}"),
    };
    let type_check_base = match &type_check_ops[3].kind {
        MachineInstKind::Load { dst, .. } => *dst,
        other => panic!("expected function-view base load in type-check block, got {other:?}"),
    };
    assert_ne!(type_check_scaled, type_check_base);
    let type_canon_offset = match &type_check_ops[6].kind {
        MachineInstKind::Load { addr, .. } => addr.offset,
        other => panic!("expected type-canon base load in type-check block, got {other:?}"),
    };
    assert_eq!(
        type_canon_offset,
        crate::vm::native::runtime::context::ctx_offset::TYPE_CANON_BASE as i32
    );
    assert!(matches!(
        type_check_ops[5].kind,
        MachineInstKind::Load {
            width: crate::vm::native::ir::machine::MachineMemWidth::U32,
            extension: crate::vm::native::ir::machine::MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[3].terminator,
        MachineTerminator::Branch {
            cond: crate::vm::native::ir::machine::MachineBranchCond::IntCompare {
                rhs: MachineValue::Reg(_),
                ..
            },
            ..
        }
    ));

    let dispatch_ops = &program.blocks[5].ops;
    let dispatch_scaled = match &dispatch_ops[1].kind {
        MachineInstKind::Move { dst, .. } => *dst,
        other => panic!("expected scaled-index move in dispatch block, got {other:?}"),
    };
    let dispatch_base = match &dispatch_ops[3].kind {
        MachineInstKind::Load { dst, .. } => *dst,
        other => panic!("expected function-view base load in dispatch block, got {other:?}"),
    };
    assert_ne!(dispatch_scaled, dispatch_base);
    assert!(matches!(
        dispatch_ops[5].kind,
        MachineInstKind::Load {
            width: crate::vm::native::ir::machine::MachineMemWidth::U32,
            extension: crate::vm::native::ir::machine::MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        dispatch_ops[6].kind,
        MachineInstKind::Load {
            width: crate::vm::native::ir::machine::MachineMemWidth::U32,
            extension: crate::vm::native::ir::machine::MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[8].ops.as_slice(),
        [crate::vm::native::ir::machine::MachineInst {
            kind: MachineInstKind::CallHelper(_)
        }]
    ));
    assert!(matches!(
        program.blocks[8].terminator,
        MachineTerminator::Jump(crate::vm::native::ir::machine::MachineEdge {
            target: MachineBlockId(9),
            ..
        })
    ));
    assert!(matches!(
        program.blocks[9].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::Unreachable
        }
    ));
}

#[test]
fn lowers_call_indirect_with_gp_word_width_on_32_bit_target() {
    let frame = plan_frame_layout(0, 6, 4);
    let call_base = frame.operand_slot(0);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::CallIndirect {
                    type_idx: 3,
                    table_idx: 0,
                    index_slot: call_base.advance(2),
                    args: crate::vm::plan::frame::FrameSpan::new(call_base, 2),
                    results: crate::vm::plan::frame::FrameSpan::new(call_base, 1),
                    skip_reload: alloc::vec![],
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("call_indirect lowering should stay GP-word-sized on 32-bit targets");

    let program = &lowered.module.functions[0].program;
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch {
            cond: crate::vm::native::ir::machine::MachineBranchCond::IntCompare {
                width: crate::vm::native::ir::machine::MachineIntWidth::I32,
                ..
            },
            ..
        }
    ));
    assert!(program.blocks[1].ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U32,
            ..
        }
    )));
    assert!(program.blocks[1].ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::MachineIntWidth::I32,
            op: MachineIntBinaryOp::Mul,
            ..
        }
    )));
    assert!(matches!(
        program.blocks[7].ops[0].kind,
        MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn uses_canonical_u64_width_for_gp_word_frame_slots_on_32bit_targets() {
    let frame = plan_frame_layout(0, 1, 2);
    let slot = frame.local_slot(0);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 7 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot,
                        src: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot,
                        dst: LirValue(1),
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                        args: alloc::vec![LirValue(1)],
                        results: alloc::vec![],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![ValueType::I32, ValueType::I32],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit GP-word slot accesses should use canonical slot width");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        MachineInstKind::Store {
            width: MachineMemWidth::U64,
            ..
        }
    )));
    assert!(ops.iter().any(|op| matches!(
        op.kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U64,
            ..
        }
    )));
}

#[test]
fn lowers_direct_local_call_call_link_with_canonical_frame_width_on_32bit_targets() {
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![LirInst {
                kind: LirInstKind::Boundary(LirBoundaryOp::CallInternal {
                    callee: 1,
                    args: crate::vm::plan::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::plan::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1,
                    ),
                    skip_reload: alloc::vec![],
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
        value_types: alloc::vec![],
    };
    let callee = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(0),
                frame: caller_frame,
                lir: &caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::native::ir::machine::MachineFuncId(1),
                frame: callee_frame,
                lir: &callee,
                result_count: 0,
            },
        ],
    })
    .expect("32-bit direct local call lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    let store_widths: alloc::vec::Vec<_> = ops
        .iter()
        .filter_map(|op| match op.kind {
            MachineInstKind::Store { width, .. } => Some(width),
            _ => None,
        })
        .collect();
    assert_eq!(
        store_widths,
        alloc::vec![
            MachineMemWidth::U32,
            MachineMemWidth::U32,
            MachineMemWidth::U32,
            MachineMemWidth::U32,
            MachineMemWidth::U32,
        ]
    );
}

#[test]
fn lowers_global_get_and_set_without_helpers() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::GlobalSet { idx: 3 })
                            .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::GlobalGet { idx: 3 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("global get/set should lower directly");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 5);
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(9),
            ..
        }
    ));
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Store { .. }));
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_table_get_with_explicit_oob_trap_block() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 0 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::TableGet { table_idx: 1 })
                            .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("table.get should lower with an explicit trap split");

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 3);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[1].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::TableOutOfBounds
        }
    ));
    assert!(matches!(
        program.blocks[2].ops.last().unwrap().kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        program.blocks[2].terminator,
        MachineTerminator::Return
    ));
}

#[test]
fn lowers_i32_load_with_inline_trap_if() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 4,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("i32.load should lower with an inline trap");

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 1);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Return
    ));
    let ops = &program.blocks[0].ops;
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Convert {
            op: crate::vm::native::ir::machine::MachineConvertOp::I64ExtendI32U,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::TrapIf {
            kind: crate::vm::native::ir::machine::MachineTrapKind::MemoryOutOfBounds,
            ..
        }
    )));
    assert!(matches!(
        ops.last().unwrap().kind,
        MachineInstKind::Load {
            width: crate::vm::native::ir::machine::MachineMemWidth::U32,
            ..
        }
    ));
}

#[test]
fn lowers_i32_load_with_gp_word_bounds_ops_on_32_bit_target() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 4,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("i32.load should use GP-word bounds ops on 32-bit targets");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(!ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Convert {
            op: crate::vm::native::ir::machine::MachineConvertOp::I64ExtendI32U,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::TrapIf {
            cond: crate::vm::native::ir::machine::MachineBranchCond::IntCompare {
                width: crate::vm::native::ir::machine::MachineIntWidth::I32,
                ..
            },
            ..
        }
    )));
}

#[test]
fn lowers_32bit_memory_bounds_checks_with_wraparound_traps() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 1,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit memory lowering should emit wraparound traps");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    let mut saw_offset_wrap = false;
    let mut saw_access_wrap = false;
    for inst in ops {
        if let MachineInstKind::TrapIf {
            cond:
                crate::vm::native::ir::machine::MachineBranchCond::IntCompare {
                    width: crate::vm::native::ir::machine::MachineIntWidth::I32,
                    kind: MachineCompareKind::Lt,
                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                    rhs: MachineValue::Imm64(value),
                    ..
                },
            kind: crate::vm::native::ir::machine::MachineTrapKind::MemoryOutOfBounds,
        } = &inst.kind
        {
            if *value == 1 {
                saw_offset_wrap = true;
            }
            if *value == 4 {
                saw_access_wrap = true;
            }
        }
    }
    assert!(saw_offset_wrap, "missing offset wraparound trap");
    assert!(saw_access_wrap, "missing access-size wraparound trap");
}

#[cfg(has_guard_pages)]
#[test]
fn keeps_explicit_mem0_bounds_checks_for_32bit_multiword_gp_accesses_with_guard_pages() {
    let frame = plan_frame_layout(0, 2, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x8877_6655_4433_2211,
                        })
                        .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Store {
                            offset: 0,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(0), LirValue(1)],
                        results: alloc::vec![],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        use_guard_pages: true,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("32-bit multiword mem0 access should keep explicit bounds checks");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    let mut saw_access_wrap = false;
    let mut saw_bounds_trap = false;
    for inst in ops {
        if let MachineInstKind::TrapIf {
            cond:
                crate::vm::native::ir::machine::MachineBranchCond::IntCompare {
                    width: crate::vm::native::ir::machine::MachineIntWidth::I32,
                    kind: MachineCompareKind::Lt,
                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                    rhs: MachineValue::Imm64(value),
                    ..
                },
            kind: crate::vm::native::ir::machine::MachineTrapKind::MemoryOutOfBounds,
        } = &inst.kind
        {
            if *value == 8 {
                saw_access_wrap = true;
            }
        }
        if let MachineInstKind::TrapIf {
            cond:
                crate::vm::native::ir::machine::MachineBranchCond::IntCompare {
                    width: crate::vm::native::ir::machine::MachineIntWidth::I32,
                    kind: MachineCompareKind::Gt,
                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                    ..
                },
            kind: crate::vm::native::ir::machine::MachineTrapKind::MemoryOutOfBounds,
        } = &inst.kind
        {
            saw_bounds_trap = true;
        }
    }
    assert!(saw_access_wrap, "missing 32-bit multiword wraparound trap");
    assert!(saw_bounds_trap, "missing explicit mem0 bounds trap");
}

#[test]
fn lowers_ref_null_and_is_null_with_gp_word_width_on_32_bit_target() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::RefNull).unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::RefIsNull).unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new_with_gp_unit_bytes(0, 4, 0, 2, 4),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("ref.is_null should use GP-word null constants on 32-bit targets");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord,
            src: MachineValue::Imm64(value),
            ..
        } if value == u32::MAX as u64
    ));
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::IntCompare {
            width: crate::vm::native::ir::machine::MachineIntWidth::I32,
            rhs: MachineValue::Imm64(value),
            ..
        } if value == u32::MAX as u64
    ));
}

#[test]
fn omits_zero_offset_add_in_bounds_check_setup() {
    let frame = plan_frame_layout(0, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 0,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![LirValue(1)],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("offset-zero load should lower without a no-op add");

    let setup_ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(!setup_ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::IntBinary {
                op: MachineIntBinaryOp::Add,
                rhs: MachineValue::Imm64(0),
                ..
            }
        )
    }));
}

#[test]
fn threads_live_transients_through_split_continuation_params() {
    let frame = plan_frame_layout(1, 2, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 3 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::TableGet { table_idx: 1 })
                            .unwrap(),
                        args: alloc::vec![LirValue(0)],
                        results: alloc::vec![LirValue(2)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I64Add).unwrap(),
                        args: alloc::vec![LirValue(1), LirValue(2)],
                        results: alloc::vec![LirValue(3)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(3),
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("split continuation params should be threaded explicitly");

    let program = &lowered.module.functions[0].program;
    let MachineTerminator::Branch { else_edge, .. } = &program.blocks[0].terminator else {
        panic!("expected split branch terminator");
    };
    let continuation = &program.blocks[2];
    let expected_args = continuation
        .params
        .iter()
        .map(|param| MachineValue::Reg(param.reg))
        .collect::<Vec<_>>();

    assert_eq!(else_edge.target, continuation.id);
    assert_eq!(else_edge.args, expected_args);
    assert!(continuation
        .params
        .iter()
        .any(|param| param.reg == MachineReg(5)));
    assert!(matches!(
        continuation.ops[0].kind,
        MachineInstKind::Convert {
            op: crate::vm::native::ir::machine::MachineConvertOp::I64ExtendI32U,
            ..
        }
    ));
}

#[test]
fn lowers_f32_store_inline_with_trap_if_preserving_fp_transient_width() {
    let frame = plan_frame_layout(0, 0, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::F32Const {
                            value: 0x3f800000,
                        })
                        .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(1)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::F32Abs).unwrap(),
                        args: alloc::vec![LirValue(1)],
                        results: alloc::vec![LirValue(2)],
                    },
                },
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::F32Store {
                            offset: 4,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: alloc::vec![LirValue(0), LirValue(2)],
                        results: alloc::vec![],
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("inline memory store should preserve FP transient widths");

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 1);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Return
    ));
    assert!(program.blocks[0].ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::TrapIf {
                kind: crate::vm::native::ir::machine::MachineTrapKind::MemoryOutOfBounds,
                ..
            }
        )
    }));
    assert!(program.blocks[0].ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Store {
                src: MachineValue::Reg(reg),
                ..
            } if reg.0 >= program.first_fp_reg
        )
    }));
}

#[test]
fn lowers_f32_const_to_fp_machine_const() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(0, 1, 1);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::Value {
                        op: LirLeafOp::from_primitive(PrimitiveOpKind::F32Const {
                            value: 0x4120_0000,
                        })
                        .unwrap(),
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0)],
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.operand_slot(0),
                        src: LirValue(0),
                    },
                },
            ],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![ValueType::F32],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 1,
        }],
    })
    .expect("f32 const should lower");

    let program = &lowered.module.functions[0].program;
    assert!(program.blocks[0].ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::FloatConst {
                width: crate::vm::native::ir::machine::MachineFloatWidth::F32,
                bits: 0x4120_0000,
                dst,
            } if dst.0 >= program.first_fp_reg
        ) || matches!(
            inst.kind,
            MachineInstKind::Store {
                ty: MachineStorageType::Fp32,
                width: MachineMemWidth::U32,
                src: MachineValue::Imm64(0x4120_0000),
                ..
            }
        )
    }));
}

#[test]
fn float_slot_load_routes_to_fp_bank_when_typed() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(1, 2, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: frame.local_slot(0),
                        dst: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.operand_slot(0),
                        src: LirValue(0),
                    },
                },
            ],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![ValueType::F64],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 1,
        }],
    })
    .expect("typed float slot load should lower");

    let program = &lowered.module.functions[0].program;
    // The Load destination for the float slot must be an FP register.
    let load_dst = program.blocks[0]
        .ops
        .iter()
        .find_map(|inst| match &inst.kind {
            MachineInstKind::Load { dst, .. } => Some(*dst),
            _ => None,
        })
        .expect("there should be a Load instruction");
    assert!(
        program.is_fp_reg(load_dst),
        "typed F64 LoadSlot must allocate into FP bank, got GP reg {}",
        load_dst.0,
    );
}

#[test]
fn untyped_slot_load_stays_in_gp_bank() {
    let frame = plan_frame_layout(1, 2, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: frame.local_slot(0),
                        dst: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.operand_slot(0),
                        src: LirValue(0),
                    },
                },
            ],
            terminator: LirTerminator::Return {
                results: Some(crate::vm::plan::frame::FrameSpan::new(
                    frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: alloc::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 1,
        }],
    })
    .expect("untyped slot load should lower");

    let program = &lowered.module.functions[0].program;
    let load_dst = program.blocks[0]
        .ops
        .iter()
        .find_map(|inst| match &inst.kind {
            MachineInstKind::Load { dst, .. } => Some(*dst),
            _ => None,
        })
        .expect("there should be a Load instruction");
    assert!(
        program.is_gp_reg(load_dst),
        "untyped LoadSlot must stay in GP bank, got FP reg {}",
        load_dst.0,
    );
}

#[test]
fn f32_block_params_keep_f32_width() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(1, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs::default(),
        blocks: alloc::vec![
            LirBlock {
                id: LirTarget(0),
                params: alloc::vec![],
                ops: alloc::vec![LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: frame.local_slot(0),
                        dst: LirValue(0),
                    },
                }],
                terminator: LirTerminator::Goto(LirEdge {
                    target: LirTarget(1),
                    bindings: alloc::vec![LirBinding {
                        param: LirValue(1),
                        value: LirValue(0),
                    }],
                }),
            },
            LirBlock {
                id: LirTarget(1),
                params: alloc::vec![LirValue(1)],
                ops: alloc::vec![],
                terminator: LirTerminator::Return { results: None },
            },
        ],
        value_types: alloc::vec![ValueType::F32, ValueType::F32],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 0, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("typed f32 block params should lower");

    let params = &lowered.module.functions[0].program.blocks[1].params;
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].ty, MachineStorageType::Fp32);
}

#[test]
fn f32_cached_locals_use_f32_slot_widths() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(1, 1, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            gp_preferred_slots: alloc::vec![],
            gp_preferred_types: alloc::vec![],
            fp_preferred_slots: alloc::vec![frame.local_slot(0)],
            fp_preferred_types: alloc::vec![ValueType::F32],
            gp_local_info: alloc::vec![],
            fp_local_info: alloc::vec![CachedLocalInfo {
                is_param: true,
                reads_before_write: true
            }],
        },
        blocks: alloc::vec![LirBlock {
            id: LirTarget(0),
            params: alloc::vec![],
            ops: alloc::vec![
                LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: frame.local_slot(0),
                        dst: LirValue(0),
                    },
                },
                LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: frame.local_slot(0),
                        src: LirValue(0),
                    },
                },
            ],
            terminator: LirTerminator::Return { results: None },
        }],
        value_types: alloc::vec![ValueType::F32],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4, 1, 2),
        #[cfg(has_guard_pages)]
        use_guard_pages: false,
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("typed f32 cached locals should lower");

    let program = &lowered.module.functions[0].program;
    assert_eq!(
        program.fp_reg_init_widths,
        alloc::vec![None, None, Some(MachineFloatWidth::F32)],
    );

    let ops = &program.blocks[0].ops;
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U32,
            ..
        }
    ));
}
