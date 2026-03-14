use crate::vm::{
    backend::BackendConfig,
    lir::{
        ir::{
            LirBinding, LirBlock, LirBoundaryOp, LirEdge, LirInst, LirInstKind, LirLocalCachePrefs,
            LirProgram, LirTerminator, LirValue,
        },
        leaf::LirLeafOp,
        target::LirTarget,
    },
    native::{
        ir::machine::{
            MachineBlockId, MachineFunction, MachineInstKind, MachineIntBinaryOp, MachineModule,
            MachineReg, MachineTerminator, MachineValue,
        },
        ir::runtime::MachineHelperSymbol,
        lower::{lower_module, LowerFunctionInput, LowerModuleInput},
    },
    plan::frame::plan_frame_layout,
    wasm::primitive_op::PrimitiveOpKind,
};

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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 3),
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
fn native_backend_requires_at_least_one_lir_lane() {
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
    };

    let err = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 0),
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect_err("zero-lane native backend should be rejected");

    assert!(alloc::format!("{err}").contains("at least one LIR lane register"));
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let err = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let err = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
fn lowers_cached_local_reads_and_writes_through_cache_regs() {
    let frame = plan_frame_layout(1, 2, 2);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            preferred_slots: alloc::vec![frame.local_slot(0)],
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4),
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(4)),
        }
    ));
    assert!(matches!(
        ops[2].kind,
        MachineInstKind::Move {
            dst: MachineReg(6),
            src: MachineValue::Imm64(7),
        }
    ));
    assert!(matches!(
        ops[3].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            src: MachineValue::Reg(MachineReg(6)),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
fn flushes_and_reloads_cached_locals_around_call_external() {
    let frame = plan_frame_layout(1, 2, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            preferred_slots: alloc::vec![frame.local_slot(0)],
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
                    }),
                },
            ],
            terminator: LirTerminator::TrapUnreachable,
        }],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4),
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("external helper lowering should succeed with cached locals");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 8);
    assert!(matches!(ops[0].kind, MachineInstKind::Load { .. }));
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(9),
            ..
        }
    ));
    assert!(matches!(ops[2].kind, MachineInstKind::Move { .. }));
    assert!(matches!(ops[3].kind, MachineInstKind::Store { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::CallHelper(_)));
    assert!(matches!(ops[5].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[6].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[7].kind, MachineInstKind::Load { .. }));
}

#[test]
fn flushes_and_reloads_cached_locals_around_runtime_helpers() {
    let frame = plan_frame_layout(1, 2, 3);
    let lir = LirProgram {
        entry: LirTarget(0),
        local_cache: LirLocalCachePrefs {
            preferred_slots: alloc::vec![frame.local_slot(0)],
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4),
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("runtime helper lowering should succeed with cached locals");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 8);
    assert!(matches!(ops[0].kind, MachineInstKind::Load { .. }));
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(5),
            ..
        }
    ));
    assert!(matches!(ops[2].kind, MachineInstKind::Move { .. }));
    assert!(matches!(ops[3].kind, MachineInstKind::Store { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::CallHelper(_)));
    assert!(matches!(ops[5].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[6].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[7].kind, MachineInstKind::Load { .. }));
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
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
            rhs: MachineValue::Imm64(_),
            ..
        } if dst == callee_frame_base
    ));
    assert_eq!(call_block.ops.len(), 9);
    assert!(matches!(
        call_block.ops[1].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        call_block.ops[2].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[3].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[4].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(4)),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[5].kind,
        MachineInstKind::Store {
            src: MachineValue::Imm64(0),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[6].kind,
        MachineInstKind::Store {
            src: MachineValue::Imm64(1),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[7].kind,
        MachineInstKind::Store {
            src: MachineValue::Reg(MachineReg(1)),
            ..
        }
    ));
    assert!(matches!(
        call_block.ops[8].kind,
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
            preferred_slots: alloc::vec![caller_frame.local_slot(0)],
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
                    }),
                },
            ],
            terminator: LirTerminator::TrapUnreachable,
        }],
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4),
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
    assert!(matches!(
        second_call_block.ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    assert!(matches!(
        second_call_block.ops[1].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        second_call_block.ops[2].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            ..
        }
    ));
    assert!(matches!(
        second_call_block.ops[3].kind,
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
            preferred_slots: alloc::vec![frame.local_slot(0)],
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(1, 4),
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("block-edge cache preservation lowering should succeed");

    let program = &lowered.module.functions[0].program;
    assert!(matches!(
        program.blocks[0].ops[0].kind,
        MachineInstKind::Load {
            dst: MachineReg(4),
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].ops[1].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(9),
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].ops[2].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Jump(_)
    ));
    assert!(matches!(
        program.blocks[1].ops[0].kind,
        MachineInstKind::Move {
            dst: MachineReg(5),
            src: MachineValue::Reg(MachineReg(4)),
        }
    ));
    assert!(matches!(
        program.blocks[1].ops[1].kind,
        MachineInstKind::Move {
            dst: MachineReg(4),
            src: MachineValue::Reg(MachineReg(5)),
        }
    ));
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
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
                }),
            }],
            terminator: LirTerminator::TrapUnreachable,
        }],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
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
fn lowers_i32_load_with_explicit_oob_trap_block() {
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
    };

    let lowered = lower_module(LowerModuleInput {
        backend: BackendConfig::new(0, 4),
        functions: &[LowerFunctionInput {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            frame,
            lir: &lir,
            result_count: 0,
        }],
    })
    .expect("i32.load should lower with an explicit trap split");

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 3);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[0].ops[1].kind,
        MachineInstKind::Convert {
            op: crate::vm::native::ir::machine::MachineConvertOp::I64ExtendI32U,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].ops[2].kind,
        MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[1].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::native::ir::machine::MachineTrapKind::MemoryOutOfBounds
        }
    ));
    assert!(matches!(
        program.blocks[2].ops.last().unwrap().kind,
        MachineInstKind::Load {
            width: crate::vm::native::ir::machine::MachineMemWidth::U32,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[2].terminator,
        MachineTerminator::Return
    ));
}
