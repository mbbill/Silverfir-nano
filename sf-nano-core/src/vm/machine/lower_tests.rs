use crate::collections;
use crate::value_type::ValueType;

use crate::vm::{
    backend::BackendConfig,
    machine::{
        lower_module,
        machine_ir::{
            MachineBlockId, MachineCompareKind, MachineFunction, MachineInstKind,
            MachineIntBinaryOp, MachineMemWidth, MachineModule, MachineReg, MachineStorageType,
            MachineTerminator, MachineValue, MACHINE_FIXED_REG_COUNT,
        },
        LowerFunctionInput, LowerModuleInput,
    },
    middle::{
        frame::plan_frame_layout,
        ssa_ir::{
            ir::{
                LocalSlotInfo, SsaBinding, SsaBlock, SsaCallOp, SsaEdge, SsaInst, SsaInstKind,
                SsaOperand, SsaProgram, SsaTerminator, SsaValue,
            },
            leaf::SsaLeafOp,
            target::SsaTarget,
        },
    },
    wasm::primitive_op::PrimitiveOpKind,
};

const HOST_GP_UNIT_BYTES: u8 = core::mem::size_of::<usize>() as u8;
const HOST_CALL_SCRATCH_SLOTS: u16 = if HOST_GP_UNIT_BYTES == 4 { 8 } else { 3 };

fn total_gp_budget_for_allocatable(allocatable_gp_budget: u8, gp_unit_bytes: u8) -> u8 {
    if allocatable_gp_budget == 0 {
        return 0;
    }
    if gp_unit_bytes == 4 {
        allocatable_gp_budget.saturating_add(if allocatable_gp_budget == 1 { 1 } else { 2 })
    } else {
        allocatable_gp_budget.saturating_add(1)
    }
}

fn host_backend_config(
    gp_cache_budget: u8,
    gp_linear_budget: u8,
    fp_cache_budget: u8,
    fp_linear_budget: u8,
) -> BackendConfig {
    let gp_dynamic_budget = total_gp_budget_for_allocatable(
        gp_cache_budget.saturating_add(gp_linear_budget),
        HOST_GP_UNIT_BYTES,
    );
    let fp_dynamic_budget = fp_cache_budget.saturating_add(fp_linear_budget);
    BackendConfig::new(
        gp_dynamic_budget,
        fp_dynamic_budget,
        HOST_GP_UNIT_BYTES,
        HOST_CALL_SCRATCH_SLOTS,
    )
}

fn gp32_backend_config(
    gp_cache_budget: u8,
    gp_linear_budget: u8,
    fp_cache_budget: u8,
    fp_linear_budget: u8,
) -> BackendConfig {
    // 32-bit GP targets need at least 5 GP dynamic budget units to satisfy
    // the backend-wide worst-case operand pressure invariant.
    let gp_linear_budget = gp_linear_budget.max(5);
    let gp_dynamic_budget =
        total_gp_budget_for_allocatable(gp_cache_budget.saturating_add(gp_linear_budget), 4);
    let fp_dynamic_budget = fp_cache_budget.saturating_add(fp_linear_budget);
    BackendConfig::new(gp_dynamic_budget, fp_dynamic_budget, 4, 8)
}

fn assert_valid_32bit_gp_target(module: &MachineModule, backend: BackendConfig) {
    assert!(backend.is_32bit_gp_target());
    let max_gp_regs = MACHINE_FIXED_REG_COUNT + backend.gp_dynamic_budget as u16;
    module
        .validate_32bit_gp_target(max_gp_regs)
        .unwrap_or_else(|err| panic!("32-bit lowered module must already validate: {err}"));
}

#[test]
fn lowers_simple_slot_and_add_block() {
    let frame = plan_frame_layout(1, 4, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: frame.local_slot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(0)),
                            SsaOperand::Value(SsaValue(1))
                        ],
                        results: collections::vec![SsaValue(2)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot: frame.local_slot(0),
                        src: SsaValue(2),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let MachineModule { functions, .. } = lowered.module;
    let MachineFunction { program, .. } = &functions[0];
    assert_eq!(lowered.abi.functions.len(), 1);
    assert_eq!(lowered.abi.functions[0].frame_prefix_slots, 1);
    assert_eq!(
        lowered.abi.functions[0].total_frame_slots,
        frame.total_slots()
    );
    // Under the new local-call ABI the dead call-link half of the frame
    // plan is gone; what `FrameLayoutPlan::call_scratch` carries today is
    // the live helper-scratch half, exposed unchanged on the per-function
    // ABI as `helper_scratch`.
    assert_eq!(
        lowered.abi.functions[0].helper_scratch,
        frame.call_scratch.map(|span| {
            crate::vm::machine::machine_ir::MachineFrameRegion {
                base_slot: span.start.0,
                slots: span.count,
            }
        })
    );
    assert_eq!(lowered.abi.functions[0].return_results, None);
    assert_eq!(program.entry, MachineBlockId(0));
    assert_eq!(program.blocks.len(), 1);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Return
    ));
    // LocalGetCache now source-aliases the value to the cache register
    // without emitting a Move, so the block is one op shorter.
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 11 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 22 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(2)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::Select).unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(0)),
                            SsaOperand::Value(SsaValue(1)),
                            SsaOperand::Value(SsaValue(2))
                        ],
                        results: collections::vec![SsaValue(3)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 3, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
fn native_backend_requires_at_least_one_gp_linear_value_register() {
    let frame = plan_frame_layout(0, 0, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let err = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 0, 0, 0),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect_err("zero GP-dynamic native backend must be rejected");

    assert!(
        alloc::format!("{err}").contains("at least one GP dynamic register"),
        "unexpected error: {err}"
    );
}

#[test]
fn projects_return_results_and_helper_scratch_from_frame_plan() {
    let frame = plan_frame_layout(2, 6, 6);
    let result_span = crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(result_span),
            },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let runtime = &lowered.abi;
    // The dead call-link half of the frame plan is gone; the helper_scratch
    // exposed by the per-function ABI is now the entire span the frame
    // planner reserved.
    assert_eq!(
        runtime.functions[0].helper_scratch,
        Some(crate::vm::machine::machine_ir::MachineFrameRegion {
            base_slot: frame.call_scratch.unwrap().start.0,
            slots: frame.call_scratch.unwrap().count,
        })
    );
    assert_eq!(
        runtime.functions[0].return_results,
        Some(crate::vm::machine::machine_ir::MachineFrameRegion {
            base_slot: result_span.start.0,
            slots: result_span.count,
        })
    );
}

#[test]
fn rejects_inconsistent_return_result_spans() {
    let frame = plan_frame_layout(0, 4, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: SsaTerminator::Return {
                    results: Some(crate::vm::middle::frame::FrameSpan::new(
                        frame.operand_slot(0),
                        1
                    )),
                },
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: SsaTerminator::Return {
                    results: Some(crate::vm::middle::frame::FrameSpan::new(
                        frame.operand_slot(1),
                        1
                    )),
                },
            },
        ],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let err = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect_err("inconsistent return spans should be rejected");

    assert!(err.message().contains("inconsistent return result spans"));
}

#[test]
fn rejects_mixed_void_and_value_returns() {
    let frame = plan_frame_layout(0, 4, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: SsaTerminator::Return { results: None },
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: SsaTerminator::Return {
                    results: Some(crate::vm::middle::frame::FrameSpan::new(
                        frame.operand_slot(0),
                        1,
                    )),
                },
            },
        ],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let err = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect_err("mixed void and value returns should be rejected");

    assert!(err.message().contains("inconsistent return result spans"));
}

#[test]
fn lowers_branch_edge_bindings_into_machine_edge_args() {
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![SsaValue(0), SsaValue(1)],
                ops: collections::vec![],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![
                        SsaBinding {
                            param: SsaValue(3),
                            value: SsaValue(1),
                        },
                        SsaBinding {
                            param: SsaValue(2),
                            value: SsaValue(0),
                        },
                    ],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![SsaValue(2), SsaValue(3)],
                ops: collections::vec![],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame: plan_frame_layout(0, 2, 2),
            ssa: ssa,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![SsaValue(0)],
                ops: collections::vec![],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![SsaBinding {
                        param: SsaValue(1),
                        value: SsaValue(0),
                    }],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![SsaValue(1)],
                ops: collections::vec![],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![ValueType::I64, ValueType::I64],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame: plan_frame_layout(0, 2, 2),
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 param lowering should succeed");

    let block0 = &lowered.module.functions[0].program.blocks[0];
    assert_eq!(block0.params.len(), 2);
    assert!(matches!(
        block0.params[0],
        crate::vm::machine::machine_ir::MachineBlockParam {
            reg: MachineReg(4),
            ty: MachineStorageType::GpWord,
            owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
        }
    ));
    assert!(matches!(
        block0.params[1],
        crate::vm::machine::machine_ir::MachineBlockParam {
            reg: MachineReg(5),
            ty: MachineStorageType::GpWord,
            owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
        }
    ));

    let MachineTerminator::Jump(edge) = &block0.terminator else {
        panic!("expected jump terminator");
    };
    assert_eq!(
        edge.args,
        collections::vec![
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
    let backend = gp32_backend_config(0, 6, 0, 2);
    let frame = plan_frame_layout(1, 4, 4);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x0123_4567_89ab_cdef,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot: frame.local_slot(0),
                        dst: SsaValue(1),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x1111_2222_3333_4444,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(2)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Add).unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(1)),
                            SsaOperand::Value(SsaValue(2))
                        ],
                        results: collections::vec![SsaValue(3)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![
            ValueType::I64,
            ValueType::I64,
            ValueType::I64,
            ValueType::I64
        ],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
fn gp32_i64_slot_get_stays_frame_based_for_explicit_cache_candidate() {
    let backend = gp32_backend_config(1, 4, 0, 2);
    let frame = plan_frame_layout(1, 2, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I64],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot,
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Spill {
                        slot: frame.operand_slot(0),
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalReserveCache { slot },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I64],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 slot load should not force a cache binding");

    assert_valid_32bit_gp_target(&lowered.module, backend);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Load {
            owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U32,
            ..
        }
    ));
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::Load {
            owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U32,
            ..
        }
    ));
}

#[test]
fn gp32_i64_slot_set_stays_frame_based_for_explicit_cache_candidate() {
    let backend = gp32_backend_config(1, 4, 0, 2);
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I64],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x0123_4567_89ab_cdef,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot,
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalReserveCache { slot },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I64],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 slot store should not force a cache binding");

    assert_valid_32bit_gp_target(&lowered.module, backend);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                width: MachineMemWidth::U32,
                ..
            }
        )
    }));
}

#[test]
fn lowers_i64_global_get_set_directly_to_legal_32bit_machineir() {
    let backend = gp32_backend_config(0, 4, 0, 2);
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::GlobalSet { idx: 3 })
                            .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::GlobalGet { idx: 3 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I64, ValueType::I64],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit global get/set lowering should succeed");

    assert_valid_32bit_gp_target(&lowered.module, backend);
}

#[test]
fn lowers_i64_memory_load_store_directly_to_legal_32bit_machineir() {
    let backend = gp32_backend_config(0, 5, 0, 2);
    let frame = plan_frame_layout(0, 3, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x8877_6655_4433_2211,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Store {
                            offset: 0,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(0)),
                            SsaOperand::Value(SsaValue(1))
                        ],
                        results: collections::vec![],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(2)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Load {
                            offset: 0,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(2))],
                        results: collections::vec![SsaValue(3)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![
            ValueType::I32,
            ValueType::I64,
            ValueType::I32,
            ValueType::I64
        ],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 memory load/store lowering should succeed");

    assert_valid_32bit_gp_target(&lowered.module, backend);
}

#[test]
fn lowers_direct_local_call_to_legal_32bit_machineir() {
    let backend = gp32_backend_config(0, 4, 0, 2);
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                    callee: 1,
                    args: crate::vm::middle::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::middle::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1,
                    ),
                }),
            }],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };
    let callee = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: frame.local_slot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 7 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: frame.local_slot(0),
                        src: SsaValue(1),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    let cache_reg = ops
        .iter()
        .find_map(|inst| match inst.kind {
            MachineInstKind::Load { dst, .. } => Some(dst),
            _ => None,
        })
        .expect("expected LocalGetCache to load the local into a cache reg");
    // LocalGetCache now source-aliases the value to the cache register
    // without emitting a Move — no snapshot_reg to find.
    let const_reg = ops
        .iter()
        .find_map(|inst| match inst.kind {
            MachineInstKind::Move {
                dst,
                src: MachineValue::Imm64(7),
                ..
            } => Some(dst),
            _ => None,
        })
        .expect("expected i32.const 7 to materialize into a linear-value reg");
    assert!(
        ops.iter().any(|inst| matches!(
            inst.kind,
            MachineInstKind::Move {
                dst,
                src: MachineValue::Reg(src),
                ..
            } if dst == cache_reg && src == const_reg
        )),
        "expected LocalSetCache to write the new value back into the cache reg"
    );
}

#[test]
fn local_set_cache_reuses_dying_source_linear_value_when_no_extra_reg_is_free() {
    let frame = plan_frame_layout(1, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 7 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 1, 0, 0),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("LocalSetCache should be able to bind the dying source reg as cache");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 1, "no extra cache copy should be needed");
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(7),
            ..
        }
    ));
}

#[test]
fn does_not_zero_unread_cached_locals_at_entry_on_32bit_targets() {
    let backend = gp32_backend_config(1, 4, 0, 2);
    let frame = plan_frame_layout(1, 2, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 9 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit lowering should not zero unread cached locals");

    assert_valid_32bit_gp_target(&lowered.module, backend);

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 1);
    // LocalSetCache should be able to bind the dying const reg directly as the cache reg.
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(9),
            ..
        }
    ));
}

#[test]
fn lowers_call_external_through_frame_metadata_without_helper_scratch() {
    let frame = plan_frame_layout(0, 2, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                    callee: 7,
                    args: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 2),
                    results: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 1),
                }),
            }],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("external helper lowering should succeed");

    assert_eq!(lowered.module.consts.len(), 1);
    // Under the new local-call ABI the dead call-link half of the frame
    // plan is gone, so the helper_scratch field directly mirrors whatever
    // the frame planner reserved. For 1A this still includes the
    // historical 3-slot reservation from `HOST_CALL_SCRATCH_SLOTS`; a
    // future cleanup can drop it to 0 when no helper actually needs it.
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0].kind, MachineInstKind::CallExternal(_)));
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Load { .. }));
}

#[test]
fn coalesces_dead_i64_const_directly_into_uncached_store_slot() {
    let frame = plan_frame_layout(0, 1, 1);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x0123_4567_89ab_cdef,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![ValueType::I64],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalDropCache {
                        slot: frame.local_slot(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                        callee: 7,
                        args: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 1),
                        results: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 0),
                    }),
                },
            ],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("external helper lowering should succeed with cached locals");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 5);
    // ops[0]: I64Const(9) into linear-value reg
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(9),
            ..
        }
    ));
    // ops[1]: explicit LocalDropCache flushes cache to frame before the call
    assert!(matches!(ops[1].kind, MachineInstKind::Store { .. }));
    // ops[2]: external call
    assert!(matches!(ops[2].kind, MachineInstKind::CallExternal(_)));
    // ops[3-4]: mem0 cache reg refresh only; there is no implicit cached-local reload
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn skips_dead_cached_local_reload_after_direct_external_call() {
    let frame = plan_frame_layout(1, 2, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalDropCache {
                        slot: frame.local_slot(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                        callee: 7,
                        args: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 1),
                        results: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 0),
                    }),
                },
            ],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("direct external call lowering should preserve explicit cache protocol");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 5);
    assert!(matches!(ops[0].kind, MachineInstKind::Move { .. }));
    assert!(matches!(ops[1].kind, MachineInstKind::Store { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::CallExternal(_)));
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn flushes_and_reloads_cached_locals_around_runtime_helpers() {
    let frame = plan_frame_layout(1, 2, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 5 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalDropCache {
                        slot: frame.local_slot(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                        callee: 7,
                        args: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 1),
                        results: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 0),
                    }),
                },
            ],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("runtime helper lowering should succeed with cached locals");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 5);
    // ops[0]: I64Const(5) into linear-value reg
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(5),
            ..
        }
    ));
    // ops[1]: explicit LocalDropCache flushes before the runtime helper call
    assert!(matches!(ops[1].kind, MachineInstKind::Store { .. }));
    // ops[2]: call helper (call_external)
    assert!(matches!(ops[2].kind, MachineInstKind::CallExternal(_)));
    // ops[3-4]: reload mem0 cache regs only
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_direct_local_call_with_continuation_block() {
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                    callee: 1,
                    args: crate::vm::middle::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::middle::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1
                    ),
                }),
            }],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };
    let callee = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call lowering should succeed");

    let caller_program = &lowered.module.functions[0].program;
    assert_eq!(caller_program.blocks.len(), 2);
    let call_block = &caller_program.blocks[0];
    let (callee_frame_base, caller_result_base) = match call_block.terminator {
        MachineTerminator::CallDirect {
            callee,
            callee_frame_base,
            caller_result_base,
            continuation,
        } => {
            assert_eq!(callee, crate::vm::machine::machine_ir::MachineFuncId(1));
            assert_eq!(continuation, MachineBlockId(1));
            (callee_frame_base, caller_result_base)
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
    // After the new local-call ABI rewrite, the call site emits:
    //   0. add callee_frame_base, fp, args_offset
    //   1. load stack_end
    //   2. sub stack_limit, callee_total_frame_bytes
    //   3. trap_if Gt callee_frame_base, stack_limit
    //   4. add caller_result_base, fp, results_offset
    // — five ops total, with no call-link memory writes. The continuation
    // block id and the caller frame pointer travel via the backend-private
    // call record set up by the per-arch lowering of `CallDirect`, not via
    // any MIR-visible store.
    assert_eq!(call_block.ops.len(), 5);
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
                crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                    width: crate::vm::machine::machine_ir::MachineIntWidth::I64,
                    kind: MachineCompareKind::Gt,
                    sign: crate::vm::machine::machine_ir::MachineSign::Unsigned,
                    lhs: MachineValue::Reg(lhs),
                    ..
                },
            kind: crate::vm::machine::machine_ir::MachineTrapKind::StackOverflow,
        } if lhs == callee_frame_base
    ));
    // The fifth op computes `caller_result_base = caller_fp + results_offset`
    // and is consumed by the `CallDirect` terminator below.
    assert!(matches!(
        call_block.ops[4].kind,
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(MachineReg(1)),
            ..
        } if dst == caller_result_base
    ));

    let continuation = &caller_program.blocks[1];
    assert!(continuation.params.is_empty());
    assert!(continuation.ops.is_empty());
    assert!(matches!(
        continuation.terminator,
        MachineTerminator::Trap {
            kind: crate::vm::machine::machine_ir::MachineTrapKind::Unreachable
        }
    ));
}

#[test]
fn flushes_cached_local_before_second_direct_call() {
    let caller_frame = plan_frame_layout(2, 1, 4);
    let callee_frame = plan_frame_layout(0, 1, 4);

    let caller = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                        callee: 1,
                        args: crate::vm::middle::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            0,
                        ),
                        results: crate::vm::middle::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            1,
                        ),
                    }),
                },
                SsaInst {
                    kind: SsaInstKind::Fill {
                        slot: caller_frame.operand_slot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: caller_frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalDropCache {
                        slot: caller_frame.local_slot(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                        callee: 1,
                        args: crate::vm::middle::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            0,
                        ),
                        results: crate::vm::middle::frame::FrameSpan::new(
                            caller_frame.operand_slot(0),
                            1,
                        ),
                    }),
                },
            ],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };
    let callee = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call lowering should succeed with cached locals");

    let caller_program = &lowered.module.functions[0].program;
    assert_eq!(caller_program.blocks.len(), 3);

    let second_call_block = &caller_program.blocks[1];
    let result_slot_offset = i32::from(caller_frame.operand_slot(0).0) * 8;
    let local_slot_offset = i32::from(caller_frame.local_slot(0).0) * 8;
    let result_reload_indices = second_call_block
        .ops
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| match &inst.kind {
            MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                addr,
                ..
            } if *addr
                == crate::vm::machine::machine_ir::MachineAddr {
                    base: MachineReg(1),
                    offset: result_slot_offset,
                } =>
            {
                Some(idx)
            }
            _ => None,
        })
        .collect::<collections::Vec<_>>();
    let local_flush_indices = second_call_block
        .ops
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| match inst.kind {
            MachineInstKind::Store { addr, .. }
                if addr
                    == crate::vm::machine::machine_ir::MachineAddr {
                        base: MachineReg(1),
                        offset: local_slot_offset,
                    } =>
            {
                Some(idx)
            }
            _ => None,
        })
        .collect::<collections::Vec<_>>();

    assert_eq!(
        result_reload_indices.len(),
        1,
        "the continuation for the second direct call should reload exactly one linear call result from its frame slot; ops={:?}",
        second_call_block.ops
    );
    assert_eq!(
        local_flush_indices.len(),
        1,
        "the continuation for the second direct call should flush exactly one cached local to its frame slot before the next call; ops={:?}",
        second_call_block.ops
    );
    assert!(
        result_reload_indices[0] < local_flush_indices[0],
        "the filled call result must be reloaded before the cached local is flushed; ops={:?}",
        second_call_block.ops
    );
    assert!(matches!(
        second_call_block.terminator,
        MachineTerminator::CallDirect {
            callee: crate::vm::machine::machine_ir::MachineFuncId(1),
            continuation: MachineBlockId(2),
            ..
        }
    ));
}

#[test]
fn preserves_cached_locals_across_block_edges() {
    let frame = plan_frame_layout(1, 1, 4);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(0)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: frame.local_slot(0),
                            src: SsaValue(0),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalDropCache {
                            slot: frame.local_slot(0),
                        },
                    },
                ],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::LocalGetCache {
                            slot: frame.local_slot(0),
                            dst: SsaValue(1),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: frame.local_slot(0),
                            src: SsaValue(1),
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("block-edge cache preservation lowering should succeed");

    let program = &lowered.module.functions[0].program;
    let block0_move_index = program.blocks[0]
        .ops
        .iter()
        .position(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    src: MachineValue::Imm64(9),
                    ..
                }
            )
        })
        .expect("block 0 should materialize the constant into one linear machine value");
    let block0_store_index = program.blocks[0]
        .ops
        .iter()
        .position(|inst| matches!(inst.kind, MachineInstKind::Store { .. }))
        .expect("block 0 should flush the dropped cached local to the frame before the edge");
    assert!(
        block0_move_index < block0_store_index,
        "the constant must be materialized before the cache flush; ops={:?}",
        program.blocks[0].ops
    );
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Jump(_)
    ));
    // Block 1: cache is re-established explicitly from the frame slot.
    // Then LocalGetCache aliases v1 to the cache register, and LocalSetCache
    // of v1 back to the same slot is a no-op (value already in cache_reg).
    let block1 = &program.blocks[1];
    let (_load_idx, _cache_reg) = block1
        .ops
        .iter()
        .enumerate()
        .find_map(|(idx, inst)| match inst.kind {
            MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                dst,
                ..
            } => Some((idx, dst)),
            _ => None,
        })
        .expect("block 1 should re-establish the cached local with one CachedLocal load");
    // With source-aliasing, LocalGetCache+LocalSetCache of the same slot
    // produces zero additional Move ops beyond the initial Load.
    assert!(
        !block1.ops.iter().any(|inst| matches!(
            inst.kind,
            MachineInstKind::Move {
                src: MachineValue::Reg(_),
                ..
            }
        )),
        "identity get+set on same cached local should not emit any reg-to-reg Move; ops={:?}",
        block1.ops
    );
}

#[test]
fn threads_cached_locals_through_block_edge_params() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 9 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(0)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot,
                            src: SsaValue(0),
                        },
                    },
                ],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::LocalGetCache {
                            slot,
                            dst: SsaValue(1),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot,
                            src: SsaValue(1),
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![ValueType::I32, ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![], collections::vec![slot]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("block-edge cache carry lowering should succeed");

    let program = &lowered.module.functions[0].program;
    assert!(
        program.blocks[0]
            .ops
            .iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Store { .. })),
        "carried cache edge should not spill the cached local to frame"
    );
    let carried_reg = match &program.blocks[0].terminator {
        MachineTerminator::Jump(edge) => {
            assert_eq!(edge.args.len(), 1, "expected one hidden cache edge arg");
            match edge.args[0] {
                MachineValue::Reg(reg) => reg,
                other => panic!("expected carried cache edge arg register, got {other:?}"),
            }
        }
        other => panic!("expected jump terminator, got {other:?}"),
    };
    assert_eq!(program.blocks[1].params.len(), 1);
    // With source-aliasing, LocalGetCache aliases v1 to the carried cache
    // register (block param), and the subsequent LocalSetCache of v1 back to
    // the same slot is a no-op. Block 1 should have zero Move ops.
    assert!(
        program.blocks[1].ops.iter().all(|inst| !matches!(
            inst.kind,
            MachineInstKind::Move {
                src: MachineValue::Reg(_),
                ..
            }
        )),
        "identity get+set through carried cache should produce no reg-to-reg Move; ops={:?}",
        program.blocks[1].ops
    );
    assert!(
        program.blocks[1]
            .ops
            .iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Load { .. })),
        "carried cache edge should not reload the cached local from frame"
    );
    assert_ne!(carried_reg.0, 0);
}

#[test]
fn keeps_shared_cache_lane_when_earlier_local_drops_on_edge() {
    let frame = plan_frame_layout(2, 1, 4);
    let slot0 = frame.local_slot(0);
    let slot1 = frame.local_slot(1);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32, ValueType::I32],
        local_slot_info: collections::vec![
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
        ],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(0)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: slot0,
                            src: SsaValue(0),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 2 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(1)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: slot1,
                            src: SsaValue(1),
                        },
                    },
                ],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::LocalGetCache {
                            slot: slot1,
                            dst: SsaValue(2),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: slot1,
                            src: SsaValue(2),
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![ValueType::I32, ValueType::I32, ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![], collections::vec![slot1]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("lane-preserving cache edge lowering should succeed");

    let program = &lowered.module.functions[0].program;
    let carried_reg = match &program.blocks[0].terminator {
        MachineTerminator::Jump(edge) => match edge.args.as_slice() {
            [MachineValue::Reg(reg)] => *reg,
            other => panic!("expected one real carried cache edge arg, got {other:?}"),
        },
        other => panic!("expected jump terminator, got {other:?}"),
    };

    assert_eq!(
        program.blocks[1].params.len(),
        1,
        "target block should receive one carried cache param"
    );
    assert_eq!(
        program.blocks[1].params[0].reg, carried_reg,
        "dropping an earlier cached local should leave a hole, not compact the shared local onto a new lane"
    );
}

#[test]
fn local_reserve_cache_does_not_reload_old_slot_value() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalReserveCache { slot },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 9 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot,
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("LocalReserveCache should lower without reloading the old slot value");

    let block = &lowered.module.functions[0].program.blocks[0];
    assert!(
        block
            .ops
            .iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Load { .. })),
        "LocalReserveCache should not trigger a frame reload before the later LocalSetCache; ops={:?}",
        block.ops
    );
    assert!(
        block.ops.iter().any(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                    ..
                }
            )
        }),
        "the later LocalSetCache should still materialize cached-local ownership"
    );
}

#[test]
fn reserved_cache_edge_threads_without_reload_into_target_block() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![SsaInst {
                    kind: SsaInstKind::LocalReserveCache { slot },
                }],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(2),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(2),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 9 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(0)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot,
                            src: SsaValue(0),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalGetCache {
                            slot,
                            dst: SsaValue(1),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot,
                            src: SsaValue(1),
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![ValueType::I32, ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![
            collections::vec![],
            collections::vec![],
            collections::vec![slot]
        ],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("reserve repair edge should carry cache state without reloading");

    let program = &lowered.module.functions[0].program;
    let reserve_block = &program.blocks[1];
    let target_block = &program.blocks[2];

    assert!(
        reserve_block
            .ops
            .iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Load { .. })),
        "LocalReserveCache edge block should not reload the old slot value; ops={:?}",
        reserve_block.ops
    );
    match &reserve_block.terminator {
        MachineTerminator::Jump(edge) => {
            assert_eq!(edge.args.len(), 1, "expected one hidden cache edge arg");
            assert!(
                !matches!(edge.args[0], MachineValue::Reg(_)),
                "reserve-only cached locals must not be threaded as real incoming values; edge args={:?}",
                edge.args
            );
        }
        other => panic!("expected jump terminator from reserve edge block, got {other:?}"),
    }
    assert_eq!(target_block.params.len(), 1);
    assert!(
        target_block
            .ops
            .iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Load { .. })),
        "target block should receive reserved cache state through params, not reload it from frame; ops={:?}",
        target_block.ops
    );
}

#[test]
fn reserved_cache_edge_aligns_each_reserved_arg_with_target_param_reg() {
    let frame = plan_frame_layout(3, 1, 6);
    let slot0 = frame.local_slot(0);
    let slot1 = frame.local_slot(1);
    let slot2 = frame.local_slot(2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32, ValueType::I32, ValueType::I32],
        local_slot_info: collections::vec![
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
        ],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![
                    // Reverse first-mention order so explicit cached-local
                    // metadata does not accidentally match the target entry
                    // slot order.
                    SsaInst {
                        kind: SsaInstKind::LocalReserveCache { slot: slot2 },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalReserveCache { slot: slot1 },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalReserveCache { slot: slot0 },
                    },
                ],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(0)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: slot0,
                            src: SsaValue(0),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 2 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(1)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: slot1,
                            src: SsaValue(1),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 3 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(2)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: slot2,
                            src: SsaValue(2),
                        },
                    },
                ],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![ValueType::I32, ValueType::I32, ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![
            collections::vec![],
            collections::vec![slot0, slot1, slot2],
        ],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(3, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("multi-slot reserve edge should lower");

    let program = &lowered.module.functions[0].program;
    let reserve_block = &program.blocks[0];
    let target_block = &program.blocks[1];
    let MachineTerminator::Jump(edge) = &reserve_block.terminator else {
        panic!("expected jump terminator for reserve edge");
    };
    assert_eq!(edge.args.len(), target_block.params.len());
    for (arg, param) in edge.args.iter().zip(target_block.params.iter()) {
        assert!(
            matches!(arg, MachineValue::ReservedReg(reg) if *reg == param.reg),
            "reserve-only edge args must line up with target cached-local params in-place; args={:?} params={:?}",
            edge.args,
            target_block.params
        );
    }
}

#[test]
fn local_drop_cache_skips_writeback_when_cache_is_clean() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalEnsureCache { slot },
                },
                SsaInst {
                    kind: SsaInstKind::LocalDropCache { slot },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("clean LocalDropCache should not write back");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(
        ops.iter()
            .filter(|inst| matches!(inst.kind, MachineInstKind::Load { .. }))
            .count(),
        1,
        "LocalEnsureCache should materialize one frame load"
    );
    assert!(
        ops.iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Store { .. })),
        "dropping a clean cached local should not write its frame slot back"
    );
}

#[test]
fn does_not_save_clean_carried_cache_before_external_call() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![SsaInst {
                    kind: SsaInstKind::LocalEnsureCache { slot },
                }],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![SsaInst {
                    kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                        callee: 7,
                        args: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 0),
                        results: crate::vm::middle::frame::FrameSpan::new(frame.operand_slot(0), 0),
                    }),
                }],
                terminator: SsaTerminator::TrapUnreachable,
            },
        ],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![], collections::vec![slot]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("clean carried cache should not be saved before external call");

    let program = &lowered.module.functions[0].program;
    assert!(matches!(
        &program.blocks[0].terminator,
        MachineTerminator::Jump(_)
    ));
    assert!(
        program.blocks[1]
            .ops
            .iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Store { .. })),
        "clean carried cache should not be spilled before the external call"
    );
    assert!(
        program.blocks[1]
            .ops
            .iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::CallExternal(_))),
        "expected external call in continuation block"
    );
}

#[test]
fn saves_only_dirty_cached_locals_before_external_call() {
    let frame = plan_frame_layout(2, 1, 4);
    let dirty_slot = frame.local_slot(0);
    let clean_slot = frame.local_slot(1);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32, ValueType::I32],
        local_slot_info: collections::vec![
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true,
            },
        ],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![SsaInst {
                    kind: SsaInstKind::LocalEnsureCache { slot: clean_slot },
                }],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![],
                ops: collections::vec![
                    SsaInst {
                        kind: SsaInstKind::Value {
                            op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 11 })
                                .unwrap(),
                            args: collections::vec![],
                            results: collections::vec![SsaValue(0)],
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: dirty_slot,
                            src: SsaValue(0),
                        },
                    },
                    SsaInst {
                        kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                            callee: 7,
                            args: crate::vm::middle::frame::FrameSpan::new(
                                frame.operand_slot(0),
                                0
                            ),
                            results: crate::vm::middle::frame::FrameSpan::new(
                                frame.operand_slot(0),
                                0
                            ),
                        }),
                    },
                ],
                terminator: SsaTerminator::TrapUnreachable,
            },
        ],
        value_types: collections::vec![ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![
            collections::vec![],
            collections::vec![clean_slot]
        ],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("only dirty caches should be saved before external call");

    let ops = &lowered.module.functions[0].program.blocks[1].ops;
    let store_count = ops
        .iter()
        .filter(|inst| matches!(inst.kind, MachineInstKind::Store { .. }))
        .count();
    assert_eq!(
        store_count, 1,
        "one dirty cached local and one clean carried local should produce exactly one save store before the external call; ops={ops:?}"
    );
    assert!(
        ops.iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::CallExternal(_))),
        "expected external call in lowered block"
    );
}

#[test]
fn entry_block_cached_locals_are_loaded_in_prologue_not_passed_as_params() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::LocalEnsureCache { slot },
            }],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![collections::vec![slot]],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("entry block cached locals should lower via prologue loads");

    let block = &lowered.module.functions[0].program.blocks[0];
    assert!(
        block.params.is_empty(),
        "entry block must not gain hidden cache params"
    );
    assert!(
        matches!(block.ops[0].kind, MachineInstKind::Load { .. }),
        "entry cached local should be materialized with an explicit frame load"
    );
    assert!(
        block
            .ops
            .iter()
            .skip(1)
            .all(|inst| !matches!(inst.kind, MachineInstKind::Load { .. })),
        "LocalEnsureCache should not reload the same cached local again"
    );
}

#[test]
fn rejects_cache_store_with_incompatible_gp_storage_types() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(1, 2, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: frame.local_slot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: frame.local_slot(0),
                        src: SsaValue(1),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32, ValueType::I64],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let err = lower_module(LowerModuleInput {
        backend: gp32_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect_err("typed i64 store into cached i32 local must be rejected");

    let message = alloc::format!("{err}");
    assert!(
        message.contains("LocalSetCache src for cached local slot FrameSlot(0)")
            || message.contains("typed SSA-IR store to cached local slot"),
        "unexpected error: {message}"
    );
}

#[test]
fn lowers_direct_local_call_with_sparse_machine_function_ids() {
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                    callee: 2,
                    args: crate::vm::middle::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::middle::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1,
                    ),
                }),
            }],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };
    let callee = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(2),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("sparse-id local call lowering should succeed");

    assert_eq!(lowered.module.functions.len(), 3);
    assert_eq!(lowered.abi.functions.len(), 3);
    assert!(matches!(
        lowered.module.functions[1].program.blocks[0].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::machine::machine_ir::MachineTrapKind::Unreachable
        }
    ));
    assert!(matches!(
        lowered.module.functions[0].program.blocks[0].terminator,
        MachineTerminator::CallDirect {
            callee: crate::vm::machine::machine_ir::MachineFuncId(2),
            ..
        }
    ));
}

#[test]
fn lowers_memory_size_without_helper_boundary() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::MemorySize { mem_idx: 1 })
                        .unwrap(),
                    args: collections::vec![],
                    results: collections::vec![SsaValue(0)],
                },
            }],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Value {
                    op: SsaLeafOp::from_primitive(PrimitiveOpKind::MemorySize { mem_idx: 1 })
                        .unwrap(),
                    args: collections::vec![],
                    results: collections::vec![SsaValue(0)],
                },
            }],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Call(SsaCallOp::CallIndirect {
                    type_idx: 3,
                    table_idx: 0,
                    index_slot: call_base.advance(2),
                    args: crate::vm::middle::frame::FrameSpan::new(call_base, 2),
                    results: crate::vm::middle::frame::FrameSpan::new(call_base, 1),
                }),
            }],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("call_indirect lowering should succeed");

    assert_eq!(lowered.module.consts.len(), 1);

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 12);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[2].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::machine::machine_ir::MachineTrapKind::TableOutOfBounds
        }
    ));
    assert!(matches!(
        program.blocks[4].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::machine::machine_ir::MachineTrapKind::InvalidFunctionReference
        }
    ));
    assert!(matches!(
        program.blocks[6].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::machine::machine_ir::MachineTrapKind::IndirectCallTypeMismatch
        }
    ));
    assert!(matches!(
        program.blocks[7].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert_eq!(program.blocks[7].params.len(), 1);
    assert!(matches!(
        program.blocks[7].ops[0].kind,
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[8].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[8].ops[0].kind,
        MachineInstKind::Store { .. }
    ));
    assert!(matches!(
        program.blocks[9].terminator,
        MachineTerminator::CallIndirect {
            continuation: MachineBlockId(11),
            ..
        }
    ));
    let type_check_ops = &program.blocks[3].ops;
    assert!(!type_check_ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Move {
            dst,
            src: MachineValue::Reg(src),
            ..
        } if dst == src
    )));
    let type_check_scaled = match &type_check_ops[1].kind {
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Mul,
            dst,
            ..
        } => *dst,
        other => panic!("expected scaled-index multiply in type-check block, got {other:?}"),
    };
    let type_check_base = match &type_check_ops[2].kind {
        MachineInstKind::Load { dst, .. } => *dst,
        other => panic!("expected function-view base load in type-check block, got {other:?}"),
    };
    assert_ne!(type_check_scaled, type_check_base);
    let type_canon_offset = match &type_check_ops[5].kind {
        MachineInstKind::Load { addr, .. } => addr.offset,
        other => panic!("expected type-canon base load in type-check block, got {other:?}"),
    };
    assert_eq!(
        type_canon_offset,
        crate::vm::runtime::context::ctx_offset::TYPE_CANON_BASE as i32
    );
    assert!(matches!(
        type_check_ops[4].kind,
        MachineInstKind::Load {
            width: crate::vm::machine::machine_ir::MachineMemWidth::U32,
            extension: crate::vm::machine::machine_ir::MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[3].terminator,
        MachineTerminator::Branch {
            cond: crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                rhs: MachineValue::Reg(_),
                ..
            },
            ..
        }
    ));

    let dispatch_ops = &program.blocks[5].ops;
    assert!(!dispatch_ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Move {
            dst,
            src: MachineValue::Reg(src),
            ..
        } if dst == src
    )));
    let dispatch_scaled = match &dispatch_ops[1].kind {
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Mul,
            dst,
            ..
        } => *dst,
        other => panic!("expected scaled-index multiply in dispatch block, got {other:?}"),
    };
    let dispatch_base = match &dispatch_ops[2].kind {
        MachineInstKind::Load { dst, .. } => *dst,
        other => panic!("expected function-view base load in dispatch block, got {other:?}"),
    };
    assert_ne!(dispatch_scaled, dispatch_base);
    assert!(matches!(
        dispatch_ops[4].kind,
        MachineInstKind::Load {
            width: crate::vm::machine::machine_ir::MachineMemWidth::U32,
            extension: crate::vm::machine::machine_ir::MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        dispatch_ops[5].kind,
        MachineInstKind::Load {
            width: crate::vm::machine::machine_ir::MachineMemWidth::U32,
            extension: crate::vm::machine::machine_ir::MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[10].ops[0].kind,
        MachineInstKind::CallExternal(_)
    ));
    assert!(matches!(
        program.blocks[10].ops[1].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        program.blocks[10].ops[2].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        program.blocks[10].terminator,
        MachineTerminator::Jump(crate::vm::machine::machine_ir::MachineEdge {
            target: MachineBlockId(11),
            ..
        })
    ));
    assert!(matches!(
        program.blocks[11].terminator,
        MachineTerminator::Trap {
            kind: crate::vm::machine::machine_ir::MachineTrapKind::Unreachable
        }
    ));
}

#[test]
fn lowers_call_indirect_with_gp_word_width_on_32_bit_target() {
    let frame = plan_frame_layout(0, 6, 4);
    let call_base = frame.operand_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Call(SsaCallOp::CallIndirect {
                    type_idx: 3,
                    table_idx: 0,
                    index_slot: call_base.advance(2),
                    args: crate::vm::middle::frame::FrameSpan::new(call_base, 2),
                    results: crate::vm::middle::frame::FrameSpan::new(call_base, 1),
                }),
            }],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("call_indirect lowering should stay GP-word-sized on 32-bit targets");

    let program = &lowered.module.functions[0].program;
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch {
            cond: crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
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
            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
            op: MachineIntBinaryOp::Mul,
            ..
        }
    )));
    assert!(matches!(
        program.blocks[7].ops[0].kind,
        MachineInstKind::IntBinary {
            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn uses_canonical_u64_width_for_gp_word_frame_slots_on_32bit_targets() {
    let frame = plan_frame_layout(0, 1, 2);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 7 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot,
                        src: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot,
                        dst: SsaValue(1),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(1))],
                        results: collections::vec![],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32, ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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

    let caller = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![SsaInst {
                kind: SsaInstKind::Call(SsaCallOp::CallDirect {
                    callee: 1,
                    args: crate::vm::middle::frame::FrameSpan::new(caller_frame.operand_slot(1), 2),
                    results: crate::vm::middle::frame::FrameSpan::new(
                        caller_frame.operand_slot(0),
                        1,
                    ),
                }),
            }],
            terminator: SsaTerminator::TrapUnreachable,
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };
    let callee = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    callee_frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: crate::vm::machine::machine_ir::MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("32-bit direct local call lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    let store_widths: collections::Vec<_> = ops
        .iter()
        .filter_map(|op| match op.kind {
            MachineInstKind::Store { width, .. } => Some(width),
            _ => None,
        })
        .collect();
    // After the new local-call ABI rewrite, the call site emits no MIR-visible
    // stores at all: there is no software call-link record to write, so no
    // stores appear in the caller's pre-terminator op stream. The continuation
    // and caller-state propagation now travels via the backend-private call
    // record set up by the per-arch terminator lowering.
    assert!(
        store_widths.is_empty(),
        "expected no MIR-visible stores in the new call ABI, got {store_widths:?}"
    );
}

#[test]
fn lowers_global_get_and_set_without_helpers() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 9 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::GlobalSet { idx: 3 })
                            .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::GlobalGet { idx: 3 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 0 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::TableGet { table_idx: 1 })
                            .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
            kind: crate::vm::machine::machine_ir::MachineTrapKind::TableOutOfBounds
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 4,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
            op: crate::vm::machine::machine_ir::MachineConvertOp::I64ExtendI32U,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::IntBinary {
            width: crate::vm::machine::machine_ir::MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::TrapIf {
            kind: crate::vm::machine::machine_ir::MachineTrapKind::MemoryOutOfBounds,
            ..
        }
    )));
    assert!(matches!(
        ops.last().unwrap().kind,
        MachineInstKind::Load {
            width: crate::vm::machine::machine_ir::MachineMemWidth::U32,
            ..
        }
    ));
}

#[test]
fn lowers_i32_load_with_gp_word_bounds_ops_on_32_bit_target() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 4,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("i32.load should use GP-word bounds ops on 32-bit targets");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(!ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Convert {
            op: crate::vm::machine::machine_ir::MachineConvertOp::I64ExtendI32U,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::IntBinary {
            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::TrapIf {
            cond: crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                ..
            },
            ..
        }
    )));
}

#[test]
fn lowers_32bit_memory_bounds_checks_with_wraparound_traps() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 1,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
                crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                    width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                    kind: MachineCompareKind::Lt,
                    sign: crate::vm::machine::machine_ir::MachineSign::Unsigned,
                    rhs: MachineValue::Imm64(value),
                    ..
                },
            kind: crate::vm::machine::machine_ir::MachineTrapKind::MemoryOutOfBounds,
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

#[cfg(sf_has_guard_pages)]
#[test]
fn keeps_explicit_mem0_bounds_checks_for_32bit_multiword_gp_accesses_with_guard_pages() {
    let frame = plan_frame_layout(0, 2, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const {
                            value: 0x8877_6655_4433_2211,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Store {
                            offset: 0,
                            memidx: 0,
                        })
                        .unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(0)),
                            SsaOperand::Value(SsaValue(1))
                        ],
                        results: collections::vec![],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        use_guard_pages: true,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
                crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                    width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                    kind: MachineCompareKind::Lt,
                    sign: crate::vm::machine::machine_ir::MachineSign::Unsigned,
                    rhs: MachineValue::Imm64(value),
                    ..
                },
            kind: crate::vm::machine::machine_ir::MachineTrapKind::MemoryOutOfBounds,
        } = &inst.kind
        {
            if *value == 8 {
                saw_access_wrap = true;
            }
        }
        if let MachineInstKind::TrapIf {
            cond:
                crate::vm::machine::machine_ir::MachineBranchCond::IntCompare {
                    width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
                    kind: MachineCompareKind::Gt,
                    sign: crate::vm::machine::machine_ir::MachineSign::Unsigned,
                    ..
                },
            kind: crate::vm::machine::machine_ir::MachineTrapKind::MemoryOutOfBounds,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::RefNull).unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::RefIsNull).unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("ref.is_null should use GP-word null constants on 32-bit targets");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            src: MachineValue::Imm64(value),
            ..
        } if value == u32::MAX as u64
    ));
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::IntCompare {
            width: crate::vm::machine::machine_ir::MachineIntWidth::I32,
            rhs: MachineValue::Imm64(value),
            ..
        } if value == u32::MAX as u64
    ));
}

#[test]
fn omits_zero_offset_add_in_bounds_check_setup() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Load {
                            offset: 0,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![SsaValue(1)],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
fn threads_live_linear_values_through_split_continuation_params() {
    let frame = plan_frame_layout(1, 2, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Const { value: 3 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::TableGet { table_idx: 1 })
                            .unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(0))],
                        results: collections::vec![SsaValue(2)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I64Add).unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(1)),
                            SsaOperand::Value(SsaValue(2))
                        ],
                        results: collections::vec![SsaValue(3)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot: frame.local_slot(0),
                        src: SsaValue(3),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
        .collect::<collections::Vec<_>>();

    assert_eq!(else_edge.target, continuation.id);
    assert_eq!(else_edge.args, expected_args);
    assert!(continuation
        .params
        .iter()
        .any(|param| param.reg == MachineReg(5)));
    assert!(matches!(
        continuation.ops[0].kind,
        MachineInstKind::Convert {
            op: crate::vm::machine::machine_ir::MachineConvertOp::I64ExtendI32U,
            ..
        }
    ));
}

#[test]
fn lowers_f32_store_inline_with_trap_if_preserving_fp_linear_value_width() {
    let frame = plan_frame_layout(0, 0, 3);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 8 })
                            .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::F32Const {
                            value: 0x3f800000,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::F32Abs).unwrap(),
                        args: collections::vec![SsaOperand::Value(SsaValue(1))],
                        results: collections::vec![SsaValue(2)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::F32Store {
                            offset: 4,
                            memidx: 1,
                        })
                        .unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(0)),
                            SsaOperand::Value(SsaValue(2))
                        ],
                        results: collections::vec![],
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("inline memory store should preserve FP linear-value widths");

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
                kind: crate::vm::machine::machine_ir::MachineTrapKind::MemoryOutOfBounds,
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
            } if reg.0 >= lowered.module.config.first_fp_reg()
        )
    }));
}

#[test]
fn lowers_f32_const_to_fp_machine_const() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(0, 1, 1);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::F32Const {
                            value: 0x4120_0000,
                        })
                        .unwrap(),
                        args: collections::vec![],
                        results: collections::vec![SsaValue(0)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Spill {
                        slot: frame.operand_slot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![ValueType::F32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 1,
        }],
    })
    .expect("f32 const should lower");

    let program = &lowered.module.functions[0].program;
    assert!(program.blocks[0].ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::FloatConst {
                width: crate::vm::machine::machine_ir::MachineFloatWidth::F32,
                bits: 0x4120_0000,
                dst,
            } if dst.0 >= lowered.module.config.first_fp_reg()
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot: frame.local_slot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Spill {
                        slot: frame.operand_slot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![ValueType::F64],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
        crate::vm::machine::machine_ir::is_fp_reg(load_dst, lowered.module.config),
        "typed F64 LocalGet must allocate into FP bank, got GP reg {}",
        load_dst.0,
    );
}

#[test]
fn untyped_slot_load_stays_in_gp_bank() {
    let frame = plan_frame_layout(1, 2, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot: frame.local_slot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::Spill {
                        slot: frame.operand_slot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::Return {
                results: Some(crate::vm::middle::frame::FrameSpan::new(
                    frame.operand_slot(0),
                    1,
                )),
            },
        }],
        value_types: collections::vec![],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
        crate::vm::machine::machine_ir::is_gp_reg(load_dst, lowered.module.config),
        "untyped LocalGet must stay in GP bank, got FP reg {}",
        load_dst.0,
    );
}

#[test]
fn f32_block_params_keep_f32_width() {
    use crate::value_type::ValueType;

    let frame = plan_frame_layout(1, 1, 2);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![],
        local_slot_info: collections::vec![],
        blocks: collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::vec![],
                ops: collections::vec![SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot: frame.local_slot(0),
                        dst: SsaValue(0),
                    },
                }],
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![SsaBinding {
                        param: SsaValue(1),
                        value: SsaValue(0),
                    }],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![SsaValue(1)],
                ops: collections::vec![],
                terminator: SsaTerminator::Return { results: None },
            },
        ],
        value_types: collections::vec![ValueType::F32, ValueType::F32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
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
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::F32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot: frame.local_slot(0),
                        dst: SsaValue(0),
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot: frame.local_slot(0),
                        src: SsaValue(0),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::F32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 1, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("typed f32 cached locals should lower");

    let program = &lowered.module.functions[0].program;
    assert_eq!(
        program.fp_reg_init_widths,
        collections::vec![None, None, None],
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

// --- Source-aliasing regression tests ---

/// LocalGetCache should NOT emit a Move — the SSA value should alias the
/// cache register directly so that downstream ops use it without a copy.
#[test]
fn local_get_cache_source_aliases_without_move() {
    let frame = plan_frame_layout(1, 2, 2);
    let slot = frame.local_slot(0);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32],
        local_slot_info: collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                // get_cache produces v0 aliased to the cache register
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot,
                        dst: SsaValue(0),
                    },
                },
                // i32.add(v0, #1) should use cache_reg directly as an operand
                SsaInst {
                    kind: SsaInstKind::Value {
                        op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                        args: collections::vec![
                            SsaOperand::Value(SsaValue(0)),
                            SsaOperand::Const(1),
                        ],
                        results: collections::vec![SsaValue(1)],
                    },
                },
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot,
                        src: SsaValue(1),
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32, ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    // Expected: Load(ensure cache) + Move(const 1) + IntBinary(Add) + Store
    // NO Move from cache→linear for LocalGetCache.
    let cache_to_linear_moves: collections::Vec<_> = ops
        .iter()
        .filter(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    src: MachineValue::Reg(_),
                    ..
                }
            )
        })
        .collect();
    assert!(
        cache_to_linear_moves.is_empty(),
        "LocalGetCache must source-alias without emitting a cache-to-linear Move; found {:?}",
        cache_to_linear_moves
            .iter()
            .map(|i| &i.kind)
            .collect::<collections::Vec<_>>()
    );
}

/// When LocalSetCache overwrites a cache register that still has a live
/// aliased value, materialize_cache_aliases must copy the alias out first.
#[test]
fn local_set_cache_materializes_live_alias_before_overwrite() {
    let frame = plan_frame_layout(2, 2, 2);
    let slot0 = frame.local_slot(0);
    let slot1 = frame.local_slot(1);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32, ValueType::I32],
        local_slot_info: collections::vec![
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
        ],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                // v0 aliases cache_reg of fp[0]
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: slot0,
                        dst: SsaValue(0)
                    },
                },
                // v1 aliases cache_reg of fp[1]
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: slot1,
                        dst: SsaValue(1)
                    },
                },
                // Now overwrite fp[0] with v1 — v0 is still live (used below)
                // This must materialize v0 out of cache_reg(fp[0]) first.
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: slot0,
                        src: SsaValue(1)
                    },
                },
                // Use v0 — if alias wasn't materialized, this would read the
                // new value (wrong).
                SsaInst {
                    kind: SsaInstKind::LocalSetSlot {
                        slot: slot1,
                        src: SsaValue(0)
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32, ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    // There must be at least one LinearValue Move that materializes the
    // aliased value out of the cache register before the overwrite.
    let materialization_moves: collections::Vec<_> = ops
        .iter()
        .filter(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    src: MachineValue::Reg(_),
                    ..
                }
            )
        })
        .collect();
    assert!(
        !materialization_moves.is_empty(),
        "must materialize aliased live value before cache overwrite; ops={:?}",
        ops.iter().map(|i| &i.kind).collect::<collections::Vec<_>>()
    );
}

/// A local.get → local.set to a different slot (copy pattern) should
/// produce exactly one Move (cache→cache), not two moves through a
/// linear register.
#[test]
fn local_get_cache_to_set_cache_different_slot_single_move() {
    let frame = plan_frame_layout(2, 2, 2);
    let slot0 = frame.local_slot(0);
    let slot1 = frame.local_slot(1);
    let ssa = SsaProgram {
        entry: SsaTarget(0),
        local_slot_types: collections::vec![ValueType::I32, ValueType::I32],
        local_slot_info: collections::vec![
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
        ],
        blocks: collections::vec![SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![
                // v0 aliases cache_reg of fp[0]
                SsaInst {
                    kind: SsaInstKind::LocalGetCache {
                        slot: slot0,
                        dst: SsaValue(0)
                    },
                },
                // set fp[1] = v0 (v0's last use)
                SsaInst {
                    kind: SsaInstKind::LocalSetCache {
                        slot: slot1,
                        src: SsaValue(0)
                    },
                },
            ],
            terminator: SsaTerminator::Return { results: None },
        }],
        value_types: collections::vec![ValueType::I32],
        value_sink_local: collections::vec![],
        block_entry_cached_slots: collections::vec![],
        block_cfg_origins: collections::vec![],
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: crate::vm::machine::machine_ir::MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("lowering should succeed");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    // Expected: Load(fp[0]) + Load(fp[1]) + Move(CachedLocal, cache1 <- cache0)
    // The Move from LocalGetCache→LocalSetCache should be a single CachedLocal move.
    let reg_moves: collections::Vec<_> = ops
        .iter()
        .filter(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    src: MachineValue::Reg(_),
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        reg_moves.len(),
        1,
        "get_cache(fp[0]) → set_cache(fp[1]) should produce exactly one reg-to-reg Move; got {:?}",
        reg_moves
            .iter()
            .map(|i| &i.kind)
            .collect::<collections::Vec<_>>()
    );
}
