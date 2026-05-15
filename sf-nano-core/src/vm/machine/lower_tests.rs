use crate::collections;
#[cfg(sf_has_simd)]
use crate::opcodes::OpcodeFD;
use crate::value_type::ValueType;

use crate::vm::{
    backend::BackendConfig,
    machine::{
        lower_module,
        machine_ir::{
            is_fp_reg, is_gp_reg, MachineAddr, MachineBlockId, MachineBlockParam,
            MachineBranchCond, MachineCallResults, MachineCallTarget, MachineCompareKind,
            MachineConvertOp, MachineEdge, MachineFloatWidth, MachineFrameRegion, MachineFuncId,
            MachineFunction, MachineInstKind, MachineIntBinaryOp, MachineIntWidth,
            MachineLoadExtension, MachineMemWidth, MachineModule, MachineReg, MachineRegOwner,
            MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
            MACHINE_FIXED_REG_COUNT,
        },
        LowerFunctionInput, LowerModuleInput,
    },
    middle::{
        frame::{plan_frame_layout, FrameSpan},
        ssa_ir::{
            ir::{
                LocalSlotInfo, SsaBinding, SsaBlock, SsaCallArgs, SsaCallOp, SsaCallOperandLoc,
                SsaEdge, SsaInst, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
            },
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
        HOST_GP_UNIT_BYTES,
        gp_dynamic_budget,
        fp_dynamic_budget,
        HOST_CALL_SCRATCH_SLOTS,
    )
}

fn host_backend_config_with_preserved(
    gp_volatile_dynamic: u8,
    gp_preserved_dynamic: u8,
    gp_internal_scratch: u8,
) -> BackendConfig {
    let mut config = BackendConfig::with_volatility(
        HOST_GP_UNIT_BYTES,
        gp_volatile_dynamic,
        gp_preserved_dynamic,
        gp_internal_scratch,
        0,
        0,
        gp_volatile_dynamic.min(4),
        0,
        false,
        HOST_CALL_SCRATCH_SLOTS,
    );
    config.preserved_cache_min_local_call_crosses = 2;
    config
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
    BackendConfig::new(4, gp_dynamic_budget, fp_dynamic_budget, 8)
}

fn assert_valid_32bit_gp_target(module: &MachineModule, backend: BackendConfig) {
    assert!(backend.is_32bit_gp_target());
    let max_gp_regs = MACHINE_FIXED_REG_COUNT + backend.gp_dynamic_budget as u16;
    module
        .validate_32bit_gp_target(max_gp_regs)
        .unwrap_or_else(|err| panic!("32-bit lowered module must already validate: {err}"));
}

fn test_call_args(span: FrameSpan, param_types: &[ValueType]) -> SsaCallArgs {
    let mut types = collections::Vec::new();
    for idx in 0..span.count as usize {
        types.push(param_types.get(idx).copied().unwrap_or(ValueType::I64));
    }
    SsaCallArgs {
        frame_base: span.start,
        total_params: span.count,
        param_types: types,
        stack_prefix_count: span.count,
        live_suffix: collections::Vec::new(),
    }
}

fn test_i64_call_args(span: FrameSpan) -> SsaCallArgs {
    test_call_args(span, &[])
}

#[test]
fn lowers_simple_slot_and_add_block() {
    let frame = plan_frame_layout(1, 4, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_cache(frame.local_slot(0), SsaValue(0)));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 1 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::I32Add).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(2),
                [
                    SsaOperand::value(SsaValue(0)),
                    SsaOperand::value(SsaValue(1)),
                ],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_slot(frame.local_slot(0), SsaValue(2)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
            MachineFrameRegion {
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 11 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 22 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 1 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(2),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __eidx = __blk0.push_extra_arg(SsaOperand::value(SsaValue(2)));
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::Select).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(3),
                [
                    SsaOperand::value(SsaValue(0)),
                    SsaOperand::value(SsaValue(1)),
                ],
                __eidx,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 3, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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

#[cfg(sf_has_simd)]
#[test]
fn lowers_v128_binary_and_publishes_raw_slot_handle() {
    let frame = plan_frame_layout(1, 0, HOST_CALL_SCRATCH_SLOTS);
    let slot = frame.local_slot(0);
    let mut ssa = SsaProgram::default();
    ssa.entry = SsaTarget(0);
    ssa.local_slot_types = collections::vec![ValueType::V128];
    ssa.local_slot_info = collections::vec![LocalSlotInfo::default()];
    ssa.block_entry_cached_slots = collections::vec![collections::Vec::new()];
    ssa.block_cfg_origins = collections::vec![];
    ssa.value_types = collections::vec![ValueType::V128, ValueType::V128, ValueType::V128];
    ssa.value_sink_local = collections::vec![];

    let lhs = ssa
        .intern_primitive(PrimitiveOpKind::V128Const { value: [1; 16] })
        .expect("lhs const");
    let rhs = ssa
        .intern_primitive(PrimitiveOpKind::V128Const { value: [2; 16] })
        .expect("rhs const");
    let add = ssa
        .intern_primitive(PrimitiveOpKind::I32x4Add)
        .expect("simd add");

    ssa.blocks.push(SsaBlock {
        id: SsaTarget(0),
        params: collections::Vec::new(),
        ops: collections::vec![
            SsaInst::primitive(lhs, SsaValue(0), [SsaOperand::NONE, SsaOperand::NONE], 0),
            SsaInst::primitive(rhs, SsaValue(1), [SsaOperand::NONE, SsaOperand::NONE], 0),
            SsaInst::primitive(
                add,
                SsaValue(2),
                [
                    SsaOperand::value(SsaValue(0)),
                    SsaOperand::value(SsaValue(1))
                ],
                0,
            ),
            SsaInst::local_set_slot(slot, SsaValue(2)),
        ],
        extra_args: collections::Vec::new(),
        terminator: SsaTerminator::Return { results: None },
    });

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 6),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa,
            result_count: 0,
        }],
    })
    .expect("SIMD MIR lowering should succeed before native codegen");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::V128Const { bytes, .. } if bytes == [1; 16]
        )
    }));
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::V128Const { bytes, .. } if bytes == [2; 16]
        )
    }));
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::SimdBinary {
                opcode,
                dst_ty: MachineStorageType::V128,
                ..
            } if opcode == OpcodeFD::I32X4_ADD as u32
        )
    }));
    assert!(ops
        .iter()
        .any(|inst| matches!(inst.kind, MachineInstKind::V128ToRaw { .. })));
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Store {
                width: MachineMemWidth::U64,
                ..
            }
        )
    }));
}

#[cfg(sf_has_simd)]
#[test]
fn lowers_v128_local_get_slot_through_raw_handle_boundary() {
    let frame = plan_frame_layout(1, 0, HOST_CALL_SCRATCH_SLOTS);
    let slot = frame.local_slot(0);
    let mut ssa = SsaProgram::default();
    ssa.entry = SsaTarget(0);
    ssa.local_slot_types = collections::vec![ValueType::V128];
    ssa.local_slot_info = collections::vec![LocalSlotInfo::default()];
    ssa.block_entry_cached_slots = collections::vec![collections::Vec::new()];
    ssa.block_cfg_origins = collections::vec![];
    ssa.value_types = collections::vec![ValueType::V128, ValueType::V128];
    ssa.value_sink_local = collections::vec![];

    let not = ssa
        .intern_primitive(PrimitiveOpKind::V128Not)
        .expect("simd unary");
    ssa.blocks.push(SsaBlock {
        id: SsaTarget(0),
        params: collections::Vec::new(),
        ops: collections::vec![
            SsaInst::local_get_slot(slot, SsaValue(0)),
            SsaInst::primitive(
                not,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0,
            ),
            SsaInst::local_set_slot(slot, SsaValue(1)),
        ],
        extra_args: collections::Vec::new(),
        terminator: SsaTerminator::Return { results: None },
    });

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 6),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa,
            result_count: 0,
        }],
    })
    .expect("SIMD local boundary lowering should succeed before native codegen");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                width: MachineMemWidth::U64,
                ..
            }
        )
    }));
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::V128FromRaw {
                owner: MachineRegOwner::LinearValue,
                ..
            }
        )
    }));
    assert!(ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::SimdUnary {
                opcode,
                dst_ty: MachineStorageType::V128,
                ..
            } if opcode == OpcodeFD::V128_NOT as u32
        )
    }));
    assert!(ops
        .iter()
        .any(|inst| matches!(inst.kind, MachineInstKind::V128ToRaw { .. })));
}

#[test]
fn native_backend_requires_at_least_one_gp_linear_value_register() {
    let frame = plan_frame_layout(0, 0, 3);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        ssa.blocks.push(__blk0);
        ssa
    };

    let err = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 0, 0, 0),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect_err("zero GP-dynamic native backend must be rejected");

    assert!(
        err.message().contains("at least one GP dynamic register"),
        "unexpected error: {err}"
    );
}

#[test]
fn projects_return_results_and_helper_scratch_from_frame_plan() {
    let frame = plan_frame_layout(2, 6, 6);
    let result_span = FrameSpan::new(frame.operand_slot(0), 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(result_span),
            },
        };
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
        Some(MachineFrameRegion {
            base_slot: frame.call_scratch.unwrap().start.0,
            slots: frame.call_scratch.unwrap().count,
        })
    );
    assert_eq!(
        runtime.functions[0].return_results,
        Some(MachineFrameRegion {
            base_slot: result_span.start.0,
            slots: result_span.count,
        })
    );
}

#[test]
fn rejects_inconsistent_return_result_spans() {
    let frame = plan_frame_layout(0, 4, 3);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(frame.operand_slot(0), 1)),
            },
        };
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(frame.operand_slot(1), 1)),
            },
        };
        ssa.blocks.push(__blk1);
        ssa
    };

    let err = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(frame.operand_slot(0), 1)),
            },
        };
        ssa.blocks.push(__blk1);
        ssa
    };

    let err = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![SsaValue(0), SsaValue(1)],
            ops: collections::vec![],
            extra_args: collections::vec![],
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
        };
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![SsaValue(2), SsaValue(3)],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I64, ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![SsaValue(0)],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![SsaBinding {
                    param: SsaValue(1),
                    value: SsaValue(0),
                }],
            }),
        };
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![SsaValue(1)],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
        MachineBlockParam {
            reg: MachineReg(4),
            ty: MachineStorageType::GpWord,
            owner: MachineRegOwner::LinearValue,
        }
    ));
    assert!(matches!(
        block0.params[1],
        MachineBlockParam {
            reg: MachineReg(5),
            ty: MachineStorageType::GpWord,
            owner: MachineRegOwner::LinearValue,
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![
            ValueType::I64,
            ValueType::I64,
            ValueType::I64,
            ValueType::I64
        ];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const {
                    value: 0x0123_4567_89ab_cdef,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_slot(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_get_slot(frame.local_slot(0), SsaValue(1)));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const {
                    value: 0x1111_2222_3333_4444,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(2),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::I64Add).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(3),
                [
                    SsaOperand::value(SsaValue(1)),
                    SsaOperand::value(SsaValue(2)),
                ],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I64];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: true,
        }];
        ssa.block_entry_cached_slots = collections::vec![collections::vec![]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0.ops.push(SsaInst::local_get_slot(slot, SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::spill(frame.operand_slot(0), SsaValue(0)));
        __blk0.ops.push(SsaInst::local_reserve_cache(slot));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 slot load should not force a cache binding");

    assert_valid_32bit_gp_target(&lowered.module, backend);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(
        ops.iter()
            .filter(|inst| {
                matches!(
                    inst.kind,
                    MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        width: MachineMemWidth::U32,
                        ..
                    }
                )
            })
            .count(),
        2,
        "32-bit i64 local.get should load its two frame lanes directly"
    );
}

#[test]
fn gp32_i64_slot_set_stays_frame_based_for_explicit_cache_candidate() {
    let backend = gp32_backend_config(1, 4, 0, 2);
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I64];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }];
        ssa.block_entry_cached_slots = collections::vec![collections::vec![]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const {
                    value: 0x0123_4567_89ab_cdef,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0.ops.push(SsaInst::local_set_slot(slot, SsaValue(0)));
        __blk0.ops.push(SsaInst::local_reserve_cache(slot));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
fn gp32_i64_local_get_set_cache_same_slot_emits_no_self_moves() {
    let backend = gp32_backend_config(1, 4, 0, 2);
    let frame = plan_frame_layout(1, 1, 2);
    let slot = frame.local_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I64];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0.ops.push(SsaInst::local_get_cache(slot, SsaValue(0)));
        __blk0.ops.push(SsaInst::local_set_cache(slot, SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 cache identity set should lower without self moves");

    assert_valid_32bit_gp_target(&lowered.module, backend);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert!(!ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Move {
            owner: MachineRegOwner::CachedLocal,
            src: MachineValue::Reg(_),
            ..
        }
    )), "i64 LocalGetCache -> LocalSetCache of the same slot should not emit cache self moves; ops={ops:?}");
}

#[test]
fn gp32_i64_local_set_cache_materializes_live_pair_alias_before_overwrite() {
    let backend = gp32_backend_config(1, 6, 0, 2);
    let frame = plan_frame_layout(2, 2, 2);
    let slot0 = frame.local_slot(0);
    let slot1 = frame.local_slot(1);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I64, ValueType::I64];
        ssa.local_slot_info = collections::vec![
            LocalSlotInfo {
                is_param: false,
                reads_before_write: true,
            },
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
        ];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I64, ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_cache(slot0, SsaValue(0)));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 7 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(slot0, SsaValue(1)));
        __blk0.ops.push(SsaInst::local_set_slot(slot1, SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("32-bit i64 cache overwrite should materialize live pair aliases");

    assert_valid_32bit_gp_target(&lowered.module, backend);
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    let cached_regs: collections::Vec<_> = ops
        .iter()
        .filter_map(|inst| match inst.kind {
            MachineInstKind::Load {
                owner: MachineRegOwner::CachedLocal,
                dst,
                ..
            } => Some(dst),
            _ => None,
        })
        .collect();
    assert_eq!(
        cached_regs.len(),
        2,
        "expected cached i64 pair loads; ops={ops:?}"
    );
    let materialized_halves = ops
        .iter()
        .filter(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    src: MachineValue::Reg(src),
                    ..
                } if cached_regs.contains(&src)
            )
        })
        .count();
    assert_eq!(
        materialized_halves, 2,
        "live i64 cache alias should be materialized as both halves before overwrite; ops={ops:?}"
    );
}

#[test]
fn lowers_i64_global_get_set_directly_to_legal_32bit_machineir() {
    let backend = gp32_backend_config(0, 4, 0, 2);
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I64, ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::GlobalSet { idx: 3 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::GlobalGet { idx: 3 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![
            ValueType::I32,
            ValueType::I64,
            ValueType::I32,
            ValueType::I64
        ];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const {
                    value: 0x8877_6655_4433_2211,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Store {
                    offset: 0,
                    memidx: 0,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [
                    SsaOperand::value(SsaValue(0)),
                    SsaOperand::value(SsaValue(1)),
                ],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(2),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Load {
                    offset: 0,
                    memidx: 0,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(3),
                [SsaOperand::value(SsaValue(2)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![];
        caller.local_slot_info = collections::vec![];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![];
        caller.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = caller.push_call_op(SsaCallOp::CallDirect {
                callee: 1,
                args: test_i64_call_args(FrameSpan::new(caller_frame.operand_slot(1), 2)),
                results: FrameSpan::new(caller_frame.operand_slot(0), 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        caller.blocks.push(__blk0);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(callee_frame.operand_slot(0), 1)),
            },
        };
        callee.blocks.push(__blk0);
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_cache(frame.local_slot(0), SsaValue(0)));
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::Drop).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 7 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(1)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 7 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 1, 0, 0),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
fn lowers_call_runtime_through_frame_metadata_without_helper_scratch() {
    let frame = plan_frame_layout(0, 2, 3);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallDirect {
                callee: 7,
                args: test_i64_call_args(FrameSpan::new(frame.operand_slot(0), 2)),
                results: FrameSpan::new(frame.operand_slot(0), 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("runtime-call lowering should succeed");

    assert_eq!(lowered.module.consts.len(), 1);
    // Under the new local-call ABI the dead call-link half of the frame
    // plan is gone, so the helper_scratch field directly mirrors whatever
    // the frame planner reserved. For 1A this still includes the
    // historical 3-slot reservation from `HOST_CALL_SCRATCH_SLOTS`; a
    // future cleanup can drop it to 0 when no helper actually needs it.
    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0].kind, MachineInstKind::CallRuntime(_)));
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Load { .. }));
}

#[test]
fn coalesces_dead_i64_const_directly_into_uncached_store_slot() {
    let frame = plan_frame_layout(0, 1, 1);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const {
                    value: 0x0123_4567_89ab_cdef,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_slot(frame.local_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
fn flushes_and_reloads_cached_locals_around_call_runtime() {
    let frame = plan_frame_layout(1, 2, 3);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_drop_cache(frame.local_slot(0)));
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallDirect {
                callee: 7,
                args: test_i64_call_args(FrameSpan::new(frame.operand_slot(0), 1)),
                results: FrameSpan::new(frame.operand_slot(0), 0),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("runtime-call lowering should succeed with cached locals");

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
    // ops[2]: runtime call
    assert!(matches!(ops[2].kind, MachineInstKind::CallRuntime(_)));
    // ops[3-4]: mem0 cache reg refresh only; there is no implicit cached-local reload
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn skips_dead_cached_local_reload_after_direct_runtime_call() {
    let frame = plan_frame_layout(1, 2, 3);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_drop_cache(frame.local_slot(0)));
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallDirect {
                callee: 7,
                args: test_i64_call_args(FrameSpan::new(frame.operand_slot(0), 1)),
                results: FrameSpan::new(frame.operand_slot(0), 0),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("direct runtime call lowering should preserve explicit cache protocol");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    assert_eq!(ops.len(), 5);
    assert!(matches!(ops[0].kind, MachineInstKind::Move { .. }));
    assert!(matches!(ops[1].kind, MachineInstKind::Store { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::CallRuntime(_)));
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn flushes_and_reloads_cached_locals_around_runtime_helpers() {
    let frame = plan_frame_layout(1, 2, 3);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 5 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_drop_cache(frame.local_slot(0)));
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallDirect {
                callee: 7,
                args: test_i64_call_args(FrameSpan::new(frame.operand_slot(0), 1)),
                results: FrameSpan::new(frame.operand_slot(0), 0),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    // ops[2]: call helper (call_runtime)
    assert!(matches!(ops[2].kind, MachineInstKind::CallRuntime(_)));
    // ops[3-4]: reload mem0 cache regs only
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_direct_local_call_with_continuation_block() {
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![];
        caller.local_slot_info = collections::vec![];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![];
        caller.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = caller.push_call_op(SsaCallOp::CallDirect {
                callee: 1,
                args: test_i64_call_args(FrameSpan::new(caller_frame.operand_slot(1), 2)),
                results: FrameSpan::new(caller_frame.operand_slot(0), 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        caller.blocks.push(__blk0);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(callee_frame.operand_slot(0), 1)),
            },
        };
        callee.blocks.push(__blk0);
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
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
    match call_block.terminator {
        MachineTerminator::Call {
            target: MachineCallTarget::Direct(callee),
            frame_delta,
            ref args,
            ref results,
            ref success,
        } => {
            assert_eq!(callee, MachineFuncId(1));
            assert_eq!(frame_delta, u32::from(caller_frame.operand_slot(1).0) * 8);
            assert_eq!(args.frame_params.base_slot, caller_frame.operand_slot(1).0);
            assert_eq!(args.frame_params.slots, 2);
            assert!(matches!(
                results,
                MachineCallResults::FrameFallback {
                    callee_results,
                    caller_results,
                } if *callee_results == (MachineFrameRegion {
                    base_slot: callee_frame.operand_slot(0).0,
                    slots: 1,
                }) && *caller_results == (MachineFrameRegion {
                    base_slot: caller_frame.operand_slot(0).0,
                    slots: 1,
                })
            ));
            assert_eq!(success.target, MachineBlockId(1));
            assert!(success.args.is_empty());
        }
        ref other => panic!("expected direct call terminator, got {other:?}"),
    };
    assert!(matches!(
        call_block.ops[0].kind,
        MachineInstKind::IntBinary {
            lhs: MachineValue::Reg(MachineReg(1)),
            rhs: MachineValue::Imm64(offset),
            ..
        } if offset == u64::from(caller_frame.operand_slot(1).0) * 8
    ));
    assert!(
        call_block.ops.iter().any(|inst| matches!(
            inst.kind,
            MachineInstKind::TrapIf {
                kind: MachineTrapKind::StackOverflow,
                ..
            }
        )),
        "direct-call lowering should emit a stack-overflow precheck"
    );
    let continuation = &caller_program.blocks[1];
    assert!(continuation.params.is_empty());
    assert!(continuation.ops.is_empty());
    assert!(matches!(
        continuation.terminator,
        MachineTerminator::Trap {
            kind: MachineTrapKind::Unreachable
        }
    ));
}

#[test]
fn flushes_cached_local_before_second_direct_call() {
    let caller_frame = plan_frame_layout(2, 1, 4);
    let callee_frame = plan_frame_layout(0, 1, 4);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![ValueType::I32];
        caller.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![];
        caller.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = caller.push_call_op(SsaCallOp::CallDirect {
                callee: 1,
                args: test_i64_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0)),
                results: FrameSpan::new(caller_frame.operand_slot(0), 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        __blk0
            .ops
            .push(SsaInst::fill(caller_frame.operand_slot(0), SsaValue(0)));
        __blk0.ops.push(SsaInst::local_set_cache(
            caller_frame.local_slot(0),
            SsaValue(0),
        ));
        __blk0
            .ops
            .push(SsaInst::local_drop_cache(caller_frame.local_slot(0)));
        {
            let __idx = caller.push_call_op(SsaCallOp::CallDirect {
                callee: 1,
                args: test_i64_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0)),
                results: FrameSpan::new(caller_frame.operand_slot(0), 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        caller.blocks.push(__blk0);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(callee_frame.operand_slot(0), 1)),
            },
        };
        callee.blocks.push(__blk0);
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
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
                owner: MachineRegOwner::LinearValue,
                addr,
                ..
            } if *addr
                == MachineAddr {
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
                    == MachineAddr {
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
        MachineTerminator::Call {
            target: MachineCallTarget::Direct(MachineFuncId(1)),
            success: MachineEdge {
                target: MachineBlockId(2),
                ..
            },
            ..
        }
    ));
}

#[test]
fn direct_local_call_carries_clean_non_ref_cache_in_preserved_reg() {
    let caller_frame = plan_frame_layout(1, 1, 4);
    let callee_frame = plan_frame_layout(0, 0, 4);
    let slot = caller_frame.local_slot(0);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![ValueType::I32];
        caller.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![ValueType::I32, ValueType::I32, ValueType::I32];
        caller.value_sink_local = collections::vec![];
        let mut block = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(0)));
        let drop_idx = caller.intern_primitive(PrimitiveOpKind::Drop).unwrap();
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
            0,
        ));
        let call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(1)));
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(1)), SsaOperand::NONE],
            0,
        ));
        let second_call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(second_call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(2)));
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(2)), SsaOperand::NONE],
            0,
        ));
        caller.blocks.push(block);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        callee.blocks.push(SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        });
        callee
    };

    let backend = host_backend_config_with_preserved(0, 3, 2);
    let lowered = lower_module(LowerModuleInput {
        backend,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call should carry preserved cached local");

    let caller_program = &lowered.module.functions[0].program;
    assert_eq!(caller_program.blocks.len(), 3);
    let call_block = &caller_program.blocks[0];
    let continuation = &caller_program.blocks[1];
    let MachineTerminator::Call {
        ref success,
        target: MachineCallTarget::Direct(MachineFuncId(1)),
        ..
    } = call_block.terminator
    else {
        panic!(
            "expected direct local call terminator, got {:?}",
            call_block.terminator
        );
    };
    assert_eq!(success.args.len(), 1);
    assert!(
        matches!(success.args[0], MachineValue::Reg(reg) if backend.gp_preserved_dynamic > 0 && reg.0 == MACHINE_FIXED_REG_COUNT)
    );
    assert_eq!(continuation.params.len(), 1);
    assert_eq!(continuation.params[0].owner, MachineRegOwner::CachedLocal);
    assert!(
        call_block
            .ops
            .iter()
            .chain(continuation.ops.iter())
            .chain(caller_program.blocks[2].ops.iter())
            .all(|inst| !matches!(inst.kind, MachineInstKind::Store { .. })),
        "clean non-ref cache in a preserved reg should not be stored around the local call; call_ops={:?}, cont_ops={:?}",
        call_block.ops,
        continuation.ops
    );
    assert!(
        continuation.ops.iter().all(|inst| !matches!(
            inst.kind,
            MachineInstKind::Load {
                owner: MachineRegOwner::CachedLocal,
                ..
            }
        )),
        "continued local.get_cache should use the carried cache register, not reload; ops={:?}",
        continuation.ops
    );
}

#[test]
fn direct_local_call_places_call_live_clean_cache_in_preserved_reg_when_available() {
    let caller_frame = plan_frame_layout(1, 0, 4);
    let callee_frame = plan_frame_layout(0, 0, 4);
    let slot = caller_frame.local_slot(0);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![ValueType::I32];
        caller.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: true,
        }];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![ValueType::I32, ValueType::I32, ValueType::I32];
        caller.value_sink_local = collections::vec![];
        let mut block = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(0)));
        let drop_idx = caller.intern_primitive(PrimitiveOpKind::Drop).unwrap();
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
            0,
        ));
        let call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(1)));
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(1)), SsaOperand::NONE],
            0,
        ));
        let second_call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(second_call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(2)));
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(2)), SsaOperand::NONE],
            0,
        ));
        caller.blocks.push(block);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        callee.blocks.push(SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::Vec::new(),
            extra_args: collections::Vec::new(),
            terminator: SsaTerminator::Return { results: None },
        });
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config_with_preserved(3, 3, 0),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call should lower with a preserved cache lane");

    let caller_function = &lowered.module.functions[0];
    assert!(
        caller_function
            .preserved_clobbers
            .contains(&MachineReg(MACHINE_FIXED_REG_COUNT + 3)),
        "call-live clean cache should use the first preserved lane when one is available: {:?}",
        caller_function.preserved_clobbers
    );
    let caller_program = &caller_function.program;
    assert_eq!(caller_program.blocks.len(), 3);
    let call_block = &caller_program.blocks[0];
    let continuation = &caller_program.blocks[1];
    let MachineTerminator::Call { ref success, .. } = call_block.terminator else {
        panic!(
            "expected direct local call terminator, got {:?}",
            call_block.terminator
        );
    };
    assert_eq!(success.args.len(), 1);
    assert!(matches!(
        success.args[0],
        MachineValue::Reg(MachineReg(reg)) if reg == MACHINE_FIXED_REG_COUNT + 3
    ));
    assert_eq!(continuation.params.len(), 1);
    assert_eq!(continuation.params[0].owner, MachineRegOwner::CachedLocal);
    assert!(
        continuation.ops.iter().all(|inst| !matches!(
            inst.kind,
            MachineInstKind::Load {
                owner: MachineRegOwner::CachedLocal,
                ..
            }
        )),
        "continued local.get_cache should use the carried preserved cache register"
    );
}

#[test]
fn direct_local_call_materializes_call_live_register_param_cache_in_preserved_home() {
    let caller_frame = plan_frame_layout(1, 0, 4);
    let callee_frame = plan_frame_layout(0, 0, 4);
    let slot = caller_frame.local_slot(0);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![ValueType::I32];
        caller.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![ValueType::I32, ValueType::I32, ValueType::I32];
        caller.value_sink_local = collections::vec![];
        let mut block = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(0)));
        let drop_idx = caller.intern_primitive(PrimitiveOpKind::Drop).unwrap();
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
            0,
        ));
        let call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(1)));
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(1)), SsaOperand::NONE],
            0,
        ));
        let second_call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(second_call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(2)));
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(2)), SsaOperand::NONE],
            0,
        ));
        caller.blocks.push(block);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        callee.blocks.push(SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::Vec::new(),
            extra_args: collections::Vec::new(),
            terminator: SsaTerminator::Return { results: None },
        });
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config_with_preserved(3, 3, 0),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect(
        "direct local call should materialize a call-live register param cache in a preserved lane",
    );

    let caller_program = &lowered.module.functions[0].program;
    let call_block = &caller_program.blocks[0];
    let continuation = &caller_program.blocks[1];
    let preserved = MachineReg(MACHINE_FIXED_REG_COUNT + 3);
    assert!(
        call_block.ops.iter().any(|inst| {
            matches!(
                inst.kind,
                MachineInstKind::Move {
                    owner: MachineRegOwner::CachedLocal,
                    dst,
                    src: MachineValue::Reg(MachineReg(src)),
                    ..
                } if dst == preserved && src == MACHINE_FIXED_REG_COUNT
            )
        }),
        "register param cache should move from volatile arg lane into preserved lane; ops={:?}",
        call_block.ops
    );
    let MachineTerminator::Call { ref success, .. } = call_block.terminator else {
        panic!(
            "expected direct local call terminator, got {:?}",
            call_block.terminator
        );
    };
    assert_eq!(
        success.args,
        collections::vec![MachineValue::Reg(preserved)]
    );
    assert_eq!(continuation.params.len(), 1);
    assert_eq!(continuation.params[0].owner, MachineRegOwner::CachedLocal);
    assert!(
        call_block
            .ops
            .iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::Store { .. })),
        "dirty register param cache must be frame-published before the local call"
    );
    assert!(
        continuation.ops.iter().all(|inst| !matches!(
            inst.kind,
            MachineInstKind::Load {
                owner: MachineRegOwner::CachedLocal,
                ..
            }
        )),
        "continued local.get_cache should use the carried preserved cache register"
    );
}

#[test]
fn direct_local_call_carries_dirty_non_ref_cache_after_publishing_it() {
    let caller_frame = plan_frame_layout(1, 1, 4);
    let callee_frame = plan_frame_layout(0, 0, 4);
    let slot = caller_frame.local_slot(0);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![ValueType::I32];
        caller.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![ValueType::I32, ValueType::I32, ValueType::I32];
        caller.value_sink_local = collections::vec![];
        let mut block = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        let const_idx = caller
            .intern_primitive(PrimitiveOpKind::I32Const { value: 7 })
            .unwrap();
        block.ops.push(SsaInst::primitive(
            const_idx,
            SsaValue(0),
            [SsaOperand::NONE, SsaOperand::NONE],
            0,
        ));
        block.ops.push(SsaInst::local_set_cache(slot, SsaValue(0)));
        let call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(1)));
        let drop_idx = caller.intern_primitive(PrimitiveOpKind::Drop).unwrap();
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(1)), SsaOperand::NONE],
            0,
        ));
        let second_call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(second_call_idx));
        block.ops.push(SsaInst::local_get_cache(slot, SsaValue(2)));
        block.ops.push(SsaInst::primitive(
            drop_idx,
            SsaValue::NONE,
            [SsaOperand::value(SsaValue(2)), SsaOperand::NONE],
            0,
        ));
        caller.blocks.push(block);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        callee.blocks.push(SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        });
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config_with_preserved(1, 3, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call should publish dirty non-ref cache");

    let caller_program = &lowered.module.functions[0].program;
    assert_eq!(caller_program.blocks.len(), 3);
    let call_block = &caller_program.blocks[0];
    let continuation = &caller_program.blocks[1];
    let MachineTerminator::Call { ref success, .. } = call_block.terminator else {
        panic!(
            "expected direct local call terminator, got {:?}",
            call_block.terminator
        );
    };
    assert_eq!(success.args.len(), 1);
    assert!(matches!(
        success.args[0],
        MachineValue::Reg(MachineReg(reg)) if reg == MACHINE_FIXED_REG_COUNT + 1
    ));
    assert_eq!(continuation.params.len(), 1);
    assert_eq!(continuation.params[0].owner, MachineRegOwner::CachedLocal);
    assert!(
        call_block
            .ops
            .iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::Store { .. })),
        "dirty non-ref cache must be frame-published before the local call; ops={:?}",
        call_block.ops
    );
    assert!(
        continuation.ops.iter().all(|inst| !matches!(
            inst.kind,
            MachineInstKind::Load {
                owner: MachineRegOwner::CachedLocal,
                ..
            }
        )),
        "continued local.get_cache should use the carried preserved cache register; ops={:?}",
        continuation.ops
    );
}

#[test]
fn direct_local_call_publishes_dirty_ref_cache_even_in_preserved_reg() {
    let caller_frame = plan_frame_layout(1, 1, 4);
    let callee_frame = plan_frame_layout(0, 0, 4);
    let slot = caller_frame.local_slot(0);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![ValueType::funcref()];
        caller.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![ValueType::funcref()];
        caller.value_sink_local = collections::vec![];
        let mut block = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        let ref_idx = caller
            .intern_primitive(PrimitiveOpKind::RefFunc { func_idx: 1 })
            .unwrap();
        block.ops.push(SsaInst::primitive(
            ref_idx,
            SsaValue(0),
            [SsaOperand::NONE, SsaOperand::NONE],
            0,
        ));
        block.ops.push(SsaInst::local_set_cache(slot, SsaValue(0)));
        let call_idx = caller.push_call_op(SsaCallOp::CallDirect {
            callee: 1,
            args: test_call_args(FrameSpan::new(caller_frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(caller_frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        block.ops.push(SsaInst::call(call_idx));
        caller.blocks.push(block);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        callee.blocks.push(SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        });
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config_with_preserved(0, 3, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
                frame: callee_frame,
                ssa: callee,
                result_count: 0,
            },
        ],
    })
    .expect("direct local call should publish dirty ref cache");

    let caller_program = &lowered.module.functions[0].program;
    let call_block = &caller_program.blocks[0];
    let MachineTerminator::Call { ref success, .. } = call_block.terminator else {
        panic!(
            "expected direct local call terminator, got {:?}",
            call_block.terminator
        );
    };
    assert!(
        success.args.is_empty(),
        "ref caches must not be carried through the local-call success edge"
    );
    assert!(
        call_block
            .ops
            .iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::Store { .. })),
        "dirty ref cache must be frame-published before the local call; ops={:?}",
        call_block.ops
    );
}

#[test]
fn preserves_cached_locals_across_block_edges() {
    let frame = plan_frame_layout(1, 1, 4);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_drop_cache(frame.local_slot(0)));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk1
            .ops
            .push(SsaInst::local_get_cache(frame.local_slot(0), SsaValue(1)));
        __blk1
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(1)));
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                    owner: MachineRegOwner::LinearValue,
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
                owner: MachineRegOwner::CachedLocal,
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true
        }];
        ssa.block_entry_cached_slots =
            collections::vec![collections::vec![], collections::vec![slot]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0.ops.push(SsaInst::local_set_cache(slot, SsaValue(0)));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk1.ops.push(SsaInst::local_get_cache(slot, SsaValue(1)));
        __blk1.ops.push(SsaInst::local_set_cache(slot, SsaValue(1)));
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.local_slot_info = collections::vec![
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
        ];
        ssa.block_entry_cached_slots =
            collections::vec![collections::vec![], collections::vec![slot1]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I32, ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 1 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(slot0, SsaValue(0)));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 2 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(slot1, SsaValue(1)));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk1
            .ops
            .push(SsaInst::local_get_cache(slot1, SsaValue(2)));
        __blk1
            .ops
            .push(SsaInst::local_set_cache(slot1, SsaValue(2)));
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }];
        ssa.block_entry_cached_slots = collections::vec![collections::vec![]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0.ops.push(SsaInst::local_reserve_cache(slot));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0.ops.push(SsaInst::local_set_cache(slot, SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                    owner: MachineRegOwner::CachedLocal,
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false,
        }];
        ssa.block_entry_cached_slots = collections::vec![
            collections::vec![],
            collections::vec![],
            collections::vec![slot]
        ];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(2),
                bindings: collections::vec![],
            }),
        };
        __blk1.ops.push(SsaInst::local_reserve_cache(slot));
        ssa.blocks.push(__blk1);
        let mut __blk2 = SsaBlock {
            id: SsaTarget(2),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 9 })
                .unwrap();
            __blk2.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk2.ops.push(SsaInst::local_set_cache(slot, SsaValue(0)));
        __blk2.ops.push(SsaInst::local_get_cache(slot, SsaValue(1)));
        __blk2.ops.push(SsaInst::local_set_cache(slot, SsaValue(1)));
        ssa.blocks.push(__blk2);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32, ValueType::I32, ValueType::I32];
        ssa.local_slot_info = collections::vec![
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
        ];
        ssa.block_entry_cached_slots =
            collections::vec![collections::vec![], collections::vec![slot0, slot1, slot2],];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I32, ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        __blk0.ops.push(SsaInst::local_reserve_cache(slot2));
        __blk0.ops.push(SsaInst::local_reserve_cache(slot1));
        __blk0.ops.push(SsaInst::local_reserve_cache(slot0));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 1 })
                .unwrap();
            __blk1.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk1
            .ops
            .push(SsaInst::local_set_cache(slot0, SsaValue(0)));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 2 })
                .unwrap();
            __blk1.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk1
            .ops
            .push(SsaInst::local_set_cache(slot1, SsaValue(1)));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 3 })
                .unwrap();
            __blk1.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(2),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk1
            .ops
            .push(SsaInst::local_set_cache(slot2, SsaValue(2)));
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(3, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: true,
        }];
        ssa.block_entry_cached_slots = collections::vec![collections::vec![]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0.ops.push(SsaInst::local_ensure_cache(slot));
        __blk0.ops.push(SsaInst::local_drop_cache(slot));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
        ops.iter().all(|inst| !matches!(
            inst.kind,
            MachineInstKind::Store {
                src: MachineValue::Reg(_),
                ..
            }
        )),
        "dropping a clean cached local should not write its frame slot back"
    );
}

#[test]
fn does_not_save_clean_carried_cache_before_runtime_call() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: true
        }];
        ssa.block_entry_cached_slots =
            collections::vec![collections::vec![], collections::vec![slot]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        __blk0.ops.push(SsaInst::local_ensure_cache(slot));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallDirect {
                callee: 7,
                args: test_i64_call_args(FrameSpan::new(frame.operand_slot(0), 0)),
                results: FrameSpan::new(frame.operand_slot(0), 0),
                result_types: collections::Vec::new(),
            });
            __blk1.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("clean carried cache should not be saved before runtime call");

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
        "clean carried cache should not be spilled before the runtime call"
    );
    assert!(
        program.blocks[1]
            .ops
            .iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::CallRuntime(_))),
        "expected runtime call in continuation block"
    );
}

#[test]
fn preserved_call_keeps_carried_cache_live_across_edge() {
    let frame = plan_frame_layout(2, 1, 4);
    let slot0 = frame.local_slot(0);
    let slot1 = frame.local_slot(1);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32, ValueType::funcref()];
        ssa.local_slot_info = collections::vec![
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true,
            },
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
        ];
        ssa.block_entry_cached_slots =
            collections::vec![collections::vec![slot0], collections::vec![slot0]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::funcref(), ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::RefFunc { func_idx: 0 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0.ops.push(SsaInst::local_set_slot(slot1, SsaValue(0)));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk1
            .ops
            .push(SsaInst::local_get_cache(slot0, SsaValue(1)));
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::Drop).unwrap();
            __blk1.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [SsaOperand::value(SsaValue(1)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa,
            result_count: 0,
        }],
    })
    .expect("preserved helper call should keep carried cache live across the edge");

    let program = &lowered.module.functions[0].program;
    let source_block = &program.blocks[0];
    let target_block = &program.blocks[1];

    assert!(
        source_block
            .ops
            .iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::RefFunc { .. })),
        "expected ref.func MIR op in source block"
    );
    let MachineTerminator::Jump(edge) = &source_block.terminator else {
        panic!("expected jump terminator from preserved-call block");
    };
    assert_eq!(edge.args.len(), 1, "expected one hidden cache edge arg");
    assert!(
        matches!(edge.args[0], MachineValue::Reg(_)),
        "carried cache should be threaded as a real register value; edge args={:?}",
        edge.args
    );
    assert_eq!(target_block.params.len(), 1);
    assert!(
        target_block
            .ops
            .iter()
            .all(|inst| !matches!(inst.kind, MachineInstKind::Load { .. })),
        "target block should receive the carried cache through params, not reload it from frame; ops={:?}",
        target_block.ops
    );
}

#[test]
fn saves_only_dirty_cached_locals_before_runtime_call() {
    let frame = plan_frame_layout(2, 1, 4);
    let dirty_slot = frame.local_slot(0);
    let clean_slot = frame.local_slot(1);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.local_slot_info = collections::vec![
            LocalSlotInfo {
                is_param: false,
                reads_before_write: false,
            },
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true,
            },
        ];
        ssa.block_entry_cached_slots =
            collections::vec![collections::vec![], collections::vec![clean_slot]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        __blk0.ops.push(SsaInst::local_ensure_cache(clean_slot));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 11 })
                .unwrap();
            __blk1.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk1
            .ops
            .push(SsaInst::local_set_cache(dirty_slot, SsaValue(0)));
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallDirect {
                callee: 7,
                args: test_i64_call_args(FrameSpan::new(frame.operand_slot(0), 0)),
                results: FrameSpan::new(frame.operand_slot(0), 0),
                result_types: collections::Vec::new(),
            });
            __blk1.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("only dirty caches should be saved before runtime call");

    let ops = &lowered.module.functions[0].program.blocks[1].ops;
    let store_count = ops
        .iter()
        .filter(|inst| matches!(inst.kind, MachineInstKind::Store { .. }))
        .count();
    assert_eq!(
        store_count, 1,
        "one dirty cached local and one clean carried local should produce exactly one save store before the runtime call; ops={ops:?}"
    );
    assert!(
        ops.iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::CallRuntime(_))),
        "expected runtime call in lowered block"
    );
}

#[test]
fn entry_block_cached_locals_are_loaded_in_prologue_not_passed_as_params() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }];
        ssa.block_entry_cached_slots = collections::vec![collections::vec![slot]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0.ops.push(SsaInst::local_ensure_cache(slot));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
        block.ops.is_empty(),
        "entry cached parameter should alias its incoming argument lane without a frame load"
    );
}

#[test]
fn register_passed_entry_param_cache_stays_dirty_across_edges() {
    let frame = plan_frame_layout(1, 1, 4);
    let slot = frame.local_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }];
        ssa.block_entry_cached_slots =
            collections::vec![collections::vec![slot], collections::vec![slot]];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut entry = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![],
            }),
        };
        entry.ops.push(SsaInst::local_ensure_cache(slot));
        ssa.blocks.push(entry);
        let mut call_block = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        let call_idx = ssa.push_call_op(SsaCallOp::CallDirect {
            callee: 7,
            args: test_call_args(FrameSpan::new(frame.operand_slot(0), 0), &[]),
            results: FrameSpan::new(frame.operand_slot(0), 0),
            result_types: collections::Vec::new(),
        });
        call_block.ops.push(SsaInst::call(call_idx));
        ssa.blocks.push(call_block);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa,
            result_count: 0,
        }],
    })
    .expect("register-passed cached param should save before runtime call");

    let ops = &lowered.module.functions[0].program.blocks[1].ops;
    assert!(
        ops.iter().any(|inst| matches!(
            inst.kind,
            MachineInstKind::Store {
                addr: MachineAddr { offset: 0, .. },
                ..
            }
        )),
        "register-passed param cache must be saved before the runtime call; ops={ops:?}"
    );
    assert!(
        ops.iter()
            .any(|inst| matches!(inst.kind, MachineInstKind::CallRuntime(_))),
        "expected runtime call in successor block"
    );
}

#[test]
fn rejects_cache_store_with_incompatible_gp_storage_types() {
    use ValueType;

    let frame = plan_frame_layout(1, 2, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: false
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_cache(frame.local_slot(0), SsaValue(0)));
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_cache(frame.local_slot(0), SsaValue(1)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let err = lower_module(LowerModuleInput {
        backend: gp32_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect_err("typed i64 store into cached i32 local must be rejected");

    let message = err.message();
    assert!(
        message.contains("LocalSetCache src for cached local slot FrameSlot(0)")
            || message.contains("typed SSA-IR store to cached local slot")
            || message.contains("cached local slot uses incompatible value type"),
        "unexpected error: {message}"
    );
}

#[test]
fn lowers_direct_local_call_with_sparse_machine_function_ids() {
    let caller_frame = plan_frame_layout(1, 4, 4);
    let callee_frame = plan_frame_layout(3, 2, 4);

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![];
        caller.local_slot_info = collections::vec![];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![];
        caller.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = caller.push_call_op(SsaCallOp::CallDirect {
                callee: 2,
                args: test_i64_call_args(FrameSpan::new(caller_frame.operand_slot(1), 2)),
                results: FrameSpan::new(caller_frame.operand_slot(0), 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        caller.blocks.push(__blk0);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(callee_frame.operand_slot(0), 1)),
            },
        };
        callee.blocks.push(__blk0);
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(2),
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
            kind: MachineTrapKind::Unreachable
        }
    ));
    assert!(matches!(
        lowered.module.functions[0].program.blocks[0].terminator,
        MachineTerminator::Call {
            target: MachineCallTarget::Direct(MachineFuncId(2)),
            ..
        }
    ));
}

#[test]
fn lowers_memory_size_without_helper_boundary() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::MemorySize { mem_idx: 1 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::MemorySize { mem_idx: 1 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::ShrU,
            rhs: MachineValue::Imm64(16),
            ..
        }
    ));
}

#[test]
fn lowers_call_indirect_with_local_and_runtime_dispatch_paths() {
    let frame = plan_frame_layout(0, 6, 4);
    let call_base = frame.operand_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallIndirect {
                type_idx: 3,
                table_idx: 0,
                index: SsaCallOperandLoc::Stack {
                    slot: call_base.advance(2),
                },
                args: test_i64_call_args(FrameSpan::new(call_base, 2)),
                results: FrameSpan::new(call_base, 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 8, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("call_indirect lowering should succeed");

    assert_eq!(lowered.module.consts.len(), 1);

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 11);
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[2].terminator,
        MachineTerminator::Trap {
            kind: MachineTrapKind::TableOutOfBounds
        }
    ));
    assert!(matches!(
        program.blocks[4].terminator,
        MachineTerminator::Trap {
            kind: MachineTrapKind::InvalidFunctionReference
        }
    ));
    assert!(matches!(
        program.blocks[6].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert_eq!(program.blocks[6].params.len(), 1);
    assert!(matches!(
        program.blocks[6].ops[0].kind,
        MachineInstKind::IntBinary {
            op: MachineIntBinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[7].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[7].ops[0].kind,
        MachineInstKind::Store { .. }
    ));
    assert!(matches!(
        program.blocks[8].terminator,
        MachineTerminator::Call {
            target: MachineCallTarget::Indirect { .. },
            success: MachineEdge {
                target: MachineBlockId(10),
                ..
            },
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
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[3].terminator,
        MachineTerminator::Branch {
            cond: MachineBranchCond::IntCompare {
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
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        dispatch_ops[5].kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[9].ops[0].kind,
        MachineInstKind::CallRuntime(_)
    ));
    assert!(matches!(
        program.blocks[9].ops[1].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        program.blocks[9].ops[2].kind,
        MachineInstKind::Load { .. }
    ));
    assert!(matches!(
        program.blocks[9].terminator,
        MachineTerminator::Jump(MachineEdge {
            target: MachineBlockId(10),
            ..
        })
    ));
    assert!(matches!(
        program.blocks[10].terminator,
        MachineTerminator::Trap {
            kind: MachineTrapKind::Unreachable
        }
    ));
}

#[test]
fn lowers_call_ref_with_local_and_runtime_dispatch_paths() {
    let frame = plan_frame_layout(0, 6, 4);
    let call_base = frame.operand_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallRef {
                type_idx: 3,
                callee_ref: SsaCallOperandLoc::Stack {
                    slot: call_base.advance(2),
                },
                args: test_i64_call_args(FrameSpan::new(call_base, 2)),
                results: FrameSpan::new(call_base, 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 8, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa,
            result_count: 0,
        }],
    })
    .expect("call_ref lowering should succeed");

    assert_eq!(lowered.module.consts.len(), 1);

    let program = &lowered.module.functions[0].program;
    assert_eq!(program.blocks.len(), 9);
    assert!(matches!(
        program.blocks[0].ops[0].kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
            ..
        }
    ));
    assert!(matches!(
        program.blocks[0].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[1].terminator,
        MachineTerminator::Trap {
            kind: MachineTrapKind::InvalidFunctionReference
        }
    ));
    assert!(matches!(
        program.blocks[2].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[3].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert_eq!(program.blocks[4].params.len(), 1);
    assert!(matches!(
        program.blocks[4].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[5].terminator,
        MachineTerminator::Branch { .. }
    ));
    assert!(matches!(
        program.blocks[6].terminator,
        MachineTerminator::Call {
            target: MachineCallTarget::Indirect { .. },
            success: MachineEdge {
                target: MachineBlockId(8),
                ..
            },
            ..
        }
    ));
    assert!(matches!(
        program.blocks[7].ops[0].kind,
        MachineInstKind::CallRuntime(_)
    ));
    assert!(matches!(
        program.blocks[7].terminator,
        MachineTerminator::Jump(MachineEdge {
            target: MachineBlockId(8),
            ..
        })
    ));
    assert!(matches!(
        program.blocks[8].terminator,
        MachineTerminator::Trap {
            kind: MachineTrapKind::Unreachable
        }
    ));
}

#[test]
fn lowers_call_indirect_with_gp_word_width_on_32_bit_target() {
    let frame = plan_frame_layout(0, 6, 4);
    let call_base = frame.operand_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = ssa.push_call_op(SsaCallOp::CallIndirect {
                type_idx: 3,
                table_idx: 0,
                index: SsaCallOperandLoc::Stack {
                    slot: call_base.advance(2),
                },
                args: test_i64_call_args(FrameSpan::new(call_base, 2)),
                results: FrameSpan::new(call_base, 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 8, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
            cond: MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
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
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Mul,
            ..
        }
    )));
    assert!(
        program
            .blocks
            .iter()
            .any(|block| block.ops.iter().any(|inst| matches!(
                inst.kind,
                MachineInstKind::IntBinary {
                    width: MachineIntWidth::I32,
                    op: MachineIntBinaryOp::Add,
                    ..
                }
            ))),
        "expected 32-bit call_indirect lowering to use GP-word I32 add ops"
    );
}

#[test]
fn uses_canonical_u64_width_for_gp_word_frame_slots_on_32bit_targets() {
    let frame = plan_frame_layout(0, 1, 2);
    let slot = frame.local_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 7 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0.ops.push(SsaInst::local_set_slot(slot, SsaValue(0)));
        __blk0.ops.push(SsaInst::local_get_slot(slot, SsaValue(1)));
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::Drop).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [SsaOperand::value(SsaValue(1)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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

    let caller = {
        let mut caller = SsaProgram::default();
        caller.entry = SsaTarget(0);
        caller.local_slot_types = collections::vec![];
        caller.local_slot_info = collections::vec![];
        caller.block_entry_cached_slots = collections::vec![];
        caller.block_cfg_origins = collections::vec![];
        caller.value_types = collections::vec![];
        caller.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::TrapUnreachable,
        };
        {
            let __idx = caller.push_call_op(SsaCallOp::CallDirect {
                callee: 1,
                args: test_i64_call_args(FrameSpan::new(caller_frame.operand_slot(1), 2)),
                results: FrameSpan::new(caller_frame.operand_slot(0), 1),
                result_types: collections::Vec::new(),
            });
            __blk0.ops.push(SsaInst::call(__idx));
        }
        caller.blocks.push(__blk0);
        caller
    };
    let callee = {
        let mut callee = SsaProgram::default();
        callee.entry = SsaTarget(0);
        callee.local_slot_types = collections::vec![];
        callee.local_slot_info = collections::vec![];
        callee.block_entry_cached_slots = collections::vec![];
        callee.block_cfg_origins = collections::vec![];
        callee.value_types = collections::vec![];
        callee.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(callee_frame.operand_slot(0), 1)),
            },
        };
        callee.blocks.push(__blk0);
        callee
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![
            LowerFunctionInput {
                id: MachineFuncId(0),
                frame: caller_frame,
                ssa: caller,
                result_count: 0,
            },
            LowerFunctionInput {
                id: MachineFuncId(1),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 9 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::GlobalSet { idx: 3 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::GlobalGet { idx: 3 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
            frame,
            ssa: ssa,
            result_count: 0,
        }],
    })
    .expect("global get/set should lower directly");

    let ops = &lowered.module.functions[0].program.blocks[0].ops;
    // Each global access emits two MIR ops now: an indexed load of the raw
    // pointer from the inline ptrs tail at `[runtime_base +
    // globals_ptrs_inline_offset + idx*ptr]`, then a load or store through
    // that pointer. Plus the const-9 setup, the block totals five ops.
    assert_eq!(ops.len(), 5);
    assert!(matches!(
        ops[0].kind,
        MachineInstKind::Move {
            src: MachineValue::Imm64(9),
            ..
        }
    ));
    // global.set: load raw_ptr, then store through it.
    assert!(matches!(ops[1].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[2].kind, MachineInstKind::Store { .. }));
    // global.get: load raw_ptr, then deref.
    assert!(matches!(ops[3].kind, MachineInstKind::Load { .. }));
    assert!(matches!(ops[4].kind, MachineInstKind::Load { .. }));
}

#[test]
fn lowers_table_get_with_explicit_oob_trap_block() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 0 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::TableGet { table_idx: 1 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
            kind: MachineTrapKind::TableOutOfBounds
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Load {
                    offset: 4,
                    memidx: 1,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
            op: MachineConvertOp::I64ExtendI32U,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::TrapIf {
            kind: MachineTrapKind::MemoryOutOfBounds,
            ..
        }
    )));
    assert!(matches!(
        ops.last().unwrap().kind,
        MachineInstKind::Load {
            width: MachineMemWidth::U32,
            ..
        }
    ));
}

#[test]
fn lowers_i32_load_with_gp_word_bounds_ops_on_32_bit_target() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Load {
                    offset: 4,
                    memidx: 1,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
            op: MachineConvertOp::I64ExtendI32U,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::TrapIf {
            cond: MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                ..
            },
            ..
        }
    )));
}

#[test]
fn lowers_32bit_memory_bounds_checks_with_wraparound_traps() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Load {
                    offset: 1,
                    memidx: 0,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                MachineBranchCond::IntCompare {
                    width: MachineIntWidth::I32,
                    kind: MachineCompareKind::Lt,
                    sign: MachineSign::Unsigned,
                    rhs: MachineValue::Imm64(value),
                    ..
                },
            kind: MachineTrapKind::MemoryOutOfBounds,
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const {
                    value: 0x8877_6655_4433_2211,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Store {
                    offset: 0,
                    memidx: 0,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [
                    SsaOperand::value(SsaValue(0)),
                    SsaOperand::value(SsaValue(1)),
                ],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        use_guard_pages: true,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                MachineBranchCond::IntCompare {
                    width: MachineIntWidth::I32,
                    kind: MachineCompareKind::Lt,
                    sign: MachineSign::Unsigned,
                    rhs: MachineValue::Imm64(value),
                    ..
                },
            kind: MachineTrapKind::MemoryOutOfBounds,
        } = &inst.kind
        {
            if *value == 8 {
                saw_access_wrap = true;
            }
        }
        if let MachineInstKind::TrapIf {
            cond:
                MachineBranchCond::IntCompare {
                    width: MachineIntWidth::I32,
                    kind: MachineCompareKind::Gt,
                    sign: MachineSign::Unsigned,
                    ..
                },
            kind: MachineTrapKind::MemoryOutOfBounds,
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::RefNull).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::RefIsNull).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: gp32_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            src: MachineValue::Imm64(value),
            ..
        } if value == u32::MAX as u64
    ));
    assert!(matches!(
        ops[1].kind,
        MachineInstKind::IntCompare {
            width: MachineIntWidth::I32,
            rhs: MachineValue::Imm64(value),
            ..
        } if value == u32::MAX as u64
    ));
}

#[test]
fn omits_zero_offset_add_in_bounds_check_setup() {
    let frame = plan_frame_layout(0, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Load {
                    offset: 0,
                    memidx: 1,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I64Const { value: 3 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::TableGet { table_idx: 1 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(2),
                [SsaOperand::value(SsaValue(0)), SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::I64Add).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(3),
                [
                    SsaOperand::value(SsaValue(1)),
                    SsaOperand::value(SsaValue(2)),
                ],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::local_set_slot(frame.local_slot(0), SsaValue(3)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    assert!(continuation.params.iter().any(|param| {
        continuation.ops.iter().any(|op| {
            matches!(
                op.kind,
                MachineInstKind::IntBinary {
                    op: MachineIntBinaryOp::Mul,
                    lhs: MachineValue::Reg(reg),
                    ..
                } if reg == param.reg
            )
        })
    }));
}

#[test]
fn lowers_f32_store_inline_with_trap_if_preserving_fp_linear_value_width() {
    let frame = plan_frame_layout(0, 0, 3);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::I32Const { value: 8 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::F32Const { value: 0x3f800000 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::F32Abs).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(2),
                [SsaOperand::value(SsaValue(1)), SsaOperand::NONE],
                0u16,
            ));
        }
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::F32Store {
                    offset: 4,
                    memidx: 1,
                })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue::NONE,
                [
                    SsaOperand::value(SsaValue(0)),
                    SsaOperand::value(SsaValue(2)),
                ],
                0u16,
            ));
        }
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                kind: MachineTrapKind::MemoryOutOfBounds,
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
    use ValueType;

    let frame = plan_frame_layout(0, 1, 1);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::F32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(frame.operand_slot(0), 1)),
            },
        };
        {
            let __pidx = ssa
                .intern_primitive(PrimitiveOpKind::F32Const { value: 0x4120_0000 })
                .unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(0),
                [SsaOperand::NONE, SsaOperand::NONE],
                0u16,
            ));
        }
        __blk0
            .ops
            .push(SsaInst::spill(frame.operand_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                width: MachineFloatWidth::F32,
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
    use ValueType;

    let frame = plan_frame_layout(1, 2, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::F64];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(frame.operand_slot(0), 1)),
            },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_slot(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::spill(frame.operand_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
        is_fp_reg(load_dst, lowered.module.config),
        "typed F64 LocalGet must allocate into FP bank, got GP reg {}",
        load_dst.0,
    );
}

#[test]
fn untyped_slot_load_stays_in_gp_bank() {
    let frame = plan_frame_layout(1, 2, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return {
                results: Some(FrameSpan::new(frame.operand_slot(0), 1)),
            },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_slot(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::spill(frame.operand_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
        is_gp_reg(load_dst, lowered.module.config),
        "untyped LocalGet must stay in GP bank, got FP reg {}",
        load_dst.0,
    );
}

#[test]
fn f32_block_params_keep_f32_width() {
    use ValueType;

    let frame = plan_frame_layout(1, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![];
        ssa.local_slot_info = collections::vec![];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::F32, ValueType::F32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Goto(SsaEdge {
                target: SsaTarget(1),
                bindings: collections::vec![SsaBinding {
                    param: SsaValue(1),
                    value: SsaValue(0),
                }],
            }),
        };
        __blk0
            .ops
            .push(SsaInst::local_get_slot(frame.local_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        let mut __blk1 = SsaBlock {
            id: SsaTarget(1),
            params: collections::vec![SsaValue(1)],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        ssa.blocks.push(__blk1);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    use ValueType;

    let frame = plan_frame_layout(1, 1, 2);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::F32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: false,
            reads_before_write: true
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::F32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_slot(frame.local_slot(0), SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_set_slot(frame.local_slot(0), SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(0, 4, 1, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Load {
            ty: MachineStorageType::Fp32,
            width: MachineMemWidth::U32,
            ..
        }
    )));
    assert!(ops.iter().any(|inst| matches!(
        inst.kind,
        MachineInstKind::Store {
            ty: MachineStorageType::Fp32,
            width: MachineMemWidth::U32,
            ..
        }
    )));
}

// --- Source-aliasing regression tests ---

/// LocalGetCache should NOT emit a Move — the SSA value should alias the
/// cache register directly so that downstream ops use it without a copy.
#[test]
fn local_get_cache_source_aliases_without_move() {
    let frame = plan_frame_layout(1, 2, 2);
    let slot = frame.local_slot(0);
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32];
        ssa.local_slot_info = collections::vec![LocalSlotInfo {
            is_param: true,
            reads_before_write: true,
        }];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0.ops.push(SsaInst::local_get_cache(slot, SsaValue(0)));
        {
            let __c1 = ssa.intern_const((1) as u64);
            let __pidx = ssa.intern_primitive(PrimitiveOpKind::I32Add).unwrap();
            __blk0.ops.push(SsaInst::primitive(
                __pidx,
                SsaValue(1),
                [SsaOperand::value(SsaValue(0)), __c1],
                0u16,
            ));
        }
        __blk0.ops.push(SsaInst::local_set_slot(slot, SsaValue(1)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(1, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                    owner: MachineRegOwner::LinearValue,
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.local_slot_info = collections::vec![
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
        ];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_cache(slot0, SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_get_cache(slot1, SsaValue(1)));
        __blk0
            .ops
            .push(SsaInst::local_set_cache(slot0, SsaValue(1)));
        __blk0.ops.push(SsaInst::local_set_slot(slot1, SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
                    owner: MachineRegOwner::LinearValue,
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
    let ssa = {
        let mut ssa = SsaProgram::default();
        ssa.entry = SsaTarget(0);
        ssa.local_slot_types = collections::vec![ValueType::I32, ValueType::I32];
        ssa.local_slot_info = collections::vec![
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
            LocalSlotInfo {
                is_param: true,
                reads_before_write: true
            },
        ];
        ssa.block_entry_cached_slots = collections::vec![];
        ssa.block_cfg_origins = collections::vec![];
        ssa.value_types = collections::vec![ValueType::I32];
        ssa.value_sink_local = collections::vec![];
        let mut __blk0 = SsaBlock {
            id: SsaTarget(0),
            params: collections::vec![],
            ops: collections::vec![],
            extra_args: collections::vec![],
            terminator: SsaTerminator::Return { results: None },
        };
        __blk0
            .ops
            .push(SsaInst::local_get_cache(slot0, SsaValue(0)));
        __blk0
            .ops
            .push(SsaInst::local_set_cache(slot1, SsaValue(0)));
        ssa.blocks.push(__blk0);
        ssa
    };

    let lowered = lower_module(LowerModuleInput {
        backend: host_backend_config(2, 4, 0, 2),
        #[cfg(sf_has_guard_pages)]
        use_guard_pages: false,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages: false,
        functions: collections::vec![LowerFunctionInput {
            id: MachineFuncId(0),
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
