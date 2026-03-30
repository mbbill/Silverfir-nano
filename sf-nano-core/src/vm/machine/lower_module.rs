// ---------------------------------------------------------------------------
// Lowering: prepared SSA-IR → MachineIR
// ---------------------------------------------------------------------------

use alloc::{vec, vec::Vec};

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
            MachineCallLinkLayout, MachineCompareKind, MachineEdge, MachineFloatWidth,
            MachineFrameRegion, MachineFuncId, MachineFunction, MachineFunctionRuntime,
            MachineInst, MachineInstKind, MachineIntBinaryOp, MachineLoadExtension,
            MachineMemWidth, MachineModule, MachineProgram, MachineReg, MachineRuntimeContract,
            MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
        },
        middle::{
            frame::{FrameLayoutPlan, FrameSpan},
            ssa_ir::{
                ir::{
                    SsaBoundaryOp, SsaInstKind, SsaLocalCachePrefs, SsaProgram, SsaTerminator,
                    SsaValue,
                },
                validate::validate_program,
            },
        },
        runtime::{
            context::function_kind,
            layout::{function_view_abi_layout, native_runtime_abi_layout},
        },
    },
};

use crate::vm::middle::frame::FrameSlot;
use crate::vm::middle::ssa_ir::ir::{SsaBlock, SsaTerminator as SsaTerm};
use crate::vm::middle::ssa_ir::target::SsaTarget;

use super::{
    gp32::Gp32Lowering,
    lower_context::{BlockLowerContext, ValueRegs},
    lower_i64::I64Lowering,
    lower_i64_gp64::Gp64Lowering,
    lower_inst::LeafLowering,
    lower_regalloc::{machine_block_params_for_value, MachineRegFile},
    lower_sidecar::SidecarBuilder,
};

/// One prepared function ready for SSA-IR -> MachineIR lowering.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LowerFunctionInput<'a> {
    pub id: MachineFuncId,
    pub frame: FrameLayoutPlan,
    pub ssa: &'a SsaProgram,
    /// Declared result count from the function type signature.
    /// Used as fallback when all return paths are unreachable.
    pub result_count: u16,
}

/// One lowering request for a whole machine module.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LowerModuleInput<'a> {
    pub backend: BackendConfig,
    pub functions: &'a [LowerFunctionInput<'a>],
    #[cfg(has_guard_pages)]
    pub use_guard_pages: bool,
}

/// Result of lowering prepared SSA-IR into MachineIR plus runtime-side contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredMachineModule {
    pub module: MachineModule,
    pub runtime: MachineRuntimeContract,
}

pub(crate) fn lower_module(input: LowerModuleInput<'_>) -> Result<LoweredMachineModule, WasmError> {
    let max_regfile = MachineRegFile::new(input.backend)?;
    let call_link = MachineCallLinkLayout {
        continuation_offset: 0,
        caller_frame_offset: 8,
        caller_result_base_offset: 16,
        slot_count: 3,
    };

    let function_count = input
        .functions
        .iter()
        .map(|function| function.id.0 as usize)
        .max()
        .map(|max| max + 1)
        .unwrap_or(0);
    let mut functions = alloc::vec![None; function_count];
    let mut function_runtime = (0..function_count)
        .map(|index| MachineFunctionRuntime {
            id: MachineFuncId(index as u32),
            ..MachineFunctionRuntime::default()
        })
        .collect::<Vec<_>>();
    let mut sidecar = SidecarBuilder::new();
    for function in input.functions {
        validate_program(function.ssa)?;
        function_runtime[function.id.0 as usize] = lower_function_runtime(*function, call_link)?;
    }
    #[cfg(has_guard_pages)]
    let guard_pages = input.use_guard_pages;
    #[cfg(not(has_guard_pages))]
    let guard_pages = false;
    for function in input.functions {
        functions[function.id.0 as usize] = Some(lower_function(
            *function,
            input.backend,
            &max_regfile,
            &function_runtime,
            call_link,
            &mut sidecar,
            guard_pages,
        )?);
    }
    let runtime = MachineRuntimeContract {
        call_link,
        functions: function_runtime,
    };

    let (consts, externs) = sidecar.finish();
    let functions = functions
        .into_iter()
        .enumerate()
        .map(|(index, function)| {
            function.unwrap_or_else(|| stub_machine_function(MachineFuncId(index as u32)))
        })
        .collect();
    let module = MachineModule {
        config: input.backend,
        functions,
        consts,
        externs,
    };
    module.validate()?;

    Ok(LoweredMachineModule { module, runtime })
}

fn lower_function_runtime(
    input: LowerFunctionInput<'_>,
    call_link: MachineCallLinkLayout,
) -> Result<MachineFunctionRuntime, WasmError> {
    let call_scratch = input.frame.call_scratch.map(frame_span_region);
    let helper_scratch = input.frame.call_scratch.and_then(|span| {
        let link_slots = call_link.slot_count;
        (span.count > link_slots).then(|| MachineFrameRegion {
            base_slot: span.start.0 + link_slots,
            slots: span.count - link_slots,
        })
    });
    let mut return_results = derive_return_results(input.ssa)?;
    // Fallback: if no Return terminators exist (all paths trap/unreachable),
    // use the declared result count from the type signature so that callers
    // see the correct contract.
    if return_results.is_none() && input.result_count > 0 {
        let result_span = input.frame.return_results(input.result_count);
        return_results = Some(frame_span_region(result_span));
    }

    Ok(MachineFunctionRuntime {
        id: input.id,
        frame_prefix_slots: input.frame.frame_prefix_size,
        total_frame_slots: input.frame.total_slots(),
        call_scratch,
        helper_scratch,
        return_results,
    })
}

fn lower_function(
    input: LowerFunctionInput<'_>,
    config: BackendConfig,
    regfile: &MachineRegFile,
    runtime: &[MachineFunctionRuntime],
    call_link: MachineCallLinkLayout,
    sidecar: &mut SidecarBuilder,
    guard_pages: bool,
) -> Result<MachineFunction, WasmError> {
    let gp_reg_width = config.gp_unit_bytes;
    let original_block_count = input.ssa.blocks.len();
    let mut original_blocks = alloc::vec![None; original_block_count];
    let mut extra_blocks = Vec::new();
    let mut extra_block_ids = ExtraBlockAllocator::new(original_block_count as u32);
    let i64_ops: &'static dyn I64Lowering = if gp_reg_width == 4 {
        &Gp32Lowering
    } else {
        &Gp64Lowering
    };

    let block_entry_dirty =
        compute_block_entry_dirty(&input.ssa.blocks, input.ssa.entry, &input.ssa.local_cache);

    for block in &input.ssa.blocks {
        let target = block.id;
        let entry_dirty = block_entry_dirty
            .get(target.as_u32() as usize)
            .map(|v| v.as_slice());
        let mut lower = BlockLowerContext::new(
            regfile,
            input.ssa,
            &input.ssa.local_cache,
            block,
            runtime,
            call_link,
            gp_reg_width,
            i64_ops,
            target == input.ssa.entry,
            entry_dirty,
            #[cfg(has_guard_pages)]
            guard_pages,
        )?;
        let mut current_block = MachineBlockId(block.id.as_u32());
        let mut current_params = Vec::new();
        for (regs, value) in lower
            .machine_params()
            .iter()
            .copied()
            .zip(block.params.iter().copied())
        {
            current_params.extend(machine_block_params_for_value(
                regs,
                program_value_storage_type(input.ssa, value),
            ));
        }

        for inst in &block.ops {
            match &inst.kind {
                SsaInstKind::Value { op, args, results } => {
                    lower.apply_sink_premap(args, results)?;
                    if let Some(lowered) = lower.lower_leaf_special(
                        op,
                        args,
                        results,
                        extra_block_ids.peek(1),
                        extra_block_ids.peek(0),
                    )? {
                        match lowered {
                            LeafLowering::InPlace => {
                                lower.release_dead_values()?;
                            }
                            LeafLowering::Split {
                                continuation,
                                trap,
                                trap_kind,
                                mut terminator,
                                continuation_ops,
                            } => {
                                let continuation_params =
                                    lower.split_continuation_params(&continuation_ops, &terminator);
                                lower.release_dead_values()?;
                                attach_continuation_args(
                                    &mut terminator,
                                    continuation,
                                    &continuation_params,
                                )?;
                                extra_block_ids.reserve(2);
                                push_lowered_block(
                                    current_block,
                                    &mut original_blocks,
                                    &mut extra_blocks,
                                    current_params,
                                    lower.take_ops(),
                                    terminator,
                                )?;
                                extra_blocks.push(MachineBlock {
                                    id: trap,
                                    params: Vec::new(),
                                    ops: Vec::new(),
                                    terminator: MachineTerminator::Trap { kind: trap_kind },
                                });
                                current_block = continuation;
                                current_params = continuation_params;
                                lower.emit_machine_ops(continuation_ops);
                            }
                        }
                        continue;
                    }
                    lower.lower_inst(inst)?;
                }
                SsaInstKind::Boundary(boundary) => match boundary {
                    SsaBoundaryOp::MemoryGrow { .. }
                    | SsaBoundaryOp::MemoryFill { .. }
                    | SsaBoundaryOp::MemoryCopy { .. }
                    | SsaBoundaryOp::TableGrow { .. }
                    | SsaBoundaryOp::TableFill { .. }
                    | SsaBoundaryOp::TableCopy { .. }
                    | SsaBoundaryOp::MemoryInit { .. }
                    | SsaBoundaryOp::DataDrop { .. }
                    | SsaBoundaryOp::TableInit { .. }
                    | SsaBoundaryOp::ElemDrop { .. } => {
                        lower.lower_runtime(boundary, sidecar)?;
                    }
                    SsaBoundaryOp::CallExternal {
                        func_idx,
                        args,
                        results,
                        ..
                    } => {
                        lower.lower_call_external(*func_idx, *args, *results, sidecar)?;
                    }
                    SsaBoundaryOp::CallInternal {
                        callee,
                        args,
                        results,
                        skip_reload,
                        ..
                    } => {
                        let continuation = extra_block_ids.alloc();
                        let terminator =
                            lower.lower_call_internal(*callee, *args, *results, continuation)?;
                        push_lowered_block(
                            current_block,
                            &mut original_blocks,
                            &mut extra_blocks,
                            current_params,
                            lower.take_ops(),
                            terminator,
                        )?;
                        current_block = continuation;
                        current_params = Vec::new();
                        lower.begin_continuation_block_selective(Some(skip_reload))?;
                    }
                    SsaBoundaryOp::CallIndirect { .. } => {
                        let SsaBoundaryOp::CallIndirect {
                            type_idx,
                            table_idx,
                            index_slot,
                            args,
                            results,
                            skip_reload,
                        } = boundary
                        else {
                            unreachable!("matched call_indirect boundary");
                        };
                        let type_idx = *type_idx;
                        let table_idx = *table_idx;
                        let index_slot = *index_slot;
                        let args = *args;
                        let results = *results;
                        lower.ensure_no_live_values(
                            "prepared SSA-IR call_indirect reached native lowering with live transient SSA values; values must be published before the call",
                        )?;
                        // After the checked block resolves the table entry, this canonical frame
                        // slot is reused to carry the resolved function index through the rest of
                        // the indirect dispatch path.
                        let func_idx_slot = index_slot;

                        let checked = extra_block_ids.alloc();
                        let trap_oob = extra_block_ids.alloc();
                        let type_check = extra_block_ids.alloc();
                        let trap_invalid_ref = extra_block_ids.alloc();
                        let dispatch = extra_block_ids.alloc();
                        let trap_type = extra_block_ids.alloc();
                        let local_call = extra_block_ids.alloc();
                        let external_call = extra_block_ids.alloc();
                        let continuation = extra_block_ids.alloc();
                        let indirect_temps = call_indirect_gp_temps(&lower)?;
                        let local_call_target_param = indirect_temps.lane0;

                        lower.emit_save_all_cached_locals()?;
                        emit_call_indirect_bounds_check_setup(
                            &mut lower,
                            table_idx,
                            func_idx_slot,
                        )?;
                        push_lowered_block(
                            current_block,
                            &mut original_blocks,
                            &mut extra_blocks,
                            current_params,
                            lower.take_ops(),
                            MachineTerminator::Branch {
                                cond: MachineBranchCond::IntCompare {
                                    width: lower.gp_word_int_width(),
                                    kind: MachineCompareKind::Ge,
                                    sign: MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(indirect_temps.lane0),
                                    rhs: MachineValue::Reg(indirect_temps.lane2),
                                },
                                then_edge: MachineEdge {
                                    target: trap_oob,
                                    args: Vec::new(),
                                },
                                else_edge: MachineEdge {
                                    target: checked,
                                    args: Vec::new(),
                                },
                            },
                        )?;

                        push_lowered_block(
                            checked,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            build_call_indirect_checked_block(&lower, table_idx, func_idx_slot)?,
                            MachineTerminator::Branch {
                                cond: MachineBranchCond::IntCompare {
                                    width: lower.gp_word_int_width(),
                                    kind: MachineCompareKind::Ge,
                                    sign: MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(indirect_temps.lane2),
                                    rhs: MachineValue::Reg(indirect_temps.lane1),
                                },
                                then_edge: MachineEdge {
                                    target: trap_invalid_ref,
                                    args: Vec::new(),
                                },
                                else_edge: MachineEdge {
                                    target: type_check,
                                    args: Vec::new(),
                                },
                            },
                        )?;
                        push_lowered_block(
                            trap_oob,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            Vec::new(),
                            MachineTerminator::Trap {
                                kind: MachineTrapKind::TableOutOfBounds,
                            },
                        )?;
                        push_lowered_block(
                            type_check,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            build_call_indirect_type_check_block(&lower, type_idx, func_idx_slot)?,
                            MachineTerminator::Branch {
                                cond: MachineBranchCond::IntCompare {
                                    width: lower.gp_word_int_width(),
                                    kind: MachineCompareKind::Ne,
                                    sign: MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(indirect_temps.lane0),
                                    rhs: MachineValue::Reg(indirect_temps.lane2),
                                },
                                then_edge: MachineEdge {
                                    target: trap_type,
                                    args: Vec::new(),
                                },
                                else_edge: MachineEdge {
                                    target: dispatch,
                                    args: Vec::new(),
                                },
                            },
                        )?;
                        push_lowered_block(
                            trap_invalid_ref,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            Vec::new(),
                            MachineTerminator::Trap {
                                kind: MachineTrapKind::InvalidFunctionReference,
                            },
                        )?;
                        push_lowered_block(
                            dispatch,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            build_call_indirect_dispatch_block(&lower, func_idx_slot)?,
                            MachineTerminator::Branch {
                                cond: MachineBranchCond::IntCompare {
                                    width: lower.gp_word_int_width(),
                                    kind: MachineCompareKind::Eq,
                                    sign: MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(indirect_temps.lane0),
                                    rhs: MachineValue::Imm64(function_kind::LOCAL as u64),
                                },
                                then_edge: MachineEdge {
                                    target: local_call,
                                    args: vec![MachineValue::Reg(indirect_temps.lane2)],
                                },
                                else_edge: MachineEdge {
                                    target: external_call,
                                    args: Vec::new(),
                                },
                            },
                        )?;
                        push_lowered_block(
                            trap_type,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            Vec::new(),
                            MachineTerminator::Trap {
                                kind: MachineTrapKind::IndirectCallTypeMismatch,
                            },
                        )?;
                        push_lowered_block(
                            local_call,
                            &mut original_blocks,
                            &mut extra_blocks,
                            vec![MachineBlockParam::gp_word(local_call_target_param)],
                            build_call_indirect_local_block(&mut lower, args)?,
                            MachineTerminator::CallIndirect {
                                callee_target: MachineValue::Reg(local_call_target_param),
                                callee_frame_base: indirect_temps.lane1,
                                arg_slots: args.count,
                                caller_result_base: results.start.0,
                                continuation,
                            },
                        )?;
                        let helper_call = lower.call_indirect_external_site(
                            func_idx_slot,
                            args,
                            results,
                            sidecar,
                        );
                        push_lowered_block(
                            external_call,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            vec![MachineInst {
                                kind: MachineInstKind::CallHelper(helper_call),
                            }],
                            MachineTerminator::Jump(MachineEdge {
                                target: continuation,
                                args: Vec::new(),
                            }),
                        )?;

                        current_block = continuation;
                        current_params = Vec::new();
                        lower.begin_continuation_block_selective(Some(skip_reload))?;
                    }
                },
                _ => lower.lower_inst(inst)?,
            }
        }

        let terminator = lower.lower_terminator()?;
        push_lowered_block(
            current_block,
            &mut original_blocks,
            &mut extra_blocks,
            current_params,
            lower.take_ops(),
            terminator,
        )?;
    }

    let mut blocks = Vec::with_capacity(original_block_count + extra_blocks.len());
    for (index, block) in original_blocks.into_iter().enumerate() {
        blocks.push(block.ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "machine lowering did not produce original block {}",
                index
            ))
        })?);
    }
    blocks.extend(extra_blocks);

    let program = MachineProgram {
        entry: MachineBlockId(input.ssa.entry.as_u32()),
        fp_reg_init_widths: fp_reg_init_widths(&regfile, &input.ssa.local_cache)?,
        blocks,
    };
    program.validate(config)?;

    Ok(MachineFunction {
        id: input.id,
        program,
    })
}

fn stub_machine_function(id: MachineFuncId) -> MachineFunction {
    MachineFunction {
        id,
        program: MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: Vec::new(),
            blocks: vec![MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: MachineTerminator::Trap {
                    kind: MachineTrapKind::Unreachable,
                },
            }],
        },
    }
}

#[inline]
pub(super) fn slot_offset_bytes(
    slot: crate::vm::middle::frame::FrameSlot,
) -> Result<i32, WasmError> {
    let bytes = i32::from(slot.0)
        .checked_mul(8)
        .ok_or_else(|| WasmError::internal("frame slot byte offset overflow".into()))?;
    Ok(bytes)
}

fn derive_return_results(program: &SsaProgram) -> Result<Option<MachineFrameRegion>, WasmError> {
    let mut derived: Option<Option<MachineFrameRegion>> = None;
    for block in &program.blocks {
        let SsaTerminator::Return { results } = &block.terminator else {
            continue;
        };
        let region = results.map(frame_span_region);
        match derived {
            None => derived = Some(region),
            Some(current) if current == region => {}
            Some(_) => {
                return Err(WasmError::internal(
                    "prepared SSA-IR uses inconsistent return result spans across blocks".into(),
                ));
            }
        }
    }
    Ok(derived.unwrap_or(None))
}

#[inline]
fn frame_span_region(span: FrameSpan) -> MachineFrameRegion {
    MachineFrameRegion {
        base_slot: span.start.0,
        slots: span.count,
    }
}

fn push_lowered_block(
    id: MachineBlockId,
    original_blocks: &mut [Option<MachineBlock>],
    continuation_blocks: &mut Vec<MachineBlock>,
    params: Vec<MachineBlockParam>,
    ops: Vec<MachineInst>,
    terminator: MachineTerminator,
) -> Result<(), WasmError> {
    let block = MachineBlock {
        id,
        params,
        ops,
        terminator,
    };
    let original_len = original_blocks.len();
    if id.as_usize() < original_len {
        let slot = &mut original_blocks[id.as_usize()];
        if slot.is_some() {
            return Err(WasmError::internal(
                "machine lowering attempted to assign one original block twice".into(),
            ));
        }
        *slot = Some(block);
    } else {
        let expected = original_len + continuation_blocks.len();
        if id.as_usize() != expected {
            return Err(WasmError::internal(
                "machine continuation blocks must be appended in id order".into(),
            ));
        }
        continuation_blocks.push(block);
    }
    Ok(())
}

fn attach_continuation_args(
    terminator: &mut MachineTerminator,
    continuation: MachineBlockId,
    params: &[MachineBlockParam],
) -> Result<(), WasmError> {
    let args = params
        .iter()
        .map(|param| MachineValue::Reg(param.reg))
        .collect::<Vec<_>>();
    let attached = match terminator {
        MachineTerminator::Jump(edge) => attach_edge_args(edge, continuation, args),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            let then_args = args.clone();
            attach_edge_args(then_edge, continuation, then_args)
                | attach_edge_args(else_edge, continuation, args)
        }
        MachineTerminator::JumpTable { entries, .. } => {
            let mut attached = false;
            for edge in entries {
                attached |= attach_edge_args(edge, continuation, args.clone());
            }
            attached
        }
        MachineTerminator::CallDirect { .. }
        | MachineTerminator::CallIndirect { .. }
        | MachineTerminator::Return
        | MachineTerminator::Trap { .. } => false,
    };
    if attached {
        Ok(())
    } else {
        Err(WasmError::internal(
            "split continuation terminator does not branch to the continuation block".into(),
        ))
    }
}

fn attach_edge_args(
    edge: &mut MachineEdge,
    continuation: MachineBlockId,
    args: Vec<MachineValue>,
) -> bool {
    if edge.target != continuation {
        return false;
    }
    edge.args = args;
    true
}

#[inline]
pub(super) fn target_param_regs(
    params: &[SsaValue],
    program: &SsaProgram,
    regfile: &MachineRegFile,
    gp_reg_width: u8,
) -> Result<Vec<ValueRegs>, WasmError> {
    let mut regs = Vec::with_capacity(params.len());
    let mut gp_index = 0usize;
    let mut fp_index = 0usize;
    for param in params {
        let ty = program_value_storage_type(program, *param);
        if ty.is_fp() {
            regs.push(ValueRegs {
                lo: regfile.fp_transient(fp_index).ok_or_else(|| {
                    WasmError::internal("target params exceed FP transient register budget".into())
                })?,
                hi: None,
            });
            fp_index += 1;
        } else if gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
            let lo = regfile.gp_transient(gp_index).ok_or_else(|| {
                WasmError::internal("target params exceed GP transient register budget".into())
            })?;
            let hi = regfile.gp_transient(gp_index + 1).ok_or_else(|| {
                WasmError::internal("target i64 params exceed GP transient pair budget".into())
            })?;
            regs.push(ValueRegs { lo, hi: Some(hi) });
            gp_index += 2;
        } else {
            regs.push(ValueRegs {
                lo: regfile.gp_transient(gp_index).ok_or_else(|| {
                    WasmError::internal("target params exceed GP transient register budget".into())
                })?,
                hi: None,
            });
            gp_index += 1;
        }
    }
    Ok(regs)
}

fn program_value_storage_type(program: &SsaProgram, value: SsaValue) -> MachineStorageType {
    match program.value_types.get(value.0 as usize).copied() {
        Some(ValueType::F32) => MachineStorageType::Fp32,
        Some(ValueType::F64) => MachineStorageType::Fp64,
        Some(ValueType::I64) => MachineStorageType::GpI64,
        Some(_) | None => MachineStorageType::GpWord,
    }
}

fn fp_reg_init_widths(
    regfile: &MachineRegFile,
    cache_prefs: &SsaLocalCachePrefs,
) -> Result<Vec<Option<MachineFloatWidth>>, WasmError> {
    let mut widths = vec![None; regfile.fp_transient_count() + regfile.fp_local_cache_count()];
    if cache_prefs.fp_preferred_slots.len() != cache_prefs.fp_preferred_types.len() {
        return Err(WasmError::internal(
            "FP cached-local slot/type metadata length mismatch".into(),
        ));
    }
    for (index, ty) in cache_prefs.fp_preferred_types.iter().copied().enumerate() {
        let width = match ty {
            ValueType::F32 => MachineFloatWidth::F32,
            ValueType::F64 => MachineFloatWidth::F64,
            _ => {
                return Err(WasmError::internal(
                    "FP cached-local preference contains a non-float type".into(),
                ))
            }
        };
        let slot = regfile.fp_transient_count() + index;
        let Some(entry) = widths.get_mut(slot) else {
            return Err(WasmError::internal(
                "FP cached-local preferences exceed declared FP cache register count".into(),
            ));
        };
        *entry = Some(width);
    }
    Ok(widths)
}

struct ExtraBlockAllocator {
    next: u32,
}

impl ExtraBlockAllocator {
    #[inline]
    fn new(first_extra_block: u32) -> Self {
        Self {
            next: first_extra_block,
        }
    }

    #[inline]
    fn alloc(&mut self) -> MachineBlockId {
        let id = MachineBlockId(self.next);
        self.next += 1;
        id
    }

    #[inline]
    fn peek(&self, offset: u32) -> MachineBlockId {
        MachineBlockId(self.next + offset)
    }

    #[inline]
    fn reserve(&mut self, count: u32) {
        self.next += count;
    }
}

#[derive(Clone, Copy, Debug)]
struct CallIndirectGpTemps {
    lane0: MachineReg,
    lane1: MachineReg,
    lane2: MachineReg,
}

// `call_indirect` is the structured MachineIR exception that intentionally
// threads a fixed GP transient bundle across synthetic blocks.
fn call_indirect_gp_temps(lower: &BlockLowerContext<'_>) -> Result<CallIndirectGpTemps, WasmError> {
    Ok(CallIndirectGpTemps {
        lane0: lower.reserved_gp_transient(0, "call_indirect control lane 0")?,
        lane1: lower.reserved_gp_transient(1, "call_indirect control lane 1")?,
        lane2: lower.reserved_gp_transient(2, "call_indirect control lane 2")?,
    })
}

fn emit_call_indirect_bounds_check_setup(
    lower: &mut BlockLowerContext<'_>,
    table_idx: u32,
    index_slot: crate::vm::middle::frame::FrameSlot,
) -> Result<(), WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let index = temps.lane0;
    let table_views = temps.lane1;
    let table_len = temps.lane2;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: index,
            addr: lower.frame_addr(index_slot)?,
            width: lower.canonical_gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: table_views,
            addr: lower.runtime_addr(runtime_layout.context.table_views_base_offset),
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: table_len,
            addr: indexed_const_addr(
                table_views,
                table_idx,
                runtime_layout.pointer_len_view.stride as usize,
                runtime_layout.pointer_len_view.len_offset,
            )?,
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    Ok(())
}

fn build_call_indirect_checked_block(
    lower: &BlockLowerContext<'_>,
    table_idx: u32,
    index_slot: crate::vm::middle::frame::FrameSlot,
) -> Result<Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let index = temps.lane0;
    let table_base = temps.lane1;
    let func_idx = temps.lane2;
    Ok(vec![
        MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: index,
                addr: lower.frame_addr(index_slot)?,
                width: lower.canonical_gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: table_base,
                addr: lower.runtime_addr(runtime_layout.context.table_views_base_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: table_base,
                addr: indexed_const_addr(
                    table_base,
                    table_idx,
                    runtime_layout.pointer_len_view.stride as usize,
                    runtime_layout.pointer_len_view.base_offset,
                )?,
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Mul,
                dst: index,
                lhs: MachineValue::Reg(index),
                rhs: MachineValue::Imm64(u64::from(runtime_layout.ref_handle_stride)),
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: table_base,
                lhs: MachineValue::Reg(table_base),
                rhs: MachineValue::Reg(index),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: func_idx,
                addr: MachineAddr {
                    base: table_base,
                    offset: 0,
                },
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: lower.frame_addr(index_slot)?,
                width: lower.canonical_gp_word_mem_width(),
                src: MachineValue::Reg(func_idx),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: table_base,
                addr: lower.runtime_addr(runtime_layout.context.function_views_len_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
    ])
}

fn build_call_indirect_type_check_block(
    lower: &BlockLowerContext<'_>,
    expected_type_idx: u32,
    index_slot: crate::vm::middle::frame::FrameSlot,
) -> Result<Vec<MachineInst>, WasmError> {
    let function_view_layout = function_view_abi_layout();
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let actual_type = temps.lane0;
    let function_views = temps.lane1;
    let scaled_index = temps.lane2;
    let expected_type = temps.lane2;
    let mut ops = dynamic_function_view_load(
        lower,
        index_slot,
        scaled_index,
        function_views,
        scaled_index,
        function_view_layout.type_canon_offset,
        MachineMemWidth::U32,
        MachineLoadExtension::ZeroExtend,
        actual_type,
    )?;
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: function_views,
            addr: lower.runtime_addr(runtime_layout.context.type_canon_base_offset),
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: expected_type,
            addr: indexed_const_addr(
                function_views,
                expected_type_idx,
                core::mem::size_of::<u32>(),
                0,
            )?,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
        },
    });
    Ok(ops)
}

fn build_call_indirect_dispatch_block(
    lower: &BlockLowerContext<'_>,
    index_slot: crate::vm::middle::frame::FrameSlot,
) -> Result<Vec<MachineInst>, WasmError> {
    let function_view_layout = function_view_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let kind = temps.lane0;
    let function_views = temps.lane1;
    let scaled_index = temps.lane2;
    let local_target = temps.lane2;
    let mut ops = dynamic_function_view_load(
        lower,
        index_slot,
        scaled_index,
        function_views,
        scaled_index,
        function_view_layout.kind_offset,
        MachineMemWidth::U32,
        MachineLoadExtension::ZeroExtend,
        kind,
    )?;
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: local_target,
            addr: MachineAddr {
                base: function_views,
                offset: function_view_layout.local_target_offset as i32,
            },
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
        },
    });
    Ok(ops)
}

fn build_call_indirect_local_block(
    lower: &mut BlockLowerContext<'_>,
    args: FrameSpan,
) -> Result<Vec<MachineInst>, WasmError> {
    let callee_frame_base = call_indirect_gp_temps(lower)?.lane1;

    let ops = vec![MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: callee_frame_base,
            lhs: MachineValue::Reg(lower.frame_base_reg()),
            rhs: MachineValue::Imm64(slot_offset_bytes(args.start)? as u64),
        },
    }];
    // By the time the local branch reaches this block, the dispatch path has
    // already resolved and validated the local callee target above MachineIR.
    // The remaining dynamic work below MachineIR is just the local-call
    // transfer mechanics for that resolved target. Arguments are already laid
    // out in the caller operand window, so the callee frame starts there.
    lower.emit_machine_ops(ops);
    Ok(lower.take_ops())
}

fn dynamic_function_view_load(
    lower: &BlockLowerContext<'_>,
    index_slot: crate::vm::middle::frame::FrameSlot,
    func_idx_dst: MachineReg,
    base_reg: MachineReg,
    scaled_index_reg: MachineReg,
    field_offset: u32,
    field_width: MachineMemWidth,
    field_extension: MachineLoadExtension,
    dst: MachineReg,
) -> Result<Vec<MachineInst>, WasmError> {
    let runtime_layout = native_runtime_abi_layout(lower.gp_reg_width());
    let mut ops = vec![MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: func_idx_dst,
            addr: lower.frame_addr(index_slot)?,
            width: lower.canonical_gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    }];
    if scaled_index_reg != func_idx_dst {
        ops.push(MachineInst {
            kind: MachineInstKind::Move {
                ty: MachineStorageType::GpWord,
                dst: scaled_index_reg,
                src: MachineValue::Reg(func_idx_dst),
            },
        });
    }
    ops.extend([
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Mul,
                dst: scaled_index_reg,
                lhs: MachineValue::Reg(scaled_index_reg),
                rhs: MachineValue::Imm64(u64::from(runtime_layout.function_view.stride)),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: base_reg,
                addr: lower.runtime_addr(runtime_layout.context.function_views_base_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: base_reg,
                lhs: MachineValue::Reg(base_reg),
                rhs: MachineValue::Reg(scaled_index_reg),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst,
                addr: MachineAddr {
                    base: base_reg,
                    offset: field_offset as i32,
                },
                width: field_width,
                extension: field_extension,
            },
        },
    ]);
    Ok(ops)
}

fn indexed_const_addr(
    base: MachineReg,
    index: u32,
    stride: usize,
    field_offset: u32,
) -> Result<MachineAddr, WasmError> {
    let scaled = (index as u64)
        .checked_mul(stride as u64)
        .and_then(|value| value.checked_add(field_offset as u64))
        .ok_or_else(|| WasmError::internal("runtime view byte offset overflow".into()))?;
    let offset = i32::try_from(scaled)
        .map_err(|_| WasmError::internal("runtime view byte offset exceeds i32".into()))?;
    Ok(MachineAddr { base, offset })
}

// ---------------------------------------------------------------------------
// Cross-block dirty-flag dataflow analysis
// ---------------------------------------------------------------------------
//
// Computes, for each SSA-IR block, which cached locals are dirty (register !=
// frame slot) at block entry.  A cached local becomes dirty when a `LocalSet`
// writes to it and becomes clean at every `Boundary` op (which saves all dirty
// locals to frame).
//
// The analysis is a forward dataflow with join = OR (dirty if ANY predecessor
// leaves it dirty).  Initialized optimistically (all-clean) and iterated to
// fixpoint.  Entry block is always all-clean.

fn compute_block_entry_dirty(
    blocks: &[SsaBlock],
    entry: SsaTarget,
    cache_prefs: &SsaLocalCachePrefs,
) -> Vec<Vec<bool>> {
    let n_blocks = blocks.len();
    let n_cached = cache_prefs.gp_preferred_slots.len() + cache_prefs.fp_preferred_slots.len();

    if n_cached == 0 || n_blocks == 0 {
        return vec![vec![]; n_blocks];
    }

    // Map FrameSlot → cached-local index (GP then FP order).
    let all_slots: Vec<FrameSlot> = cache_prefs
        .gp_preferred_slots
        .iter()
        .chain(cache_prefs.fp_preferred_slots.iter())
        .copied()
        .collect();
    let max_slot = all_slots.iter().map(|s| s.0).max().unwrap_or(0) as usize;
    let mut slot_to_index = vec![usize::MAX; max_slot + 1];
    for (i, slot) in all_slots.iter().enumerate() {
        slot_to_index[slot.0 as usize] = i;
    }

    // Build predecessor map.
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n_blocks];
    for (idx, block) in blocks.iter().enumerate() {
        let succs = terminator_successors(&block.terminator);
        for succ in succs {
            let si = succ.0 as usize;
            if si < n_blocks {
                predecessors[si].push(idx);
            }
        }
    }

    // Precompute per-block transfer summaries.
    //
    // If a block has any Boundary op:
    //   exit_dirty = locals written AFTER the last Boundary (entry state irrelevant)
    // If no Boundary:
    //   exit_dirty[i] = entry_dirty[i] OR written_in_block[i]
    let mut has_boundary = vec![false; n_blocks];
    let mut written_after_last_boundary = vec![vec![false; n_cached]; n_blocks];
    let mut written_anywhere = vec![vec![false; n_cached]; n_blocks];

    for (idx, block) in blocks.iter().enumerate() {
        for op in &block.ops {
            match &op.kind {
                SsaInstKind::LocalSet { slot, .. } => {
                    let si = slot.0 as usize;
                    if si <= max_slot {
                        let ci = slot_to_index[si];
                        if ci != usize::MAX {
                            written_after_last_boundary[idx][ci] = true;
                            written_anywhere[idx][ci] = true;
                        }
                    }
                }
                SsaInstKind::Boundary(_) => {
                    has_boundary[idx] = true;
                    for w in &mut written_after_last_boundary[idx] {
                        *w = false;
                    }
                }
                _ => {}
            }
        }
    }

    // Forward dataflow to fixpoint.
    let entry_idx = entry.0 as usize;
    let mut entry_dirty = vec![vec![false; n_cached]; n_blocks];
    let mut exit_dirty = vec![vec![false; n_cached]; n_blocks];

    let mut changed = true;
    while changed {
        changed = false;
        for idx in 0..n_blocks {
            // Join: entry_dirty = OR of predecessor exit_dirty values.
            if idx != entry_idx {
                for &pred in &predecessors[idx] {
                    for i in 0..n_cached {
                        if exit_dirty[pred][i] && !entry_dirty[idx][i] {
                            entry_dirty[idx][i] = true;
                            changed = true;
                        }
                    }
                }
            }

            // Transfer: compute exit_dirty.
            if has_boundary[idx] {
                // Last boundary cleared everything; only subsequent writes
                // contribute to exit state.
                for i in 0..n_cached {
                    let new = written_after_last_boundary[idx][i];
                    if new != exit_dirty[idx][i] {
                        exit_dirty[idx][i] = new;
                        changed = true;
                    }
                }
            } else {
                // No boundary: exit = entry OR written_anywhere.
                for i in 0..n_cached {
                    let new = entry_dirty[idx][i] || written_anywhere[idx][i];
                    if new != exit_dirty[idx][i] {
                        exit_dirty[idx][i] = new;
                        changed = true;
                    }
                }
            }
        }
    }

    entry_dirty
}

fn terminator_successors(term: &SsaTerm) -> Vec<SsaTarget> {
    match term {
        SsaTerm::Goto(edge) => vec![edge.target],
        SsaTerm::Branch {
            then_edge,
            else_edge,
            ..
        } => vec![then_edge.target, else_edge.target],
        SsaTerm::BrTable { entries, .. } => entries.iter().map(|e| e.target).collect(),
        SsaTerm::Return { .. } | SsaTerm::TrapUnreachable => vec![],
    }
}
