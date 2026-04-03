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
            MachineFrameRegion, MachineFuncId, MachineFunction, MachineFunctionAbi, MachineInst,
            MachineInstKind, MachineIntBinaryOp, MachineLoadExtension, MachineMemWidth,
            MachineModule, MachineModuleAbi, MachineProgram, MachineReg, MachineSign,
            MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
        },
        middle::{
            frame::{FrameLayoutPlan, FrameSpan},
            ssa_ir::{
                ir::{
                    SsaCallOp, SsaInstKind, SsaProgram, SsaTerminator, SsaValue,
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

use super::{
    gp32::Gp32Lowering,
    lower_const_pool::ConstPoolBuilder,
    lower_context::{explicit_cached_locals, BlockLowerContext, EntryCacheParam, ValueRegs},
    lower_i64::I64Lowering,
    lower_i64_gp64::Gp64Lowering,
    lower_inst::LeafLowering,
    lower_regalloc::{machine_block_params_for_value, MachineRegFile},
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

/// Result of lowering prepared SSA-IR into MachineIR plus backend-facing ABI
/// metadata derived from the shared frame plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredMachineModule {
    pub module: MachineModule,
    pub abi: MachineModuleAbi,
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
    let mut is_local_func = alloc::vec![false; function_count];
    let mut function_abis = (0..function_count)
        .map(|index| MachineFunctionAbi {
            id: MachineFuncId(index as u32),
            ..MachineFunctionAbi::default()
        })
        .collect::<Vec<_>>();
    let mut const_pool = ConstPoolBuilder::new();
    for function in input.functions {
        validate_program(function.ssa)?;
        is_local_func[function.id.0 as usize] = true;
        function_abis[function.id.0 as usize] = lower_function_runtime(*function, call_link)?;
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
            &function_abis,
            &is_local_func,
            call_link,
            &mut const_pool,
            guard_pages,
        )?);
    }
    let abi = MachineModuleAbi {
        call_link,
        functions: function_abis,
    };

    let consts = const_pool.finish();
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
    };
    module.validate()?;

    Ok(LoweredMachineModule { module, abi })
}

fn lower_function_runtime(
    input: LowerFunctionInput<'_>,
    call_link: MachineCallLinkLayout,
) -> Result<MachineFunctionAbi, WasmError> {
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

    Ok(MachineFunctionAbi {
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
    runtime: &[MachineFunctionAbi],
    is_local_func: &[bool],
    call_link: MachineCallLinkLayout,
    const_pool: &mut ConstPoolBuilder,
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
    let explicit_cache = explicit_cached_locals(input.ssa);

    for block in &input.ssa.blocks {
        let target = block.id;
        let mut lower = BlockLowerContext::new(
            regfile,
            input.ssa,
            &explicit_cache,
            block,
            runtime,
            call_link,
            gp_reg_width,
            i64_ops,
            target == input.ssa.entry,
            None,
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
        append_entry_cache_params(&mut current_params, lower.entry_cache_params(), &explicit_cache);

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
                SsaInstKind::Call(call) => match call {
                    // `CallDirect` preserves Wasm semantics above MachineIR:
                    // the callee is compile-time known, but target kind
                    // (local vs external) is decided here.
                    //
                    // Local targets become MIR terminators because control
                    // transfers into another compiled MachineIR function and
                    // resumes at an explicit continuation block.
                    SsaCallOp::CallDirect {
                        callee,
                        args,
                        results,
                        skip_reload,
                    } if is_local_func
                        .get(*callee as usize)
                        .copied()
                        .unwrap_or(false) =>
                    {
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
                    // External targets stay in the current block as helper-call
                    // instructions. They cross the runtime ABI boundary but do
                    // not produce a MachineIR CFG edge.
                    SsaCallOp::CallDirect {
                        callee,
                        args,
                        results,
                        skip_reload,
                    } => {
                        lower.lower_call_external(
                            *callee,
                            *args,
                            *results,
                            skip_reload,
                            const_pool,
                        )?;
                    }
                    SsaCallOp::CallIndirect { .. } => {
                        let SsaCallOp::CallIndirect {
                            type_idx,
                            table_idx,
                            index_slot,
                            args,
                            results,
                            skip_reload,
                        } = call
                        else {
                            unreachable!("matched call_indirect");
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

                        // `call_indirect` lowers to a synthetic block cluster because the
                        // MachineIR needs each runtime-visible check and target-kind split to be
                        // explicit in CFG form:
                        //
                        //   current_block
                        //     -> trap_oob
                        //     -> checked
                        //          -> trap_invalid_ref
                        //          -> type_check
                        //               -> trap_type
                        //               -> dispatch
                        //                    -> local_prepare
                        //                         -> local_transfer
                        //                         -> local_zero_loop -> local_transfer
                        //                    -> external_call
                        //
                        //   local_transfer --CallIndirect--> continuation
                        //   external_call --------Jump-----> continuation
                        //
                        // The local arm needs more blocks because it must load local-call
                        // metadata, run a dynamic stack precheck, and zero the callee frame prefix
                        // before it can commit the final transfer terminator. The external arm is
                        // a straight inline runtime-entry call that rejoins the shared
                        // continuation block immediately.
                        let checked = extra_block_ids.alloc();
                        let trap_oob = extra_block_ids.alloc();
                        let type_check = extra_block_ids.alloc();
                        let trap_invalid_ref = extra_block_ids.alloc();
                        let dispatch = extra_block_ids.alloc();
                        let trap_type = extra_block_ids.alloc();
                        let local_prepare = extra_block_ids.alloc();
                        let local_zero_loop = extra_block_ids.alloc();
                        let local_transfer = extra_block_ids.alloc();
                        let external_call = extra_block_ids.alloc();
                        let continuation = extra_block_ids.alloc();
                        let indirect_temps = call_indirect_gp_temps(&lower)?;
                        // These four reserved GP lanes are intentionally threaded across the
                        // synthetic blocks with stage-specific meanings:
                        //
                        //   lane0: table index -> dispatch/type scratch -> resolved local callee id
                        //   lane1: table base  -> callee frame base
                        //   lane2: table len / type id / entry ptr / zero-loop bound
                        //   lane3: zero-loop cursor / call-link base
                        let local_call_target_param = indirect_temps.lane0;

                        lower.emit_save_dirty_cached_locals()?;
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

                        // `checked` resolves the actual function index from the selected table
                        // element after the outer bounds check has succeeded.
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
                        // `type_check` validates the resolved target's canonical signature against
                        // the Wasm type expected by this call site.
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
                        // `dispatch` decides whether the resolved target stays inside compiled
                        // local code or crosses the external-call runtime entry.
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
                                    target: local_prepare,
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
                        // `local_prepare` is the first local-only stage. It computes the callee
                        // frame base, loads local-call metadata (entry address, frame size, local
                        // prefix length, call-scratch base), performs the dynamic stack precheck,
                        // and seeds the zero-loop cursor/bound.
                        push_lowered_block(
                            local_prepare,
                            &mut original_blocks,
                            &mut extra_blocks,
                            vec![MachineBlockParam::gp_word(local_call_target_param)],
                            build_call_indirect_local_prepare_block(&mut lower, args)?,
                            MachineTerminator::Branch {
                                cond: MachineBranchCond::IntCompare {
                                    width: lower.gp_word_int_width(),
                                    kind: MachineCompareKind::Ge,
                                    sign: MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(indirect_temps.lane3),
                                    rhs: MachineValue::Reg(indirect_temps.lane2),
                                },
                                then_edge: MachineEdge {
                                    target: local_transfer,
                                    args: Vec::new(),
                                },
                                else_edge: MachineEdge {
                                    target: local_zero_loop,
                                    args: Vec::new(),
                                },
                            },
                        )?;
                        // `local_zero_loop` clears the part of the callee local-prefix window that
                        // lies above the passed arguments. It is skipped completely when the
                        // argument span already covers the full prefix.
                        push_lowered_block(
                            local_zero_loop,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            build_call_indirect_local_zero_loop_block(&mut lower)?,
                            MachineTerminator::Branch {
                                cond: MachineBranchCond::IntCompare {
                                    width: lower.gp_word_int_width(),
                                    kind: MachineCompareKind::Lt,
                                    sign: MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(indirect_temps.lane3),
                                    rhs: MachineValue::Reg(indirect_temps.lane2),
                                },
                                then_edge: MachineEdge {
                                    target: local_zero_loop,
                                    args: Vec::new(),
                                },
                                else_edge: MachineEdge {
                                    target: local_transfer,
                                    args: Vec::new(),
                                },
                            },
                        )?;
                        // `local_transfer` writes the logical call-link fields and terminates with
                        // `CallIndirect`. By this point MachineIR has already resolved the callee
                        // entry address and chosen the call-link base; the backend only commits the
                        // final transfer.
                        push_lowered_block(
                            local_transfer,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            build_call_indirect_local_transfer_block(
                                &mut lower,
                                continuation,
                                results,
                            )?,
                            MachineTerminator::CallIndirect {
                                callee_target: indirect_temps.lane0,
                                callee_entry: indirect_temps.lane2,
                                callee_frame_base: indirect_temps.lane1,
                                call_link_base: indirect_temps.lane3,
                                continuation,
                            },
                        )?;
                        let metadata = lower.build_call_external_meta_indirect(
                            func_idx_slot,
                            args,
                            results,
                            const_pool,
                        );
                        // `external_call` is the external-target sibling of the local path. It
                        // reuses the resolved function index now stored in `func_idx_slot`,
                        // performs the inline external-call sequence, and then jumps to the shared
                        // continuation block.
                        push_lowered_block(
                            external_call,
                            &mut original_blocks,
                            &mut extra_blocks,
                            Vec::new(),
                            lower.build_external_call_ops(metadata),
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
        fp_reg_init_widths: fp_reg_init_widths(&regfile)?,
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
                lo: preferred_fp_dynamic_reg(regfile, fp_index).ok_or_else(|| {
                    WasmError::internal("target params exceed FP dynamic register budget".into())
                })?,
                hi: None,
            });
            fp_index += 1;
        } else if gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
            let lo = preferred_gp_dynamic_reg(regfile, gp_index).ok_or_else(|| {
                WasmError::internal("target params exceed GP dynamic register budget".into())
            })?;
            let hi = preferred_gp_dynamic_reg(regfile, gp_index + 1).ok_or_else(|| {
                WasmError::internal("target i64 params exceed GP dynamic pair budget".into())
            })?;
            regs.push(ValueRegs { lo, hi: Some(hi) });
            gp_index += 2;
        } else {
            regs.push(ValueRegs {
                lo: preferred_gp_dynamic_reg(regfile, gp_index).ok_or_else(|| {
                    WasmError::internal("target params exceed GP dynamic register budget".into())
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

pub(super) fn preferred_gp_dynamic_reg(regfile: &MachineRegFile, ordinal: usize) -> Option<MachineReg> {
    if ordinal < regfile.gp_transient_count() {
        regfile.gp_transient(ordinal)
    } else {
        regfile.gp_local_cache(ordinal - regfile.gp_transient_count())
    }
}

pub(super) fn preferred_fp_dynamic_reg(regfile: &MachineRegFile, ordinal: usize) -> Option<MachineReg> {
    if ordinal < regfile.fp_transient_count() {
        regfile.fp_transient(ordinal)
    } else {
        regfile.fp_local_cache(ordinal - regfile.fp_transient_count())
    }
}

fn fp_reg_init_widths(regfile: &MachineRegFile) -> Result<Vec<Option<MachineFloatWidth>>, WasmError> {
    Ok(vec![None; regfile.fp_dynamic_count()])
}

fn append_entry_cache_params(
    params: &mut Vec<MachineBlockParam>,
    entry_cache_params: &[EntryCacheParam],
    cached_locals: &[super::lower_context::CachedLocal],
) {
    for entry in entry_cache_params {
        if let Some(cached) = cached_locals.get(entry.cached_index) {
            params.extend(machine_block_params_for_value(entry.regs, cached.ty));
        }
    }
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
    lane3: MachineReg,
}

// `call_indirect` is the structured MachineIR exception that intentionally
// threads a fixed GP transient bundle across synthetic blocks.
fn call_indirect_gp_temps(lower: &BlockLowerContext<'_>) -> Result<CallIndirectGpTemps, WasmError> {
    Ok(CallIndirectGpTemps {
        lane0: lower.reserved_gp_transient(0, "call_indirect control lane 0")?,
        lane1: lower.reserved_gp_transient(1, "call_indirect control lane 1")?,
        lane2: lower.reserved_gp_transient(2, "call_indirect control lane 2")?,
        lane3: lower.reserved_gp_transient(3, "call_indirect control lane 3")?,
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

fn build_call_indirect_local_prepare_block(
    lower: &mut BlockLowerContext<'_>,
    args: FrameSpan,
) -> Result<Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let call_info = runtime_layout.local_call_info;
    let temps = call_indirect_gp_temps(lower)?;
    let callee_target = temps.lane0;
    let callee_frame_base = temps.lane1;
    let prefix_end = temps.lane2;
    let prefix_current = temps.lane3;

    // The local callee reuses the caller operand span as its frame prefix, so
    // the only dynamic work left is per-callee stack/prefix/call-link setup.
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: callee_frame_base,
            lhs: MachineValue::Reg(lower.frame_base_reg()),
            rhs: MachineValue::Imm64(slot_offset_bytes(args.start)? as u64),
        },
    });

    emit_local_call_info_entry_addr(lower, callee_target, prefix_end, prefix_current)?;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: prefix_current,
            addr: MachineAddr {
                base: prefix_end,
                offset: call_info.total_frame_bytes_offset as i32,
            },
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_dynamic_call_stack_precheck(callee_frame_base, prefix_end, prefix_current)?;

    emit_local_call_info_entry_addr(lower, callee_target, prefix_end, prefix_current)?;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: prefix_end,
            addr: MachineAddr {
                base: prefix_end,
                offset: call_info.frame_prefix_slots_offset as i32,
            },
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Mul,
            dst: prefix_end,
            lhs: MachineValue::Reg(prefix_end),
            rhs: MachineValue::Imm64(8),
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: prefix_end,
            lhs: MachineValue::Reg(prefix_end),
            rhs: MachineValue::Reg(callee_frame_base),
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: prefix_current,
            lhs: MachineValue::Reg(callee_frame_base),
            rhs: MachineValue::Imm64(u64::from(args.count) * 8),
        },
    });
    Ok(lower.take_ops())
}

fn build_call_indirect_local_zero_loop_block(
    lower: &mut BlockLowerContext<'_>,
) -> Result<Vec<MachineInst>, WasmError> {
    let temps = call_indirect_gp_temps(lower)?;
    let prefix_end = temps.lane2;
    let prefix_current = temps.lane3;
    lower.emit_zero_canonical_slot_at_addr(prefix_current)?;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: prefix_current,
            lhs: MachineValue::Reg(prefix_current),
            rhs: MachineValue::Imm64(8),
        },
    });
    let _ = prefix_end;
    Ok(lower.take_ops())
}

fn build_call_indirect_local_transfer_block(
    lower: &mut BlockLowerContext<'_>,
    continuation: MachineBlockId,
    results: FrameSpan,
) -> Result<Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let call_info = runtime_layout.local_call_info;
    let temps = call_indirect_gp_temps(lower)?;
    let callee_target = temps.lane0;
    let callee_frame_base = temps.lane1;
    let callee_entry = temps.lane2;
    let call_link_base = temps.lane3;

    emit_local_call_info_entry_addr(lower, callee_target, call_link_base, callee_entry)?;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: callee_entry,
            addr: MachineAddr {
                base: call_link_base,
                offset: call_info.entry_offset as i32,
            },
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: call_link_base,
            addr: MachineAddr {
                base: call_link_base,
                offset: call_info.call_scratch_base_slot_offset as i32,
            },
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Mul,
            dst: call_link_base,
            lhs: MachineValue::Reg(call_link_base),
            rhs: MachineValue::Imm64(8),
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: call_link_base,
            lhs: MachineValue::Reg(call_link_base),
            rhs: MachineValue::Reg(callee_frame_base),
        },
    });
    lower.store_call_link_absolute(
        call_link_base,
        continuation,
        slot_offset_bytes(results.start)? as u64,
    )?;
    Ok(lower.take_ops())
}

fn emit_local_call_info_entry_addr(
    lower: &mut BlockLowerContext<'_>,
    callee_target: MachineReg,
    info_base: MachineReg,
    scaled_index: MachineReg,
) -> Result<(), WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            ty: MachineStorageType::GpWord,
            dst: info_base,
            addr: lower.runtime_addr(runtime_layout.context.local_call_infos_base_offset),
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    if scaled_index != callee_target {
        lower.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Move {
                ty: MachineStorageType::GpWord,
                dst: scaled_index,
                src: MachineValue::Reg(callee_target),
            },
        });
    }
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Mul,
            dst: scaled_index,
            lhs: MachineValue::Reg(scaled_index),
            rhs: MachineValue::Imm64(u64::from(runtime_layout.local_call_info.stride)),
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: info_base,
            lhs: MachineValue::Reg(info_base),
            rhs: MachineValue::Reg(scaled_index),
        },
    });
    Ok(())
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
