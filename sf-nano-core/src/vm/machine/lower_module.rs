// ---------------------------------------------------------------------------
// Lowering: prepared SSA-IR → MachineIR
// ---------------------------------------------------------------------------

use crate::collections::{self, phase_span_with_function};

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        entities::TableDispatchMode,
        machine::machine_ir::{
            MachineAddr, MachineArgSrc, MachineArgSrcPair, MachineBlock, MachineBlockId,
            MachineBlockParam, MachineBranchCond, MachineCallArgs, MachineCallLaneArg,
            MachineCallTarget, MachineCompareKind, MachineConstId, MachineConvertOp, MachineEdge,
            MachineFloatWidth, MachineFrameRegion, MachineFuncId, MachineFunction,
            MachineFunctionAbi, MachineIndexExtend, MachineInst, MachineInstKind,
            MachineIntBinaryOp, MachineLoadExtension, MachineMemWidth, MachineModule,
            MachineModuleAbi, MachineParamLoc, MachineProgram, MachineReg, MachineRegOwner,
            MachineResultSrc, MachineReturnAbi, MachineReturnValue, MachineSign,
            MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
        },
        middle::{
            cell::CellId,
            frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
            ssa_ir::{
                ir::{
                    SsaBlock, SsaCallArgs, SsaCallLiveArg, SsaCallOp, SsaInstView, SsaOp,
                    SsaOperand, SsaProgram, SsaTerminator, SsaValue,
                },
                target::SsaTarget,
                validate::validate_program,
            },
        },
        runtime::{
            context::function_kind,
            layout::{
                fixed_call_table_entry_abi_layout, fixed_call_table_view_abi_layout,
                function_view_abi_layout,
            },
        },
    },
};

use super::{
    call_abi::INDIRECT_DISPATCH_CONTROL_GP_LANES,
    gp32::Gp32Lowering,
    lower_cache_layout::{compute_block_entry_cache_params, preferred_preserved_for_cached},
    lower_const_pool::ConstPoolBuilder,
    lower_context::{
        explicit_cached_cells, BlockLowerContext, CachedCell, EntryCacheParam, ValueRegs,
    },
    lower_i64::I64Lowering,
    lower_i64_gp64::Gp64Lowering,
    lower_inst::LeafLowering,
    lower_regalloc::{
        append_machine_block_params_for_value, value_type_storage_type, MachineRegFile,
    },
};

/// One prepared function borrowed for internal lowering helpers.
#[derive(Clone, Copy, Debug)]
struct BorrowedLowerFunctionInput<'a> {
    pub id: MachineFuncId,
    pub frame: FrameLayoutPlan,
    pub ssa: &'a SsaProgram,
    /// Declared result count from the function type signature.
    /// Used as fallback when all return paths are unreachable.
    pub result_count: u16,
}

/// Owned lowering input that allows the caller to release prepared SSA
/// progressively while MachineIR is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LowerFunctionInput {
    pub id: MachineFuncId,
    pub frame: FrameLayoutPlan,
    pub ssa: SsaProgram,
    pub result_count: u16,
}

impl LowerFunctionInput {
    #[inline]
    fn borrowed(&self) -> BorrowedLowerFunctionInput<'_> {
        BorrowedLowerFunctionInput {
            id: self.id,
            frame: self.frame,
            ssa: &self.ssa,
            result_count: self.result_count,
        }
    }
}

/// Owned whole-module lowering request. This is used by the production compile
/// pipeline so SSA can be dropped function-by-function during lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LowerModuleInput {
    pub backend: BackendConfig,
    pub functions: collections::Vec<LowerFunctionInput>,
    #[cfg(sf_has_guard_pages)]
    pub use_guard_pages: bool,
    #[cfg(sf_has_guard_pages)]
    pub use_stack_guard_pages: bool,
}

/// Result of lowering prepared SSA-IR into MachineIR plus backend-facing ABI
/// metadata derived from the shared frame plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredMachineModule {
    pub module: MachineModule,
    pub abi: MachineModuleAbi,
}

#[allow(dead_code)]
pub(crate) fn lower_module(input: LowerModuleInput) -> Result<LoweredMachineModule, WasmError> {
    lower_module_with_table_dispatch_modes(input, &[])
}

pub(crate) fn lower_module_with_table_dispatch_modes(
    input: LowerModuleInput,
    table_dispatch_modes: &[TableDispatchMode],
) -> Result<LoweredMachineModule, WasmError> {
    let max_regfile = MachineRegFile::new(input.backend)?;

    let function_count = input
        .functions
        .iter()
        .map(|function| function.id.0 as usize)
        .max()
        .map(|max| max + 1)
        .unwrap_or(0);
    let mut functions = collections::vec![None; function_count];
    let mut is_local_func = collections::vec![false; function_count];
    let mut function_abis = (0..function_count)
        .map(|index| MachineFunctionAbi {
            id: MachineFuncId(index as u32),
            ..MachineFunctionAbi::default()
        })
        .collect::<collections::Vec<_>>();
    let mut const_pool = ConstPoolBuilder::new();
    for function in &input.functions {
        let borrowed = function.borrowed();
        validate_program(borrowed.ssa)?;
        is_local_func[borrowed.id.0 as usize] = true;
        function_abis[borrowed.id.0 as usize] = lower_function_runtime(borrowed, input.backend)?;
    }
    #[cfg(sf_has_guard_pages)]
    let guard_pages = input.use_guard_pages;
    #[cfg(sf_has_guard_pages)]
    let stack_guard_pages = input.use_stack_guard_pages;
    for mut function in input.functions {
        let mir_lower_function_phase = phase_span_with_function("mir_lower", Some(function.id.0));
        let function_id = function.id.0 as usize;
        functions[function_id] = Some(lower_function(
            &mut function,
            input.backend,
            &max_regfile,
            &function_abis,
            &is_local_func,
            table_dispatch_modes,
            &mut const_pool,
            #[cfg(sf_has_guard_pages)]
            guard_pages,
            #[cfg(sf_has_guard_pages)]
            stack_guard_pages,
        )?);
        drop(mir_lower_function_phase);
    }
    let abi = MachineModuleAbi {
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

#[allow(dead_code)]
pub(crate) fn lower_single_function(
    backend: BackendConfig,
    function: LowerFunctionInput,
    runtime: &mut [MachineFunctionAbi],
    is_local_func: &[bool],
    const_pool: &mut ConstPoolBuilder,
    #[cfg(sf_has_guard_pages)] guard_pages: bool,
    #[cfg(sf_has_guard_pages)] stack_guard_pages: bool,
) -> Result<(MachineFunction, MachineFunctionAbi), WasmError> {
    lower_single_function_with_table_dispatch_modes(
        backend,
        function,
        runtime,
        is_local_func,
        &[],
        const_pool,
        #[cfg(sf_has_guard_pages)]
        guard_pages,
        #[cfg(sf_has_guard_pages)]
        stack_guard_pages,
    )
}

pub(crate) fn lower_single_function_with_table_dispatch_modes(
    backend: BackendConfig,
    function: LowerFunctionInput,
    runtime: &mut [MachineFunctionAbi],
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    const_pool: &mut ConstPoolBuilder,
    #[cfg(sf_has_guard_pages)] guard_pages: bool,
    #[cfg(sf_has_guard_pages)] stack_guard_pages: bool,
) -> Result<(MachineFunction, MachineFunctionAbi), WasmError> {
    let mut function = function;
    let max_regfile = MachineRegFile::new(backend)?;
    let abi = {
        let borrowed = function.borrowed();
        validate_program(borrowed.ssa)?;
        let abi = lower_function_runtime(borrowed, backend)?;
        let runtime_slot = runtime
            .get_mut(borrowed.id.0 as usize)
            .ok_or_else(|| WasmError::internal("runtime metadata slot is out of range"))?;
        *runtime_slot = abi.clone();
        abi
    };
    let machine = lower_function(
        &mut function,
        backend,
        &max_regfile,
        runtime,
        is_local_func,
        table_dispatch_modes,
        const_pool,
        #[cfg(sf_has_guard_pages)]
        guard_pages,
        #[cfg(sf_has_guard_pages)]
        stack_guard_pages,
    )?;
    Ok((machine, abi))
}

fn lower_function_runtime(
    input: BorrowedLowerFunctionInput<'_>,
    config: BackendConfig,
) -> Result<MachineFunctionAbi, WasmError> {
    // Under the new local-call ABI, the dead "call_link" half of
    // `call_scratch` is gone — `FrameLayoutPlan::call_scratch` now only
    // carries helper-scratch slots (the live half). Expose the whole region
    // as the `helper_scratch`.
    let helper_scratch = input.frame.call_scratch.map(frame_span_region);
    let mut return_results = derive_return_results(input.ssa)?;
    // Fallback: if no Return terminators exist (all paths trap/unreachable),
    // use the declared result count from the type signature so that callers
    // see the correct contract.
    if return_results.is_none() && input.result_count > 0 {
        let result_span = input.frame.return_results(input.result_count);
        return_results = Some(frame_span_region(result_span));
    }
    let return_abi = derive_return_abi(&input.ssa.result_types, return_results, config);

    let init_locals = input
        .ssa
        .cell_info
        .iter()
        .enumerate()
        .filter(|(_, info)| !info.is_param && info.reads_before_write)
        .map(|(i, _)| i as u16)
        .collect();

    Ok(MachineFunctionAbi {
        id: input.id,
        param_locs: derive_param_locs(input.ssa, config),
        frame_prefix_slots: input.frame.frame_prefix_size,
        total_frame_slots: input.frame.total_slots(),
        helper_scratch,
        return_results,
        return_abi,
        init_locals,
    })
}

fn derive_param_locs(
    program: &SsaProgram,
    config: BackendConfig,
) -> collections::Vec<MachineParamLoc> {
    let param_count = program
        .cell_info
        .iter()
        .take_while(|info| info.is_param)
        .count()
        .min(u16::MAX as usize) as u16;
    let param_types = (0..param_count)
        .map(|param_index| {
            program
                .cell_types
                .get(param_index as usize)
                .copied()
                .unwrap_or(ValueType::I64)
        })
        .collect::<collections::Vec<_>>();
    derive_param_locs_from_types(&param_types, config)
}

pub(crate) fn derive_param_locs_from_types(
    param_types: &[ValueType],
    config: BackendConfig,
) -> collections::Vec<MachineParamLoc> {
    let param_count = param_types.len().min(u16::MAX as usize) as u16;
    let mut locs = (0..param_count)
        .map(|param_index| MachineParamLoc::Frame {
            param_index,
            slot: FrameSlot(param_index),
        })
        .collect::<collections::Vec<_>>();
    let budget = internal_call_arg_budget(config);
    let mut selected = collections::Vec::new();
    let mut gp_used = 0u8;
    let mut fp_used = 0u8;
    for param_index in (0..param_count).rev() {
        let ty = param_types
            .get(param_index as usize)
            .copied()
            .unwrap_or(ValueType::I64);
        let storage = value_type_storage_type(ty);
        if matches!(storage, MachineStorageType::V128) {
            break;
        }
        if storage.is_fp() {
            if fp_used >= budget.fp_units {
                break;
            }
            selected.push((param_index, storage));
            fp_used = fp_used.saturating_add(1);
            continue;
        }
        let units = if config.gp_unit_bytes == 4 && matches!(storage, MachineStorageType::GpI64) {
            2
        } else {
            1
        };
        if gp_used.saturating_add(units) > budget.gp_units {
            break;
        }
        selected.push((param_index, storage));
        gp_used = gp_used.saturating_add(units);
    }
    selected.reverse();
    let mut gp_lane = 0u8;
    let mut fp_lane = 0u8;
    for (param_index, storage) in selected {
        let loc = if storage.is_fp() {
            let lane = fp_lane;
            fp_lane = fp_lane.saturating_add(1);
            MachineParamLoc::FpArg {
                param_index,
                lane,
                ty: storage,
            }
        } else if config.gp_unit_bytes == 4 && matches!(storage, MachineStorageType::GpI64) {
            let lo_lane = gp_lane;
            let hi_lane = gp_lane.saturating_add(1);
            gp_lane = gp_lane.saturating_add(2);
            MachineParamLoc::GpArgPair {
                param_index,
                lo_lane,
                hi_lane,
            }
        } else {
            let lane = gp_lane;
            gp_lane = gp_lane.saturating_add(1);
            MachineParamLoc::GpArg {
                param_index,
                lane,
                ty: storage,
            }
        };
        if let Some(slot) = locs.get_mut(param_index as usize) {
            *slot = loc;
        }
    }
    locs
}

fn internal_call_arg_budget(
    config: BackendConfig,
) -> crate::vm::machine::machine_ir::InternalCallArgBudget {
    crate::vm::machine::machine_ir::InternalCallArgBudget {
        gp_units: config.gp_arg_lanes,
        fp_units: config.fp_arg_lanes,
    }
}

fn lower_function(
    input: &mut LowerFunctionInput,
    config: BackendConfig,
    regfile: &MachineRegFile,
    runtime: &[MachineFunctionAbi],
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    const_pool: &mut ConstPoolBuilder,
    #[cfg(sf_has_guard_pages)] guard_pages: bool,
    #[cfg(sf_has_guard_pages)] stack_guard_pages: bool,
) -> Result<MachineFunction, WasmError> {
    let gp_reg_width = config.gp_unit_bytes;
    let original_block_count = input.ssa.blocks.len();
    let mut original_blocks = OriginalBlocks::new(original_block_count);
    let mut extra_blocks = collections::Vec::new();
    let mut extra_block_ids = ExtraBlockAllocator::new(original_block_count as u32);
    let i64_ops: &'static dyn I64Lowering = if gp_reg_width == 4 {
        &Gp32Lowering
    } else {
        &Gp64Lowering
    };
    let explicit_cache = explicit_cached_cells(&input.ssa)?;
    let current_abi = runtime
        .get(input.id.0 as usize)
        .ok_or_else(|| WasmError::internal("runtime metadata missing for current function"))?;
    // The machine owns the physical lane layout again; the plan hands it only the
    // context-free preference (per local slot) + the per-entry requirement rows.
    let entry_cache_params = compute_block_entry_cache_params(
        regfile,
        &input.ssa,
        &explicit_cache,
        gp_reg_width,
        is_local_func,
        table_dispatch_modes,
        &input.ssa.preferred_preserved,
    )?;
    let call_preserved_cache_candidates =
        preferred_preserved_for_cached(&explicit_cache, &input.ssa.preferred_preserved);
    let has_direct_self_tail_call = input.ssa.blocks.iter().any(|block| {
        matches!(
            block.terminator,
            SsaTerminator::TailCallDirect { callee, .. } if callee == input.id.0
        )
    });
    let entry_has_no_ssa_params = input
        .ssa
        .blocks
        .get(input.ssa.entry.as_usize())
        .map(|block| block.params.is_empty())
        .unwrap_or(false);
    let self_tail_entry_params = (has_direct_self_tail_call && entry_has_no_ssa_params)
        .then(|| derive_self_tail_entry_params(regfile, current_abi, &explicit_cache))
        .flatten();
    let block_entry_cache_dirty = compute_block_entry_cache_dirty(
        &input.ssa,
        &explicit_cache,
        current_abi,
        &entry_cache_params,
        is_local_func,
        &call_preserved_cache_candidates,
    );
    release_cache_planning_only_ssa_storage(&mut input.ssa);
    for block_index in 0..original_block_count {
        let block = take_block_for_lowering(&mut input.ssa.blocks[block_index]);
        let target = block.id;
        let mut lower = BlockLowerContext::new(
            regfile,
            &input.ssa,
            &explicit_cache,
            &entry_cache_params,
            &block,
            current_abi,
            runtime,
            gp_reg_width,
            i64_ops,
            target == input.ssa.entry,
            block_entry_cache_dirty
                .get(target.as_usize())
                .map(|dirty| dirty.as_slice()),
            Some(&call_preserved_cache_candidates),
            #[cfg(sf_has_guard_pages)]
            guard_pages,
            #[cfg(sf_has_guard_pages)]
            stack_guard_pages,
        )?;
        let mut current_block = MachineBlockId(block.id.as_u32());
        let mut current_params = collections::Vec::new();
        for (regs, value) in lower
            .machine_params()
            .iter()
            .copied()
            .zip(block.params.iter().copied())
        {
            append_machine_block_params_for_value(
                &mut current_params,
                regs,
                program_value_storage_type(&input.ssa, value),
                MachineRegOwner::LinearValue,
            );
        }
        if target != input.ssa.entry {
            append_entry_cache_params(
                &mut current_params,
                lower.entry_cache_params(),
                &explicit_cache,
            );
        } else {
            if let Some(params) = self_tail_entry_params.as_deref() {
                append_entry_cache_params(&mut current_params, params, &explicit_cache);
            }
            // Entry block: zero-init non-param locals that may be read before
            // being written. The caller no longer pre-zeros the callee frame;
            // each function is responsible for satisfying the wasm zero-init
            // contract for its own locals at function entry.
            let init_locals = runtime
                .get(input.id.0 as usize)
                .map(|abi| abi.init_locals.clone())
                .unwrap_or_default();
            lower.emit_zero_init_locals(&init_locals)?;
        }

        for inst_idx in 0..block.ops.len() {
            let inst = block.ops[inst_idx];
            match block.view(inst_idx, &input.ssa) {
                SsaInstView::Value { op, result, args } => {
                    // Ordinary scalar leaves have at most three operands. Keep
                    // those on the stack; only wide GC constructors need the
                    // overflow Vec path.
                    let mut inline_args = [SsaOperand::NONE; 3];
                    let overflow_args = (args.len() > inline_args.len()).then(|| args.to_vec());
                    let args_slice = if let Some(overflow) = overflow_args.as_deref() {
                        overflow
                    } else {
                        for (index, arg) in args.iter().enumerate() {
                            inline_args[index] = arg;
                        }
                        &inline_args[..args.len()]
                    };
                    let result_storage = [result];
                    let results_slice = if result.is_some() {
                        &result_storage[..]
                    } else {
                        &[]
                    };
                    lower.apply_sink_premap(args_slice, results_slice)?;
                    if let Some(lowered) = lower.lower_leaf_special(
                        op,
                        args_slice,
                        results_slice,
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
                                    params: collections::Vec::new(),
                                    ops: collections::Vec::new(),
                                    terminator: MachineTerminator::Trap { kind: trap_kind },
                                });
                                current_block = continuation;
                                current_params = continuation_params;
                                lower.emit_machine_ops(continuation_ops);
                            }
                        }
                        continue;
                    }
                    // The caller already decoded this Value instruction. Avoid
                    // lower_inst decoding it again, cloning the primitive, and
                    // rebuilding operand/result Vecs.
                    lower.lower_leaf(op, args_slice, results_slice)?;
                    lower.release_dead_values()?;
                }
                SsaInstView::Call(call) => match call {
                    // `Call` preserves Wasm semantics above MachineIR:
                    // the callee is compile-time known, but target kind
                    // (compiled local vs runtime-dispatch path) is decided here.
                    //
                    // Local targets become MIR terminators because control
                    // transfers into another compiled MachineIR function and
                    // resumes at an explicit continuation block.
                    SsaCallOp::CallDirect {
                        callee,
                        args,
                        results,
                        ..
                    } if is_local_func
                        .get(*callee as usize)
                        .copied()
                        .unwrap_or(false) =>
                    {
                        let live_result = inst.result.is_some().then_some(inst.result);
                        let continuation = extra_block_ids.alloc();
                        let (terminator, continuation_params) = lower.lower_call_internal(
                            *callee,
                            args,
                            *results,
                            live_result,
                            continuation,
                        )?;
                        push_lowered_block(
                            current_block,
                            &mut original_blocks,
                            &mut extra_blocks,
                            current_params,
                            lower.take_ops(),
                            terminator,
                        )?;
                        current_block = continuation;
                        current_params = continuation_params;
                        // `lower_call_internal` has already split cached-local
                        // state: volatile caches were saved/dropped, while
                        // non-ref caches resident in preserved regs are carried
                        // as identity edge params into this continuation.
                    }
                    // Runtime-dispatch targets stay in the current block as
                    // inline runtime-call instructions. They cross the runtime
                    // ABI boundary but do not produce a MachineIR CFG edge.
                    SsaCallOp::CallDirect {
                        callee,
                        args,
                        results,
                        ..
                    } => {
                        let live_result = inst.result.is_some().then_some(inst.result);
                        lower.lower_call_runtime(
                            *callee,
                            args,
                            *results,
                            live_result,
                            const_pool,
                        )?;
                    }
                    SsaCallOp::CallRef {
                        type_idx,
                        callee_ref,
                        args,
                        results,
                        result_types,
                    } => {
                        let type_idx = *type_idx;
                        let param_locs = derive_param_locs_from_types(&args.param_types, config);
                        let carried_args =
                            capture_indirect_carried_args(&lower, args, &param_locs)?;
                        publish_indirect_call_args_to_frame(&mut lower, args, &carried_args)?;
                        let ref_slot = lower.publish_call_operand_to_frame(callee_ref)?;
                        lower.publish_register_params_to_frame()?;
                        let args = args.frame_span();
                        let results = *results;
                        let live_result = inst.result.is_some().then_some(inst.result);
                        let (continuation, continuation_params) = lower_indirect_dispatch_cluster(
                            &mut lower,
                            IndirectCallSource::Ref { ref_slot },
                            type_idx,
                            IndirectTransfer::Call {
                                args,
                                results,
                                live_result,
                            },
                            result_types,
                            config,
                            param_locs,
                            carried_args,
                            table_dispatch_modes,
                            current_block,
                            current_params,
                            &mut original_blocks,
                            &mut extra_blocks,
                            &mut extra_block_ids,
                            const_pool,
                        )?
                        .expect("non-tail indirect call should return a continuation block");
                        current_block = continuation;
                        current_params = continuation_params;
                        lower.begin_continuation_block_selective()?;
                    }
                    SsaCallOp::CallIndirect { .. } => {
                        let SsaCallOp::CallIndirect {
                            type_idx,
                            table_idx,
                            index,
                            args,
                            results,
                            result_types,
                        } = call
                        else {
                            unreachable!("matched call_indirect");
                        };
                        let type_idx = *type_idx;
                        let table_idx = *table_idx;
                        let param_locs = derive_param_locs_from_types(&args.param_types, config);
                        let carried_args =
                            capture_indirect_carried_args(&lower, args, &param_locs)?;
                        publish_indirect_call_args_to_frame(&mut lower, args, &carried_args)?;
                        let fixed_local_table =
                            fixed_local_table_len(table_dispatch_modes, table_idx).is_some();
                        let fixed_fast_table =
                            fixed_local_table && !lower.use_explicit_stack_prechecks();
                        let index = if fixed_fast_table {
                            prepare_call_indirect_index_in_lane(&mut lower, index)?
                        } else {
                            IndirectCallIndex::Frame(lower.publish_call_operand_to_frame(index)?)
                        };
                        lower.publish_register_params_to_frame()?;
                        let args = args.frame_span();
                        let results = *results;
                        let live_result = inst.result.is_some().then_some(inst.result);
                        let (continuation, continuation_params) = lower_indirect_dispatch_cluster(
                            &mut lower,
                            IndirectCallSource::Table { table_idx, index },
                            type_idx,
                            IndirectTransfer::Call {
                                args,
                                results,
                                live_result,
                            },
                            result_types,
                            config,
                            param_locs,
                            carried_args,
                            table_dispatch_modes,
                            current_block,
                            current_params,
                            &mut original_blocks,
                            &mut extra_blocks,
                            &mut extra_block_ids,
                            const_pool,
                        )?
                        .expect("non-tail indirect call should return a continuation block");
                        current_block = continuation;
                        current_params = continuation_params;
                        if !fixed_local_table {
                            lower.begin_continuation_block_selective()?;
                        }
                    }
                },
                _ => lower.lower_inst(&inst)?,
            }
        }

        if target == input.ssa.entry && entry_terminator_may_leave_block(&block.terminator) {
            lower.publish_register_params_to_frame()?;
        }

        match &block.terminator {
            SsaTerminator::TailCallDirect {
                callee,
                args,
                return_results,
            } => {
                let self_tail_backedge = (*callee == input.id.0).then(|| {
                    try_lower_self_tail_backedge(
                        &lower,
                        args,
                        input.ssa.entry,
                        self_tail_entry_params.as_deref().unwrap_or(&[]),
                        &explicit_cache,
                    )
                });
                let terminator = if let Some(Some(edge)) = self_tail_backedge {
                    MachineTerminator::Jump(edge)
                } else if is_local_func
                    .get(*callee as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    lower.lower_tail_call_internal(*callee, args, *return_results)?
                } else {
                    lower.lower_call_runtime(
                        *callee,
                        args,
                        runtime_tail_results_span(*return_results),
                        None,
                        const_pool,
                    )?;
                    return_terminator_from_frame(&current_abi.return_abi)
                };
                push_lowered_block(
                    current_block,
                    &mut original_blocks,
                    &mut extra_blocks,
                    current_params,
                    lower.take_ops(),
                    terminator,
                )?;
                continue;
            }
            SsaTerminator::TailCallIndirect {
                type_idx,
                table_idx,
                index,
                args,
                return_results,
            } => {
                let param_locs = derive_param_locs_from_types(&args.param_types, config);
                lower.publish_call_args_to_frame(args)?;
                let type_idx = *type_idx;
                let table_idx = *table_idx;
                let fixed_local_table =
                    fixed_local_table_len(table_dispatch_modes, table_idx).is_some();
                let fixed_fast_table = fixed_local_table && !lower.use_explicit_stack_prechecks();
                let index = if fixed_fast_table {
                    prepare_call_indirect_index_in_lane(&mut lower, index)?
                } else {
                    IndirectCallIndex::Frame(lower.publish_call_operand_to_frame(index)?)
                };
                lower.publish_register_params_to_frame()?;
                lower.ensure_no_live_values(
                    "prepared SSA-IR tail call_indirect reached native lowering with live linear SSA values; values must be published before the tail transfer",
                )?;
                let args = args.frame_span();
                let return_results = *return_results;
                let _ = lower_indirect_dispatch_cluster(
                    &mut lower,
                    IndirectCallSource::Table { table_idx, index },
                    type_idx,
                    IndirectTransfer::Tail {
                        args,
                        return_results,
                    },
                    &input.ssa.result_types,
                    config,
                    param_locs,
                    IndirectCarriedArgs::default(),
                    table_dispatch_modes,
                    current_block,
                    current_params,
                    &mut original_blocks,
                    &mut extra_blocks,
                    &mut extra_block_ids,
                    const_pool,
                )?;
                continue;
            }
            SsaTerminator::TailCallRef {
                type_idx,
                callee_ref,
                args,
                return_results,
            } => {
                let param_locs = derive_param_locs_from_types(&args.param_types, config);
                lower.publish_call_args_to_frame(args)?;
                let ref_slot = lower.publish_call_operand_to_frame(callee_ref)?;
                lower.publish_register_params_to_frame()?;
                lower.ensure_no_live_values(
                    "prepared SSA-IR tail call_ref reached native lowering with live linear SSA values; values must be published before the tail transfer",
                )?;
                let type_idx = *type_idx;
                let args = args.frame_span();
                let return_results = *return_results;
                let _ = lower_indirect_dispatch_cluster(
                    &mut lower,
                    IndirectCallSource::Ref { ref_slot },
                    type_idx,
                    IndirectTransfer::Tail {
                        args,
                        return_results,
                    },
                    &input.ssa.result_types,
                    config,
                    param_locs,
                    IndirectCarriedArgs::default(),
                    table_dispatch_modes,
                    current_block,
                    current_params,
                    &mut original_blocks,
                    &mut extra_blocks,
                    &mut extra_block_ids,
                    const_pool,
                )?;
                continue;
            }
            SsaTerminator::EhThrow { tag_idx, args } => {
                let tag_idx = *tag_idx;
                let args = *args;
                lower.ensure_no_live_values(
                    "prepared SSA-IR throw reached native lowering with live linear SSA values; payload must be published before the throw",
                )?;
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::EhThrow { tag_idx, args },
                });
                let terminator = MachineTerminator::Trap {
                    kind: MachineTrapKind::Unreachable,
                };
                push_lowered_block(
                    current_block,
                    &mut original_blocks,
                    &mut extra_blocks,
                    current_params,
                    lower.take_ops(),
                    terminator,
                )?;
                continue;
            }
            SsaTerminator::EhThrowRef { exnref_slot } => {
                let exnref_slot = *exnref_slot;
                lower.ensure_no_live_values(
                    "prepared SSA-IR throw_ref reached native lowering with live linear SSA values; exnref must be published before the throw",
                )?;
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::EhThrowRef { exnref_slot },
                });
                let terminator = MachineTerminator::Trap {
                    kind: MachineTrapKind::Unreachable,
                };
                push_lowered_block(
                    current_block,
                    &mut original_blocks,
                    &mut extra_blocks,
                    current_params,
                    lower.take_ops(),
                    terminator,
                )?;
                continue;
            }
            _ => {}
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

    release_prepared_ssa_storage(&mut input.ssa);
    drop(block_entry_cache_dirty);
    drop(entry_cache_params);
    drop(explicit_cache);

    let mut blocks = original_blocks.finish()?;
    blocks.reserve_exact(extra_blocks.len());
    blocks.extend(extra_blocks);

    let entry = MachineBlockId(input.ssa.entry.as_u32());

    let program = MachineProgram {
        entry,
        fp_reg_init_widths: fp_reg_init_widths(&regfile)?,
        blocks,
    };
    program.validate(config)?;

    Ok(MachineFunction {
        id: input.id,
        program,
        // Final optimization is the single authority for this derived ABI
        // metadata because peepholes may add or remove register definitions.
        preserved_clobbers: collections::Vec::new(),
    })
}

/// Turns a register-only direct self tail call into a CFG backedge.
///
/// The ordinary tail-call path must publish arguments and cached parameters to
/// the Wasm frame, restore the native body frame, jump through the function
/// entry, and reload the parameters. A self call whose complete parameter set
/// is already live in registers can instead use the normal edge parallel-move
/// machinery and branch straight to the function's entry block.
fn try_lower_self_tail_backedge(
    lower: &BlockLowerContext<'_>,
    args: &SsaCallArgs,
    entry: SsaTarget,
    entry_cache_params: &[EntryCacheParam],
    cached_cells: &[CachedCell],
) -> Option<MachineEdge> {
    if args.stack_prefix_count != 0
        || args.live_suffix.len() != usize::from(args.total_params)
        || entry_cache_params.len() != usize::from(args.total_params)
    {
        return None;
    }

    let mut seen = collections::vec![false; usize::from(args.total_params)];
    let mut edge_args = collections::Vec::with_capacity(entry_cache_params.len());
    for entry_param in entry_cache_params {
        if !entry_param.needs_value {
            return None;
        }
        let cached = cached_cells.get(usize::from(entry_param.cached_index))?;
        if !cached.info.is_param {
            return None;
        }
        let param_index = usize::try_from(cached.cell.0).ok()?;
        if param_index >= seen.len() || seen[param_index] {
            return None;
        }
        let live = args
            .live_suffix
            .iter()
            .find(|arg| usize::from(arg.param_index) == param_index)?;
        if program_value_storage_type(lower.program(), live.value) != cached.ty {
            return None;
        }
        let (lo, hi) = lower.try_value_regs(live.value)?;
        if hi.is_some() != entry_param.regs.hi.is_some() {
            return None;
        }
        edge_args.push(MachineValue::Reg(lo));
        if let Some(hi) = hi {
            edge_args.push(MachineValue::Reg(hi));
        }
        seen[param_index] = true;
    }
    if seen.iter().any(|seen| !seen) {
        return None;
    }

    Some(MachineEdge {
        target: MachineBlockId(entry.as_u32()),
        args: edge_args,
    })
}

fn derive_self_tail_entry_params(
    regfile: &MachineRegFile,
    abi: &MachineFunctionAbi,
    cached_cells: &[CachedCell],
) -> Option<collections::Vec<EntryCacheParam>> {
    // A fresh tail-called activation may discard every non-parameter local,
    // and reference parameters must remain published in the canonical frame
    // for GC root discovery. Keep this first version to scalar parameters
    // whose complete live state can safely travel on the CFG edge.
    if cached_cells
        .iter()
        .any(|cached| !cached.info.is_param || cached.value_ty.is_ref())
    {
        return None;
    }
    let mut params = collections::Vec::with_capacity(abi.param_locs.len());
    for loc in &abi.param_locs {
        let (param_index, regs) = match *loc {
            MachineParamLoc::Frame { .. } => return None,
            MachineParamLoc::GpArg {
                param_index, lane, ..
            } => (
                param_index,
                ValueRegs {
                    lo: regfile.ordered_gp_allocatable(lane as usize)?,
                    hi: None,
                },
            ),
            MachineParamLoc::GpArgPair {
                param_index,
                lo_lane,
                hi_lane,
            } => (
                param_index,
                ValueRegs {
                    lo: regfile.ordered_gp_allocatable(lo_lane as usize)?,
                    hi: Some(regfile.ordered_gp_allocatable(hi_lane as usize)?),
                },
            ),
            MachineParamLoc::FpArg {
                param_index, lane, ..
            } => (
                param_index,
                ValueRegs {
                    lo: regfile.fp_dynamic(lane as usize)?,
                    hi: None,
                },
            ),
        };
        let cached_index = cached_cells
            .iter()
            .position(|cached| cached.info.is_param && cached.cell.0 == param_index)?;
        params.push(EntryCacheParam {
            cached_index: u16::try_from(cached_index).ok()?,
            regs,
            needs_value: true,
        });
    }
    Some(params)
}

fn release_cache_planning_only_ssa_storage(ssa: &mut SsaProgram) {
    drop(core::mem::take(&mut ssa.cell_types));
    drop(core::mem::take(&mut ssa.cell_info));
    drop(core::mem::take(&mut ssa.block_entry_cached_cells));
}

fn release_prepared_ssa_storage(ssa: &mut SsaProgram) {
    drop(core::mem::take(&mut ssa.blocks));
    drop(core::mem::take(&mut ssa.cell_types));
    drop(core::mem::take(&mut ssa.cell_info));
    drop(core::mem::take(&mut ssa.block_entry_cached_cells));
    drop(core::mem::take(&mut ssa.value_types));
    drop(core::mem::take(&mut ssa.value_sink_cell));
    drop(core::mem::take(&mut ssa.const_pool));
    drop(core::mem::take(&mut ssa.primitive_pool));
    drop(core::mem::take(&mut ssa.call_ops));
}

fn take_block_for_lowering(block: &mut SsaBlock) -> SsaBlock {
    SsaBlock {
        id: block.id,
        params: block.params.clone(),
        ops: core::mem::take(&mut block.ops),
        extra_args: core::mem::take(&mut block.extra_args),
        terminator: core::mem::replace(&mut block.terminator, SsaTerminator::TrapUnreachable),
    }
}

fn stub_machine_function(id: MachineFuncId) -> MachineFunction {
    MachineFunction {
        id,
        program: MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Trap {
                    kind: MachineTrapKind::Unreachable,
                },
            }],
        },
        preserved_clobbers: collections::Vec::new(),
    }
}

#[inline]
pub(super) fn slot_offset_bytes(slot: FrameSlot) -> Result<i32, WasmError> {
    let bytes = i32::from(slot.0)
        .checked_mul(8)
        .ok_or_else(|| WasmError::internal("frame slot byte offset overflow"))?;
    Ok(bytes)
}

fn derive_return_results(program: &SsaProgram) -> Result<Option<MachineFrameRegion>, WasmError> {
    let mut derived: Option<Option<MachineFrameRegion>> = None;
    for block in &program.blocks {
        let results = match &block.terminator {
            SsaTerminator::Return { results } => *results,
            SsaTerminator::ReturnScalar { results, .. } => Some(*results),
            SsaTerminator::TailCallDirect { return_results, .. }
            | SsaTerminator::TailCallIndirect { return_results, .. }
            | SsaTerminator::TailCallRef { return_results, .. } => *return_results,
            _ => continue,
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

pub(crate) fn derive_return_abi(
    result_types: &[ValueType],
    return_results: Option<MachineFrameRegion>,
    config: BackendConfig,
) -> MachineReturnAbi {
    let Some(results) = return_results else {
        return MachineReturnAbi::None;
    };
    if !config.scalar_return_lanes || result_types.len() != 1 || results.slots != 1 {
        return MachineReturnAbi::FrameFallback { results };
    }
    match value_type_storage_type(result_types[0]) {
        MachineStorageType::GpWord => MachineReturnAbi::ScalarGp {
            ty: MachineStorageType::GpWord,
        },
        MachineStorageType::GpI64 if config.gp_unit_bytes == 4 => MachineReturnAbi::ScalarGpPair,
        MachineStorageType::GpI64 => MachineReturnAbi::ScalarGp {
            ty: MachineStorageType::GpI64,
        },
        MachineStorageType::Fp32 => MachineReturnAbi::ScalarFp {
            ty: MachineStorageType::Fp32,
        },
        MachineStorageType::Fp64 => MachineReturnAbi::ScalarFp {
            ty: MachineStorageType::Fp64,
        },
        MachineStorageType::V128 => MachineReturnAbi::FrameFallback { results },
    }
}

fn return_terminator_from_frame(return_abi: &MachineReturnAbi) -> MachineTerminator {
    match return_abi {
        MachineReturnAbi::ScalarGp { ty } => MachineTerminator::ReturnScalar {
            value: MachineReturnValue::ScalarGp {
                src: MachineResultSrc::FrameSlot(FrameSlot(0)),
                ty: *ty,
            },
        },
        MachineReturnAbi::ScalarGpPair => MachineTerminator::ReturnScalar {
            value: MachineReturnValue::ScalarGpPair {
                lo: MachineResultSrc::FrameSlotOffset {
                    slot: FrameSlot(0),
                    byte_offset: 0,
                },
                hi: MachineResultSrc::FrameSlotOffset {
                    slot: FrameSlot(0),
                    byte_offset: 4,
                },
            },
        },
        MachineReturnAbi::ScalarFp { ty } => MachineTerminator::ReturnScalar {
            value: MachineReturnValue::ScalarFp {
                src: MachineResultSrc::FrameSlot(FrameSlot(0)),
                ty: *ty,
            },
        },
        MachineReturnAbi::None | MachineReturnAbi::FrameFallback { .. } => {
            MachineTerminator::Return
        }
    }
}

#[inline]
fn fixed_callee_return_results(slots: u16) -> MachineFrameRegion {
    MachineFrameRegion {
        base_slot: 0,
        slots,
    }
}

#[inline]
fn runtime_tail_results_span(results: Option<FrameSpan>) -> FrameSpan {
    results.unwrap_or_else(|| FrameSpan::new(FrameSlot(0), 0))
}

fn entry_terminator_may_leave_block(term: &SsaTerminator) -> bool {
    matches!(
        term,
        SsaTerminator::Goto(_)
            | SsaTerminator::Branch { .. }
            | SsaTerminator::BrTable { .. }
            | SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. }
            | SsaTerminator::EhThrow { .. }
            | SsaTerminator::EhThrowRef { .. }
    )
}

#[derive(Clone, Copy, Debug)]
enum IndirectCallSource {
    Table {
        table_idx: u32,
        index: IndirectCallIndex,
    },
    Ref {
        ref_slot: FrameSlot,
    },
}

impl IndirectCallSource {
    #[inline]
    fn frame_slot(self) -> FrameSlot {
        match self {
            IndirectCallSource::Table { index, .. } => index
                .frame_slot()
                .expect("generic indirect table dispatch requires a published frame slot"),
            IndirectCallSource::Ref { ref_slot } => ref_slot,
        }
    }

    #[inline]
    fn build_runtime_call_meta(
        self,
        lower: &BlockLowerContext<'_>,
        type_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
        const_pool: &mut ConstPoolBuilder,
    ) -> MachineConstId {
        match self {
            IndirectCallSource::Table { index, .. } => lower.build_call_runtime_meta_indirect(
                type_idx,
                index
                    .frame_slot()
                    .expect("runtime indirect table dispatch requires a published frame slot"),
                args,
                results,
                const_pool,
            ),
            IndirectCallSource::Ref { ref_slot } => {
                lower.build_call_runtime_meta_ref(type_idx, ref_slot, args, results, const_pool)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum IndirectCallIndex {
    Frame(FrameSlot),
    DispatchLane,
}

impl IndirectCallIndex {
    #[inline]
    fn frame_slot(self) -> Option<FrameSlot> {
        match self {
            IndirectCallIndex::Frame(slot) => Some(slot),
            IndirectCallIndex::DispatchLane => None,
        }
    }

    #[inline]
    fn is_dispatch_lane(self) -> bool {
        matches!(self, IndirectCallIndex::DispatchLane)
    }
}

#[inline]
fn fixed_local_table_len(modes: &[TableDispatchMode], table_idx: u32) -> Option<u32> {
    match modes
        .get(table_idx as usize)
        .copied()
        .unwrap_or(TableDispatchMode::Generic)
    {
        TableDispatchMode::FixedLocalOnly { len } => Some(len),
        TableDispatchMode::Generic => None,
    }
}

#[derive(Clone, Copy, Debug)]
enum IndirectTransfer {
    Call {
        args: FrameSpan,
        results: FrameSpan,
        live_result: Option<SsaValue>,
    },
    Tail {
        args: FrameSpan,
        return_results: Option<FrameSpan>,
    },
}

impl IndirectTransfer {
    #[inline]
    fn source_args(self) -> FrameSpan {
        match self {
            IndirectTransfer::Call { args, .. } | IndirectTransfer::Tail { args, .. } => args,
        }
    }

    #[inline]
    fn callee_args(self) -> FrameSpan {
        match self {
            IndirectTransfer::Call { args, .. } => args,
            IndirectTransfer::Tail { args, .. } => FrameSpan::new(FrameSlot(0), args.count),
        }
    }

    #[inline]
    fn runtime_results(self) -> FrameSpan {
        match self {
            IndirectTransfer::Call { results, .. } => results,
            IndirectTransfer::Tail { return_results, .. } => {
                runtime_tail_results_span(return_results)
            }
        }
    }

    #[inline]
    fn is_tail(self) -> bool {
        matches!(self, IndirectTransfer::Tail { .. })
    }
}

#[derive(Clone, Debug, Default)]
struct IndirectCarriedArgs {
    params: collections::Vec<MachineBlockParam>,
    entry_args: collections::Vec<MachineValue>,
    local_call_args: Option<MachineCallArgs>,
    runtime_arg_stores: collections::Vec<IndirectRuntimeArgStore>,
    consumed_values: collections::Vec<SsaValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndirectRuntimeArgStore {
    slot: FrameSlot,
    ty: MachineStorageType,
    lo: MachineReg,
    hi: Option<MachineReg>,
}

impl IndirectCarriedArgs {
    #[inline]
    fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    fn param_values(&self) -> collections::Vec<MachineValue> {
        self.params
            .iter()
            .map(|param| MachineValue::Reg(param.reg))
            .collect()
    }

    fn local_prepare_params(
        &self,
        local_call_target_param: MachineReg,
    ) -> collections::Vec<MachineBlockParam> {
        let mut params = collections::Vec::with_capacity(self.params.len() + 1);
        params.push(MachineBlockParam::gp_word(local_call_target_param));
        params.extend(self.params.iter().copied());
        params
    }

    fn local_prepare_args(&self, local_call_target: MachineReg) -> collections::Vec<MachineValue> {
        let mut args = collections::Vec::with_capacity(self.params.len() + 1);
        args.push(MachineValue::Reg(local_call_target));
        args.extend(self.param_values());
        args
    }

    fn should_defer_frame_publish(&self, slot: FrameSlot) -> bool {
        self.runtime_arg_stores
            .iter()
            .any(|store| store.slot == slot)
    }

    fn consume_live_values(&self, lower: &mut BlockLowerContext<'_>) -> Result<(), WasmError> {
        for value in &self.consumed_values {
            let _ = lower.use_value(*value)?;
        }
        lower.release_dead_values()
    }
}

fn prepend_gp_word_param(
    reg: MachineReg,
    tail: &[MachineBlockParam],
) -> collections::Vec<MachineBlockParam> {
    let mut params = collections::Vec::with_capacity(tail.len() + 1);
    params.push(MachineBlockParam::gp_word(reg));
    params.extend(tail.iter().copied());
    params
}

fn prepend_gp_word_arg(reg: MachineReg, tail: &[MachineValue]) -> collections::Vec<MachineValue> {
    let mut args = collections::Vec::with_capacity(tail.len() + 1);
    args.push(MachineValue::Reg(reg));
    args.extend(tail.iter().copied());
    args
}

fn prepend_gp_word_params(
    regs: &[MachineReg],
    tail: &[MachineBlockParam],
) -> collections::Vec<MachineBlockParam> {
    let mut params = collections::Vec::with_capacity(tail.len() + regs.len());
    for &reg in regs {
        params.push(MachineBlockParam::gp_word(reg));
    }
    params.extend(tail.iter().copied());
    params
}

fn prepend_gp_word_args(
    regs: &[MachineReg],
    tail: &[MachineValue],
) -> collections::Vec<MachineValue> {
    let mut args = collections::Vec::with_capacity(tail.len() + regs.len());
    for &reg in regs {
        args.push(MachineValue::Reg(reg));
    }
    args.extend(tail.iter().copied());
    args
}

fn capture_indirect_carried_args(
    lower: &BlockLowerContext<'_>,
    args: &SsaCallArgs,
    param_locs: &[MachineParamLoc],
) -> Result<IndirectCarriedArgs, WasmError> {
    let temps = call_indirect_gp_temps(lower)?;
    let control_regs = [temps.lane0, temps.lane1, temps.lane2, temps.lane3];
    let mut carried = IndirectCarriedArgs::default();
    let mut call_args = empty_call_args_for_span(args.frame_span(), param_locs);

    for loc in param_locs {
        match *loc {
            MachineParamLoc::Frame { .. } => {}
            MachineParamLoc::GpArg {
                param_index,
                lane,
                ty,
            } => {
                let slot = args.frame_base.advance(param_index);
                let src = if let Some(arg) = call_live_arg(args, param_index) {
                    if let Some(reg) = lower.try_value_reg(arg.value) {
                        if !control_regs.contains(&reg) {
                            let reg = ensure_carried_reg(&mut carried, reg, ty);
                            track_deferred_arg_store(
                                &mut carried,
                                arg.frame_slot,
                                ty,
                                reg,
                                None,
                                arg.value,
                            );
                            MachineArgSrc::Reg(reg)
                        } else {
                            MachineArgSrc::FrameSlot(slot)
                        }
                    } else {
                        MachineArgSrc::FrameSlot(slot)
                    }
                } else {
                    MachineArgSrc::FrameSlot(slot)
                };
                call_args.lane_args.push(MachineCallLaneArg::Gp {
                    param_index,
                    lane,
                    src,
                    ty,
                });
            }
            MachineParamLoc::GpArgPair {
                param_index,
                lo_lane,
                hi_lane,
            } => {
                let slot = args.frame_base.advance(param_index);
                let src = if let Some(arg) = call_live_arg(args, param_index) {
                    if let Some((lo, Some(hi))) = lower.try_value_regs(arg.value) {
                        if !control_regs.contains(&lo) && !control_regs.contains(&hi) {
                            let lo =
                                ensure_carried_reg(&mut carried, lo, MachineStorageType::GpWord);
                            let hi =
                                ensure_carried_reg(&mut carried, hi, MachineStorageType::GpWord);
                            track_deferred_arg_store(
                                &mut carried,
                                arg.frame_slot,
                                MachineStorageType::GpI64,
                                lo,
                                Some(hi),
                                arg.value,
                            );
                            MachineArgSrcPair {
                                lo: MachineArgSrc::Reg(lo),
                                hi: MachineArgSrc::Reg(hi),
                            }
                        } else {
                            frame_slot_arg_pair(slot)
                        }
                    } else {
                        frame_slot_arg_pair(slot)
                    }
                } else {
                    frame_slot_arg_pair(slot)
                };
                call_args.lane_args.push(MachineCallLaneArg::GpPair {
                    param_index,
                    lo_lane,
                    hi_lane,
                    src,
                });
            }
            MachineParamLoc::FpArg {
                param_index,
                lane,
                ty,
            } => {
                let slot = args.frame_base.advance(param_index);
                let src = if let Some(arg) = call_live_arg(args, param_index) {
                    if let Some(reg) = lower.try_value_reg(arg.value) {
                        let reg = ensure_carried_reg(&mut carried, reg, ty);
                        track_deferred_arg_store(
                            &mut carried,
                            arg.frame_slot,
                            ty,
                            reg,
                            None,
                            arg.value,
                        );
                        MachineArgSrc::Reg(reg)
                    } else {
                        MachineArgSrc::FrameSlot(slot)
                    }
                } else {
                    MachineArgSrc::FrameSlot(slot)
                };
                call_args.lane_args.push(MachineCallLaneArg::Fp {
                    param_index,
                    lane,
                    src,
                    ty,
                });
            }
        }
    }

    if !carried.is_empty() {
        carried.local_call_args = Some(call_args);
    }
    Ok(carried)
}

fn publish_indirect_call_args_to_frame(
    lower: &mut BlockLowerContext<'_>,
    args: &SsaCallArgs,
    carried: &IndirectCarriedArgs,
) -> Result<(), WasmError> {
    for arg in &args.live_suffix {
        if !carried.should_defer_frame_publish(arg.frame_slot) {
            lower.publish_value_to_frame_slot(arg.value, arg.frame_slot)?;
        }
    }
    lower.release_dead_values()
}

fn track_deferred_arg_store(
    carried: &mut IndirectCarriedArgs,
    slot: FrameSlot,
    ty: MachineStorageType,
    lo: MachineReg,
    hi: Option<MachineReg>,
    value: SsaValue,
) {
    if !carried
        .runtime_arg_stores
        .iter()
        .any(|store| store.slot == slot)
    {
        carried
            .runtime_arg_stores
            .push(IndirectRuntimeArgStore { slot, ty, lo, hi });
    }
    if !carried.consumed_values.contains(&value) {
        carried.consumed_values.push(value);
    }
}

fn emit_indirect_runtime_arg_stores(
    lower: &mut BlockLowerContext<'_>,
    stores: &[IndirectRuntimeArgStore],
) -> Result<(), WasmError> {
    for store in stores {
        match (store.ty, store.hi) {
            (MachineStorageType::GpI64, Some(hi)) => {
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: lower.frame_addr_offset(store.slot, 0)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(store.lo),
                    },
                });
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: lower.frame_addr_offset(store.slot, 4)?,
                        width: MachineMemWidth::U32,
                        src: MachineValue::Reg(hi),
                    },
                });
            }
            (MachineStorageType::GpI64, None) => {
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpI64,
                        addr: lower.frame_addr(store.slot)?,
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(store.lo),
                    },
                });
            }
            (_, Some(_)) => {
                return Err(WasmError::internal(
                    "non-i64 indirect runtime arg store cannot have a high register",
                ));
            }
            (_, None) => {
                let width = match store.ty {
                    MachineStorageType::Fp32 => MachineMemWidth::U32,
                    MachineStorageType::GpWord
                    | MachineStorageType::GpI64
                    | MachineStorageType::Fp64
                    | MachineStorageType::V128 => MachineMemWidth::U64,
                };
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: store.ty,
                        addr: lower.frame_addr(store.slot)?,
                        width,
                        src: MachineValue::Reg(store.lo),
                    },
                });
            }
        }
    }
    Ok(())
}

fn ensure_carried_reg(
    carried: &mut IndirectCarriedArgs,
    reg: MachineReg,
    ty: MachineStorageType,
) -> MachineReg {
    if carried.params.iter().any(|param| param.reg == reg) {
        return reg;
    }
    carried.params.push(MachineBlockParam {
        reg,
        ty,
        owner: MachineRegOwner::LinearValue,
    });
    carried.entry_args.push(MachineValue::Reg(reg));
    reg
}

fn empty_call_args_for_span(args: FrameSpan, param_locs: &[MachineParamLoc]) -> MachineCallArgs {
    let mut frame_count = 0u16;
    while frame_count < args.count {
        match param_locs.get(frame_count as usize) {
            Some(MachineParamLoc::Frame { .. }) | None => {
                frame_count = frame_count.saturating_add(1);
            }
            Some(_) => break,
        }
    }
    MachineCallArgs {
        frame_params: frame_span_region(FrameSpan::new(args.start, frame_count)),
        lane_args: collections::Vec::new(),
    }
}

#[inline]
fn frame_slot_arg_pair(slot: FrameSlot) -> MachineArgSrcPair {
    MachineArgSrcPair {
        lo: MachineArgSrc::FrameSlot(slot),
        hi: MachineArgSrc::FrameSlotOffset {
            slot,
            byte_offset: 4,
        },
    }
}

#[inline]
fn call_live_arg(args: &SsaCallArgs, param_index: u16) -> Option<&SsaCallLiveArg> {
    args.live_suffix
        .iter()
        .find(|arg| arg.param_index == param_index)
}

fn lower_indirect_dispatch_cluster(
    lower: &mut BlockLowerContext<'_>,
    source: IndirectCallSource,
    type_idx: u32,
    transfer: IndirectTransfer,
    result_types: &[ValueType],
    config: BackendConfig,
    param_locs: collections::Vec<MachineParamLoc>,
    carried_args: IndirectCarriedArgs,
    table_dispatch_modes: &[TableDispatchMode],
    current_block: MachineBlockId,
    current_params: collections::Vec<MachineBlockParam>,
    original_blocks: &mut OriginalBlocks,
    extra_blocks: &mut collections::Vec<MachineBlock>,
    extra_block_ids: &mut ExtraBlockAllocator,
    const_pool: &mut ConstPoolBuilder,
) -> Result<Option<(MachineBlockId, collections::Vec<MachineBlockParam>)>, WasmError> {
    let fixed_local_table_len = match source {
        IndirectCallSource::Table { table_idx, .. } => {
            fixed_local_table_len(table_dispatch_modes, table_idx)
        }
        IndirectCallSource::Ref { .. } => None,
    };
    let fixed_local_table = fixed_local_table_len.is_some();
    let fixed_fast_table = fixed_local_table && !lower.use_explicit_stack_prechecks();
    let (checked, trap_oob, trap_invalid_ref, trap_type_mismatch, type_check, dispatch) =
        match source {
            IndirectCallSource::Table { .. } => {
                let checked = Some(extra_block_ids.alloc());
                let trap_oob = Some(extra_block_ids.alloc());
                let type_check = extra_block_ids.alloc();
                let trap_invalid_ref = extra_block_ids.alloc();
                let trap_type_mismatch = fixed_local_table.then(|| extra_block_ids.alloc());
                let dispatch = (!fixed_local_table).then(|| extra_block_ids.alloc());
                (
                    checked,
                    trap_oob,
                    trap_invalid_ref,
                    trap_type_mismatch,
                    type_check,
                    dispatch,
                )
            }
            IndirectCallSource::Ref { .. } => {
                let trap_invalid_ref = extra_block_ids.alloc();
                let type_check = extra_block_ids.alloc();
                let dispatch = Some(extra_block_ids.alloc());
                (None, None, trap_invalid_ref, None, type_check, dispatch)
            }
        };
    let local_prepare = extra_block_ids.alloc();
    let local_transfer = if fixed_fast_table {
        local_prepare
    } else {
        extra_block_ids.alloc()
    };
    let runtime_call = (!fixed_local_table).then(|| extra_block_ids.alloc());
    let continuation = if matches!(transfer, IndirectTransfer::Call { .. }) {
        Some(extra_block_ids.alloc())
    } else {
        None
    };
    let indirect_temps = call_indirect_gp_temps(lower)?;
    let local_call_target_param = indirect_temps.lane0;
    let local_call_entry_param = indirect_temps.lane2;

    lower.emit_save_dirty_cached_cells()?;
    if transfer.is_tail() {
        lower.emit_repack_tail_call_args_to_frame_prefix(transfer.source_args())?;
    }

    match source {
        IndirectCallSource::Table { table_idx, index } => {
            let bounds_limit = emit_call_indirect_bounds_check_setup(
                lower,
                table_idx,
                index,
                fixed_local_table_len,
            )?;
            carried_args.consume_live_values(lower)?;
            lower.ensure_no_live_values(
                "prepared SSA-IR call_indirect reached native lowering with live linear SSA values; values must be consumed before the dispatch edge",
            )?;
            push_lowered_block(
                current_block,
                original_blocks,
                extra_blocks,
                current_params,
                lower.take_ops(),
                MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: lower.gp_word_int_width(),
                        kind: MachineCompareKind::Ge,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(indirect_temps.lane0),
                        rhs: bounds_limit,
                    },
                    then_edge: MachineEdge {
                        target: trap_oob.expect("table indirect dispatch should allocate trap_oob"),
                        args: collections::Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: checked.expect("table indirect dispatch should allocate checked"),
                        args: prepend_gp_word_arg(indirect_temps.lane0, &carried_args.entry_args),
                    },
                },
            )?;
            push_lowered_block(
                checked.expect("table indirect dispatch should allocate checked"),
                original_blocks,
                extra_blocks,
                prepend_gp_word_param(indirect_temps.lane0, &carried_args.params),
                if fixed_fast_table {
                    build_call_indirect_fixed_table_load_block(
                        lower,
                        table_idx,
                        indirect_temps.lane0,
                    )?
                } else {
                    build_call_indirect_checked_block(
                        lower,
                        table_idx,
                        index.frame_slot().expect(
                            "non-fast indirect table dispatch requires a published frame slot",
                        ),
                        index.is_dispatch_lane(),
                        fixed_local_table,
                        !fixed_local_table,
                    )?
                },
                MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: lower.gp_word_int_width(),
                        kind: if fixed_local_table {
                            MachineCompareKind::Eq
                        } else {
                            MachineCompareKind::Ge
                        },
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(indirect_temps.lane2),
                        rhs: if fixed_fast_table {
                            MachineValue::Imm64(0)
                        } else if fixed_local_table {
                            MachineValue::Imm64(u64::MAX)
                        } else {
                            MachineValue::Reg(indirect_temps.lane1)
                        },
                    },
                    then_edge: MachineEdge {
                        target: trap_invalid_ref,
                        args: collections::Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: type_check,
                        args: if fixed_fast_table {
                            prepend_gp_word_args(
                                &[
                                    indirect_temps.lane0,
                                    indirect_temps.lane2,
                                    indirect_temps.lane3,
                                ],
                                &carried_args.param_values(),
                            )
                        } else if fixed_local_table {
                            prepend_gp_word_arg(indirect_temps.lane2, &carried_args.param_values())
                        } else {
                            carried_args.param_values()
                        },
                    },
                },
            )?;
            push_lowered_block(
                trap_oob.expect("table indirect dispatch should allocate trap_oob"),
                original_blocks,
                extra_blocks,
                collections::Vec::new(),
                collections::Vec::new(),
                MachineTerminator::Trap {
                    kind: MachineTrapKind::TableOutOfBounds,
                },
            )?;
            push_lowered_block(
                type_check,
                original_blocks,
                extra_blocks,
                if fixed_fast_table {
                    prepend_gp_word_params(
                        &[
                            indirect_temps.lane0,
                            indirect_temps.lane2,
                            indirect_temps.lane3,
                        ],
                        &carried_args.params,
                    )
                } else if fixed_local_table {
                    prepend_gp_word_param(indirect_temps.lane2, &carried_args.params)
                } else {
                    carried_args.params.clone()
                },
                if fixed_fast_table {
                    build_call_indirect_expected_type_load_block(lower, type_idx)?
                } else if fixed_local_table {
                    build_call_indirect_fixed_local_type_check_block(
                        lower,
                        type_idx,
                        indirect_temps.lane2,
                    )?
                } else {
                    build_call_indirect_type_check_block(lower, type_idx, source.frame_slot())?
                },
                MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: lower.gp_word_int_width(),
                        kind: MachineCompareKind::Ne,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(if fixed_local_table {
                            indirect_temps.lane3
                        } else {
                            indirect_temps.lane0
                        }),
                        rhs: MachineValue::Reg(if fixed_fast_table {
                            indirect_temps.lane1
                        } else {
                            indirect_temps.lane2
                        }),
                    },
                    then_edge: MachineEdge {
                        target: if fixed_local_table {
                            trap_type_mismatch.expect(
                                "fixed-local table dispatch should allocate type-mismatch trap",
                            )
                        } else {
                            runtime_call.expect("generic indirect dispatch should allocate runtime")
                        },
                        args: if fixed_local_table {
                            collections::Vec::new()
                        } else {
                            carried_args.param_values()
                        },
                    },
                    else_edge: MachineEdge {
                        target: if fixed_fast_table {
                            local_transfer
                        } else if fixed_local_table {
                            local_prepare
                        } else {
                            dispatch.expect("generic indirect dispatch should allocate dispatch")
                        },
                        args: if fixed_fast_table {
                            prepend_gp_word_args(
                                &[indirect_temps.lane0, indirect_temps.lane2],
                                &carried_args.param_values(),
                            )
                        } else if fixed_local_table {
                            carried_args.local_prepare_args(indirect_temps.lane0)
                        } else {
                            carried_args.param_values()
                        },
                    },
                },
            )?;
            push_lowered_block(
                trap_invalid_ref,
                original_blocks,
                extra_blocks,
                collections::Vec::new(),
                collections::Vec::new(),
                MachineTerminator::Trap {
                    kind: MachineTrapKind::InvalidFunctionReference,
                },
            )?;
            if let Some(trap_type_mismatch) = trap_type_mismatch {
                push_lowered_block(
                    trap_type_mismatch,
                    original_blocks,
                    extra_blocks,
                    collections::Vec::new(),
                    collections::Vec::new(),
                    MachineTerminator::Trap {
                        kind: MachineTrapKind::IndirectCallTypeMismatch,
                    },
                )?;
            } else {
                push_lowered_block(
                    dispatch.expect("generic indirect dispatch should allocate dispatch"),
                    original_blocks,
                    extra_blocks,
                    carried_args.params.clone(),
                    build_call_indirect_dispatch_block(lower, source.frame_slot())?,
                    MachineTerminator::Branch {
                        cond: MachineBranchCond::IntCompare {
                            width: lower.gp_word_int_width(),
                            kind: MachineCompareKind::Ne,
                            sign: MachineSign::Unsigned,
                            lhs: MachineValue::Reg(indirect_temps.lane2),
                            rhs: MachineValue::Imm64(function_kind::LOCAL as u64),
                        },
                        then_edge: MachineEdge {
                            target: runtime_call
                                .expect("generic indirect dispatch should allocate runtime"),
                            args: carried_args.param_values(),
                        },
                        else_edge: MachineEdge {
                            target: local_prepare,
                            args: carried_args.local_prepare_args(indirect_temps.lane0),
                        },
                    },
                )?;
            }
        }
        IndirectCallSource::Ref { ref_slot } => {
            lower.emit_machine_ops(build_call_ref_validate_block(lower, ref_slot)?);
            carried_args.consume_live_values(lower)?;
            lower.ensure_no_live_values(
                "prepared SSA-IR call_ref reached native lowering with live linear SSA values; values must be consumed before the dispatch edge",
            )?;
            push_lowered_block(
                current_block,
                original_blocks,
                extra_blocks,
                current_params,
                lower.take_ops(),
                MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: lower.gp_word_int_width(),
                        kind: MachineCompareKind::Ge,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(indirect_temps.lane0),
                        rhs: MachineValue::Reg(indirect_temps.lane1),
                    },
                    then_edge: MachineEdge {
                        target: trap_invalid_ref,
                        args: collections::Vec::new(),
                    },
                    else_edge: MachineEdge {
                        target: type_check,
                        args: carried_args.entry_args.clone(),
                    },
                },
            )?;
            push_lowered_block(
                trap_invalid_ref,
                original_blocks,
                extra_blocks,
                collections::Vec::new(),
                collections::Vec::new(),
                MachineTerminator::Trap {
                    kind: MachineTrapKind::InvalidFunctionReference,
                },
            )?;
            push_lowered_block(
                type_check,
                original_blocks,
                extra_blocks,
                carried_args.params.clone(),
                build_call_indirect_type_check_block(lower, type_idx, source.frame_slot())?,
                MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: lower.gp_word_int_width(),
                        kind: MachineCompareKind::Ne,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(indirect_temps.lane0),
                        rhs: MachineValue::Reg(indirect_temps.lane2),
                    },
                    then_edge: MachineEdge {
                        target: runtime_call
                            .expect("generic indirect dispatch should allocate runtime"),
                        args: carried_args.param_values(),
                    },
                    else_edge: MachineEdge {
                        target: dispatch
                            .expect("generic indirect dispatch should allocate dispatch"),
                        args: carried_args.param_values(),
                    },
                },
            )?;
            push_lowered_block(
                dispatch.expect("generic indirect dispatch should allocate dispatch"),
                original_blocks,
                extra_blocks,
                carried_args.params.clone(),
                build_call_indirect_dispatch_block(lower, source.frame_slot())?,
                MachineTerminator::Branch {
                    cond: MachineBranchCond::IntCompare {
                        width: lower.gp_word_int_width(),
                        kind: MachineCompareKind::Ne,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(indirect_temps.lane2),
                        rhs: MachineValue::Imm64(function_kind::LOCAL as u64),
                    },
                    then_edge: MachineEdge {
                        target: runtime_call
                            .expect("generic indirect dispatch should allocate runtime"),
                        args: carried_args.param_values(),
                    },
                    else_edge: MachineEdge {
                        target: local_prepare,
                        args: carried_args.local_prepare_args(indirect_temps.lane0),
                    },
                },
            )?;
        }
    }
    if !fixed_fast_table {
        let local_prepare_ops = match transfer {
            IndirectTransfer::Call { args, .. } => {
                build_call_indirect_local_prepare_block(lower, args)?
            }
            IndirectTransfer::Tail { .. } => {
                build_tail_call_indirect_local_prepare_block(lower, transfer.callee_args())?
            }
        };
        push_lowered_block(
            local_prepare,
            original_blocks,
            extra_blocks,
            carried_args.local_prepare_params(local_call_target_param),
            local_prepare_ops,
            MachineTerminator::Jump(MachineEdge {
                target: local_transfer,
                args: carried_args.param_values(),
            }),
        )?;
    }
    let mut local_transfer_ops = match transfer {
        IndirectTransfer::Call { results, .. } => {
            if fixed_fast_table {
                collections::Vec::new()
            } else {
                build_call_indirect_local_transfer_block(lower, results)?
            }
        }
        IndirectTransfer::Tail { .. } => {
            if fixed_fast_table {
                collections::Vec::new()
            } else {
                build_call_indirect_local_tail_transfer_block(lower)?
            }
        }
    };
    let machine_call_args = match carried_args.local_call_args.clone() {
        Some(args) => args,
        None => {
            lower.prepare_internal_call_args_from_frame_span(transfer.callee_args(), &param_locs)?
        }
    };
    local_transfer_ops.extend(lower.take_ops());
    let mut result_success_args = collections::Vec::new();
    let mut continuation_params = collections::Vec::new();
    let local_transfer_term = match continuation {
        Some(continuation) => {
            let IndirectTransfer::Call {
                results,
                live_result,
                ..
            } = transfer
            else {
                unreachable!("non-tail indirect transfer must be Call");
            };
            let return_abi = derive_return_abi(
                result_types,
                Some(fixed_callee_return_results(results.count)),
                config,
            );
            let (machine_results, success_args, params) =
                lower.call_result_placement(&return_abi, results, live_result)?;
            result_success_args = success_args;
            continuation_params = params;
            MachineTerminator::Call {
                target: MachineCallTarget::Indirect {
                    callee_target: indirect_temps.lane0,
                    callee_entry: indirect_temps.lane2,
                },
                frame_delta: slot_offset_bytes(transfer.source_args().start)? as u32,
                args: machine_call_args,
                results: machine_results,
                success: MachineEdge {
                    target: continuation,
                    args: result_success_args.clone(),
                },
            }
        }
        None => MachineTerminator::TailCall {
            target: MachineCallTarget::Indirect {
                callee_target: indirect_temps.lane0,
                callee_entry: indirect_temps.lane2,
            },
            args: machine_call_args,
        },
    };
    push_lowered_block(
        local_transfer,
        original_blocks,
        extra_blocks,
        if fixed_fast_table {
            prepend_gp_word_params(
                &[local_call_target_param, local_call_entry_param],
                &carried_args.params,
            )
        } else {
            carried_args.params.clone()
        },
        local_transfer_ops,
        local_transfer_term,
    )?;
    if let Some(runtime_call) = runtime_call {
        let metadata = source.build_runtime_call_meta(
            lower,
            type_idx,
            transfer.callee_args(),
            transfer.runtime_results(),
            const_pool,
        );
        emit_indirect_runtime_arg_stores(lower, &carried_args.runtime_arg_stores)?;
        let runtime_ops_seed = lower.build_runtime_call_ops(metadata);
        lower.emit_machine_ops(runtime_ops_seed);
        if continuation.is_some() {
            if let IndirectTransfer::Call {
                results,
                live_result: Some(value),
                ..
            } = transfer
            {
                lower.materialize_runtime_call_result(value, results.start)?;
            }
        }
        let runtime_ops = lower.take_ops();
        let runtime_term = match continuation {
            Some(continuation) => MachineTerminator::Jump(MachineEdge {
                target: continuation,
                args: result_success_args.clone(),
            }),
            None => return_terminator_from_frame(lower.current_return_abi()),
        };
        push_lowered_block(
            runtime_call,
            original_blocks,
            extra_blocks,
            carried_args.params.clone(),
            runtime_ops,
            runtime_term,
        )?;
    }
    Ok(continuation.map(|id| (id, continuation_params)))
}

fn push_lowered_block(
    id: MachineBlockId,
    original_blocks: &mut OriginalBlocks,
    continuation_blocks: &mut collections::Vec<MachineBlock>,
    mut params: collections::Vec<MachineBlockParam>,
    mut ops: collections::Vec<MachineInst>,
    terminator: MachineTerminator,
) -> Result<(), WasmError> {
    params.shrink_to_fit();
    ops.shrink_to_fit();
    let block = MachineBlock {
        id,
        params,
        ops,
        terminator,
    };
    let original_len = original_blocks.expected();
    if id.as_usize() < original_len {
        original_blocks.push(id, block)?;
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
        .collect::<collections::Vec<_>>();
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
        MachineTerminator::Call { .. }
        | MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
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
    args: collections::Vec<MachineValue>,
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
) -> Result<collections::Vec<ValueRegs>, WasmError> {
    let mut regs = collections::Vec::with_capacity(params.len());
    let mut gp_index = 0usize;
    let mut fp_index = 0usize;
    for param in params {
        let ty = program_value_storage_type(program, *param);
        if ty.is_fp() {
            regs.push(ValueRegs {
                lo: preferred_fp_dynamic_reg(regfile, fp_index).ok_or_else(|| {
                    WasmError::internal("target params exceed FP dynamic register budget")
                })?,
                hi: None,
            });
            fp_index += 1;
        } else if gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
            let lo = preferred_gp_dynamic_reg(regfile, gp_index).ok_or_else(|| {
                WasmError::internal("target params exceed GP dynamic register budget")
            })?;
            let hi = preferred_gp_dynamic_reg(regfile, gp_index + 1).ok_or_else(|| {
                WasmError::internal("target i64 params exceed GP dynamic pair budget")
            })?;
            regs.push(ValueRegs { lo, hi: Some(hi) });
            gp_index += 2;
        } else {
            regs.push(ValueRegs {
                lo: preferred_gp_dynamic_reg(regfile, gp_index).ok_or_else(|| {
                    WasmError::internal("target params exceed GP dynamic register budget")
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
        Some(ValueType::V128) => MachineStorageType::V128,
        Some(_) | None => MachineStorageType::GpWord,
    }
}

pub(super) fn preferred_gp_dynamic_reg(
    regfile: &MachineRegFile,
    ordinal: usize,
) -> Option<MachineReg> {
    regfile.ordered_gp_allocatable(ordinal)
}

pub(super) fn preferred_fp_dynamic_reg(
    regfile: &MachineRegFile,
    ordinal: usize,
) -> Option<MachineReg> {
    regfile.ordered_fp_allocatable(ordinal)
}

fn fp_reg_init_widths(
    regfile: &MachineRegFile,
) -> Result<collections::Vec<Option<MachineFloatWidth>>, WasmError> {
    Ok(collections::vec![None; regfile.fp_dynamic_count()])
}

fn append_entry_cache_params(
    params: &mut collections::Vec<MachineBlockParam>,
    entry_cache_params: &[EntryCacheParam],
    cached_cells: &[super::lower_context::CachedCell],
) {
    for entry in entry_cache_params {
        if let Some(cached) = cached_cells.get(usize::from(entry.cached_index)) {
            append_machine_block_params_for_value(
                params,
                entry.regs,
                cached.ty,
                MachineRegOwner::CachedCell,
            );
        }
    }
}

struct OriginalBlocks {
    expected: usize,
    blocks: collections::Vec<MachineBlock>,
}

impl OriginalBlocks {
    #[inline]
    fn new(expected: usize) -> Self {
        Self {
            expected,
            blocks: collections::Vec::with_capacity(expected),
        }
    }

    #[inline]
    fn expected(&self) -> usize {
        self.expected
    }

    #[inline]
    fn push(&mut self, id: MachineBlockId, block: MachineBlock) -> Result<(), WasmError> {
        if id.as_usize() != self.blocks.len() {
            return Err(WasmError::internal(
                "machine lowering emitted original blocks out of id order".into(),
            ));
        }
        self.blocks.push(block);
        Ok(())
    }

    #[inline]
    fn finish(self) -> Result<collections::Vec<MachineBlock>, WasmError> {
        if self.blocks.len() != self.expected {
            return Err(WasmError::internal(
                "machine lowering did not produce every original block".into(),
            ));
        }
        Ok(self.blocks)
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
// threads a fixed GP dynamic bundle across synthetic blocks.
fn call_indirect_gp_temps(lower: &BlockLowerContext<'_>) -> Result<CallIndirectGpTemps, WasmError> {
    let base = lower.regfile().gp_arg_lane_count();
    if lower.regfile().gp_dynamic_count() < base + INDIRECT_DISPATCH_CONTROL_GP_LANES {
        return Err(WasmError::internal(
            "call_indirect lowering requires four GP dynamic control lanes",
        ));
    }
    Ok(CallIndirectGpTemps {
        lane0: lower.reserved_gp_dynamic(base, "call_indirect control lane 0")?,
        lane1: lower.reserved_gp_dynamic(base + 1, "call_indirect control lane 1")?,
        lane2: lower.reserved_gp_dynamic(base + 2, "call_indirect control lane 2")?,
        lane3: lower.reserved_gp_dynamic(base + 3, "call_indirect control lane 3")?,
    })
}

fn evict_call_indirect_control_gp_caches(
    lower: &mut BlockLowerContext<'_>,
) -> Result<(), WasmError> {
    let temps = call_indirect_gp_temps(lower)?;
    lower.evict_cached_cells_in_gp_regs(&[temps.lane0, temps.lane1, temps.lane2, temps.lane3])
}

fn prepare_call_indirect_index_in_lane(
    lower: &mut BlockLowerContext<'_>,
    loc: &crate::vm::middle::ssa_ir::ir::SsaCallOperandLoc,
) -> Result<IndirectCallIndex, WasmError> {
    evict_call_indirect_control_gp_caches(lower)?;
    let temps = call_indirect_gp_temps(lower)?;
    let index = temps.lane0;
    match loc {
        crate::vm::middle::ssa_ir::ir::SsaCallOperandLoc::Stack { slot } => {
            lower.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: index,
                    addr: lower.frame_addr(*slot)?,
                    width: MachineMemWidth::U32,
                    extension: MachineLoadExtension::ZeroExtend,
                },
            });
        }
        crate::vm::middle::ssa_ir::ir::SsaCallOperandLoc::Live { value, .. } => {
            let src = lower.use_value(*value)?;
            if lower.gp_reg_width() == 8 {
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Convert {
                        op: MachineConvertOp::I64ExtendI32U,
                        dst: index,
                        src: MachineValue::Reg(src),
                    },
                });
            } else if src != index {
                lower.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: index,
                        src: MachineValue::Reg(src),
                    },
                });
            }
            lower.release_dead_values()?;
        }
    }
    Ok(IndirectCallIndex::DispatchLane)
}

fn emit_call_indirect_bounds_check_setup(
    lower: &mut BlockLowerContext<'_>,
    table_idx: u32,
    index_source: IndirectCallIndex,
    fixed_len: Option<u32>,
) -> Result<MachineValue, WasmError> {
    evict_call_indirect_control_gp_caches(lower)?;
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let index = temps.lane0;
    let table_views = temps.lane1;
    let table_len = temps.lane2;
    if let IndirectCallIndex::Frame(index_slot) = index_source {
        lower.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: index,
                addr: lower.frame_addr(index_slot)?,
                // Wasm table indices are i32 values even on 64-bit hosts.
                // Reload them with explicit zero-extension so stale high halves
                // in a published GpWord carrier cannot perturb indirect dispatch.
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        });
    }
    if let Some(len) = fixed_len {
        return Ok(MachineValue::Imm64(u64::from(len)));
    }
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: table_views,
            addr: lower.runtime_addr(runtime_layout.context.table_views_base_offset),
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
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
    Ok(MachineValue::Reg(table_len))
}

fn build_call_indirect_checked_block(
    lower: &BlockLowerContext<'_>,
    table_idx: u32,
    index_slot: FrameSlot,
    index_in_reg: bool,
    fixed_local_table: bool,
    store_func_idx: bool,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let index = temps.lane0;
    let table_base = temps.lane1;
    let func_idx = temps.lane2;
    let mut ops = collections::Vec::new();
    if !index_in_reg {
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: index,
                addr: lower.frame_addr(index_slot)?,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        });
    }
    ops.extend([
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: table_base,
                addr: lower.runtime_addr(runtime_layout.context.table_views_base_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
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
                owner: MachineRegOwner::LinearValue,
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
    ]);
    if store_func_idx {
        ops.push(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: lower.frame_addr(index_slot)?,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(func_idx),
            },
        });
    }
    if !fixed_local_table {
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: table_base,
                addr: lower.runtime_addr(runtime_layout.context.function_views_len_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
    }
    Ok(ops)
}

fn build_call_ref_validate_block(
    lower: &BlockLowerContext<'_>,
    ref_slot: FrameSlot,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let func_idx = temps.lane0;
    let function_views_len = temps.lane1;
    Ok(collections::vec![
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: func_idx,
                addr: lower.frame_addr(ref_slot)?,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: function_views_len,
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
    index_slot: FrameSlot,
) -> Result<collections::Vec<MachineInst>, WasmError> {
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
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: function_views,
            addr: lower.runtime_addr(runtime_layout.context.type_canon_base_offset),
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
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

fn build_call_indirect_fixed_local_type_check_block(
    lower: &BlockLowerContext<'_>,
    expected_type_idx: u32,
    func_idx: MachineReg,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let function_view_layout = function_view_abi_layout();
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let local_target = temps.lane0;
    let function_views = temps.lane1;
    let scaled_index = temps.lane2;
    let actual_type = temps.lane3;
    let expected_type = temps.lane2;
    let mut ops = collections::Vec::new();
    if scaled_index != func_idx {
        ops.push(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: scaled_index,
                src: MachineValue::Reg(func_idx),
            },
        });
    }
    ops.extend([
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Mul,
                dst: scaled_index,
                lhs: MachineValue::Reg(scaled_index),
                rhs: MachineValue::Imm64(u64::from(runtime_layout.function_view.stride)),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: function_views,
                addr: lower.runtime_addr(runtime_layout.context.function_views_base_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: function_views,
                lhs: MachineValue::Reg(function_views),
                rhs: MachineValue::Reg(scaled_index),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: actual_type,
                addr: MachineAddr {
                    base: function_views,
                    offset: function_view_layout.type_canon_offset as i32,
                },
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: local_target,
                addr: MachineAddr {
                    base: function_views,
                    offset: function_view_layout.local_target_offset as i32,
                },
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: function_views,
                addr: lower.runtime_addr(runtime_layout.context.type_canon_base_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
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
        },
    ]);
    Ok(ops)
}

fn build_call_indirect_fixed_table_load_block(
    lower: &BlockLowerContext<'_>,
    table_idx: u32,
    index_reg: MachineReg,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let fixed_view = fixed_call_table_view_abi_layout();
    let fixed_entry = fixed_call_table_entry_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let local_target = temps.lane0;
    let table_base = temps.lane1;
    let callee_entry = temps.lane2;
    let actual_type = temps.lane3;
    debug_assert_eq!(fixed_entry.stride, 16);
    Ok(collections::vec![
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: table_base,
                addr: lower.runtime_addr(runtime_layout.context.fixed_call_table_views_base_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: table_base,
                addr: indexed_const_addr(
                    table_base,
                    table_idx,
                    fixed_view.stride as usize,
                    fixed_view.entry_base_offset,
                )?,
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Shl,
                dst: local_target,
                lhs: MachineValue::Reg(index_reg),
                rhs: MachineValue::Imm64(4),
            },
        },
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: actual_type,
                base: table_base,
                index: local_target,
                index_extend: MachineIndexExtend::None,
                offset: fixed_entry.type_canon_offset as i32,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: local_target,
                lhs: MachineValue::Reg(local_target),
                rhs: MachineValue::Imm64(u64::from(fixed_entry.entry_offset)),
            },
        },
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: callee_entry,
                base: table_base,
                index: local_target,
                index_extend: MachineIndexExtend::None,
                offset: 0,
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: lower.gp_word_int_width(),
                op: MachineIntBinaryOp::Sub,
                dst: local_target,
                lhs: MachineValue::Reg(local_target),
                rhs: MachineValue::Imm64(u64::from(
                    fixed_entry.entry_offset - fixed_entry.local_target_offset,
                )),
            },
        },
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: local_target,
                base: table_base,
                index: local_target,
                index_extend: MachineIndexExtend::None,
                offset: 0,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        },
    ])
}

fn build_call_indirect_expected_type_load_block(
    lower: &BlockLowerContext<'_>,
    expected_type_idx: u32,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let expected_type = temps.lane1;
    let type_canon = temps.lane1;
    Ok(collections::vec![
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: type_canon,
                addr: lower.runtime_addr(runtime_layout.context.type_canon_base_offset),
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: expected_type,
                addr: indexed_const_addr(
                    type_canon,
                    expected_type_idx,
                    core::mem::size_of::<u32>(),
                    0,
                )?,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::ZeroExtend,
            },
        },
    ])
}

fn build_call_indirect_dispatch_block(
    lower: &BlockLowerContext<'_>,
    index_slot: FrameSlot,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let function_view_layout = function_view_abi_layout();
    let temps = call_indirect_gp_temps(lower)?;
    let local_target = temps.lane0;
    let function_views = temps.lane1;
    let scaled_index = temps.lane2;
    let kind = temps.lane2;
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
            owner: MachineRegOwner::LinearValue,
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
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let call_info = runtime_layout.local_call_info;
    let temps = call_indirect_gp_temps(lower)?;
    let callee_target = temps.lane0;
    let callee_frame_base = temps.lane1;
    let call_info_base = temps.lane2;
    let total_frame_bytes = temps.lane3;

    // The local callee reuses the caller operand span as its frame prefix.
    // Non-param local initialization belongs to the callee entry block, matching
    // direct local calls; the caller only performs the optional stack precheck.
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: lower.gp_word_int_width(),
            op: MachineIntBinaryOp::Add,
            dst: callee_frame_base,
            lhs: MachineValue::Reg(lower.frame_base_reg()),
            rhs: MachineValue::Imm64(slot_offset_bytes(args.start)? as u64),
        },
    });

    if lower.use_explicit_stack_prechecks() {
        emit_local_call_info_entry_addr(lower, callee_target, call_info_base, total_frame_bytes)?;
        lower.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: total_frame_bytes,
                addr: MachineAddr {
                    base: call_info_base,
                    offset: call_info.total_frame_bytes_offset as i32,
                },
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        lower.emit_dynamic_call_stack_precheck(
            callee_frame_base,
            call_info_base,
            total_frame_bytes,
        )?;
    }

    Ok(lower.take_ops())
}

fn build_tail_call_indirect_local_prepare_block(
    lower: &mut BlockLowerContext<'_>,
    _args: FrameSpan,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let call_info = runtime_layout.local_call_info;
    let temps = call_indirect_gp_temps(lower)?;
    let callee_target = temps.lane0;
    let callee_frame_base = temps.lane1;
    let call_info_base = temps.lane2;
    let total_frame_bytes = temps.lane3;

    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: callee_frame_base,
            src: MachineValue::Reg(lower.frame_base_reg()),
        },
    });

    if lower.use_explicit_stack_prechecks() {
        emit_local_call_info_entry_addr(lower, callee_target, call_info_base, total_frame_bytes)?;
        lower.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: total_frame_bytes,
                addr: MachineAddr {
                    base: call_info_base,
                    offset: call_info.total_frame_bytes_offset as i32,
                },
                width: lower.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        lower.emit_dynamic_call_stack_precheck(
            callee_frame_base,
            call_info_base,
            total_frame_bytes,
        )?;
    }

    Ok(lower.take_ops())
}

fn build_call_indirect_local_transfer_block(
    lower: &mut BlockLowerContext<'_>,
    _results: FrameSpan,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let call_info = runtime_layout.local_call_info;
    let temps = call_indirect_gp_temps(lower)?;
    let callee_target = temps.lane0;
    // lane1 is `callee_frame_base`, populated by the earlier prepare block.
    let callee_entry = temps.lane2;
    let info_base = temps.lane3;

    emit_local_call_info_entry_addr(lower, callee_target, info_base, callee_entry)?;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: callee_entry,
            addr: MachineAddr {
                base: info_base,
                offset: call_info.entry_offset as i32,
            },
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
    Ok(lower.take_ops())
}

fn build_call_indirect_local_tail_transfer_block(
    lower: &mut BlockLowerContext<'_>,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let call_info = runtime_layout.local_call_info;
    let temps = call_indirect_gp_temps(lower)?;
    let callee_target = temps.lane0;
    let callee_entry = temps.lane2;
    let info_base = temps.lane3;

    emit_local_call_info_entry_addr(lower, callee_target, info_base, callee_entry)?;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: callee_entry,
            addr: MachineAddr {
                base: info_base,
                offset: call_info.entry_offset as i32,
            },
            width: lower.gp_word_mem_width(),
            extension: MachineLoadExtension::None,
        },
    });
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
            owner: MachineRegOwner::LinearValue,
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
                owner: MachineRegOwner::LinearValue,
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
    index_slot: FrameSlot,
    func_idx_dst: MachineReg,
    base_reg: MachineReg,
    scaled_index_reg: MachineReg,
    field_offset: u32,
    field_width: MachineMemWidth,
    field_extension: MachineLoadExtension,
    dst: MachineReg,
) -> Result<collections::Vec<MachineInst>, WasmError> {
    let runtime_layout = lower.runtime_abi_layout();
    let mut ops = collections::vec![MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: func_idx_dst,
            addr: lower.frame_addr(index_slot)?,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
        },
    }];
    if scaled_index_reg != func_idx_dst {
        ops.push(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
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
                owner: MachineRegOwner::LinearValue,
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
                owner: MachineRegOwner::LinearValue,
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
        .ok_or_else(|| WasmError::internal("runtime view byte offset overflow"))?;
    let offset = i32::try_from(scaled)
        .map_err(|_| WasmError::internal("runtime view byte offset exceeds i32"))?;
    Ok(MachineAddr { base, offset })
}

// ---------------------------------------------------------------------------
// Cross-block dirty-flag dataflow analysis
// ---------------------------------------------------------------------------
//
// Entry dirty bits are solved on the final explicit SSA program after edge
// repair blocks have been inserted. Each block carries one canonical entry
// cache state; dirty bits are propagated conservatively by OR-ing predecessor
// exit dirtiness for the slots that must already be resident on block entry.
//

fn compute_block_entry_cache_dirty(
    program: &SsaProgram,
    cached_cells: &[CachedCell],
    abi: &MachineFunctionAbi,
    entry_cache_params: &[collections::Vec<EntryCacheParam>],
    is_local_func: &[bool],
    call_preserved_cache_candidates: &[bool],
) -> collections::Vec<collections::Vec<bool>> {
    if program.blocks.is_empty() || cached_cells.is_empty() {
        return collections::vec![collections::Vec::new(); program.blocks.len()];
    }

    let predecessors = compute_ssa_predecessors(program);
    let mut successors = collections::vec![collections::Vec::new(); program.blocks.len()];
    for (successor, preds) in predecessors.iter().enumerate() {
        for &predecessor in preds {
            successors[predecessor].push(successor);
        }
    }
    let slot_index_len = cached_cells
        .iter()
        .map(|cached| cached.cell.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut slot_to_index = collections::vec![u32::MAX; slot_index_len];
    for (index, cached) in cached_cells.iter().enumerate() {
        slot_to_index[cached.cell.0 as usize] = index as u32;
    }
    let mut entry_dirty =
        collections::vec![collections::vec![false; cached_cells.len()]; program.blocks.len()];
    let register_param_slots = register_param_slots(&abi.param_locs);
    let entry_index = program.entry.as_usize();
    if let Some(entries) = entry_cache_params.get(entry_index) {
        for entry in entries {
            if !entry.needs_value {
                continue;
            }
            let cached_index = usize::from(entry.cached_index);
            let Some(cached) = cached_cells.get(cached_index) else {
                continue;
            };
            if register_param_slots.contains(&cached.cell) {
                if let Some(bits) = entry_dirty.get_mut(entry_index) {
                    if let Some(bit) = bits.get_mut(cached_index) {
                        *bit = true;
                    }
                }
            }
        }
    }

    // Cache each block's transfer result and revisit only blocks whose entry
    // state changed. The previous global sweep re-simulated a predecessor once
    // for every outgoing edge on every iteration.
    let mut exit_dirty =
        collections::vec![collections::vec![false; cached_cells.len()]; program.blocks.len()];
    let mut queued = collections::vec![true; program.blocks.len()];
    let mut worklist: collections::Vec<usize> = (0..program.blocks.len()).collect();
    let mut worklist_head = 0;
    while worklist_head < worklist.len() {
        let block_index = worklist[worklist_head];
        worklist_head += 1;
        queued[block_index] = false;

        let (_, next_exit_dirty) = simulate_block_cache_exit_state(
            program,
            &program.blocks[block_index],
            &entry_dirty[block_index],
            &slot_to_index,
            &register_param_slots,
            is_local_func,
            call_preserved_cache_candidates,
        );
        if next_exit_dirty == exit_dirty[block_index] {
            continue;
        }
        exit_dirty[block_index] = next_exit_dirty;

        for &successor in &successors[block_index] {
            if successor == entry_index {
                continue;
            }
            let Some(entry_slots) = program.block_entry_cached_cells.get(successor) else {
                continue;
            };
            if entry_slots.is_empty() {
                continue;
            }

            let mut next_dirty = collections::vec![false; cached_cells.len()];
            for &pred_index in &predecessors[successor] {
                let pred_exit_dirty = &exit_dirty[pred_index];
                for &slot in entry_slots {
                    if let Some(cached_index) = cached_cell_index(&slot_to_index, slot) {
                        next_dirty[cached_index] |= pred_exit_dirty[cached_index];
                    }
                }
            }

            if next_dirty != entry_dirty[successor] {
                entry_dirty[successor] = next_dirty;
                if !queued[successor] {
                    queued[successor] = true;
                    worklist.push(successor);
                }
            }
        }
    }

    entry_dirty
}

fn register_param_slots(param_locs: &[MachineParamLoc]) -> collections::Vec<CellId> {
    let mut slots = collections::Vec::new();
    for loc in param_locs {
        match *loc {
            MachineParamLoc::Frame { .. } => {}
            MachineParamLoc::GpArg { param_index, .. }
            | MachineParamLoc::GpArgPair { param_index, .. }
            | MachineParamLoc::FpArg { param_index, .. } => {
                slots.push(CellId(param_index));
            }
        }
    }
    slots
}

fn compute_ssa_predecessors(program: &SsaProgram) -> collections::Vec<collections::Vec<usize>> {
    let mut predecessors = collections::vec![collections::Vec::new(); program.blocks.len()];
    for block in &program.blocks {
        let from = block.id.as_usize();
        match &block.terminator {
            SsaTerminator::Goto(edge) => {
                if let Some(preds) = predecessors.get_mut(edge.target.as_usize()) {
                    preds.push(from);
                }
            }
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                if let Some(preds) = predecessors.get_mut(then_edge.target.as_usize()) {
                    preds.push(from);
                }
                if let Some(preds) = predecessors.get_mut(else_edge.target.as_usize()) {
                    preds.push(from);
                }
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    if let Some(preds) = predecessors.get_mut(edge.target.as_usize()) {
                        preds.push(from);
                    }
                }
            }
            SsaTerminator::Return { .. }
            | SsaTerminator::ReturnScalar { .. }
            | SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. }
            | SsaTerminator::TrapUnreachable
            | SsaTerminator::EhThrow { .. }
            | SsaTerminator::EhThrowRef { .. } => {}
        }
    }

    for preds in &mut predecessors {
        preds.sort_unstable();
        preds.dedup();
    }
    predecessors
}

fn simulate_block_cache_exit_state(
    program: &SsaProgram,
    block: &SsaBlock,
    entry_dirty: &[bool],
    slot_to_index: &[u32],
    register_param_slots: &[CellId],
    is_local_func: &[bool],
    call_preserved_cache_candidates: &[bool],
) -> (collections::Vec<bool>, collections::Vec<bool>) {
    let mut resident = collections::vec![false; entry_dirty.len()];
    let mut dirty = collections::vec![false; entry_dirty.len()];

    if let Some(entry_slots) = program.block_entry_cached_cells.get(block.id.as_usize()) {
        for &slot in entry_slots {
            if let Some(cached_index) = cached_cell_index(slot_to_index, slot) {
                resident[cached_index] = true;
                dirty[cached_index] = entry_dirty.get(cached_index).copied().unwrap_or(false);
            }
        }
    }

    for inst in &block.ops {
        match inst.op {
            SsaOp::CELL_GET_CACHE | SsaOp::CELL_ENSURE_CACHE => {
                let slot = CellId(inst.meta);
                if let Some(cached_index) = cached_cell_index(slot_to_index, slot) {
                    if !resident[cached_index] {
                        resident[cached_index] = true;
                        dirty[cached_index] =
                            block.id == program.entry && register_param_slots.contains(&slot);
                    }
                }
            }
            SsaOp::CELL_RESERVE_CACHE => {
                let slot = CellId(inst.meta);
                if let Some(cached_index) = cached_cell_index(slot_to_index, slot) {
                    if !resident[cached_index] {
                        resident[cached_index] = true;
                        dirty[cached_index] = false;
                    }
                }
            }
            SsaOp::CELL_SET_CACHE => {
                let slot = CellId(inst.meta);
                if let Some(cached_index) = cached_cell_index(slot_to_index, slot) {
                    resident[cached_index] = true;
                    dirty[cached_index] = true;
                }
            }
            SsaOp::CELL_DROP_CACHE => {
                let slot = CellId(inst.meta);
                if let Some(cached_index) = cached_cell_index(slot_to_index, slot) {
                    resident[cached_index] = false;
                    dirty[cached_index] = false;
                }
            }
            SsaOp::CALL => {
                if !is_direct_local_ssa_call(program, is_local_func, inst.meta) {
                    resident.fill(false);
                    dirty.fill(false);
                    continue;
                }
                for cached_index in 0..resident.len() {
                    if call_preserved_cache_candidates
                        .get(cached_index)
                        .copied()
                        .unwrap_or(false)
                    {
                        dirty[cached_index] = false;
                        continue;
                    }
                    resident[cached_index] = false;
                    dirty[cached_index] = false;
                }
            }
            _ => {}
        }
    }

    (resident, dirty)
}

#[inline]
fn cached_cell_index(slot_to_index: &[u32], cell: CellId) -> Option<usize> {
    let index = slot_to_index
        .get(cell.0 as usize)
        .copied()
        .unwrap_or(u32::MAX);
    (index != u32::MAX).then_some(index as usize)
}

/// Value-carry classification: a call whose continuation can receive caches
/// still holding their VALUES — a direct call to a local JIT body, matching
/// the residency solver's nomination, the plan's call survival, and
/// `prepare_cached_cells_for_local_call`'s emission. Distinct from the layout
/// sim's `is_local_jit_call` (lower_cache_layout), which additionally credits
/// fixed-local-only indirect calls: that one governs LANE-ASSIGNMENT stability
/// (a lane kept assigned across the call so the post-call reload lands in the
/// same lane), not value survival — measured worth 1.3k instructions on
/// sqlite, so the two classifications diverge on purpose.
fn is_direct_local_ssa_call(program: &SsaProgram, is_local_func: &[bool], call_index: u16) -> bool {
    match program.call_ops.get(call_index as usize) {
        Some(SsaCallOp::CallDirect { callee, .. }) => is_local_func
            .get(*callee as usize)
            .copied()
            .unwrap_or(false),
        Some(SsaCallOp::CallIndirect { .. } | SsaCallOp::CallRef { .. }) | None => false,
    }
}
