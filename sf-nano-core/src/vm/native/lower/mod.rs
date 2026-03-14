//! Internal lowering from prepared LIR to MachineIR.
//!
//! This is not another public IR layer. It is the internal native-side pass
//! that consumes prepared LIR plus the shared frame contract and produces the
//! final shared machine-facing IR.

mod boundary;
mod context;
mod ops;
mod regfile;
mod sidecar;
mod util;

#[cfg(test)]
mod tests;

use alloc::{vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        lir::{
            ir::{LirBoundaryOp, LirInstKind, LirProgram, LirTerminator},
            slot::FrameSpan,
            validate::validate_program,
        },
        native::ir::{
            machine::{
                MachineBlock, MachineBlockId, MachineBranchCond, MachineCompareKind,
                MachineConstData, MachineFuncId, MachineFunction, MachineInst, MachineInstKind,
                MachineLoadExtension, MachineMemWidth, MachineModule, MachineProgram, MachineReg,
                MachineTerminator, MachineTrapKind, MachineValue,
            },
            runtime::{
                MachineCallLinkLayout, MachineExternBinding, MachineFrameRegion,
                MachineFunctionRuntime, MachineRuntimeContract,
            },
        },
        native::runtime::context::{
            ctx_offset, function_kind, function_view_offset, NativeFunctionView,
        },
        plan::frame::FrameLayoutPlan,
    },
};

use self::{
    context::BlockLowerContext, ops::LeafLowering, regfile::MachineRegFile, sidecar::SidecarBuilder,
};

/// One prepared function ready for LIR -> MachineIR lowering.
#[derive(Clone, Copy, Debug)]
pub struct LowerFunctionInput<'a> {
    pub id: MachineFuncId,
    pub frame: FrameLayoutPlan,
    pub lir: &'a LirProgram,
    /// Declared result count from the function type signature.
    /// Used as fallback when all return paths are unreachable.
    pub result_count: u16,
}

/// One lowering request for a whole machine module.
#[derive(Clone, Copy, Debug)]
pub struct LowerModuleInput<'a> {
    pub backend: BackendConfig,
    pub functions: &'a [LowerFunctionInput<'a>],
}

/// Result of lowering prepared LIR into MachineIR plus runtime-side contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweredMachineModule {
    pub module: MachineModule,
    pub runtime: MachineRuntimeContract,
}

pub fn lower_module(input: LowerModuleInput<'_>) -> Result<LoweredMachineModule, WasmError> {
    let regfile = MachineRegFile::new(input.backend)?;
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
        validate_program(function.lir)?;
        function_runtime[function.id.0 as usize] = lower_function_runtime(*function, call_link)?;
    }
    for function in input.functions {
        functions[function.id.0 as usize] = Some(lower_function(
            *function,
            &regfile,
            &function_runtime,
            call_link,
            &mut sidecar,
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
            function.unwrap_or_else(|| {
                stub_machine_function(MachineFuncId(index as u32), regfile.reg_count())
            })
        })
        .collect();
    let module = MachineModule {
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
    let mut return_results = derive_return_results(input.lir)?;
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
    regfile: &MachineRegFile,
    runtime: &[MachineFunctionRuntime],
    call_link: MachineCallLinkLayout,
    sidecar: &mut SidecarBuilder,
) -> Result<MachineFunction, WasmError> {
    let caller_runtime = runtime.get(input.id.0 as usize).copied().ok_or_else(|| {
        WasmError::internal("machine runtime metadata missing for function".into())
    })?;
    let original_block_count = input.lir.blocks.len();
    let mut original_blocks = alloc::vec![None; original_block_count];
    let mut extra_blocks = Vec::new();
    let mut extra_block_ids = ExtraBlockAllocator::new(original_block_count as u32);

    for block in &input.lir.blocks {
        let target = block.id;
        let mut lower = BlockLowerContext::new(
            regfile,
            input.frame,
            input.lir,
            &input.lir.local_cache,
            block,
            caller_runtime,
            runtime,
            call_link,
            target == input.lir.entry,
        )?;
        let mut current_block = MachineBlockId(block.id.as_u32());
        let mut current_params = lower.machine_params().to_vec();

        for inst in &block.ops {
            match &inst.kind {
                LirInstKind::Value { op, args, results } => {
                    if let Some(lowered) = lower.lower_special_leaf(
                        op,
                        args,
                        results,
                        extra_block_ids.peek(1),
                        extra_block_ids.peek(0),
                    )? {
                        lower.release_dead_values()?;
                        match lowered {
                            LeafLowering::InPlace => {}
                            LeafLowering::Split {
                                continuation,
                                trap,
                                trap_kind,
                                terminator,
                                continuation_ops,
                            } => {
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
                                current_params = Vec::new();
                                lower.emit_machine_ops(continuation_ops);
                            }
                        }
                        continue;
                    }
                    lower.lower_inst(inst)?;
                }
                LirInstKind::Boundary(boundary) => match boundary {
                    LirBoundaryOp::MemoryGrow { .. }
                    | LirBoundaryOp::MemoryFill { .. }
                    | LirBoundaryOp::MemoryCopy { .. }
                    | LirBoundaryOp::TableGrow { .. }
                    | LirBoundaryOp::TableFill { .. }
                    | LirBoundaryOp::TableCopy { .. }
                    | LirBoundaryOp::MemoryInit { .. }
                    | LirBoundaryOp::DataDrop { .. }
                    | LirBoundaryOp::TableInit { .. }
                    | LirBoundaryOp::ElemDrop { .. } => {
                        lower.lower_runtime(boundary, sidecar)?;
                    }
                    LirBoundaryOp::CallExternal {
                        func_idx,
                        args,
                        results,
                    } => {
                        lower.lower_call_external(*func_idx, *args, *results, sidecar)?;
                    }
                    LirBoundaryOp::CallInternal {
                        callee,
                        args,
                        results,
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
                        lower.begin_continuation_block()?;
                    }
                    LirBoundaryOp::CallIndirect { .. } => {
                        let &LirBoundaryOp::CallIndirect {
                            type_idx,
                            table_idx,
                            index_slot,
                            args,
                            results,
                        } = boundary
                        else {
                            unreachable!("matched call_indirect boundary");
                        };
                        lower.ensure_no_live_values(
                            "prepared LIR call_indirect reached native lowering with live transient SSA values; values must be published before the call",
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
                        let local_call_target_param = lower.transient_reg(0)?;

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
                                    width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                                    kind: MachineCompareKind::Ge,
                                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(lower.transient_reg(0)?),
                                    rhs: MachineValue::Reg(lower.transient_reg(2)?),
                                },
                                then_edge: crate::vm::native::ir::machine::MachineEdge {
                                    target: trap_oob,
                                    args: Vec::new(),
                                },
                                else_edge: crate::vm::native::ir::machine::MachineEdge {
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
                                    width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                                    kind: MachineCompareKind::Ge,
                                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(lower.transient_reg(2)?),
                                    rhs: MachineValue::Reg(lower.transient_reg(1)?),
                                },
                                then_edge: crate::vm::native::ir::machine::MachineEdge {
                                    target: trap_invalid_ref,
                                    args: Vec::new(),
                                },
                                else_edge: crate::vm::native::ir::machine::MachineEdge {
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
                                    width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                                    kind: MachineCompareKind::Ne,
                                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(lower.transient_reg(0)?),
                                    rhs: MachineValue::Reg(lower.transient_reg(2)?),
                                },
                                then_edge: crate::vm::native::ir::machine::MachineEdge {
                                    target: trap_type,
                                    args: Vec::new(),
                                },
                                else_edge: crate::vm::native::ir::machine::MachineEdge {
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
                                    width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                                    kind: MachineCompareKind::Eq,
                                    sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                                    lhs: MachineValue::Reg(lower.transient_reg(0)?),
                                    rhs: MachineValue::Imm64(function_kind::LOCAL as u64),
                                },
                                then_edge: crate::vm::native::ir::machine::MachineEdge {
                                    target: local_call,
                                    args: vec![MachineValue::Reg(lower.transient_reg(2)?)],
                                },
                                else_edge: crate::vm::native::ir::machine::MachineEdge {
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
                            vec![local_call_target_param],
                            build_call_indirect_local_block(&mut lower, args)?,
                            MachineTerminator::CallIndirect {
                                callee_target: MachineValue::Reg(local_call_target_param),
                                callee_frame_base: lower.transient_reg(1)?,
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
                            MachineTerminator::Jump(crate::vm::native::ir::machine::MachineEdge {
                                target: continuation,
                                args: Vec::new(),
                            }),
                        )?;

                        current_block = continuation;
                        current_params = Vec::new();
                        lower.begin_continuation_block()?;
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
        entry: MachineBlockId(input.lir.entry.as_u32()),
        reg_count: regfile.reg_count(),
        blocks,
    };
    program.validate()?;

    Ok(MachineFunction {
        id: input.id,
        program,
    })
}

fn stub_machine_function(id: MachineFuncId, reg_count: u16) -> MachineFunction {
    MachineFunction {
        id,
        program: MachineProgram {
            entry: MachineBlockId(0),
            reg_count,
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
fn slot_offset_bytes(slot: crate::vm::plan::frame::FrameSlot) -> Result<i32, WasmError> {
    let bytes = i32::from(slot.0)
        .checked_mul(8)
        .ok_or_else(|| WasmError::internal("frame slot byte offset overflow".into()))?;
    Ok(bytes)
}

fn derive_return_results(program: &LirProgram) -> Result<Option<MachineFrameRegion>, WasmError> {
    let mut derived: Option<Option<MachineFrameRegion>> = None;
    for block in &program.blocks {
        let LirTerminator::Return { results } = &block.terminator else {
            continue;
        };
        let region = results.map(frame_span_region);
        match derived {
            None => derived = Some(region),
            Some(current) if current == region => {}
            Some(_) => {
                return Err(WasmError::internal(
                    "prepared LIR uses inconsistent return result spans across blocks".into(),
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
    params: Vec<MachineReg>,
    ops: Vec<crate::vm::native::ir::machine::MachineInst>,
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

#[inline]
fn target_param_regs(count: usize, regfile: &MachineRegFile) -> Result<Vec<MachineReg>, WasmError> {
    let mut regs = Vec::with_capacity(count);
    for index in 0..count {
        regs.push(regfile.transient(index).ok_or_else(|| {
            WasmError::internal("target params exceed transient register budget".into())
        })?);
    }
    Ok(regs)
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

fn emit_call_indirect_bounds_check_setup(
    lower: &mut BlockLowerContext<'_>,
    table_idx: u32,
    index_slot: crate::vm::plan::frame::FrameSlot,
) -> Result<(), WasmError> {
    let index = lower.transient_reg(0)?;
    let table_views = lower.transient_reg(1)?;
    let table_len = lower.transient_reg(2)?;
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            dst: index,
            addr: lower.frame_addr(index_slot)?,
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            dst: table_views,
            addr: lower.runtime_addr(ctx_offset::TABLE_VIEWS_BASE),
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        },
    });
    lower.emit_machine_inst(MachineInst {
        kind: MachineInstKind::Load {
            dst: table_len,
            addr: indexed_const_addr(
                table_views,
                table_idx,
                core::mem::size_of::<crate::vm::native::runtime::context::NativeTableView>(),
                crate::vm::native::runtime::context::table_view_offset::ELEMENTS_LEN,
            )?,
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        },
    });
    Ok(())
}

fn build_call_indirect_checked_block(
    lower: &BlockLowerContext<'_>,
    table_idx: u32,
    index_slot: crate::vm::plan::frame::FrameSlot,
) -> Result<Vec<MachineInst>, WasmError> {
    let index = lower.transient_reg(0)?;
    let table_base = lower.transient_reg(1)?;
    let func_idx = lower.transient_reg(2)?;
    Ok(vec![
        MachineInst {
            kind: MachineInstKind::Load {
                dst: index,
                addr: lower.frame_addr(index_slot)?,
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                dst: table_base,
                addr: lower.runtime_addr(ctx_offset::TABLE_VIEWS_BASE),
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                dst: table_base,
                addr: indexed_const_addr(
                    table_base,
                    table_idx,
                    core::mem::size_of::<crate::vm::native::runtime::context::NativeTableView>(),
                    crate::vm::native::runtime::context::table_view_offset::ELEMENTS_BASE,
                )?,
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Mul,
                dst: index,
                lhs: MachineValue::Reg(index),
                rhs: MachineValue::Imm64(core::mem::size_of::<crate::vm::value::RefHandle>() as u64),
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                dst: table_base,
                lhs: MachineValue::Reg(table_base),
                rhs: MachineValue::Reg(index),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                dst: func_idx,
                addr: crate::vm::native::ir::machine::MachineAddr {
                    base: table_base,
                    offset: 0,
                },
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Store {
                addr: lower.frame_addr(index_slot)?,
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(func_idx),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                dst: table_base,
                addr: lower.runtime_addr(ctx_offset::FUNCTION_VIEWS_LEN),
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        },
    ])
}

fn build_call_indirect_type_check_block(
    lower: &BlockLowerContext<'_>,
    expected_type_idx: u32,
    index_slot: crate::vm::plan::frame::FrameSlot,
) -> Result<Vec<MachineInst>, WasmError> {
    let actual_type = lower.transient_reg(0)?;
    let function_views = lower.transient_reg(1)?;
    let scaled_index = lower.transient_reg(2)?;
    let expected_type = lower.transient_reg(2)?;
    let mut ops = dynamic_function_view_load(
        lower,
        index_slot,
        scaled_index,
        function_views,
        scaled_index,
        function_view_offset::TYPE_CANON,
        MachineMemWidth::U32,
        MachineLoadExtension::ZeroExtend,
        actual_type,
    )?;
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            dst: function_views,
            addr: lower.runtime_addr(ctx_offset::TYPE_CANON_BASE),
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
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
    index_slot: crate::vm::plan::frame::FrameSlot,
) -> Result<Vec<MachineInst>, WasmError> {
    let kind = lower.transient_reg(0)?;
    let function_views = lower.transient_reg(1)?;
    let scaled_index = lower.transient_reg(2)?;
    let local_target = lower.transient_reg(2)?;
    let mut ops = dynamic_function_view_load(
        lower,
        index_slot,
        scaled_index,
        function_views,
        scaled_index,
        function_view_offset::KIND,
        MachineMemWidth::U32,
        MachineLoadExtension::ZeroExtend,
        kind,
    )?;
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            dst: local_target,
            addr: crate::vm::native::ir::machine::MachineAddr {
                base: function_views,
                offset: function_view_offset::LOCAL_TARGET as i32,
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
    let callee_frame_base = lower.transient_reg(1)?;

    let ops = vec![MachineInst {
        kind: MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::MachineIntWidth::I64,
            op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
            dst: callee_frame_base,
            lhs: MachineValue::Reg(lower.frame_base_reg()),
            rhs: MachineValue::Imm64(slot_offset_bytes(crate::vm::plan::frame::FrameSlot(
                lower.current_runtime().total_frame_slots,
            ))? as u64),
        },
    }];
    // By the time the local branch reaches this block, the dispatch path has
    // already resolved and validated the local callee target above MachineIR.
    // The remaining dynamic work below MachineIR is just the local-call
    // transfer mechanics for that resolved target.
    lower.emit_machine_ops(ops);
    lower.copy_call_args_to_frame(args, callee_frame_base)?;
    Ok(lower.take_ops())
}

fn dynamic_function_view_load(
    lower: &BlockLowerContext<'_>,
    index_slot: crate::vm::plan::frame::FrameSlot,
    func_idx_dst: MachineReg,
    base_reg: MachineReg,
    scaled_index_reg: MachineReg,
    field_offset: u32,
    field_width: MachineMemWidth,
    field_extension: MachineLoadExtension,
    dst: MachineReg,
) -> Result<Vec<MachineInst>, WasmError> {
    Ok(vec![
        MachineInst {
            kind: MachineInstKind::Load {
                dst: func_idx_dst,
                addr: lower.frame_addr(index_slot)?,
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::Move {
                dst: scaled_index_reg,
                src: MachineValue::Reg(func_idx_dst),
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Mul,
                dst: scaled_index_reg,
                lhs: MachineValue::Reg(scaled_index_reg),
                rhs: MachineValue::Imm64(core::mem::size_of::<NativeFunctionView>() as u64),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                dst: base_reg,
                addr: lower.runtime_addr(ctx_offset::FUNCTION_VIEWS_BASE),
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        },
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: crate::vm::native::ir::machine::MachineIntWidth::I64,
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                dst: base_reg,
                lhs: MachineValue::Reg(base_reg),
                rhs: MachineValue::Reg(scaled_index_reg),
            },
        },
        MachineInst {
            kind: MachineInstKind::Load {
                dst,
                addr: crate::vm::native::ir::machine::MachineAddr {
                    base: base_reg,
                    offset: field_offset as i32,
                },
                width: field_width,
                extension: field_extension,
            },
        },
    ])
}

fn indexed_const_addr(
    base: MachineReg,
    index: u32,
    stride: usize,
    field_offset: u32,
) -> Result<crate::vm::native::ir::machine::MachineAddr, WasmError> {
    let scaled = (index as u64)
        .checked_mul(stride as u64)
        .and_then(|value| value.checked_add(field_offset as u64))
        .ok_or_else(|| WasmError::internal("runtime view byte offset overflow".into()))?;
    let offset = i32::try_from(scaled)
        .map_err(|_| WasmError::internal("runtime view byte offset exceeds i32".into()))?;
    Ok(crate::vm::native::ir::machine::MachineAddr { base, offset })
}
