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

use alloc::vec::Vec;

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
                MachineBlock, MachineBlockId, MachineConstData, MachineFunction, MachineFuncId,
                MachineModule, MachineProgram, MachineReg, MachineTerminator,
            },
            runtime::{
                MachineCallLinkLayout, MachineExternBinding, MachineFrameRegion,
                MachineFunctionRuntime, MachineRuntimeContract,
            },
        },
        plan::frame::FrameLayoutPlan,
    },
};

use self::{
    context::BlockLowerContext,
    regfile::MachineRegFile,
    sidecar::SidecarBuilder,
};

/// One prepared function ready for LIR -> MachineIR lowering.
#[derive(Clone, Copy, Debug)]
pub struct LowerFunctionInput<'a> {
    pub id: MachineFuncId,
    pub frame: FrameLayoutPlan,
    pub lir: &'a LirProgram,
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

    let mut functions = Vec::with_capacity(input.functions.len());
    let mut function_runtime = Vec::with_capacity(input.functions.len());
    let mut sidecar = SidecarBuilder::new();
    for function in input.functions {
        validate_program(function.lir)?;
        function_runtime.push(lower_function_runtime(*function, call_link)?);
    }
    for function in input.functions {
        functions.push(lower_function(
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
    let return_results = derive_return_results(input.lir)?;

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
    let caller_runtime = runtime
        .get(input.id.0 as usize)
        .copied()
        .ok_or_else(|| WasmError::internal("machine runtime metadata missing for function".into()))?;
    let original_block_count = input.lir.blocks.len();
    let mut original_blocks = alloc::vec![None; original_block_count];
    let mut continuation_blocks = Vec::new();
    let mut next_continuation = original_block_count as u32;

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
                LirInstKind::Boundary(boundary) => match boundary {
                    LirBoundaryOp::MemoryGrow { .. }
                    | LirBoundaryOp::TableGrow { .. }
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
                        let continuation = MachineBlockId(next_continuation);
                        next_continuation += 1;
                        let terminator = lower.lower_call_internal(
                            *callee,
                            *args,
                            *results,
                            continuation,
                        )?;
                        push_lowered_block(
                            current_block,
                            &mut original_blocks,
                            &mut continuation_blocks,
                            current_params,
                            lower.take_ops(),
                            terminator,
                        )?;
                        current_block = continuation;
                        current_params = Vec::new();
                        lower.begin_continuation_block()?;
                    }
                    LirBoundaryOp::CallIndirect { .. } => {
                        return Err(WasmError::internal(
                            "call_indirect lowering is not implemented yet in LIR -> MachineIR"
                                .into(),
                        ));
                    }
                },
                _ => lower.lower_inst(inst)?,
            }
        }

        let terminator = lower.lower_terminator()?;
        push_lowered_block(
            current_block,
            &mut original_blocks,
            &mut continuation_blocks,
            current_params,
            lower.take_ops(),
            terminator,
        )?;
    }

    let mut blocks = Vec::with_capacity(original_block_count + continuation_blocks.len());
    for (index, block) in original_blocks.into_iter().enumerate() {
        blocks.push(block.ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "machine lowering did not produce original block {}",
                index
            ))
        })?);
    }
    blocks.extend(continuation_blocks);

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

#[inline]
fn slot_offset_bytes(slot: crate::vm::plan::frame::FrameSlot) -> Result<i32, WasmError> {
    let bytes = i32::from(slot.0)
        .checked_mul(8)
        .ok_or_else(|| WasmError::internal("frame slot byte offset overflow".into()))?;
    Ok(bytes)
}

fn derive_return_results(program: &LirProgram) -> Result<Option<MachineFrameRegion>, WasmError> {
    let mut derived: Option<MachineFrameRegion> = None;
    for block in &program.blocks {
        let LirTerminator::Return { results } = &block.terminator else {
            continue;
        };
        let region = results.map(frame_span_region);
        match derived {
            None => derived = region,
            Some(current) if region == Some(current) => {}
            Some(_) => {
                return Err(WasmError::internal(
                    "prepared LIR uses inconsistent return result spans across blocks".into(),
                ));
            }
        }
    }
    Ok(derived)
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
        regs.push(regfile
            .transient(index)
            .ok_or_else(|| WasmError::internal("target params exceed transient register budget".into()))?);
    }
    Ok(regs)
}
