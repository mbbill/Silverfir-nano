use crate::collections;

#[cfg(sf_has_debug_regions)]
use tracked_alloc::{format, string::String};

use crate::{
    error::WasmError,
    vm::{
        jit::machine::machine_ir::{
            MachineAddr, MachineArgSrc, MachineBlockParam, MachineCallArgs, MachineCallLaneArg,
            MachineFloatWidth, MachineFuncId, MachineFunction, MachineInst, MachineInstKind,
            MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner, MachineStorageType,
            MachineTrapKind, MachineValue, MACHINE_FP_REG,
        },
        jit::middle::frame::FrameSlot,
        runtime::code::CodegenModuleView,
        runtime::code_buf::CodeBuffer,
    },
};

use super::core::{CompilerCore, FunctionBody};
#[cfg(sf_has_debug_regions)]
use super::types::DebugRegion;
use super::types::{FunctionArtifact, ParallelSource};
use super::{backend::ArchBackend, template::TemplateBackend};

// ── compile_function ─────────────────────────────────────────────────────────

fn compile_function_impl<'a, A: ArchBackend<'a>>(
    compiled: &'a dyn CodegenModuleView,
    function: &'a MachineFunction,
    text: super::text_emitter::TextEmitter,
) -> Result<FunctionArtifact, WasmError> {
    super::core::CompilerCore::validate_function(
        A::NAME,
        function,
        compiled.runtime_for(function.id),
        compiled.backend(),
        A::max_total_regs(),
        A::max_fp_regs(),
    )?;

    let mut core = CompilerCore::new(compiled, FunctionBody::Mir(function));
    core.text = text;
    let mut b = A::new(core);
    #[cfg(sf_has_debug_regions)]
    let mut debug_regions = collections::Vec::new();

    // Public entry (C ABI lands here):
    //
    //   [prologue]              — save C callee-saved, set up fixed regs
    //   [root caller stub]      — push root call record, bl internal_entry
    //   [epilogue]              — restore C callee-saved, ret
    //
    // After the bl, control eventually returns through the body's unified
    // return mechanism with C_RET0 = 0 (success) or non-zero (trap kind).
    // The epilogue preserves C_RET0 and rets to the C caller.
    #[cfg(sf_has_debug_regions)]
    let public_entry_start = b.core().text.len();
    b.lower_prologue();
    b.lower_root_caller_stub();
    b.lower_epilogue();
    #[cfg(sf_has_debug_regions)]
    debug_regions.push(DebugRegion {
        offset: public_entry_start,
        len: b.core().text.len() - public_entry_start,
        label: String::from("public_entry"),
    });

    // Internal entry (local SF→SF calls and the public stub's bl land here):
    //
    //   [internal_entry_label]
    //   [body prelude]          — per-arch non-leaf setup (link save /
    //                             alignment shim). Always-on in 1A.
    //   [body blocks]
    //
    // Direct call patches resolve against `internal_entry_label`, NOT
    // against "the byte right after lower_prologue".
    let internal_entry_label = b.core().internal_entry_label;
    b.core_mut().bind_label(internal_entry_label);
    let internal_entry_offset = b.core().text.len();
    #[cfg(sf_has_debug_regions)]
    let body_prelude_start = internal_entry_offset;
    b.lower_body_prelude();
    #[cfg(sf_has_guard_pages)]
    emit_body_stack_probe(&mut b)?;
    #[cfg(sf_has_debug_regions)]
    if b.core().text.len() > body_prelude_start {
        debug_regions.push(DebugRegion {
            offset: body_prelude_start,
            len: b.core().text.len() - body_prelude_start,
            label: String::from("body_prelude"),
        });
    }

    // Blocks
    let block_layout = b.core().block_layout()?;
    for (index, block_id) in block_layout.iter().copied().enumerate() {
        let block = b
            .core()
            .mir_blocks()?
            .get(block_id.as_usize())
            .ok_or_else(|| WasmError::internal("block layout references missing block"))?;
        let label = b.core().block_label(block.id)?;
        b.core_mut().bind_label(label);
        #[cfg(sf_has_debug_regions)]
        let block_start = b.core().text.len();
        b.begin_block(block)?;
        for (op_index, inst) in block.ops.iter().enumerate() {
            b.emit_inst_at(inst, op_index)?;
        }
        let fallthrough = block_layout.get(index + 1).copied();
        b.end_block(&block.terminator, fallthrough)?;
        #[cfg(sf_has_debug_regions)]
        {
            let block_end = b.core().text.len();
            debug_regions.push(DebugRegion {
                offset: block_start,
                len: block_end - block_start,
                label: format!("b{}", block.id.0),
            });
        }
    }

    // Edge stubs (take, not clone)
    #[cfg(sf_has_debug_regions)]
    let edge_start = b.core().text.len();
    let edges = core::mem::take(&mut b.core_mut().edge_stubs);
    for edge in edges {
        b.core_mut().bind_label(edge.label);
        b.core_mut().current_block = None;
        b.core_mut().current_op_index = None;
        b.core_mut().current_edge_target = Some(edge.target);
        emit_parallel_moves::<A>(&mut b, &edge.params, &edge.args, &edge.arg_float_widths)?;
        let target_label = b.core().block_label(edge.target)?;
        b.lower_unconditional_branch(target_label);
        b.core_mut().current_edge_target = None;
    }
    #[cfg(sf_has_debug_regions)]
    {
        let edge_end = b.core().text.len();
        if edge_end > edge_start {
            debug_regions.push(DebugRegion {
                offset: edge_start,
                len: edge_end - edge_start,
                label: String::from("edges"),
            });
        }
    }

    // Per-function literal pool. Backends that accumulate deferred literals
    // (e.g. arm64's per-call patchable callee addresses) flush them here so
    // they sit at end-of-body but inside the pc-relative load range of any
    // call site. Default impl is a no-op.
    #[cfg(sf_has_debug_regions)]
    let pool_start = b.core().text.len();
    b.lower_function_literal_pool()?;
    #[cfg(sf_has_debug_regions)]
    {
        let pool_end = b.core().text.len();
        if pool_end > pool_start {
            debug_regions.push(DebugRegion {
                offset: pool_start,
                len: pool_end - pool_start,
                label: String::from("literal_pool"),
            });
        }
    }

    // Tail: body_local_error_label, stack_overflow_label, deferred traps.
    //
    // The new tail does NOT contain a `return_ok_label` or
    // `return_error_label` — the body's success-path Return is lowered
    // inline at every Return terminator (sets `C_RET0 = 0`, native return),
    // and the body's error-path tail is `body_local_error_label`, which
    // every trap stub and post-BL status check branches to.
    #[cfg(sf_has_debug_regions)]
    let tail_start = b.core().text.len();

    let body_local_error_label = b.core().body_local_error_label;
    b.core_mut().bind_label(body_local_error_label);
    b.lower_body_local_error_tail();

    let stack_overflow_label = b.core().stack_overflow_label;
    b.core_mut().bind_label(stack_overflow_label);
    b.lower_trap(MachineTrapKind::StackOverflow);

    let deferred = core::mem::take(&mut b.core_mut().deferred_traps);
    for (label, kind) in deferred {
        b.core_mut().bind_label(label);
        b.lower_trap(kind);
    }

    #[cfg(sf_has_debug_regions)]
    {
        let tail_end = b.core().text.len();
        if tail_end > tail_start {
            debug_regions.push(DebugRegion {
                offset: tail_start,
                len: tail_end - tail_start,
                label: String::from("tail"),
            });
        }
    }

    // Patch fixups
    b.patch_fixups()?;

    #[cfg(sf_has_guard_pages)]
    let body_local_error_offset = b
        .core()
        .labels
        .get(b.core().body_local_error_label)
        .and_then(|offset| *offset)
        .ok_or_else(|| WasmError::internal("body_local_error label is unresolved"))?;

    b.into_core().finish_artifact(
        internal_entry_offset,
        #[cfg(sf_has_guard_pages)]
        body_local_error_offset,
        #[cfg(sf_has_debug_regions)]
        debug_regions,
    )
}

pub(crate) fn compile_function<'a, A: ArchBackend<'a>>(
    compiled: &'a dyn CodegenModuleView,
    function: &'a MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    compile_function_impl::<A>(compiled, function, super::text_emitter::TextEmitter::new())
}

pub(crate) fn compile_function_into_buffer<'a, A: ArchBackend<'a>>(
    compiled: &'a dyn CodegenModuleView,
    function: &'a MachineFunction,
    executable: &mut CodeBuffer,
) -> Result<FunctionArtifact, WasmError> {
    compile_function_impl::<A>(
        compiled,
        function,
        super::text_emitter::TextEmitter::new_in_code_buffer(executable),
    )
}

pub(crate) fn compile_template_function_into_buffer<'a, A, F>(
    compiled: &'a dyn CodegenModuleView,
    func_id: MachineFuncId,
    executable: &mut CodeBuffer,
    emit_body: F,
) -> Result<FunctionArtifact, WasmError>
where
    A: TemplateBackend<'a>,
    F: FnOnce(&mut A) -> Result<(), WasmError>,
{
    let mut core = CompilerCore::new(compiled, FunctionBody::Template { func_id });
    core.text = super::text_emitter::TextEmitter::new_in_code_buffer(executable);
    let mut b = A::new(core);
    #[cfg(sf_has_debug_regions)]
    let mut debug_regions = collections::Vec::new();

    #[cfg(sf_has_debug_regions)]
    let public_entry_start = b.core().text.len();
    b.lower_prologue();
    b.lower_root_caller_stub();
    b.lower_epilogue();
    #[cfg(sf_has_debug_regions)]
    debug_regions.push(DebugRegion {
        offset: public_entry_start,
        len: b.core().text.len() - public_entry_start,
        label: String::from("public_entry"),
    });

    let internal_entry_label = b.core().internal_entry_label;
    b.core_mut().bind_label(internal_entry_label);
    let internal_entry_offset = b.core().text.len();
    #[cfg(sf_has_debug_regions)]
    let body_start = internal_entry_offset;
    b.lower_body_prelude();
    #[cfg(sf_has_guard_pages)]
    emit_body_stack_probe(&mut b)?;
    emit_body(&mut b)?;
    #[cfg(sf_has_debug_regions)]
    debug_regions.push(DebugRegion {
        offset: body_start,
        len: b.core().text.len() - body_start,
        label: String::from("template_body"),
    });

    #[cfg(sf_has_debug_regions)]
    let pool_start = b.core().text.len();
    b.lower_function_literal_pool()?;
    #[cfg(sf_has_debug_regions)]
    {
        let pool_end = b.core().text.len();
        if pool_end > pool_start {
            debug_regions.push(DebugRegion {
                offset: pool_start,
                len: pool_end - pool_start,
                label: String::from("literal_pool"),
            });
        }
    }

    #[cfg(sf_has_debug_regions)]
    let tail_start = b.core().text.len();
    let body_local_error_label = b.core().body_local_error_label;
    b.core_mut().bind_label(body_local_error_label);
    b.lower_body_local_error_tail();

    let stack_overflow_label = b.core().stack_overflow_label;
    b.core_mut().bind_label(stack_overflow_label);
    b.lower_trap(MachineTrapKind::StackOverflow);

    let deferred = core::mem::take(&mut b.core_mut().deferred_traps);
    for (label, kind) in deferred {
        b.core_mut().bind_label(label);
        b.lower_trap(kind);
    }
    #[cfg(sf_has_debug_regions)]
    {
        let tail_end = b.core().text.len();
        if tail_end > tail_start {
            debug_regions.push(DebugRegion {
                offset: tail_start,
                len: tail_end - tail_start,
                label: String::from("tail"),
            });
        }
    }

    b.patch_fixups()?;

    #[cfg(sf_has_guard_pages)]
    let body_local_error_offset = b
        .core()
        .labels
        .get(b.core().body_local_error_label)
        .and_then(|offset| *offset)
        .ok_or_else(|| WasmError::internal("body_local_error label is unresolved"))?;

    b.into_core().finish_artifact(
        internal_entry_offset,
        #[cfg(sf_has_guard_pages)]
        body_local_error_offset,
        #[cfg(sf_has_debug_regions)]
        debug_regions,
    )
}

#[cfg(sf_has_guard_pages)]
fn emit_body_stack_probe<'a, A: ArchBackend<'a>>(b: &mut A) -> Result<(), WasmError> {
    let Some(runtime) = b.core().current_runtime() else {
        return Ok(());
    };
    let total_slots = runtime.total_frame_slots;
    if total_slots == 0 {
        return Ok(());
    }
    let offset = frame_slot_byte_offset(FrameSlot(total_slots - 1), 0)?;
    b.lower_stack_probe(MachineAddr {
        base: MACHINE_FP_REG,
        offset,
    })
}

// ── emit_parallel_moves ──────────────────────────────────────────────────────

/// Shared cycle-resolution algorithm for block-edge parallel moves.
pub(crate) fn emit_parallel_moves<'a, A: ArchBackend<'a>>(
    backend: &mut A,
    params: &[MachineBlockParam],
    args: &[MachineValue],
    arg_float_widths: &[Option<MachineFloatWidth>],
) -> Result<(), WasmError> {
    let mut pending: collections::Vec<(MachineBlockParam, ParallelSource)> =
        collections::Vec::new();
    for ((&dst, &arg), &float_width) in params.iter().zip(args.iter()).zip(arg_float_widths.iter())
    {
        let src = match arg {
            MachineValue::Reg(reg) => ParallelSource::Reg { reg, float_width },
            MachineValue::ReservedReg(reg) => ParallelSource::ReservedReg(reg),
            MachineValue::Imm64(value) => ParallelSource::Imm(value),
        };
        if matches!(src, ParallelSource::Reg { reg, .. } if reg == dst.reg)
            || matches!(src, ParallelSource::ReservedReg(reg) if reg == dst.reg)
        {
            continue;
        }
        pending.push((dst, src));
    }

    while !pending.is_empty() {
        // Find a ready move (destination not used as source by anyone else).
        let mut ready = None;
        for index in 0..pending.len() {
            let dst = pending[index].0.reg;
            let blocked = pending.iter().enumerate().any(|(other_index, (_, src))| {
                other_index != index
                    && matches!(src, &ParallelSource::Reg { reg, .. } if reg == dst)
            });
            if !blocked {
                ready = Some(index);
                break;
            }
        }

        if let Some(index) = ready {
            let (dst, src) = pending.remove(index);
            // Free the scratch after the temp is consumed by emit_source_move.
            let free_scratch = match src {
                ParallelSource::GpTemp(id) => Some((id, false)),
                ParallelSource::FpTemp(id, _) => Some((id, true)),
                _ => None,
            };
            backend.lower_source_move(dst, src)?;
            match free_scratch {
                Some((id, false)) => backend.free_gp_scratch(id),
                Some((id, true)) => backend.free_fp_scratch(id),
                None => {}
            }
            continue;
        }

        // Cycle detected — allocate a scratch temp and break the cycle.
        let (dst, src) = pending.remove(0);
        let ParallelSource::Reg {
            reg: src_reg,
            float_width,
        } = src
        else {
            if let ParallelSource::ReservedReg(_) = src {
                continue;
            }
            backend.lower_source_move(dst, src)?;
            continue;
        };

        if dst.ty.is_fp() {
            let scratch_id = backend.alloc_fp_scratch();
            backend.lower_fp_cycle_break(dst, src_reg, float_width, scratch_id)?;
            for (_, source) in pending.iter_mut() {
                if matches!(*source, ParallelSource::Reg { reg, .. } if reg == dst.reg) {
                    *source = ParallelSource::FpTemp(
                        scratch_id,
                        dst.ty.float_width().expect("FP temp width"),
                    );
                }
            }
        } else {
            let scratch_id = backend.alloc_gp_scratch();
            backend.lower_gp_cycle_break(dst.reg, src_reg, scratch_id)?;
            for (_, source) in pending.iter_mut() {
                if matches!(*source, ParallelSource::Reg { reg, .. } if reg == dst.reg) {
                    *source = ParallelSource::GpTemp(scratch_id);
                }
            }
        }
    }
    Ok(())
}

// ── emit_call_arg_lanes ─────────────────────────────────────────────────────

/// Populate abstract internal-call lanes immediately before a native
/// wasm-to-wasm call. Register sources are handled as a true parallel move so
/// swaps do not bounce through the wasm frame; frame sources are loaded after
/// the register shuffle so a frame load cannot clobber a source register that
/// another lane still needs.
pub(crate) fn emit_call_arg_lanes<'a, A: ArchBackend<'a>>(
    backend: &mut A,
    args: &MachineCallArgs,
) -> Result<(), WasmError> {
    let mut params = collections::Vec::new();
    let mut values = collections::Vec::new();
    let mut float_widths = collections::Vec::new();
    let mut frame_loads = collections::Vec::new();

    for arg in &args.lane_args {
        match arg {
            MachineCallLaneArg::Gp { lane, src, ty, .. } => {
                let dst = backend.core().gp_arg_lane_reg(*lane);
                enqueue_call_arg_source(
                    dst,
                    *ty,
                    None,
                    *src,
                    &mut params,
                    &mut values,
                    &mut float_widths,
                    &mut frame_loads,
                );
            }
            MachineCallLaneArg::GpPair {
                lo_lane,
                hi_lane,
                src,
                ..
            } => {
                let lo = backend.core().gp_arg_lane_reg(*lo_lane);
                let hi = backend.core().gp_arg_lane_reg(*hi_lane);
                enqueue_call_arg_source(
                    lo,
                    MachineStorageType::GpWord,
                    None,
                    src.lo,
                    &mut params,
                    &mut values,
                    &mut float_widths,
                    &mut frame_loads,
                );
                enqueue_call_arg_source(
                    hi,
                    MachineStorageType::GpWord,
                    None,
                    src.hi,
                    &mut params,
                    &mut values,
                    &mut float_widths,
                    &mut frame_loads,
                );
            }
            MachineCallLaneArg::Fp { lane, src, ty, .. } => {
                let dst = backend.core().fp_arg_lane_reg(*lane);
                enqueue_call_arg_source(
                    dst,
                    *ty,
                    ty.float_width(),
                    *src,
                    &mut params,
                    &mut values,
                    &mut float_widths,
                    &mut frame_loads,
                );
            }
        }
    }

    emit_parallel_moves::<A>(backend, &params, &values, &float_widths)?;
    for load in frame_loads {
        emit_call_arg_frame_load::<A>(backend, load)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct CallArgFrameLoad {
    dst: MachineReg,
    ty: MachineStorageType,
    slot: FrameSlot,
    byte_offset: i8,
}

fn enqueue_call_arg_source(
    dst: MachineReg,
    ty: MachineStorageType,
    float_width: Option<MachineFloatWidth>,
    src: MachineArgSrc,
    params: &mut collections::Vec<MachineBlockParam>,
    values: &mut collections::Vec<MachineValue>,
    float_widths: &mut collections::Vec<Option<MachineFloatWidth>>,
    frame_loads: &mut collections::Vec<CallArgFrameLoad>,
) {
    match src {
        MachineArgSrc::Reg(reg) => {
            if reg == dst {
                return;
            }
            params.push(MachineBlockParam {
                reg: dst,
                ty,
                owner: MachineRegOwner::LinearValue,
            });
            values.push(MachineValue::Reg(reg));
            float_widths.push(float_width);
        }
        MachineArgSrc::FrameSlot(slot) => frame_loads.push(CallArgFrameLoad {
            dst,
            ty,
            slot,
            byte_offset: 0,
        }),
        MachineArgSrc::FrameSlotOffset { slot, byte_offset } => {
            frame_loads.push(CallArgFrameLoad {
                dst,
                ty,
                slot,
                byte_offset,
            });
        }
    }
}

fn emit_call_arg_frame_load<'a, A: ArchBackend<'a>>(
    backend: &mut A,
    load: CallArgFrameLoad,
) -> Result<(), WasmError> {
    let width = match load.ty {
        MachineStorageType::GpWord => {
            if backend.core().compiled.backend().gp_unit_bytes == 4 {
                MachineMemWidth::U32
            } else {
                MachineMemWidth::U64
            }
        }
        MachineStorageType::GpI64 | MachineStorageType::Fp64 => MachineMemWidth::U64,
        MachineStorageType::Fp32 => MachineMemWidth::U32,
        MachineStorageType::V128 => MachineMemWidth::U64,
    };
    backend.lower_inst(&MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: load.ty,
            dst: load.dst,
            addr: MachineAddr {
                base: MACHINE_FP_REG,
                offset: frame_slot_byte_offset(load.slot, load.byte_offset)?,
            },
            width,
            extension: MachineLoadExtension::None,
        },
    })
}

fn frame_slot_byte_offset(slot: FrameSlot, byte_offset: i8) -> Result<i32, WasmError> {
    i32::from(slot.0)
        .checked_mul(8)
        .and_then(|base| base.checked_add(i32::from(byte_offset)))
        .ok_or_else(|| WasmError::internal("call argument frame slot offset overflow"))
}
