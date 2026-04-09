#[cfg(sf_has_debug_regions)]
use alloc::{format, string::String};
use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlockParam, MachineFloatWidth, MachineTrapKind, MachineValue,
        },
        runtime::code::CompiledNativeModule,
    },
};

use super::backend::ArchBackend;
#[cfg(sf_has_debug_regions)]
use super::types::DebugRegion;
use super::types::{FunctionArtifact, ParallelSource};

// ── compile_function ─────────────────────────────────────────────────────────

pub(crate) fn compile_function<'a, A: ArchBackend<'a>>(
    compiled: &'a CompiledNativeModule,
    function: &'a crate::vm::machine::machine_ir::MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    super::core::CompilerCore::validate_function(
        A::NAME,
        function,
        compiled.backend(),
        A::max_total_regs(),
        A::max_fp_regs(),
    )?;

    let mut b = A::new(compiled, function);
    #[cfg(sf_has_debug_regions)]
    let mut debug_regions = Vec::new();

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
    #[cfg(sf_has_debug_regions)]
    if b.core().text.len() > body_prelude_start {
        debug_regions.push(DebugRegion {
            offset: body_prelude_start,
            len: b.core().text.len() - body_prelude_start,
            label: String::from("body_prelude"),
        });
    }

    // Blocks
    let block_layout = b.core().block_layout();
    for (index, block_id) in block_layout.iter().copied().enumerate() {
        let block = b
            .core()
            .function
            .program
            .blocks
            .get(block_id.as_usize())
            .ok_or_else(|| WasmError::internal("block layout references missing block".into()))?;
        let label = b.core().block_label(block.id)?;
        b.core_mut().bind_label(label);
        #[cfg(sf_has_debug_regions)]
        let block_start = b.core().text.len();
        let fallthrough = block_layout.get(index + 1).copied();
        b.lower_block(block, fallthrough)?;
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
        .ok_or_else(|| WasmError::internal("body_local_error label is unresolved".into()))?;

    b.into_core().finish_artifact(
        internal_entry_offset,
        #[cfg(sf_has_guard_pages)]
        body_local_error_offset,
        #[cfg(sf_has_debug_regions)]
        debug_regions,
    )
}

// ── emit_parallel_moves ──────────────────────────────────────────────────────

/// Shared cycle-resolution algorithm for block-edge parallel moves.
pub(crate) fn emit_parallel_moves<'a, A: ArchBackend<'a>>(
    backend: &mut A,
    params: &[MachineBlockParam],
    args: &[MachineValue],
    arg_float_widths: &[Option<MachineFloatWidth>],
) -> Result<(), WasmError> {
    let mut pending: Vec<(MachineBlockParam, ParallelSource)> = Vec::new();
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
                    && matches!(src, ParallelSource::Reg { reg, .. } if *reg == dst)
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
