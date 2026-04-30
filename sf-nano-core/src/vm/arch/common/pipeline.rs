use crate::collections;

#[cfg(sf_has_debug_regions)]
use tracked_alloc::{format, string::String};

use crate::{
    error::WasmError,
    vm::{
        machine::{
            machine_ir::{
                MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId,
                MachineFunction, MachineProgram, MachineTrapKind, MachineValue,
            },
            MirBlockSink,
        },
        runtime::code::CodegenModuleView,
    },
};

use super::backend::ArchBackend;
#[cfg(sf_has_debug_regions)]
use super::types::DebugRegion;
use super::types::{FunctionArtifact, ParallelSource};

/// Drain a fully-built `MachineFunction` through the streaming arch
/// driver. Used by static-link paths (emulator and IR-dump builds) that
/// already have a complete `MachineFunction` in hand and need a
/// per-function `FunctionArtifact` they can later concatenate into the
/// shared executable buffer.
///
/// This is *not* a separate compilation strategy — it's the exact same
/// `compile_function_streaming` driver, fed by a producer that drains
/// the function's blocks in `block_layout()` order. The buffered MIR
/// passes have already run on `function.program` by the time this is
/// called (the caller decides whether to run them).
pub(crate) fn compile_function<'a, A: ArchBackend<'a>>(
    compiled: &'a dyn CodegenModuleView,
    mut function: MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    let entry = function.program.entry;
    let fp_widths = function.program.fp_reg_init_widths.clone();
    let id = function.id;
    let layout = block_layout(&function.program);
    let mut blocks = take_function_blocks_in_layout_order(&mut function.program, &layout)?;
    // Snapshot every block's (id, params) for pre-registration so the
    // streaming driver can detect forward-reference identity edges and
    // adjacent-block fallthroughs to match the buffered baseline.
    let pre_registered: collections::Vec<(MachineBlockId, collections::Vec<MachineBlockParam>)> =
        blocks.iter().map(|b| (b.id, b.params.clone())).collect();
    let mut iter = blocks.drain(..);
    compile_function_streaming::<A, _>(
        compiled,
        id,
        entry,
        fp_widths,
        super::text_emitter::TextEmitter::new(),
        &pre_registered,
        |sink: &mut MirBlockSink<'_>| {
            let dummy_pool = crate::vm::machine::ConstPoolBuilder::new();
            for block in iter.by_ref() {
                sink(block, &dummy_pool)?;
            }
            Ok(())
        },
    )
}

// ── compile_function_streaming ───────────────────────────────────────────────

/// Streaming arch-emit driver: the **only** function-level arch entry
/// point. Builds an arch backend around an empty-blocks
/// `MachineFunction` skeleton, runs the producer callback to feed MIR
/// blocks one at a time through `produce_blocks`, and emits each
/// block's native code immediately.
///
/// Memory model: one MIR block is alive at a time inside the
/// 1-block-lookahead window; the previous block is dropped as soon as
/// the next one arrives. The lookahead exists so the previous block's
/// terminator can be lowered with a `fallthrough = Some(next_block_id)`
/// hint that lets the arch backend elide trailing branches when the
/// next physical block is the natural successor.
///
/// **Producer protocol:**
/// - The driver lands control at the entry block by either having the
///   producer emit the entry block first (no extra branch) or by
///   emitting `b entry_label` after `body_prelude` when the first
///   produced block has a non-entry id (one branch overhead per
///   function). Middleware mode does the former and matches the
///   buffered baseline byte-for-byte; tight-budget streaming pays
///   the branch.
/// - When a downstream pass requires whole-function context (e.g.
///   `fuse_compare_branch`, the 32-bit `fuse_smull_sign_ext_across_edges`,
///   any future cross-block analysis), insert a buffering middleware
///   between the producer and this driver. The middleware accumulates
///   blocks, runs the cross-block passes on a transient
///   `MachineProgram`, then re-streams in `block_layout()` order. This
///   driver doesn't know or care which mode upstream chose — it just
///   consumes the stream.
/// - The producer that doesn't install a middleware (the tight-budget
///   path) emits blocks in an order that matches its desired physical
///   layout; this driver does not reorder.
pub(crate) fn compile_function_streaming<'a, A, F>(
    compiled: &'a dyn CodegenModuleView,
    function_id: MachineFuncId,
    entry: MachineBlockId,
    fp_reg_init_widths: collections::Vec<Option<MachineFloatWidth>>,
    text: super::text_emitter::TextEmitter,
    pre_registered_blocks: &[(MachineBlockId, collections::Vec<MachineBlockParam>)],
    produce_blocks: F,
) -> Result<FunctionArtifact, WasmError>
where
    A: ArchBackend<'a>,
    F: FnOnce(&mut MirBlockSink<'_>) -> Result<(), WasmError>,
{
    // 1. Skeleton MachineFunction. Empty blocks vec — `add_streaming_block`
    //    populates it as blocks arrive. `validate_function` does not
    //    fault on the missing entry block; the entry-no-params check is
    //    re-implied by SSA validation upstream.
    let skeleton = MachineFunction {
        id: function_id,
        program: MachineProgram {
            entry,
            fp_reg_init_widths,
            blocks: collections::Vec::new(),
        },
    };
    super::core::CompilerCore::validate_function(
        A::NAME,
        &skeleton,
        compiled.backend(),
        A::max_total_regs(),
        A::max_fp_regs(),
    )?;

    let mut b = A::new(compiled, skeleton);
    b.core_mut().text = text;

    // 1b. Pre-register block params + labels. This is the "I know all
    //     blocks already exist with these params" optimization the
    //     buffering middleware uses: with every target block's params
    //     populated up front, `is_identity_edge` can correctly detect
    //     forward-reference identity edges (no edge stub needed) and
    //     fallthrough optimization can apply to forward-arrival blocks.
    //     Tight-budget producers pass an empty slice and accept the
    //     codegen-quality cost: forward-ref edge stubs aren't elided.
    for (id, params) in pre_registered_blocks {
        b.core_mut().add_streaming_block(*id, params.clone())?;
    }

    // 2. Public entry: prologue / root caller stub / epilogue.
    b.lower_prologue();
    b.lower_root_caller_stub();
    b.lower_epilogue();

    // 3. Internal entry binding + body prelude.
    let internal_entry_label = b.core().internal_entry_label;
    b.core_mut().bind_label(internal_entry_label);
    let internal_entry_offset = b.core().text.len();
    b.lower_body_prelude();

    // 4. Drive blocks via the producer. We keep a 1-block lookahead so
    //    when a new block arrives we know the previous block's
    //    fallthrough target. On arrival each block is *registered* (params
    //    placeholder so later edge-stub drains can resolve `target.params`,
    //    label *allocated* but not yet bound). On emission (when the next
    //    block arrives or production ends) we *bind* the previous block's
    //    label at the current text offset and run begin/emit/end.
    //
    //    Entry-jump policy: when the first block the producer hands us
    //    is NOT the entry block, we emit a `b entry_label` before
    //    binding it. The middleware drains in `block_layout()` order
    //    (entry first) so this jump is elided in hosted-mode
    //    compilation, matching the buffered baseline byte-for-byte.
    //    Pure streaming producers may emit the entry block at any
    //    point in their iteration; the entry-jump pays one branch per
    //    function in that case.
    {
        let mut pending: Option<(MachineBlock, usize)> = None;
        let mut first_block = true;
        let mut emit_block = |block: MachineBlock,
                              _const_pool: &crate::vm::machine::ConstPoolBuilder|
         -> Result<(), WasmError> {
            let label = b
                .core_mut()
                .add_streaming_block(block.id, block.params.clone())?;
            if first_block {
                first_block = false;
                if block.id != entry {
                    let entry_label = b.core_mut().block_label(entry)?;
                    b.lower_unconditional_branch(entry_label);
                }
            }
            if let Some((prev, prev_label)) = pending.take() {
                emit_streaming_block::<A>(&mut b, prev, prev_label, Some(block.id))?;
            }
            pending = Some((block, label));
            Ok(())
        };
        let sink: &mut MirBlockSink<'_> = &mut emit_block;
        produce_blocks(sink)?;
        if let Some((last, last_label)) = pending.take() {
            emit_streaming_block::<A>(&mut b, last, last_label, None)?;
        }
    }

    // 5. Edge stubs. By drain time every target block has arrived via
    //    `add_streaming_block`, so target params are resolvable.
    let edges = core::mem::take(&mut b.core_mut().edge_stubs);
    for edge in edges {
        b.core_mut().bind_label(edge.label);
        b.core_mut().current_block = None;
        b.core_mut().current_op_index = None;
        b.core_mut().current_edge_target = Some(edge.target);
        let target_params = b
            .core()
            .function
            .program
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| WasmError::internal("edge target block missing at streaming drain"))?
            .params
            .clone();
        emit_parallel_moves::<A>(&mut b, &target_params, &edge.args, &edge.arg_float_widths)?;
        let target_label = b.core_mut().block_label(edge.target)?;
        b.lower_unconditional_branch(target_label);
        b.core_mut().current_edge_target = None;
    }

    // 6. Literal pool.
    b.lower_function_literal_pool()?;

    // 7. Tail: body_local_error, stack_overflow, deferred traps.
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

    // 8. Patch fixups.
    b.patch_fixups()?;

    #[cfg(sf_has_guard_pages)]
    let body_local_error_offset = b
        .core()
        .labels
        .get(b.core().body_local_error_label)
        .and_then(|offset| *offset)
        .ok_or_else(|| WasmError::internal("body_local_error label is unresolved"))?;

    // 9. Finish artifact. Streaming mode does not yet emit per-block
    //    debug regions; a future change can extend this if `sf_jitdump`
    //    needs them.
    b.into_core().finish_artifact(
        internal_entry_offset,
        #[cfg(sf_has_guard_pages)]
        body_local_error_offset,
        #[cfg(sf_has_debug_regions)]
        collections::Vec::new(),
    )
}

/// Bind the previously-allocated label for a buffered block at the
/// current text offset, then run the arch backend's
/// begin_block/emit_inst_at/end_block triplet to lower the block. The
/// `fallthrough` hint is forwarded to the terminator so the backend can
/// elide trailing branches when the next physical block is the natural
/// successor.
fn emit_streaming_block<'a, A: ArchBackend<'a>>(
    b: &mut A,
    block: MachineBlock,
    label: usize,
    fallthrough: Option<MachineBlockId>,
) -> Result<(), WasmError> {
    b.core_mut().bind_label(label);
    b.begin_block(&block)?;
    for (op_index, inst) in block.ops.iter().enumerate() {
        b.emit_inst_at(inst, op_index)?;
    }
    b.end_block(&block.terminator, fallthrough)?;
    // `block` is dropped at end of scope, freeing per-block heap content.
    drop(block);
    Ok(())
}

/// Fallthrough-aware block ordering for a fully-built `MachineProgram`.
/// Used by the static-link path (`compile_function`) and by the
/// buffering middleware in `vm/build.rs` to feed blocks to the
/// streaming driver in the order that maximizes natural-successor
/// fallthroughs.
///
/// Algorithm: depth-first chain extension from the entry, preferring
/// identity-mapped fallthrough edges. Unreachable blocks are appended
/// at the end so every block in `program.blocks` appears exactly once.
pub(crate) fn block_layout(program: &MachineProgram) -> collections::Vec<MachineBlockId> {
    let block_count = program.blocks.len();
    let mut order = collections::Vec::with_capacity(block_count);
    let mut seen = collections::vec![false; block_count];
    let mut worklist = collections::vec![program.entry];

    while let Some(start) = worklist.pop() {
        extend_block_trace(program, start, &mut seen, &mut order, &mut worklist);
    }
    for block in &program.blocks {
        let idx = block.id.as_usize();
        if idx >= seen.len() || seen[idx] {
            continue;
        }
        worklist.push(block.id);
        while let Some(start) = worklist.pop() {
            extend_block_trace(program, start, &mut seen, &mut order, &mut worklist);
        }
    }
    order
}

fn extend_block_trace(
    program: &MachineProgram,
    start: MachineBlockId,
    seen: &mut [bool],
    order: &mut collections::Vec<MachineBlockId>,
    worklist: &mut collections::Vec<MachineBlockId>,
) {
    let mut current = Some(start);
    while let Some(block_id) = current {
        let Some(block) = program.blocks.get(block_id.as_usize()) else {
            break;
        };
        if seen[block_id.as_usize()] {
            break;
        }
        seen[block_id.as_usize()] = true;
        order.push(block_id);

        let mut fallthrough = None;
        match &block.terminator {
            crate::vm::machine::machine_ir::MachineTerminator::Jump(edge) => {
                if is_identity_edge(program, edge.target, &edge.args) {
                    fallthrough = Some(edge.target);
                } else {
                    worklist.push(edge.target);
                }
            }
            crate::vm::machine::machine_ir::MachineTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                if is_identity_edge(program, else_edge.target, &else_edge.args) {
                    fallthrough = Some(else_edge.target);
                    worklist.push(then_edge.target);
                } else {
                    worklist.push(else_edge.target);
                    worklist.push(then_edge.target);
                }
            }
            crate::vm::machine::machine_ir::MachineTerminator::JumpTable { entries, .. } => {
                for edge in entries.iter().rev() {
                    worklist.push(edge.target);
                }
            }
            crate::vm::machine::machine_ir::MachineTerminator::Call { continuation, .. } => {
                fallthrough = Some(*continuation);
            }
            crate::vm::machine::machine_ir::MachineTerminator::TailCall { .. }
            | crate::vm::machine::machine_ir::MachineTerminator::Return
            | crate::vm::machine::machine_ir::MachineTerminator::Trap { .. } => {}
        }

        current = fallthrough.filter(|target| !seen[target.as_usize()]);
    }
}

fn is_identity_edge(
    program: &MachineProgram,
    target: MachineBlockId,
    args: &[MachineValue],
) -> bool {
    let Some(block) = program.blocks.get(target.as_usize()) else {
        return false;
    };
    if block.params.len() != args.len() {
        return false;
    }
    block
        .params
        .iter()
        .zip(args.iter())
        .all(|(param, arg)| match arg {
            MachineValue::Reg(r) | MachineValue::ReservedReg(r) => *r == param.reg,
            MachineValue::Imm64(_) => false,
        })
}

/// Take blocks out of `program` in the order specified by `layout`,
/// returning a `Vec<MachineBlock>` ready to be drained block-by-block
/// through the streaming driver. The original program slots are left
/// as `Return`-terminated tombstones with cloned params (so any
/// residual edge-stub params lookups work).
fn take_function_blocks_in_layout_order(
    program: &mut MachineProgram,
    layout: &[MachineBlockId],
) -> Result<collections::Vec<MachineBlock>, WasmError> {
    let mut out = collections::Vec::with_capacity(layout.len());
    for &id in layout {
        let idx = id.as_usize();
        let slot = program
            .blocks
            .get_mut(idx)
            .ok_or_else(|| WasmError::internal("block layout references missing block"))?;
        let placeholder = MachineBlock {
            id: slot.id,
            params: slot.params.clone(),
            ops: collections::Vec::new(),
            terminator: crate::vm::machine::machine_ir::MachineTerminator::Return,
        };
        out.push(core::mem::replace(slot, placeholder));
    }
    Ok(out)
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
