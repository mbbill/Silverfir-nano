use crate::collections;
use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth,
        MachineInst, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineParamLoc,
        MachineReg, MachineRegOwner, MachineStorageType, MachineTerminator, MachineTrapKind,
        MACHINE_FP_REG,
    },
};

use super::core::CompilerCore;
use super::types::ParallelSource;

use crate::vm::runtime::code_buf::CodeBuffer;

/// Trait that each architecture backend implements.
///
/// ## Naming convention
///
/// - `lower_*` — translate MachineIR into encoded instructions (lowering).
/// - `emit_nop_padding` — the one exception: writes raw padding bytes into
///   the code buffer, not MachineIR lowering.
///
/// Backends must follow the register-ownership and boundary rules documented in
/// `vm/arch/abi.md`. In particular: ordinary lowering may touch only explicit
/// operands, fixed regs when semantically required, and backend-owned temp regs
/// that were explicitly claimed from the relevant ownership tracker.
///
/// The backend MUST have a `pub core: CompilerCore<'a>` field.
/// Pipeline functions access it directly for shared state.
pub(crate) trait ArchBackend<'a>: Sized {
    /// Architecture name for error messages (e.g. "arm64").
    const NAME: &'static str;

    // ── Capacity queries (no &self — pure arch facts) ────────────────────

    fn max_total_regs() -> usize;
    fn max_fp_regs() -> usize;

    // ── Construction ─────────────────────────────────────────────────────

    fn new(core: CompilerCore<'a>) -> Self;

    /// Access the shared CompilerCore.
    fn core(&self) -> &CompilerCore<'a>;
    fn core_mut(&mut self) -> &mut CompilerCore<'a>;

    /// Consume the backend and return the owned CompilerCore.
    fn into_core(self) -> CompilerCore<'a>;

    // ── Prologue / epilogue / tail ───────────────────────────────────────
    //
    // The pipeline emits these in order at the start of every function:
    //   1. `lower_prologue`           — C-ABI public entry: save C
    //                                    callee-saved regs, set up MachineIR
    //                                    fixed registers from C arg regs.
    //   2. `lower_root_caller_stub`   — public-entry caller stub: load root
    //                                    register params, `bl/call`
    //                                    internal_entry_label, publish public
    //                                    result slots if needed, then fall
    //                                    through to `lower_epilogue`.
    //   3. `lower_epilogue`           — C-ABI epilogue: restore C
    //                                    callee-saved regs, native return
    //                                    to the C caller. `C_RET0` already
    //                                    holds 0 (success) or the trap
    //                                    kind (error) — see §9.
    //   4. (internal_entry_label binding)
    //   5. `lower_body_prelude`       — body entry prelude: per-arch setup
    //                                    such as link-save state on
    //                                    link-register targets or stack
    //                                    alignment shims where needed.
    //   6. (body blocks + edge stubs)
    //   7. `lower_body_local_error_tail` — bound at `body_local_error_label`,
    //                                    restores preserved dynamic clobbers,
    //                                    pops body prelude state, native
    //                                    return without touching `C_RET0`.
    //
    // The pipeline tail also binds the per-trap-kind labels (stack overflow
    // and deferred). Their lowerings call `raise_trap` and then branch to
    // `body_local_error_label`.
    fn lower_prologue(&mut self);
    fn lower_root_caller_stub(&mut self);
    fn lower_epilogue(&mut self);
    fn lower_body_prelude(&mut self);
    fn lower_body_local_error_tail(&mut self);

    /// Faulting read of one frame address used to validate a guarded wasm
    /// stack frame at body entry. Emitted only when `sf_has_guard_pages` is
    /// active and the runtime stack allocation has a protected guard range.
    #[cfg(sf_has_guard_pages)]
    fn lower_stack_probe(&mut self, addr: MachineAddr) -> Result<(), WasmError>;

    /// Populate the abstract internal-call argument lanes from the public
    /// entry frame before the root stub calls the same internal body entry
    /// used by wasm-to-wasm calls.
    fn lower_root_param_lanes_from_frame(&mut self) {
        let param_locs = self
            .core()
            .current_runtime()
            .map(|runtime| runtime.param_locs.clone())
            .unwrap_or_default();
        for loc in param_locs {
            for inst in root_param_lane_loads(self.core(), loc) {
                self.lower_inst(&inst)
                    .expect("root parameter lane load must lower");
            }
        }
    }

    /// Flush any per-function literal pool the backend has accumulated
    /// while lowering blocks. Called by the pipeline after edge stubs and
    /// before the body-local error tail is bound, so the literals sit at
    /// the end of the body region but before the tail labels — within the
    /// pc-relative load range of every call site that referenced them.
    ///
    /// Returns an error if a deferred load cannot reach its literal in
    /// the pool (e.g. arm64 `ldr_lit_64` only has ±1 MiB range). The
    /// default impl is a no-op and never fails. arm64 uses this hook to
    /// defer per-call literals out of the inline path so that
    /// `lower_call_direct` can elide its trailing `b continuation` when
    /// the next emitted block is the continuation.
    fn lower_function_literal_pool(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    // ── Block lowering (streaming-shaped) ────────────────────────────────
    //
    // The pipeline drives lowering one instruction at a time:
    //
    //   begin_block(block)
    //   for (i, inst) in block.ops.iter().enumerate():
    //       emit_inst_at(inst, i)
    //   end_block(terminator, fallthrough)
    //
    // The triplet replaces the older single-shot `lower_block`. Backends
    // with a per-block lookahead window (e.g. arm64 pair fusion) override
    // `emit_inst_at` and `end_block` to maintain a small instruction
    // buffer; the default impls emit each instruction immediately.

    /// Begin a new block. Default sets `current_block`, clears
    /// `current_edge_target`, and resets the block-entry FP state.
    fn begin_block(&mut self, block: &MachineBlock) -> Result<(), WasmError> {
        self.core_mut().current_block = Some(block.id);
        self.core_mut().current_edge_target = None;
        self.core_mut().reset_block_fp_state(block)?;
        Ok(())
    }

    /// Emit one instruction. `index` is the block-local op index — used by
    /// 32-bit GP backends to query `current_op_index`-keyed liveness facts.
    /// Default sets `current_op_index` and dispatches to `lower_inst`.
    fn emit_inst_at(&mut self, inst: &'a MachineInst, index: usize) -> Result<(), WasmError> {
        self.core_mut().current_op_index = Some(index);
        self.lower_inst(inst)
    }

    /// End the current block: emit the terminator and clear block state.
    /// Backends with a per-instruction lookahead buffer override this to
    /// drain remaining buffered instructions before the terminator.
    fn end_block(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core_mut().current_op_index = None;
        let result = self.lower_terminator(term, fallthrough);
        self.core_mut().current_block = None;
        result
    }

    // ── Instruction & terminator lowering (fully arch-specific) ──────────

    fn lower_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError>;
    fn lower_terminator(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError>;

    // ── Trap & branch ────────────────────────────────────────────────────

    fn lower_trap(&mut self, kind: MachineTrapKind);
    fn lower_unconditional_branch(&mut self, label: usize);

    /// Resolve all branch fixups after code generation is complete.
    fn patch_fixups(&mut self) -> Result<(), WasmError>;

    // ── Scratch allocation for parallel-move protocol ───────────────────

    fn alloc_gp_scratch(&mut self) -> u8;
    fn free_gp_scratch(&mut self, id: u8);
    fn alloc_fp_scratch(&mut self) -> u8;
    fn free_fp_scratch(&mut self, id: u8);

    // ── Parallel move primitives ─────────────────────────────────────────

    fn lower_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError>;

    fn lower_gp_cycle_break(
        &mut self,
        dst: MachineReg,
        src: MachineReg,
        scratch_id: u8,
    ) -> Result<(), WasmError>;

    fn lower_fp_cycle_break(
        &mut self,
        dst: MachineBlockParam,
        src: MachineReg,
        float_width: Option<MachineFloatWidth>,
        scratch_id: u8,
    ) -> Result<(), WasmError>;

    // ── Linking ──────────────────────────────────────────────────────────

    /// Write NOP/INT3 padding into the executable code buffer (true emission).
    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize);
}

fn root_param_lane_loads(
    core: &CompilerCore<'_>,
    loc: MachineParamLoc,
) -> collections::Vec<MachineInst> {
    match loc {
        MachineParamLoc::Frame { .. } => collections::Vec::new(),
        MachineParamLoc::GpArg {
            param_index,
            lane,
            ty,
        } => collections::vec![root_param_load(
            core.gp_arg_lane_reg(lane),
            ty,
            param_index,
            0,
            root_gp_lane_width(core),
        )],
        MachineParamLoc::GpArgPair {
            param_index,
            lo_lane,
            hi_lane,
        } => collections::vec![
            root_param_load(
                core.gp_arg_lane_reg(lo_lane),
                MachineStorageType::GpWord,
                param_index,
                0,
                MachineMemWidth::U32,
            ),
            root_param_load(
                core.gp_arg_lane_reg(hi_lane),
                MachineStorageType::GpWord,
                param_index,
                4,
                MachineMemWidth::U32,
            ),
        ],
        MachineParamLoc::FpArg {
            param_index,
            lane,
            ty,
        } => collections::vec![root_param_load(
            core.fp_arg_lane_reg(lane),
            ty,
            param_index,
            0,
            match ty {
                MachineStorageType::Fp32 => MachineMemWidth::U32,
                MachineStorageType::Fp64 => MachineMemWidth::U64,
                _ => MachineMemWidth::U64,
            },
        )],
    }
}

fn root_param_load(
    dst: MachineReg,
    ty: MachineStorageType,
    param_index: u16,
    extra_offset: i32,
    width: MachineMemWidth,
) -> MachineInst {
    MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty,
            dst,
            addr: MachineAddr {
                base: MACHINE_FP_REG,
                offset: i32::from(param_index) * 8 + extra_offset,
            },
            width,
            extension: MachineLoadExtension::None,
        },
    }
}

fn root_gp_lane_width(core: &CompilerCore<'_>) -> MachineMemWidth {
    if core.compiled.backend().gp_unit_bytes == 4 {
        MachineMemWidth::U32
    } else {
        MachineMemWidth::U64
    }
}
