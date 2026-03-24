use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineInst,
        MachineReg, MachineTerminator, MachineTrapKind,
    },
};

use super::core::CompilerCore;
use super::types::ParallelSource;

use crate::vm::runtime::code::CompiledNativeModule;
use crate::vm::runtime::code_buf::CodeBuffer;

/// Trait that each architecture backend implements.
///
/// The backend MUST have a `pub core: CompilerCore<'a>` field.
/// Pipeline functions access it directly for shared state.
///
/// Only truly arch-specific behaviour lives on this trait: instruction
/// encoding, register mapping, prologue/epilogue, branch mechanics.
pub(crate) trait ArchBackend<'a>: Sized {
    /// Architecture name for error messages (e.g. "arm64").
    const NAME: &'static str;

    // ── Capacity queries (no &self — pure arch facts) ────────────────────

    fn max_total_regs() -> usize;
    fn max_fp_regs() -> usize;

    // ── Construction ─────────────────────────────────────────────────────

    fn new(compiled: &'a CompiledNativeModule, function: &'a crate::vm::machine::machine_ir::MachineFunction) -> Self;

    /// Access the shared CompilerCore.
    fn core(&self) -> &CompilerCore<'a>;
    fn core_mut(&mut self) -> &mut CompilerCore<'a>;

    /// Consume the backend and return the owned CompilerCore.
    fn into_core(self) -> CompilerCore<'a>;

    // ── Prologue / epilogue / tail ───────────────────────────────────────

    fn emit_prologue(&mut self);
    fn emit_epilogue(&mut self);

    /// Emit the return-ok status value (e.g. `MOV X0, #0` or `XOR EAX, EAX`).
    fn emit_return_ok_status(&mut self);

    // ── Block emission ───────────────────────────────────────────────────

    /// Emit a single block's instructions and terminator.
    ///
    /// Default: iterate `block.ops` calling `emit_inst`, then `emit_terminator`.
    /// Backends override this for peephole patterns (e.g. arm64 float-compare
    /// fusion, zero-store pair fusion).
    fn emit_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core_mut().current_block = Some(block.id);
        self.core_mut().current_edge_target = None;
        self.core_mut().reset_block_fp_state(block)?;
        for (index, inst) in block.ops.iter().enumerate() {
            self.core_mut().current_op_index = Some(index);
            self.emit_inst(inst)?;
        }
        self.core_mut().current_op_index = None;
        let result = self.emit_terminator(&block.terminator, fallthrough);
        self.core_mut().current_block = None;
        result
    }

    // ── Instruction & terminator emission (fully arch-specific) ──────────

    fn emit_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError>;
    fn emit_terminator(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError>;

    // ── Trap & branch ────────────────────────────────────────────────────

    fn emit_trap(&mut self, kind: MachineTrapKind);

    /// Emit an unconditional branch to a label (with fixup).
    fn emit_unconditional_branch(&mut self, label: usize);

    /// Resolve all branch fixups after code generation is complete.
    fn patch_fixups(&mut self) -> Result<(), WasmError>;

    // ── Parallel move primitives ─────────────────────────────────────────
    // The cycle-resolution algorithm is shared (in pipeline.rs);
    // these are the arch-specific move primitives it calls.

    fn emit_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError>;

    fn emit_gp_cycle_break(
        &mut self,
        dst: MachineReg,
        src: MachineReg,
    ) -> Result<(), WasmError>;

    fn emit_fp_cycle_break(
        &mut self,
        dst: MachineBlockParam,
        src: MachineReg,
        float_width: Option<MachineFloatWidth>,
    ) -> Result<(), WasmError>;

    // ── Linking ──────────────────────────────────────────────────────────

    /// Emit NOP/INT3 padding in the executable code buffer.
    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize);

    /// The per-function compiled entry type (e.g. CompiledArm64Entry).
    type CompiledEntry: Clone + core::fmt::Debug;

    /// Construct a CompiledEntry from emitted function metadata.
    fn make_entry(
        buf: &CodeBuffer,
        emitted: &super::pipeline::EmittedFunction,
    ) -> Self::CompiledEntry;
}
