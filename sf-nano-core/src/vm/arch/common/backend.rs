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
/// ## Naming convention
///
/// - `lower_*` — translate MachineIR into encoded instructions (lowering).
/// - `emit_nop_padding` — the one exception: writes raw padding bytes into
///   the code buffer, not MachineIR lowering.
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

    fn new(compiled: &'a CompiledNativeModule, function: &'a crate::vm::machine::machine_ir::MachineFunction) -> Self;

    /// Access the shared CompilerCore.
    fn core(&self) -> &CompilerCore<'a>;
    fn core_mut(&mut self) -> &mut CompilerCore<'a>;

    /// Consume the backend and return the owned CompilerCore.
    fn into_core(self) -> CompilerCore<'a>;

    // ── Prologue / epilogue / tail ───────────────────────────────────────

    fn lower_prologue(&mut self);
    fn lower_epilogue(&mut self);
    fn lower_return_ok_status(&mut self);

    // ── Block lowering ───────────────────────────────────────────────────

    /// Lower a single block's instructions and terminator.
    ///
    /// Default: iterate `block.ops` calling `lower_inst`, then `lower_terminator`.
    /// Backends override this for peephole patterns (e.g. arm64 float-compare
    /// fusion, zero-store pair fusion).
    fn lower_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core_mut().current_block = Some(block.id);
        self.core_mut().current_edge_target = None;
        self.core_mut().reset_block_fp_state(block)?;
        for (index, inst) in block.ops.iter().enumerate() {
            self.core_mut().current_op_index = Some(index);
            self.lower_inst(inst)?;
        }
        self.core_mut().current_op_index = None;
        let result = self.lower_terminator(&block.terminator, fallthrough);
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

    type CompiledEntry: Clone + core::fmt::Debug;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &super::pipeline::EmittedFunction,
    ) -> Self::CompiledEntry;
}
