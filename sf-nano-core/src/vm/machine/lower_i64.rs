//! I64 lowering strategy trait -- the contract for GP-width-dependent i64 handling.
//!
//! Two implementations:
//! - `Gp64Lowering` (scalar, thin) in `lower_i64_gp64.rs`
//! - `Gp32Lowering` (pair, complex) in `gp32/`

use crate::error::WasmError;

use super::{
    lower_context::{BlockLowerContext, CachedLocal},
    lower_leaf_special::{MemoryLoadSpec, MemoryStoreSpec},
};

use crate::vm::middle::ssa_ir::ir::{SsaOperand, SsaValue};
use crate::vm::wasm::primitive_op::PrimitiveOpKind;

use super::lower_inst::LeafLowering;

/// Operations that differ between 32-bit and 64-bit GP targets for i64 values.
/// Implemented by `Gp64Lowering` (scalar) and `Gp32Lowering` (pair).
pub(super) trait I64Lowering {
    /// Load an i64 value from a frame slot (LocalGet / Fill).
    fn emit_load_slot_i64(
        &self,
        ctx: &mut BlockLowerContext,
        slot: crate::vm::middle::frame::FrameSlot,
        dst: SsaValue,
    ) -> Result<(), WasmError>;

    /// Store an i64 value to a frame slot (LocalSet / Spill).
    fn emit_store_slot_i64(
        &self,
        ctx: &mut BlockLowerContext,
        slot: crate::vm::middle::frame::FrameSlot,
        src: SsaValue,
    ) -> Result<(), WasmError>;

    /// Lower an i64 primitive leaf op. Returns true if handled.
    fn lower_i64_leaf(
        &self,
        ctx: &mut BlockLowerContext,
        primitive: &PrimitiveOpKind,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<bool, WasmError>;

    /// Reload a cached i64 local from its frame slot into cache register(s).
    fn emit_reload_cached_i64(
        &self,
        ctx: &mut BlockLowerContext,
        cached: &CachedLocal,
    ) -> Result<(), WasmError>;

    /// Save a cached i64 local from cache register(s) to its frame slot.
    fn emit_save_cached_i64(
        &self,
        ctx: &mut BlockLowerContext,
        cached: &CachedLocal,
    ) -> Result<(), WasmError>;

    /// Initialize a cached i64 local at function entry.
    /// `is_param` == true: load from frame (caller wrote a value).
    /// `is_param` == false: zero-init (Wasm locals start at zero).
    fn emit_entry_cached_i64(
        &self,
        ctx: &mut BlockLowerContext,
        cached: &CachedLocal,
        is_param: bool,
    ) -> Result<(), WasmError>;

    /// Load an i64 global value (global.get).
    fn emit_global_get_i64(
        &self,
        ctx: &mut BlockLowerContext,
        idx: u32,
        result: SsaValue,
    ) -> Result<(), WasmError>;

    /// Store an i64 global value (global.set).
    fn emit_global_set_i64(
        &self,
        ctx: &mut BlockLowerContext,
        idx: u32,
        src: SsaValue,
    ) -> Result<(), WasmError>;

    /// Emit i64 memory load.
    fn emit_memory_load_i64(
        &self,
        ctx: &mut BlockLowerContext,
        spec: MemoryLoadSpec,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError>;

    /// Emit i64 memory store.
    fn emit_memory_store_i64(
        &self,
        ctx: &mut BlockLowerContext,
        spec: MemoryStoreSpec,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError>;
}
