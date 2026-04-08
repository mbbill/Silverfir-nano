//! 64-bit GP i64 lowering -- scalar, thin passthrough.
//!
//! On 64-bit targets, i64 values are handled by the normal scalar register
//! allocator and instruction emitter. The trait methods here are trivially
//! thin: load/store delegate to the scalar path, and `lower_i64_leaf` always
//! returns `Ok(false)` (no pair ops).

use crate::error::WasmError;
use crate::vm::middle::frame::FrameSlot;
use crate::vm::middle::ssa_ir::ir::{SsaOperand, SsaValue};
use crate::vm::wasm::primitive_op::PrimitiveOpKind;

use super::{
    lower_context::{BlockLowerContext, BoundCachedLocal},
    lower_i64::I64Lowering,
    lower_inst::LeafLowering,
    lower_leaf_special::{MemoryLoadSpec, MemoryStoreSpec},
    lower_regalloc::{canonical_cached_local_mem_width, canonical_value_mem_width_for_value},
};

use crate::vm::machine::machine_ir::{
    MachineInst, MachineInstKind, MachineLoadExtension, MachineValue,
};

pub(super) struct Gp64Lowering;

impl I64Lowering for Gp64Lowering {
    fn emit_load_slot_i64(
        &self,
        ctx: &mut BlockLowerContext,
        slot: FrameSlot,
        dst: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = ctx.value_storage_type(dst);
        let dst_reg = ctx.alloc_slot_load_value(dst)?;
        let width = canonical_value_mem_width_for_value(ctx.program(), dst);
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                ty,
                dst: dst_reg,
                addr: ctx.frame_addr(slot)?,
                width,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn emit_store_slot_i64(
        &self,
        ctx: &mut BlockLowerContext,
        slot: FrameSlot,
        src: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = ctx.value_storage_type(src);
        let src_reg = ctx.use_value(src)?;
        let width = canonical_value_mem_width_for_value(ctx.program(), src);
        let addr = ctx.frame_addr(slot)?;
        if !ctx.try_coalesce_last_store_immediate(src, src_reg, ty, addr, width) {
            ctx.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Store {
                    ty,
                    addr,
                    width,
                    src: MachineValue::Reg(src_reg),
                },
            });
        }
        ctx.release_dead_values()?;
        Ok(())
    }

    fn lower_i64_leaf(
        &self,
        _ctx: &mut BlockLowerContext,
        _primitive: &PrimitiveOpKind,
        _args: &[SsaOperand],
        _results: &[SsaValue],
    ) -> Result<bool, WasmError> {
        // On 64-bit targets, i64 operations are handled by the normal scalar
        // arithmetic lowering path. Nothing to do here.
        Ok(false)
    }

    fn emit_reload_cached_i64(
        &self,
        ctx: &mut BlockLowerContext,
        cached: &BoundCachedLocal,
    ) -> Result<(), WasmError> {
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                ty: cached.ty,
                dst: cached.reg,
                addr: ctx.frame_addr(cached.slot)?,
                width: canonical_cached_local_mem_width(cached.ty),
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn emit_save_cached_i64(
        &self,
        ctx: &mut BlockLowerContext,
        cached: &BoundCachedLocal,
    ) -> Result<(), WasmError> {
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: cached.ty,
                addr: ctx.frame_addr(cached.slot)?,
                width: canonical_cached_local_mem_width(cached.ty),
                src: MachineValue::Reg(cached.reg),
            },
        });
        Ok(())
    }

    fn emit_global_get_i64(
        &self,
        ctx: &mut BlockLowerContext,
        idx: u32,
        result: SsaValue,
    ) -> Result<(), WasmError> {
        // On 64-bit targets the normal scalar path handles this.
        // Delegate back to the non-i64-specific global_get path.
        ctx.lower_global_get_scalar(idx, result)
    }

    fn emit_global_set_i64(
        &self,
        ctx: &mut BlockLowerContext,
        idx: u32,
        src: SsaValue,
    ) -> Result<(), WasmError> {
        ctx.lower_global_set_scalar(idx, src)
    }

    fn emit_memory_load_i64(
        &self,
        ctx: &mut BlockLowerContext,
        spec: MemoryLoadSpec,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        // On 64-bit targets the normal scalar memory-load path handles this.
        ctx.lower_memory_load_scalar(spec, args, results)
    }

    fn emit_memory_store_i64(
        &self,
        ctx: &mut BlockLowerContext,
        spec: MemoryStoreSpec,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError> {
        ctx.lower_memory_store_scalar(spec, args)
    }
}
