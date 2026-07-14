//! 32-bit GP i64 lowering -- register pair strategy.
//!
//! On 32-bit targets, i64 values are split into lo/hi word register pairs.
//! This module implements the `I64Lowering` trait for `Gp32Lowering`.

mod lower_leaf;

use crate::error::WasmError;
use crate::vm::middle::frame::FrameSlot;
use crate::vm::middle::ssa_ir::ir::{SsaOperand, SsaValue};
use crate::vm::wasm::primitive_op::PrimitiveOpKind;

use super::{
    lower_context::{BlockLowerContext, BoundCachedCell},
    lower_i64::I64Lowering,
    lower_inst::LeafLowering,
    lower_leaf_special::{MemoryLoadSpec, MemoryStoreSpec},
};

use crate::vm::machine::machine_ir::{
    MachineInst, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineRegOwner,
    MachineStorageType, MachineValue,
};

pub(super) struct Gp32Lowering;

impl I64Lowering for Gp32Lowering {
    fn emit_load_slot_i64(
        &self,
        ctx: &mut BlockLowerContext,
        slot: FrameSlot,
        dst: SsaValue,
    ) -> Result<(), WasmError> {
        let (dst_lo, dst_hi) = ctx.alloc_i64_value_pair(dst)?;
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: dst_lo,
                addr: ctx.frame_addr_offset(slot, 0)?,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        });
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: dst_hi,
                addr: ctx.frame_addr_offset(slot, 4)?,
                width: MachineMemWidth::U32,
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
        let (src_lo, src_hi) = ctx.use_i64_value_pair(src)?;
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: ctx.frame_addr_offset(slot, 0)?,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(src_lo),
            },
        });
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: ctx.frame_addr_offset(slot, 4)?,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(src_hi),
            },
        });
        ctx.release_dead_values()?;
        Ok(())
    }

    fn lower_i64_leaf(
        &self,
        ctx: &mut BlockLowerContext,
        primitive: &PrimitiveOpKind,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<bool, WasmError> {
        ctx.lower_i64_pair_leaf(primitive, args, results)
    }

    fn emit_reload_cached_i64(
        &self,
        ctx: &mut BlockLowerContext,
        cached: &BoundCachedCell,
    ) -> Result<(), WasmError> {
        let cached_hi = cached.hi_reg.ok_or_else(|| {
            WasmError::internal("cached i64 local is missing a high-half register")
        })?;
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::CachedCell,
                ty: MachineStorageType::GpWord,
                dst: cached.reg,
                addr: ctx.frame_addr_offset(cached.home, 0)?,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        });
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::CachedCell,
                ty: MachineStorageType::GpWord,
                dst: cached_hi,
                addr: ctx.frame_addr_offset(cached.home, 4)?,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn emit_save_cached_i64(
        &self,
        ctx: &mut BlockLowerContext,
        cached: &BoundCachedCell,
    ) -> Result<(), WasmError> {
        let cached_hi = cached.hi_reg.ok_or_else(|| {
            WasmError::internal("cached i64 local is missing a high-half register")
        })?;
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: ctx.frame_addr_offset(cached.home, 0)?,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(cached.reg),
            },
        });
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: ctx.frame_addr_offset(cached.home, 4)?,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(cached_hi),
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
        ctx.lower_global_get_i64_pair(idx, result)
    }

    fn emit_global_set_i64(
        &self,
        ctx: &mut BlockLowerContext,
        idx: u32,
        src: SsaValue,
    ) -> Result<(), WasmError> {
        ctx.lower_global_set_i64_pair(idx, src)
    }

    fn emit_memory_load_i64(
        &self,
        ctx: &mut BlockLowerContext,
        spec: MemoryLoadSpec,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        ctx.lower_i64_memory_load(spec, args, results)
    }

    fn emit_memory_store_i64(
        &self,
        ctx: &mut BlockLowerContext,
        spec: MemoryStoreSpec,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError> {
        ctx.lower_i64_memory_store(spec, args)
    }
}
