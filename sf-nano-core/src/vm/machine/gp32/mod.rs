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
    lower_context::{BlockLowerContext, BoundCachedLocal},
    lower_i64::I64Lowering,
    lower_inst::LeafLowering,
    lower_leaf_special::{MemoryLoadSpec, MemoryStoreSpec},
    lower_regalloc::lir_value_storage_type,
};

use crate::vm::machine::machine_ir::{
    MachineInst, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineStorageType,
    MachineValue,
};

pub(super) struct Gp32Lowering;

impl I64Lowering for Gp32Lowering {
    fn emit_load_slot_i64(
        &self,
        ctx: &mut BlockLowerContext,
        slot: FrameSlot,
        dst: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(ctx.program(), dst);
        let (dst_lo, dst_hi) = ctx.alloc_i64_value_pair(dst)?;
        if let Some(cached_index) = ctx.cached_local_index(slot) {
            let cached = ctx.ensure_bound_cached_local(cached_index)?;
            if cached.ty != ty {
                return Err(WasmError::internal(alloc::format!(
                    "typed SSA-IR load from cached local slot {:?} expects {:?} for value {:?}, but cached local is {:?}",
                    slot, ty, dst, cached.ty,
                )));
            }
            let cached_hi = cached.hi_reg.ok_or_else(|| {
                WasmError::internal("cached i64 local is missing a high-half register".into())
            })?;
            ctx.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_lo,
                    src: MachineValue::Reg(cached.reg),
                },
            });
            ctx.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    src: MachineValue::Reg(cached_hi),
                },
            });
        } else {
            ctx.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_lo,
                    addr: ctx.frame_addr_offset(slot, 0)?,
                    width: MachineMemWidth::U32,
                    extension: MachineLoadExtension::None,
                },
            });
            ctx.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    addr: ctx.frame_addr_offset(slot, 4)?,
                    width: MachineMemWidth::U32,
                    extension: MachineLoadExtension::None,
                },
            });
        }
        Ok(())
    }

    fn emit_store_slot_i64(
        &self,
        ctx: &mut BlockLowerContext,
        slot: FrameSlot,
        src: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = lir_value_storage_type(ctx.program(), src);
        let (src_lo, src_hi) = ctx.use_i64_value_pair(src)?;
        if let Some(cached_index) = ctx.cached_local_index(slot) {
            ctx.mark_cache_dirty(cached_index);
            let cached = ctx.ensure_bound_cached_local(cached_index)?;
            if cached.ty != ty {
                return Err(WasmError::internal(alloc::format!(
                    "typed SSA-IR store to cached local slot {:?} uses {:?} value {:?}, but cached local is {:?}",
                    slot, ty, src, cached.ty,
                )));
            }
            let cache_hi = cached.hi_reg.ok_or_else(|| {
                WasmError::internal("cached i64 local is missing a high-half register".into())
            })?;
            ctx.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                    ty: MachineStorageType::GpWord,
                    dst: cached.reg,
                    src: MachineValue::Reg(src_lo),
                },
            });
            ctx.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                    ty: MachineStorageType::GpWord,
                    dst: cache_hi,
                    src: MachineValue::Reg(src_hi),
                },
            });
        } else {
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
        }
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
        cached: &BoundCachedLocal,
    ) -> Result<(), WasmError> {
        let cached_hi = cached.hi_reg.ok_or_else(|| {
            WasmError::internal("cached i64 local is missing a high-half register".into())
        })?;
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                ty: MachineStorageType::GpWord,
                dst: cached.reg,
                addr: ctx.frame_addr_offset(cached.slot, 0)?,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        });
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal,
                ty: MachineStorageType::GpWord,
                dst: cached_hi,
                addr: ctx.frame_addr_offset(cached.slot, 4)?,
                width: MachineMemWidth::U32,
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
        let cached_hi = cached.hi_reg.ok_or_else(|| {
            WasmError::internal("cached i64 local is missing a high-half register".into())
        })?;
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: ctx.frame_addr_offset(cached.slot, 0)?,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(cached.reg),
            },
        });
        ctx.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: ctx.frame_addr_offset(cached.slot, 4)?,
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
