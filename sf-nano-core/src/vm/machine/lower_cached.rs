//! Cached local management -- explicit drop / save helpers.

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineAddr, MachineInst, MachineInstKind, MachineLoadExtension, MachineMemWidth,
            MachineStorageType, MachineValue,
        },
        middle::frame::FrameSlot,
    },
};

use super::{
    lower_context::BlockLowerContext, lower_module::slot_offset_bytes,
    lower_regalloc::canonical_cached_local_mem_width,
};

impl<'a> BlockLowerContext<'a> {
    /// Save only cached locals that have been written since the last save.
    pub(super) fn emit_save_dirty_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals().len() {
            if !self.is_cache_live(index) {
                continue;
            }
            if !self.is_cache_dirty(index) {
                continue;
            }
            let cached = self.bound_cached_local(index).ok_or_else(|| {
                WasmError::internal("cached local binding missing during cache save".into())
            })?;
            if matches!(cached.ty, MachineStorageType::GpI64) {
                let ops = self.i64_ops();
                ops.emit_save_cached_i64(self, &cached)?;
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: cached.ty,
                        addr: self.frame_addr(cached.slot)?,
                        width: canonical_cached_local_mem_width(cached.ty),
                        src: MachineValue::Reg(cached.reg),
                    },
                });
            }
        }
        self.clear_cache_dirty();
        self.clear_cache_live();
        Ok(())
    }

    pub(super) fn emit_drop_cached_local(&mut self, index: usize) -> Result<(), WasmError> {
        if !self.is_cache_live(index) {
            return Ok(());
        }
        let cached = self.bound_cached_local(index).ok_or_else(|| {
            WasmError::internal("cached local binding missing during cache drop".into())
        })?;
        self.materialize_cache_aliases(cached.reg, &[])?;
        if let Some(hi_reg) = cached.hi_reg {
            self.materialize_cache_aliases(hi_reg, &[])?;
        }
        if self.is_cache_dirty(index) {
            if matches!(cached.ty, MachineStorageType::GpI64) {
                let ops = self.i64_ops();
                ops.emit_save_cached_i64(self, &cached)?;
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: cached.ty,
                        addr: self.frame_addr(cached.slot)?,
                        width: canonical_cached_local_mem_width(cached.ty),
                        src: MachineValue::Reg(cached.reg),
                    },
                });
            }
        }
        self.set_cache_live(index, false);
        self.set_cache_has_value(index, false);
        self.set_cache_dirty(index, false);
        self.clear_cache_binding(index);
        Ok(())
    }

    /// Emit zero stores for the listed local slots into the function's own
    /// frame. Used at the entry block to satisfy the wasm zero-init contract
    /// for non-param locals that may be read before being written. Locals not
    /// in `slots` are guaranteed to be written before any read, so they need
    /// no explicit init store.
    pub(super) fn emit_zero_init_locals(&mut self, slots: &[u16]) -> Result<(), WasmError> {
        if slots.is_empty() {
            return Ok(());
        }
        let base = self.frame_base_reg();
        let gp_reg_width = self.gp_reg_width();
        for &slot_idx in slots {
            let slot = FrameSlot(slot_idx);
            let base_offset = slot_offset_bytes(slot)?;
            if gp_reg_width == 4 {
                // 32-bit target: two 4-byte stores per i64-sized slot.
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base,
                            offset: base_offset,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Imm64(0),
                    },
                });
                let hi_offset = base_offset.checked_add(4).ok_or_else(|| {
                    WasmError::internal("frame slot zero-init offset overflow".into())
                })?;
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base,
                            offset: hi_offset,
                        },
                        width: MachineMemWidth::U32,
                        src: MachineValue::Imm64(0),
                    },
                });
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpI64,
                        addr: MachineAddr {
                            base,
                            offset: base_offset,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Imm64(0),
                    },
                });
            }
        }
        Ok(())
    }

    pub(super) fn emit_reload_mem0_cache_regs(&mut self) {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: self.regfile().mem0_base(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: crate::vm::machine::machine_ir::MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: self.regfile().mem0_size(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_size_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
    }
}
