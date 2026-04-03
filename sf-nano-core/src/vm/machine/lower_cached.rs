//! Cached local management -- explicit drop / save helpers.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineInst, MachineInstKind, MachineLoadExtension, MachineStorageType, MachineValue,
    },
};

use super::{lower_context::BlockLowerContext, lower_regalloc::canonical_cached_local_mem_width};

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
        self.set_cache_dirty(index, false);
        self.clear_cache_binding(index);
        Ok(())
    }

    pub(super) fn emit_reload_mem0_cache_regs(&mut self) {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: self.regfile().mem0_base(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst: self.regfile().mem0_size(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_size_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
    }
}
