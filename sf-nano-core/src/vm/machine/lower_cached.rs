//! Cached local management -- reload / save / entry initialization.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineInst, MachineInstKind, MachineLoadExtension, MachineStorageType,
        MachineValue,
    },
};

use super::{
    lower_context::BlockLowerContext,
    lower_regalloc::canonical_cached_local_mem_width,
};

impl<'a> BlockLowerContext<'a> {
    /// Reload all cached locals from the frame. Used after calls and at
    /// non-entry block boundaries where the cache may be stale.
    pub(super) fn emit_reload_cached_locals(&mut self) -> Result<(), WasmError> {
        self.emit_reload_cached_locals_selective(None)
    }

    /// Reload cached locals from the frame, optionally skipping locals that
    /// are known to be written before read at this continuation point.
    ///
    /// `skip_reload` is parallel to the cached_locals vec (GP then FP order).
    /// When `skip_reload[i]` is `true`, the reload for that cached local is
    /// elided because the local will be overwritten before anyone reads it.
    pub(super) fn emit_reload_cached_locals_selective(
        &mut self,
        skip_reload: Option<&[bool]>,
    ) -> Result<(), WasmError> {
        for index in 0..self.cached_locals().len() {
            if let Some(skip) = skip_reload {
                if index < skip.len() && skip[index] {
                    continue;
                }
            }
            let cached = self.cached_locals()[index];
            if matches!(cached.ty, MachineStorageType::GpI64) {
                let ops = self.i64_ops();
                ops.emit_reload_cached_i64(self, &cached)?;
            } else {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Load {
                        ty: cached.ty,
                        dst: cached.reg,
                        addr: self.frame_addr(cached.slot)?,
                        width: canonical_cached_local_mem_width(cached.ty),
                        extension: MachineLoadExtension::None,
                    },
                });
            }
        }
        Ok(())
    }

    /// Initialize cached locals at function entry.
    ///
    /// Parameters are loaded from the frame (the caller already wrote them).
    /// Non-parameter locals that may be read before written need a zero
    /// materialisation (Wasm locals start at zero). Locals that are definitely
    /// written before any read can be left undefined; the `reads_before_write`
    /// analysis in `local_cache.rs` is a whole-function dataflow pass that is
    /// sound for this purpose on both 32-bit and 64-bit targets.
    pub(super) fn emit_entry_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals().len() {
            let cached = self.cached_locals()[index];
            if cached.info.is_param {
                // Argument -- caller wrote a real value, must load from frame.
                if matches!(cached.ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_entry_cached_i64(self, &cached, true)?;
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            ty: cached.ty,
                            dst: cached.reg,
                            addr: self.frame_addr(cached.slot)?,
                            width: canonical_cached_local_mem_width(cached.ty),
                            extension: MachineLoadExtension::None,
                        },
                    });
                }
            } else if cached.info.reads_before_write {
                // Non-param local that may be read before written -- zero the
                // register (Wasm locals are initialised to zero).
                if matches!(cached.ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_entry_cached_i64(self, &cached, false)?;
                } else if let Some(width) = cached.ty.float_width() {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::FloatConst {
                            width,
                            dst: cached.reg,
                            bits: 0,
                        },
                    });
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Move {
                            ty: cached.ty,
                            dst: cached.reg,
                            src: MachineValue::Imm64(0),
                        },
                    });
                }
            }
            // else: non-param, written before read -- skip entirely.
        }
        Ok(())
    }

    /// Save only cached locals that have been written since the last save.
    pub(super) fn emit_save_dirty_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals().len() {
            if !self.is_cache_dirty(index) {
                continue;
            }
            let cached = self.cached_locals()[index];
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
