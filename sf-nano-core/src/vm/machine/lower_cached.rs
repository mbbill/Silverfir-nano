//! Cached local management -- reload / save / entry initialization.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineInst, MachineInstKind, MachineLoadExtension, MachineStorageType,
            MachineValue,
        },
        middle::ssa_ir::ir::SsaLocalCachePrefs,
    },
};

use super::{
    lower_context::{BlockLowerContext, CachedLocal},
    lower_regalloc::{canonical_cached_local_mem_width, value_type_storage_type},
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

    pub(super) fn emit_save_all_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals().len() {
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
        Ok(())
    }

    /// Switch the cached-local set to new preferences. Used at call
    /// continuation points where the post-call block may benefit from
    /// caching different locals. Must be called AFTER `emit_save_all_cached_locals()`
    /// (the old set is already saved to frame) and BEFORE
    /// `emit_reload_cached_locals_selective()` (which reloads the new set).
    pub(super) fn switch_cache_prefs(
        &mut self,
        new_prefs: &SsaLocalCachePrefs,
    ) -> Result<(), WasmError> {
        let regfile = self.regfile();
        let gp_reg_width = self.gp_reg_width();
        let mut cached_locals = Vec::new();
        let mut gp_cache_index = 0usize;

        for (index, slot) in new_prefs.gp_preferred_slots.iter().copied().enumerate() {
            let Some(reg) = regfile.gp_local_cache(gp_cache_index) else {
                break;
            };
            let ty = new_prefs
                .gp_preferred_types
                .get(index)
                .copied()
                .map(value_type_storage_type)
                .ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "switch_cache_prefs: GP cached local {:?} is missing a type entry",
                        slot
                    ))
                })?;
            let info = new_prefs
                .gp_local_info
                .get(index)
                .copied()
                .unwrap_or_default();
            let hi_reg = if gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
                Some(regfile.gp_local_cache(gp_cache_index + 1).ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "switch_cache_prefs: GP cached local {:?} needs second register on 32-bit",
                        slot
                    ))
                })?)
            } else {
                None
            };
            cached_locals.push(CachedLocal {
                slot,
                reg,
                hi_reg,
                ty,
                info,
            });
            gp_cache_index += if hi_reg.is_some() { 2 } else { 1 };
        }

        for (index, slot) in new_prefs.fp_preferred_slots.iter().copied().enumerate() {
            let Some(reg) = regfile.fp_local_cache(index) else {
                break;
            };
            let ty = new_prefs
                .fp_preferred_types
                .get(index)
                .copied()
                .map(value_type_storage_type)
                .ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "switch_cache_prefs: FP cached local {:?} is missing a type entry",
                        slot
                    ))
                })?;
            let info = new_prefs
                .fp_local_info
                .get(index)
                .copied()
                .unwrap_or_default();
            cached_locals.push(CachedLocal {
                slot,
                reg,
                hi_reg: None,
                ty,
                info,
            });
        }

        self.replace_cached_locals(cached_locals);
        Ok(())
    }

    /// Emit remap instructions to transition from `old_set` to `new_set`.
    /// Stores evicted locals to frame, loads promoted locals from frame.
    /// Used in remap trampoline blocks.
    ///
    /// Precondition: both sets use the same register file layout (same
    /// indices map to same physical registers).
    pub(super) fn emit_cache_remap(
        &mut self,
        old_set: &[CachedLocal],
        new_set: &[CachedLocal],
    ) -> Result<(), WasmError> {
        // Phase 1: Store evicted locals (in old, not in new) to frame.
        for old in old_set {
            let still_cached = new_set.iter().any(|n| n.slot == old.slot);
            if !still_cached {
                if matches!(old.ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_save_cached_i64(self, old)?;
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Store {
                            ty: old.ty,
                            addr: self.frame_addr(old.slot)?,
                            width: canonical_cached_local_mem_width(old.ty),
                            src: MachineValue::Reg(old.reg),
                        },
                    });
                }
            }
        }

        // Phase 2: Load promoted locals (in new, not in old) from frame.
        for new in new_set {
            let was_cached = old_set.iter().any(|o| o.slot == new.slot);
            if !was_cached {
                if matches!(new.ty, MachineStorageType::GpI64) {
                    let ops = self.i64_ops();
                    ops.emit_reload_cached_i64(self, new)?;
                } else {
                    self.emit_machine_inst(MachineInst {
                        kind: MachineInstKind::Load {
                            ty: new.ty,
                            dst: new.reg,
                            addr: self.frame_addr(new.slot)?,
                            width: canonical_cached_local_mem_width(new.ty),
                            extension: MachineLoadExtension::None,
                        },
                    });
                }
            }
        }

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
