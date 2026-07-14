//! Cached local management -- explicit drop / save helpers.

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineAddr, MachineBlockParam, MachineInst, MachineInstKind, MachineLoadExtension,
            MachineMemWidth, MachineRegOwner, MachineStorageType, MachineValue,
        },
        middle::frame::FrameSlot,
    },
};

use super::{
    lower_context::{BlockLowerContext, BoundCachedLocal, ValueRegs},
    lower_module::slot_offset_bytes,
    lower_regalloc::{canonical_cached_local_mem_width, machine_block_params_for_value},
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
                WasmError::internal("cached local binding missing during cache save")
            })?;
            self.emit_save_bound_cached_local(&cached)?;
        }
        self.clear_cache_dirty();
        self.clear_cache_live();
        Ok(())
    }

    /// Prepare cached locals for a compiled local call. The plan nominates
    /// non-ref caches to survive the call; a nominated cache holding a value
    /// must sit in preserved dynamic regs here — the nomination clamp equals
    /// the bank's preserved-lane count and lane assignment places
    /// preference-first, so a volatile lane at this point means the plan
    /// contract was broken upstream and is a hard error. Dirty carried caches
    /// are published once before the call so the frame is authoritative at
    /// the safepoint, then continue in the preserved lane on the success
    /// edge.
    pub(super) fn prepare_cached_locals_for_local_call(
        &mut self,
    ) -> Result<
        (
            crate::collections::Vec<MachineValue>,
            crate::collections::Vec<MachineBlockParam>,
        ),
        WasmError,
    > {
        let mut success_args = crate::collections::Vec::new();
        let mut continuation_params = crate::collections::Vec::new();

        for index in 0..self.cached_locals().len() {
            if !self.is_cache_live(index) {
                continue;
            }
            let cached = self.bound_cached_local(index).ok_or_else(|| {
                WasmError::internal("cached local binding missing during local-call cache prep")
            })?;
            if self.plan_carries_cached_local_across_local_call(index, &cached) {
                if !self.cached_local_regs_are_preserved(&cached) {
                    return Err(WasmError::internal(
                        "preserved-nominated cache sits in a volatile lane at a survivable call",
                    ));
                }
                if self.is_cache_dirty(index) {
                    self.emit_save_bound_cached_local(&cached)?;
                    self.set_cache_dirty(index, false);
                }
                success_args.extend(cached_local_success_args(&cached));
                continuation_params.extend(cached_local_continuation_params(&cached));
                continue;
            }
            if self.is_cache_dirty(index) {
                self.emit_save_bound_cached_local(&cached)?;
            }
            self.set_cache_live(index, false);
            self.set_cache_has_value(index, false);
            self.set_cache_dirty(index, false);
            self.clear_cache_binding(index);
        }

        Ok((success_args, continuation_params))
    }

    pub(super) fn emit_drop_cached_local(&mut self, index: usize) -> Result<(), WasmError> {
        if !self.is_cache_live(index) {
            return Ok(());
        }
        let cached = self
            .bound_cached_local(index)
            .ok_or_else(|| WasmError::internal("cached local binding missing during cache drop"))?;
        self.materialize_cache_aliases(cached.reg, &[])?;
        if let Some(hi_reg) = cached.hi_reg {
            self.materialize_cache_aliases(hi_reg, &[])?;
        }
        if self.is_cache_dirty(index) {
            self.emit_save_bound_cached_local(&cached)?;
        }
        self.set_cache_live(index, false);
        self.set_cache_has_value(index, false);
        self.set_cache_dirty(index, false);
        self.clear_cache_binding(index);
        Ok(())
    }

    pub(super) fn evict_cached_locals_in_gp_regs(
        &mut self,
        regs: &[crate::vm::machine::machine_ir::MachineReg],
    ) -> Result<(), WasmError> {
        for index in 0..self.cached_locals().len() {
            if !self.is_cache_live(index) {
                continue;
            }
            let Some(cached) = self.bound_cached_local(index) else {
                continue;
            };
            if regs.contains(&cached.reg)
                || cached
                    .hi_reg
                    .map(|reg| regs.contains(&reg))
                    .unwrap_or(false)
            {
                self.emit_drop_cached_local(index)?;
            }
        }
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
                let hi_offset = base_offset
                    .checked_add(4)
                    .ok_or_else(|| WasmError::internal("frame slot zero-init offset overflow"))?;
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
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: self.regfile().mem0_base(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: self.regfile().mem0_size(),
                addr: self.runtime_addr(self.runtime_abi_layout().context.mem0_size_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
    }

    fn plan_carries_cached_local_across_local_call(
        &self,
        index: usize,
        cached: &BoundCachedLocal,
    ) -> bool {
        self.cache_has_value(index)
            && self.prefers_preserved_cache_binding(index)
            && !cached.value_ty.is_ref()
    }

    fn cached_local_regs_are_preserved(&self, cached: &BoundCachedLocal) -> bool {
        self.regfile().is_preserved_dynamic_reg(cached.reg)
            && cached
                .hi_reg
                .map(|reg| self.regfile().is_preserved_dynamic_reg(reg))
                .unwrap_or(true)
    }

    fn emit_save_bound_cached_local(&mut self, cached: &BoundCachedLocal) -> Result<(), WasmError> {
        if matches!(cached.ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            ops.emit_save_cached_i64(self, cached)?;
            return Ok(());
        }
        #[cfg(sf_has_simd)]
        if matches!(cached.ty, MachineStorageType::V128) {
            self.emit_store_frame_v128(cached.slot, cached.reg)?;
            return Ok(());
        }
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: cached.ty,
                addr: self.frame_addr(cached.slot)?,
                width: canonical_cached_local_mem_width(cached.ty),
                src: MachineValue::Reg(cached.reg),
            },
        });
        Ok(())
    }
}

fn cached_local_success_args(cached: &BoundCachedLocal) -> crate::collections::Vec<MachineValue> {
    let mut args = crate::collections::Vec::with_capacity(1 + usize::from(cached.hi_reg.is_some()));
    args.push(MachineValue::Reg(cached.reg));
    if let Some(hi_reg) = cached.hi_reg {
        args.push(MachineValue::Reg(hi_reg));
    }
    args
}

fn cached_local_continuation_params(
    cached: &BoundCachedLocal,
) -> crate::collections::Vec<MachineBlockParam> {
    machine_block_params_for_value(
        ValueRegs {
            lo: cached.reg,
            hi: cached.hi_reg,
        },
        cached.ty,
    )
    .into_iter()
    .map(|param| param.with_owner(MachineRegOwner::CachedLocal))
    .collect()
}
