use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{backend::BackendConfig, native::ir::machine::MachineReg},
};

/// Fixed machine-register partition used by lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MachineRegFile {
    runtime_base: MachineReg,
    frame_base: MachineReg,
    local_cache: Vec<MachineReg>,
    transient: Vec<MachineReg>,
    temps: Vec<MachineReg>,
    reg_count: u16,
}

impl MachineRegFile {
    pub(super) fn new(config: BackendConfig) -> Result<Self, WasmError> {
        if config.ctx_register_count == 0 {
            return Err(WasmError::internal(
                "native lowering requires one runtime-base register".into(),
            ));
        }
        if config.fp_register_count == 0 {
            return Err(WasmError::internal(
                "native lowering requires one frame-base register".into(),
            ));
        }

        let mut next = 0u16;
        let runtime_base = MachineReg(next);
        next += config.ctx_register_count as u16;
        let frame_base = MachineReg(next);
        next += config.fp_register_count as u16;

        let local_cache = collect_regs(&mut next, config.hot_local_count);
        let transient = collect_regs(&mut next, config.tos_register_count);
        let temps = collect_regs(&mut next, config.tmp_register_count);

        Ok(Self {
            runtime_base,
            frame_base,
            local_cache,
            transient,
            temps,
            reg_count: next,
        })
    }

    #[inline]
    pub(super) const fn runtime_base(&self) -> MachineReg {
        self.runtime_base
    }

    #[inline]
    pub(super) const fn frame_base(&self) -> MachineReg {
        self.frame_base
    }

    #[inline]
    pub(super) fn local_cache(&self, index: usize) -> Option<MachineReg> {
        self.local_cache.get(index).copied()
    }

    #[inline]
    pub(super) fn transient(&self, index: usize) -> Option<MachineReg> {
        self.transient.get(index).copied()
    }

    #[inline]
    pub(super) fn temp(&self, index: usize) -> Option<MachineReg> {
        self.temps.get(index).copied()
    }

    #[inline]
    pub(super) fn transient_count(&self) -> usize {
        self.transient.len()
    }

    #[inline]
    pub(super) fn reg_count(&self) -> u16 {
        self.reg_count
    }
}

fn collect_regs(next: &mut u16, count: u8) -> Vec<MachineReg> {
    let mut regs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        regs.push(MachineReg(*next));
        *next += 1;
    }
    regs
}

