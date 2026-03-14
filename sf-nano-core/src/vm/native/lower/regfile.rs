use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        native::ir::machine::{
            MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
    },
};

/// Fixed machine-register partition used by lowering.
///
/// `ctx`, `fp`, and the pinned `mem0` view regs are fixed MachineIR roles.
/// The remaining cache and lane partitions are a logical ownership model chosen
/// for lowering; they may be reused for other temporary purposes when the
/// owning values are proven dead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MachineRegFile {
    local_cache: Vec<MachineReg>,
    transient: Vec<MachineReg>,
    reg_count: u16,
}

impl MachineRegFile {
    pub(super) fn new(config: BackendConfig) -> Result<Self, WasmError> {
        if config.lir_lane_count == 0 {
            return Err(WasmError::internal(
                "native lowering requires at least one LIR lane register".into(),
            ));
        }

        let mut next = MACHINE_FIXED_REG_COUNT;
        let local_cache = collect_regs(&mut next, config.hot_local_count);
        let transient = collect_regs(&mut next, config.lir_lane_count);

        Ok(Self {
            local_cache,
            transient,
            reg_count: next,
        })
    }

    #[inline]
    pub(super) const fn runtime_base(&self) -> MachineReg {
        MACHINE_CTX_REG
    }

    #[inline]
    pub(super) const fn frame_base(&self) -> MachineReg {
        MACHINE_FP_REG
    }

    #[inline]
    pub(super) const fn mem0_base(&self) -> MachineReg {
        MACHINE_MEM0_BASE_REG
    }

    #[inline]
    pub(super) const fn mem0_size(&self) -> MachineReg {
        MACHINE_MEM0_SIZE_REG
    }

    #[inline]
    pub(super) fn local_cache(&self, index: usize) -> Option<MachineReg> {
        self.local_cache.get(index).copied()
    }

    #[inline]
    pub(super) fn transient(&self, index: usize) -> Option<MachineReg> {
        self.transient.get(index).copied()
    }

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
