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
    gp_local_cache: Vec<MachineReg>,
    gp_transient: Vec<MachineReg>,
    fp_transient: Vec<MachineReg>,
    fp_local_cache: Vec<MachineReg>,
    first_fp_reg: u16,
    reg_count: u16,
}

impl MachineRegFile {
    pub(super) fn new(config: BackendConfig) -> Result<Self, WasmError> {
        if config.gp_transient_budget == 0 {
            return Err(WasmError::internal(
                "native lowering requires at least one GP transient register".into(),
            ));
        }

        let mut next = MACHINE_FIXED_REG_COUNT;
        let gp_local_cache = collect_regs(&mut next, config.gp_local_cache_budget);
        let gp_transient = collect_regs(&mut next, config.gp_transient_budget);
        let first_fp_reg = next;
        let fp_transient = collect_regs(&mut next, config.fp_transient_budget);
        let fp_local_cache = collect_regs(&mut next, config.fp_local_cache_budget);

        // Layout: [fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache]
        //                                                              ^ first_fp_reg
        Ok(Self {
            gp_local_cache,
            gp_transient,
            fp_transient,
            fp_local_cache,
            first_fp_reg,
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
    pub(super) fn gp_local_cache(&self, index: usize) -> Option<MachineReg> {
        self.gp_local_cache.get(index).copied()
    }

    #[inline]
    pub(super) fn gp_transient(&self, index: usize) -> Option<MachineReg> {
        self.gp_transient.get(index).copied()
    }

    pub(super) fn gp_transient_count(&self) -> usize {
        self.gp_transient.len()
    }

    #[inline]
    pub(super) fn fp_transient(&self, index: usize) -> Option<MachineReg> {
        self.fp_transient.get(index).copied()
    }

    pub(super) fn fp_transient_count(&self) -> usize {
        self.fp_transient.len()
    }

    #[inline]
    pub(super) fn fp_local_cache(&self, index: usize) -> Option<MachineReg> {
        self.fp_local_cache.get(index).copied()
    }

    pub(super) fn fp_local_cache_count(&self) -> usize {
        self.fp_local_cache.len()
    }

    #[inline]
    pub(super) fn first_fp_reg(&self) -> u16 {
        self.first_fp_reg
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
