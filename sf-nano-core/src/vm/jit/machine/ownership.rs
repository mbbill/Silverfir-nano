//! Dynamic-register ownership tracking for MachineIR.
//!
//! The machine backend no longer has a static "linear-value bank" versus
//! "cached-local bank" split. Instead, each dynamic register has a semantic
//! owner at each program point:
//! - `LinearValue` for one linear SSA-like machine value
//! - `CachedCell` for one cached local binding
//!
//! This module tracks that ownership through one block by starting from block
//! params and then replaying instruction defs. Late peepholes use it to guard
//! optimizations that are only sound for linear values.

use crate::collections;

use crate::vm::{
    backend::BackendConfig,
    jit::machine::machine_ir::{
        is_dynamic_reg, MachineBlock, MachineInstKind, MachineReg, MachineRegOwner,
    },
};

/// One mutable ownership tracker for the current block scan.
pub(super) struct DynamicOwnershipTracker {
    owners: collections::Vec<Option<MachineRegOwner>>,
}

impl DynamicOwnershipTracker {
    pub(super) fn new(reg_count: usize) -> Self {
        Self {
            owners: collections::vec![None; reg_count],
        }
    }

    pub(super) fn reset_for_block(&mut self, block: &MachineBlock, config: BackendConfig) {
        for owner in &mut self.owners {
            *owner = None;
        }
        for param in &block.params {
            if is_dynamic_reg(param.reg, config) {
                self.owners[param.reg.0 as usize] = Some(param.owner);
            }
        }
    }

    #[inline]
    pub(super) fn owner(&self, reg: MachineReg, config: BackendConfig) -> Option<MachineRegOwner> {
        if !is_dynamic_reg(reg, config) {
            return None;
        }
        self.owners.get(reg.0 as usize).copied().flatten()
    }

    #[inline]
    pub(super) fn is_linear_value_reg(&self, reg: MachineReg, config: BackendConfig) -> bool {
        self.owner(reg, config) == Some(MachineRegOwner::LinearValue)
    }

    /// Apply one instruction's defs to the current ownership state.
    ///
    /// All defined registers from one instruction share the same semantic
    /// owner. Pair-producing instructions still define one linear machine
    /// value, just spread across two GP dynamic lanes.
    pub(super) fn apply_inst(&mut self, kind: &MachineInstKind, config: BackendConfig) {
        let owner = kind.def_owner();
        kind.for_each_defined_reg(|reg| {
            if is_dynamic_reg(reg, config) {
                self.owners[reg.0 as usize] = owner;
            }
        });
    }
}
