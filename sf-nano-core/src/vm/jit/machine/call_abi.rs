//! Target-independent policy for the internal wasm-to-wasm call ABI.
//!
//! This module owns abstract MachineIR facts: volatility classification,
//! preserved-clobber discovery, and small call-policy constants. Architecture
//! backends remain responsible for mapping abstract registers onto physical
//! registers and emitting the save/restore instructions.

use crate::collections;

use crate::vm::{
    jit::backend::BackendConfig,
    jit::machine::machine_ir::{
        is_preserved_dynamic_reg, MachineCallResults, MachineProgram, MachineReg, MachineResultDst,
        MachineTerminator,
    },
};

/// Temporary GP lanes needed by the current indirect/ref dispatch cluster.
///
/// These are abstract MachineIR lanes. They are not physical backend
/// registers, and this constant should disappear once dispatch state is carried
/// entirely by explicit MachineIR block params and edge args.
pub(super) const INDIRECT_DISPATCH_CONTROL_GP_LANES: usize = 4;

pub(super) fn collect_preserved_clobbers(
    program: &MachineProgram,
    config: BackendConfig,
) -> collections::Vec<MachineReg> {
    let mut regs = collections::Vec::new();
    for block in &program.blocks {
        for param in &block.params {
            push_preserved_clobber(&mut regs, param.reg, config);
        }
        for inst in &block.ops {
            inst.kind.for_each_defined_reg(|reg| {
                push_preserved_clobber(&mut regs, reg, config);
            });
        }
        collect_terminator_preserved_defs(&mut regs, &block.terminator, config);
    }
    regs.sort_by_key(|reg| reg.0);
    regs
}

fn collect_terminator_preserved_defs(
    regs: &mut collections::Vec<MachineReg>,
    term: &MachineTerminator,
    config: BackendConfig,
) {
    let MachineTerminator::Call { results, .. } = term else {
        return;
    };
    match results {
        MachineCallResults::ScalarGp { dst, .. } | MachineCallResults::ScalarFp { dst, .. } => {
            if let MachineResultDst::Reg(reg) = dst {
                push_preserved_clobber(regs, *reg, config);
            }
        }
        MachineCallResults::ScalarGpPair { lo, hi } => {
            if let MachineResultDst::Reg(reg) = lo {
                push_preserved_clobber(regs, *reg, config);
            }
            if let MachineResultDst::Reg(reg) = hi {
                push_preserved_clobber(regs, *reg, config);
            }
        }
        MachineCallResults::None | MachineCallResults::FrameFallback { .. } => {}
    }
}

fn push_preserved_clobber(
    regs: &mut collections::Vec<MachineReg>,
    reg: MachineReg,
    config: BackendConfig,
) {
    if is_preserved_dynamic_reg(reg, config) && !regs.iter().any(|existing| *existing == reg) {
        regs.push(reg);
    }
}
