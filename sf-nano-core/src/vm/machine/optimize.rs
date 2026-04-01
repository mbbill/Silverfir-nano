use crate::vm::machine::{machine_ir::MachineModule, peephole};

/// Run ISA-agnostic MachineIR optimization passes on every function in a
/// module. This lives in the outer machine layer so `machine_ir/` remains a
/// passive definition package for backend-facing data structures.
pub(crate) fn optimize_module(module: &mut MachineModule) {
    let config = module.config;
    for func in &mut module.functions {
        peephole::optimize(&mut func.program, config);
    }
}
