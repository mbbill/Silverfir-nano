use crate::vm::{
    backend::BackendConfig,
    machine::{
        machine_ir::{MachineFunction, MachineModule},
        peephole,
    },
};

pub(crate) fn optimize_function(function: &mut MachineFunction, config: BackendConfig) {
    peephole::optimize(&mut function.program, config);
}

/// Run ISA-agnostic MachineIR optimization passes on every function in a
/// module. This lives in the outer machine layer so `machine_ir/` remains a
/// passive definition package for backend-facing data structures.
pub(crate) fn optimize_module(module: &mut MachineModule) {
    let config = module.config;
    for func in &mut module.functions {
        optimize_function(func, config);
    }
}
