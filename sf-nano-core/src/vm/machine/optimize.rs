use crate::vm::{
    backend::BackendConfig,
    machine::{
        machine_ir::{MachineEdge, MachineFunction, MachineModule, MachineTerminator},
        peephole,
    },
};

pub(crate) fn optimize_function(function: &mut MachineFunction, config: BackendConfig) {
    peephole::optimize(&mut function.program, config);
    shrink_machine_function_storage(function);
}

/// Run ISA-agnostic MachineIR optimization passes on every function in a
/// module. This lives in the outer machine layer so `machine_ir/` remains a
/// passive definition package for backend-facing data structures.
pub(crate) fn optimize_module(module: &mut MachineModule) {
    let config = module.config;
    for func in &mut module.functions {
        optimize_function(func, config);
    }
    module.functions.shrink_to_fit();
    module.consts.shrink_to_fit();
}

fn shrink_machine_function_storage(function: &mut MachineFunction) {
    function.program.fp_reg_init_widths.shrink_to_fit();
    function.program.blocks.shrink_to_fit();
    for block in &mut function.program.blocks {
        block.params.shrink_to_fit();
        block.ops.shrink_to_fit();
        shrink_machine_terminator_storage(&mut block.terminator);
    }
}

fn shrink_machine_terminator_storage(terminator: &mut MachineTerminator) {
    match terminator {
        MachineTerminator::Jump(edge) => shrink_machine_edge_storage(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            shrink_machine_edge_storage(then_edge);
            shrink_machine_edge_storage(else_edge);
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries.iter_mut() {
                shrink_machine_edge_storage(edge);
            }
            entries.shrink_to_fit();
        }
        MachineTerminator::Call { .. }
        | MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => {}
    }
}

fn shrink_machine_edge_storage(edge: &mut MachineEdge) {
    edge.args.shrink_to_fit();
}
