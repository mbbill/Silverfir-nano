use crate::vm::{
    backend::BackendConfig,
    machine::{
        machine_ir::{MachineEdge, MachineFunction, MachineModule, MachineTerminator},
        peephole,
    },
};

pub(crate) fn optimize_function(
    function: &mut MachineFunction,
    config: BackendConfig,
    full_optimization: bool,
) {
    peephole::optimize(&mut function.program, config, full_optimization);
    shrink_machine_function_storage(function);
}

/// Run ISA-agnostic MachineIR optimization passes on every function in a
/// module. Per-function `full_optimization` flags are supplied by the
/// caller so each function can take its own pipeline (full vs block-by-
/// block) based on the platform's RAM budget. The slice must be at least
/// as long as `module.functions`.
pub(crate) fn optimize_module(module: &mut MachineModule, full_optimization: &[bool]) {
    debug_assert!(
        full_optimization.len() >= module.functions.len(),
        "optimize_module: full_optimization slice ({} entries) is shorter than module.functions ({} entries)",
        full_optimization.len(),
        module.functions.len(),
    );
    let config = module.config;
    for (i, func) in module.functions.iter_mut().enumerate() {
        let full_opt = full_optimization.get(i).copied().unwrap_or(true);
        optimize_function(func, config, full_opt);
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
        | MachineTerminator::Trap { .. } => {}
    }
}

fn shrink_machine_edge_storage(edge: &mut MachineEdge) {
    edge.args.shrink_to_fit();
}
