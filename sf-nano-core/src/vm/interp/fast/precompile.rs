//! Two-pass fast IR precompilation for a module.
//!
//! Pass 1: Compile all internal functions with call_internal.
//! Pass 2: Patch call_internal → call_local for same-module calls.

use crate::error::WasmError;
use crate::vm::entities::FunctionInst;
use crate::vm::interp::fast::{
    builder::build_for_function,
    frame_layout,
    handlers::full_set::{op_call_internal, op_call_local},
    instruction::Instruction,
};
use crate::vm::store::Store;

/// Precompile fast IR for all internal functions using two passes.
pub fn precompile_module_two_pass(store: &Store) -> Result<(), WasmError> {
    let module = store.module();

    // If all internal functions are already compiled, skip.
    let all_compiled = module
        .functions
        .iter()
        .filter(|f| !f.is_external())
        .all(|f| {
            f.spec()
                .map(|s| s.has_fast_code())
                .unwrap_or(true)
        });
    if all_compiled {
        return Ok(());
    }

    // Pass 1: Compile all internal functions.
    for (i, func_inst) in module.functions.iter().enumerate().filter(|(_, f)| !f.is_external()) {
        let spec = match func_inst.spec() {
            Some(s) => s,
            None => continue,
        };
        if spec.has_fast_code() {
            continue;
        }
        let _ = build_for_function(spec, Some(&module.types), store, module, i as u32);
    }

    // Pass 2: Patch call_internal → call_local for same-module calls.
    optimize_internal_calls(store);

    Ok(())
}

/// Patch `call_internal` instructions to `call_local` for same-module calls.
fn optimize_internal_calls(store: &Store) {
    let module = store.module();

    for func_inst in module.functions.iter().filter(|f| !f.is_external()) {
        let spec = match func_inst.spec() {
            Some(s) => s,
            None => continue,
        };
        let Some(fast_code) = spec.get_fast_code() else {
            continue;
        };

        // Compute caller's frame layout
        let caller_ft = spec.func_type();
        let caller_params_count = caller_ft.params().len();
        let caller_locals_count = spec.locals().len();
        let caller_frame_size = caller_params_count + caller_locals_count;
        let caller_operand_base_slots = frame_layout::operand_stack_base(caller_frame_size);
        let caller_operand_base_offset = (caller_operand_base_slots * 8) as u32;

        let code = fast_code.code();

        for inst in code.iter() {
            if inst.handler as usize != op_call_internal as usize {
                continue;
            }

            let callee_func_ptr = inst.imm0 as *const FunctionInst;
            let delta = inst.imm1 as u16;
            let callee = unsafe { &*callee_func_ptr };

            let callee_spec = match callee.spec() {
                Some(s) => s,
                None => continue,
            };
            if !callee_spec.has_fast_code() {
                continue;
            }
            let callee_entry = callee_spec.fast_cache().entry() as u64;
            let callee_ft = callee_spec.func_type();
            let callee_params_count = callee_ft.params().len() as u16;
            let callee_locals_count = callee_spec.locals().len() as u16;

            let height = (delta as usize + callee_params_count as usize)
                .saturating_sub(caller_operand_base_slots) as u16;

            let new_imm1 = (callee_params_count as u64)
                | ((callee_locals_count as u64) << 16)
                | ((caller_operand_base_offset as u64) << 32);

            let inst_ptr = inst as *const Instruction as *mut Instruction;
            unsafe {
                (*inst_ptr).handler = op_call_local;
                (*inst_ptr).imm0 = callee_entry;
                (*inst_ptr).imm1 = new_imm1;
                (*inst_ptr).imm2 = height as u64;
            }
        }
    }
}
