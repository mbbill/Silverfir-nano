use crate::error::WasmError;
use crate::vm::store::Store;

use super::compiler::build_for_function;

pub fn precompile_module(store: &Store) -> Result<(), WasmError> {
    let module = store.module();

    for (i, func_inst) in module.functions.iter().enumerate().filter(|(_, f)| !f.is_external()) {
        let Some(spec) = func_inst.spec() else {
            continue;
        };
        if spec.has_native_code() {
            continue;
        }
        let _ = build_for_function(spec, Some(&module.types), store, module, i as u32);
    }

    Ok(())
}
