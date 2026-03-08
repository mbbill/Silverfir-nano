use crate::error::WasmError;
use crate::vm::store::Store;

use super::compiler::build_for_function;

pub fn precompile_module(store: &Store) -> Result<(), WasmError> {
    let module = store.module();
    #[cfg(feature = "std")]
    let trace = std::env::var_os("SF_NATIVE_TRACE").is_some();

    for (i, func_inst) in module.functions.iter().enumerate().filter(|(_, f)| !f.is_external()) {
        let Some(spec) = func_inst.spec() else {
            continue;
        };
        if spec.has_native_code() {
            continue;
        }
        #[cfg(feature = "std")]
        if trace {
            std::eprintln!("[native] compiling func {}", i);
        }
        let result = build_for_function(spec, Some(&module.types), store, module, i as u32);
        #[cfg(feature = "std")]
        if trace {
            match &result {
                Ok(_) => std::eprintln!("[native] func {} compiled", i),
                Err(e) => std::eprintln!("[native] func {} unavailable: {}", i, e),
            }
        }
    }

    Ok(())
}
