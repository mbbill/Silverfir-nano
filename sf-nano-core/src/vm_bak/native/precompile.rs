use crate::error::WasmError;
use crate::vm::entities::FunctionInst;
use crate::vm::store::Store;

use super::{bridge, build::build_for_function};

const CALL_LOCAL_PATCHED_BIT: u64 = 1u64 << 63;

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

    optimize_internal_calls(store)?;

    Ok(())
}

fn optimize_internal_calls(store: &Store) -> Result<(), WasmError> {
    let module = store.module();
    let mut buf = module
        .native_code_buffer()
        .map_err(|err| WasmError::invalid(err.into()))?;
    let mut patch_start = usize::MAX;
    let mut patch_end = 0usize;
    let mut has_code_patches = false;
    buf.begin_write();

    for func_inst in module.functions.iter().filter(|f| !f.is_external()) {
        let Some(spec) = func_inst.spec() else {
            continue;
        };
        spec.with_native_code_mut(|native_code| {
            for meta in native_code.helper_metadata_mut() {
                if meta.helper as usize != bridge::call_internal_helper as usize {
                    continue;
                }

                let callee = meta.data0 as *const FunctionInst;
                let Some(callee_spec) = (unsafe { &*callee }).spec() else {
                    continue;
                };
                if !callee_spec.has_native_code() {
                    continue;
                }

                let params = callee_spec.func_type().params().len() as u64;
                let locals = callee_spec.locals().len() as u64;
                meta.data0 = callee_spec.native_cache().entry() as usize as u64;
                meta.data2 = CALL_LOCAL_PATCHED_BIT | params | (locals << 16);
            }

            for patch in native_code.direct_call_entry_patches() {
                let callee = unsafe { &*patch.callee };
                let Some(callee_spec) = callee.spec() else {
                    continue;
                };
                if !callee_spec.has_native_code() {
                    continue;
                }

                buf.patch_u64(
                    patch.literal_off,
                    callee_spec.native_cache().entry() as usize as u64,
                );
                patch_start = patch_start.min(patch.literal_off);
                patch_end = patch_end.max(patch.literal_off + 8);
                has_code_patches = true;
            }
        });
    }

    if has_code_patches {
        buf.finish_write(patch_start, patch_end - patch_start);
    } else {
        buf.finish_write(0, 0);
    }

    Ok(())
}
