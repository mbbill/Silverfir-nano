use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

use crate::error::WasmError;
use crate::vm::backend::active_backend;
use crate::vm::entities::FunctionInst;
use crate::vm::interp::stack::InterpreterStack;
use crate::vm::store::Store;
use crate::vm::value::Value;

#[cfg(feature = "micro-jit")]
use crate::vm::backend::BackendKind;
#[cfg(feature = "micro-jit")]
use crate::vm::native;

pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    #[cfg(feature = "micro-jit")]
    {
        if matches!(active_backend(), Ok(BackendKind::Native)) {
            if let FunctionInst::Local { spec, .. } = func_inst {
                #[cfg(feature = "lockstep-debug")]
                {
                    let func_idx = find_local_function_index(store, spec)?;
                    return eval_native_with_lockstep(func_idx, store, args);
                }

                #[cfg(not(feature = "lockstep-debug"))]
                {
                if !spec.has_native_code() {
                    native::precompile::precompile_module(store)?;
                }
                if !spec.has_native_code() {
                    let module = store.module();
                    if let Some((func_idx, _)) = module
                        .functions
                        .iter()
                        .enumerate()
                        .find(|(_, candidate)| match candidate {
                            FunctionInst::Local { spec: candidate_spec, .. } => {
                                core::ptr::eq(candidate_spec, spec)
                            }
                            FunctionInst::External { .. } => false,
                        })
                    {
                        native::compiler::build_for_function(
                            spec,
                            Some(&module.types),
                            store,
                            module,
                            func_idx as u32,
                        )?;
                    }
                }
                if spec.has_native_code() {
                    return native::runtime::eval(func_inst, store, args);
                }
                return Err(WasmError::invalid(
                    "requested backend native is unavailable for this function".into(),
                ));
                }
            }
        }
    }

    crate::vm::interp::fast::runtime::eval(func_inst, store, args)
}

#[cfg(feature = "micro-jit")]
fn find_local_function_index(
    store: &Store,
    spec: &crate::module::entities::FunctionSpec,
) -> Result<u32, WasmError> {
    store
        .module()
        .functions
        .iter()
        .enumerate()
        .find_map(|(idx, candidate)| match candidate {
            FunctionInst::Local { spec: candidate_spec, .. } if core::ptr::eq(candidate_spec, spec) => {
                Some(idx as u32)
            }
            _ => None,
        })
        .ok_or_else(|| WasmError::internal("local function missing from module".into()))
}

#[cfg(all(feature = "micro-jit", feature = "lockstep-debug"))]
fn eval_native_with_lockstep(
    func_idx: u32,
    store: &mut Store,
    args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    let mut fast_store = store.clone();

    let fast_func_ptr = fast_store.function(func_idx as usize) as *const FunctionInst;
    let fast_func = unsafe { &*fast_func_ptr };
    let native_func_ptr = store.function(func_idx as usize) as *const FunctionInst;
    let native_func = unsafe { &*native_func_ptr };

    let fast_result = crate::vm::interp::fast::runtime::eval(fast_func, &mut fast_store, args);
    let native_result = crate::vm::native::runtime::eval(native_func, store, args);

    let results_len = native_func.func_type().results().len();
    let fast_snapshot = capture_final_snapshot(&fast_store, fast_result.as_ref(), results_len);
    let native_snapshot = capture_final_snapshot(store, native_result.as_ref(), results_len);

    if fast_snapshot != native_snapshot {
        return Err(WasmError::internal(format!(
            "lockstep divergence: fast={:?}, native={:?}",
            fast_snapshot, native_snapshot
        )));
    }

    native_result
}

#[cfg(all(feature = "micro-jit", feature = "lockstep-debug"))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalSnapshot {
    result_values: Option<Vec<u64>>,
    globals: Vec<u64>,
    memory_page_hashes: Vec<u64>,
    error: Option<String>,
}

#[cfg(all(feature = "micro-jit", feature = "lockstep-debug"))]
fn capture_final_snapshot(
    store: &Store,
    result: Result<&InterpreterStack, &WasmError>,
    results_len: usize,
) -> FinalSnapshot {
    let result_values = result
        .as_ref()
        .ok()
        .map(|stack| (0..results_len).map(|i| stack.peek_at_index(i)).collect());
    let error = result.err().map(|err| err.message());
    let globals = store
        .module()
        .globals
        .iter()
        .map(|g| g.value.to_raw())
        .collect();
    let mut memory_page_hashes = Vec::new();
    for mem in &store.module().memories {
        for page in mem.data.chunks(crate::constants::WASM_PAGE_SIZE) {
            memory_page_hashes.push(hash_page(page));
        }
    }

    FinalSnapshot {
        result_values,
        globals,
        memory_page_hashes,
        error,
    }
}

#[cfg(all(feature = "micro-jit", feature = "lockstep-debug"))]
fn hash_page(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}
