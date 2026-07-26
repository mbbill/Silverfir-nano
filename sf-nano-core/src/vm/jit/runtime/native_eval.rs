use crate::collections;
use crate::{
    error::WasmError,
    vm::{
        entities::{Caller, FunctionInst},
        jit::arch,
        jit::build,
        result_buffer::ResultBuffer,
        store::Store,
        value::Value,
        value_encoding::value_to_raw_in_store,
    },
};

pub(super) fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<ResultBuffer, WasmError> {
    match func_inst {
        FunctionInst::Host {
            func_type,
            callback,
        } => {
            if args.len() != func_type.params().len() {
                return Err(WasmError::invalid("invalid argument count"));
            }
            let mut returns = collections::vec![Value::default(); func_type.results().len()];
            let mem_slice = if store.module().memories.is_empty() {
                None
            } else {
                let mem = store.memory_mut(0);
                let ptr = mem.memory_ptr();
                let len = mem.memory_len();
                Some(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
            };
            let mut caller = Caller::new(mem_slice);
            callback.call(&mut caller, args, &mut returns)?;

            let mut out = ResultBuffer::with_exact_capacity(returns.len());
            for value in returns {
                out.push(value_to_raw_in_store(value, store));
            }
            Ok(out)
        }
        FunctionInst::Linked { handle, .. } => {
            let entry = store
                .function_entry_for_handle(*handle)
                .ok_or_else(|| WasmError::internal("linked function handle not found"))?;
            let owner_store = unsafe { entry.store.as_mut() }
                .ok_or_else(|| WasmError::internal("linked function owner no longer available"))?;
            let callee_ptr = owner_store
                .module()
                .functions
                .get(entry.local_index)
                .ok_or_else(|| WasmError::internal("linked function index out of range"))?
                as *const FunctionInst;
            let callee = unsafe { &*callee_ptr };
            eval(callee, owner_store, args)
        }
        FunctionInst::Local { spec, .. } => {
            let active_config = arch::backend_config();
            let needs_compile = spec
                .get_native_code()
                .map(|code| code.compiled().backend() != active_config)
                .unwrap_or(true);
            if needs_compile {
                build::ensure_module_compiled(store)?;
            }
            let code = spec.get_native_code().ok_or_else(|| {
                WasmError::internal("native runtime is missing compiled machine code")
            })?;
            arch::dispatch_eval(spec, code, store, args)
        }
    }
}
