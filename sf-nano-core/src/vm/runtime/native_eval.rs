use crate::{
    error::WasmError,
    vm::{
        arch,
        build,
        entities::{Caller, FunctionInst},
        stack::InterpreterStack,
        store::Store,
        value::Value,
    },
};

pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    match func_inst {
        FunctionInst::External {
            func_type,
            callback,
        } => {
            if args.len() != func_type.params().len() {
                return Err(WasmError::invalid(alloc::format!(
                    "invalid argument count: got {}, expected {}",
                    args.len(),
                    func_type.params().len()
                )));
            }
            let mut returns = alloc::vec![Value::default(); func_type.results().len()];
            let mem_slice = if store.module().memories.is_empty() {
                None
            } else {
                Some(store.memory_mut(0).data.as_mut_slice())
            };
            let mut caller = Caller::new(mem_slice);
            callback(&mut caller, args, &mut returns)?;

            let mut out = InterpreterStack::with_exact_capacity(returns.len());
            for value in returns {
                out.push(value.to_raw());
            }
            Ok(out)
        }
        FunctionInst::Local { spec, .. } => {
            let active_backend = arch::active_native_backend().map_err(|err| {
                WasmError::invalid(alloc::format!("native backend unavailable: {err}"))
            })?;
            let active_config = arch::active_backend_config().map_err(|err| {
                WasmError::invalid(alloc::format!("native backend unavailable: {err}"))
            })?;
            let backend_name = arch::backend_display_name(active_backend);
            let needs_compile = spec
                .get_native_code()
                .map(|code| {
                    code.compiled().backend_kind() != active_backend
                        || code.compiled().backend() != active_config
                })
                .unwrap_or(true);
            if needs_compile {
                build::ensure_module_compiled(store)?;
            }
            let code = spec.get_native_code().ok_or_else(|| {
                WasmError::internal("native runtime is missing compiled machine code".into())
            })?;
            match active_backend {
                #[cfg(target_arch = "aarch64")]
                arch::NativeBackend::Arm64 => {
                    arch::arm64::eval(spec, code, store, args, arch::NativeBackend::Arm64.as_str())
                }
                #[cfg(target_arch = "arm")]
                arch::NativeBackend::Armv7a => arch::armv7a::eval(
                    spec,
                    code,
                    store,
                    args,
                    arch::NativeBackend::Armv7a.as_str(),
                ),
                #[cfg(target_arch = "x86_64")]
                arch::NativeBackend::X86_64 => arch::x86_64::eval(
                    spec,
                    code,
                    store,
                    args,
                    arch::NativeBackend::X86_64.as_str(),
                ),
                #[cfg(any(
                    debug_assertions,
                    not(any(target_arch = "aarch64", target_arch = "x86_64"))
                ))]
                arch::NativeBackend::Reference => {
                    arch::emulator::eval(spec, code, store, args, backend_name)
                }
            }
        }
    }
}
