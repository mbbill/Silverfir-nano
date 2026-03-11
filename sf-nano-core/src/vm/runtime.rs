use crate::error::WasmError;
use crate::vm::backend::{active_backend_mode, resolve_backend_mode, BackendKind};
use crate::vm::entities::FunctionInst;
use crate::vm::store::Store;
use crate::vm::value::Value;

#[cfg(feature = "micro-jit")]
use crate::vm::native;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeEngine {
    Interpreter,
    MicroJit(&'static str),
}

impl RuntimeEngine {
    #[inline]
    pub const fn is_micro_jit(self) -> bool {
        matches!(self, Self::MicroJit(_))
    }

    #[inline]
    pub const fn native_backend(self) -> Option<&'static str> {
        match self {
            Self::Interpreter => None,
            Self::MicroJit(backend) => Some(backend),
        }
    }
}

pub fn active_runtime_engine() -> Result<RuntimeEngine, &'static str> {
    match resolve_backend_mode(active_backend_mode())? {
        BackendKind::Native => {
            #[cfg(feature = "micro-jit")]
            {
                Ok(RuntimeEngine::MicroJit(
                    native::arch::active_native_backend()?.as_str(),
                ))
            }

            #[cfg(not(feature = "micro-jit"))]
            {
                Err("native backend not compiled in")
            }
        }
        _ => Ok(RuntimeEngine::Interpreter),
    }
}

pub fn set_reference_backend(enabled: bool) -> Result<(), &'static str> {
    #[cfg(feature = "micro-jit")]
    {
        native::arch::set_reference_backend(enabled)
    }

    #[cfg(not(feature = "micro-jit"))]
    {
        if enabled {
            Err("reference backend requires micro-jit")
        } else {
            Ok(())
        }
    }
}

pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<crate::vm::interp::stack::InterpreterStack, WasmError> {
    let engine = active_runtime_engine()
        .map_err(|err| WasmError::invalid(alloc::format!("runtime backend unavailable: {err}")))?;

    #[cfg(feature = "micro-jit")]
    {
        if engine.is_micro_jit() {
            return native::runtime::eval(func_inst, store, args);
        }
    }

    crate::vm::interp::runtime::eval(func_inst, store, args)
}
