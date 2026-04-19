use crate::error::WasmError;
use crate::vm::backend::{active_backend_mode, resolve_backend_mode, BackendKind};
use crate::vm::entities::FunctionInst;
use crate::vm::result_buffer::ResultBuffer;
use crate::vm::store::Store;
use crate::vm::value::Value;
use crate::vm::value_encoding::normalize_machine_raw_in_store;

// --- Native runtime infrastructure (jit only) ---

#[cfg(sf_jit)]
pub(crate) mod code;
#[cfg(sf_jit)]
pub(crate) mod code_buf;
#[cfg(sf_jit)]
pub(crate) mod common;
#[cfg(sf_jit)]
pub(crate) mod context;
#[cfg(sf_jit)]
pub(crate) mod dispatch_view;
#[cfg(sf_jit)]
#[cfg(sf_has_guard_pages)]
pub(crate) mod guard_pages;
#[cfg(sf_jit)]
pub(crate) mod layout;
#[cfg(sf_jit)]
pub(crate) mod os;
#[cfg(sf_jit)]
pub(crate) mod preserved;
#[cfg(sf_jit)]
pub(crate) mod runtime_call;
#[cfg(sf_jit)]
pub(crate) mod trap;
#[cfg(sf_jit)]
#[cfg(sf_has_guard_pages)]
pub(crate) mod trap_signal;

#[cfg(sf_jit)]
mod native_eval;

// --- Dispatch ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeEngine {
    Jit(&'static str),
}

impl RuntimeEngine {
    #[inline]
    pub const fn is_jit(self) -> bool {
        matches!(self, Self::Jit(_))
    }

    #[inline]
    pub const fn native_backend(self) -> Option<&'static str> {
        match self {
            Self::Jit(backend) => Some(backend),
        }
    }
}

pub fn active_runtime_engine() -> Result<RuntimeEngine, &'static str> {
    match resolve_backend_mode(active_backend_mode())? {
        BackendKind::Native => {
            #[cfg(sf_jit)]
            {
                Ok(RuntimeEngine::Jit(
                    crate::vm::arch::active_native_backend_name()?,
                ))
            }

            #[cfg(not(sf_jit))]
            {
                Err("native backend not compiled in")
            }
        }
    }
}

pub(crate) fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<ResultBuffer, WasmError> {
    let engine = active_runtime_engine()
        .map_err(|_err| WasmError::invalid("runtime backend unavailable"))?;

    #[cfg(sf_jit)]
    {
        if engine.is_jit() {
            return native_eval::eval(func_inst, store, args);
        }
    }

    let _ = (engine, func_inst, store, args);
    Err(WasmError::invalid(
        "no execution backend compiled in".into(),
    ))
}

#[cfg(sf_jit)]
use crate::value_type::ValueType;

#[cfg(sf_jit)]
#[inline]
pub(crate) unsafe fn collect_native_results_from_stack(
    stack_base: *const u64,
    result_types: &[ValueType],
    gp_unit_bytes: u8,
    store: &mut Store,
) -> Result<ResultBuffer, WasmError> {
    let mut out = ResultBuffer::with_exact_capacity(result_types.len());
    for (index, ty) in result_types.iter().enumerate() {
        let raw = normalize_machine_raw_in_store(
            unsafe { *stack_base.add(index) },
            *ty,
            gp_unit_bytes,
            store,
        )?;
        out.push(raw);
    }
    Ok(out)
}
