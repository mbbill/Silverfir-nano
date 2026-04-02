use crate::error::WasmError;
use crate::vm::backend::{active_backend_mode, resolve_backend_mode, BackendKind};
use crate::vm::entities::FunctionInst;
use crate::vm::result_buffer::ResultBuffer;
use crate::vm::store::Store;
use crate::vm::value::Value;

#[cfg(feature = "micro-jit")]
pub use crate::vm::arch::ReferenceBackendMode;
#[cfg(feature = "interp")]
use crate::vm::interp;
#[cfg(not(feature = "micro-jit"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceBackendMode {
    Disabled,
    Emu64,
    Emu32,
}

#[cfg(not(feature = "micro-jit"))]
impl ReferenceBackendMode {
    #[inline]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

// --- Native runtime infrastructure (micro-jit only) ---

#[cfg(feature = "micro-jit")]
pub(crate) mod code;
#[cfg(feature = "micro-jit")]
pub(crate) mod code_buf;
#[cfg(feature = "micro-jit")]
pub(crate) mod common;
#[cfg(feature = "micro-jit")]
pub(crate) mod context;
#[cfg(feature = "micro-jit")]
pub(crate) mod dispatch_view;
#[cfg(feature = "micro-jit")]
pub(crate) mod external;
#[cfg(feature = "micro-jit")]
#[cfg(has_guard_pages)]
pub(crate) mod guard_pages;
#[cfg(feature = "micro-jit")]
pub(crate) mod layout;
#[cfg(feature = "micro-jit")]
pub(crate) mod preserved;
#[cfg(feature = "micro-jit")]
pub(crate) mod profiler;
#[cfg(feature = "micro-jit")]
pub(crate) mod trap;
#[cfg(feature = "micro-jit")]
#[cfg(has_guard_pages)]
pub(crate) mod trap_signal;

#[cfg(feature = "micro-jit")]
mod native_eval;

// --- Dispatch ---

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
                    crate::vm::arch::active_native_backend_name()?,
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
        crate::vm::arch::set_reference_backend(enabled)
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

pub fn set_reference_backend_mode(mode: ReferenceBackendMode) -> Result<(), &'static str> {
    #[cfg(feature = "micro-jit")]
    {
        crate::vm::arch::set_reference_backend_mode(mode)
    }

    #[cfg(not(feature = "micro-jit"))]
    {
        if mode.is_enabled() {
            Err("reference backend requires micro-jit")
        } else {
            Ok(())
        }
    }
}

pub(crate) fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<ResultBuffer, WasmError> {
    let engine = active_runtime_engine()
        .map_err(|err| WasmError::invalid(alloc::format!("runtime backend unavailable: {err}")))?;

    #[cfg(feature = "micro-jit")]
    {
        if engine.is_micro_jit() {
            return native_eval::eval(func_inst, store, args);
        }
    }

    #[cfg(feature = "interp")]
    {
        return interp::runtime::eval(func_inst, store, args);
    }

    #[cfg(not(feature = "interp"))]
    {
        let _ = (func_inst, store, args);
        Err(WasmError::invalid(
            "interpreter backend not compiled in".into(),
        ))
    }
}

#[cfg(feature = "micro-jit")]
use crate::value_type::ValueType;

#[cfg(feature = "micro-jit")]
#[inline]
pub(crate) unsafe fn collect_native_results_from_stack(
    stack_base: *const u64,
    result_types: &[ValueType],
    gp_unit_bytes: u8,
) -> ResultBuffer {
    let mut out = ResultBuffer::with_exact_capacity(result_types.len());
    for (index, ty) in result_types.iter().enumerate() {
        let mut raw = unsafe { *stack_base.add(index) };
        if gp_unit_bytes == 4 && matches!(ty, ValueType::Ref(_)) && raw == u64::from(u32::MAX) {
            raw = usize::MAX as u64;
        }
        out.push(raw);
    }
    out
}
