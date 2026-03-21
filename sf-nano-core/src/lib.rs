#![no_std]
#![allow(unused)]

extern crate alloc;

#[cfg(any(feature = "wasi", feature = "std", test))]
extern crate std;

// No-op log macros (compile to nothing)
macro_rules! log_trace {
    ($($t:tt)*) => {};
}
macro_rules! log_debug {
    ($($t:tt)*) => {};
}
macro_rules! log_info {
    ($($t:tt)*) => {};
}
macro_rules! log_warn {
    ($($t:tt)*) => {};
}
macro_rules! log_error {
    ($($t:tt)*) => {};
}

pub mod constants;
pub mod error;
pub mod module;
pub mod op_decoder;
pub mod opcodes;
pub(crate) mod utils;
pub mod value_type;
pub mod vm;

#[cfg(feature = "wasi")]
pub mod wasi;

// Public re-exports for ergonomic API
pub use error::WasmError;
pub use module::type_defs::FunctionType;
pub use utils::limits::Limitable;
pub use vm::backend::{active_backend, backend_mode, set_backend_mode, BackendKind, BackendMode};
pub use vm::entities::ExternalFn;
pub use vm::instance::{Import, ImportValue, Instance};
#[cfg(feature = "micro-jit")]
pub use vm::machine::{
    native_capacity_skips, native_capacity_skips as jit_capacity_skips, native_stats,
    native_stats as jit_stats, native_stats_snapshot, native_stats_snapshot as jit_stats_snapshot,
    NativeStatsSnapshot, NativeStatsSnapshot as JitStatsSnapshot,
};
pub use vm::runtime::{
    active_runtime_engine, set_reference_backend, set_reference_backend_mode, ReferenceBackendMode,
    RuntimeEngine,
};
pub use vm::value::Value;

/// Reset process-global native runtime state that does not track module
/// lifetimes on its own. Harnesses that repeatedly construct and drop native
/// modules can call this between independent runs.
pub fn reset_native_runtime_state() {
    #[cfg(has_guard_pages)]
    {
        crate::vm::runtime::trap_signal::reset_debug_state();
        crate::vm::runtime::trap_signal::clear_registered_jit_ranges();
    }
}
