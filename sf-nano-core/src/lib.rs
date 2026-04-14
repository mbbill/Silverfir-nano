#![no_std]
#![warn(unreachable_pub)]

#[cfg(any(sf_has_std, test))]
extern crate std;

pub(crate) mod collections;
pub mod constants;
pub mod error;
pub mod module;
pub mod op_decoder;
pub mod opcodes;
pub(crate) mod utils;
pub mod value_type;
pub mod vm;

#[cfg(sf_wasi_host)]
pub mod wasi;

// Public re-exports for ergonomic API
pub use error::WasmError;
pub use module::type_defs::FunctionType;
pub use utils::limits::Limitable;
pub use vm::backend::{active_backend, backend_mode, set_backend_mode, BackendKind, BackendMode};
#[cfg(sf_jit)]
pub use vm::build::{
    native_capacity_skips, native_capacity_skips as jit_capacity_skips, native_stats,
    native_stats as jit_stats, native_stats_snapshot, native_stats_snapshot as jit_stats_snapshot,
    NativeStatsSnapshot, NativeStatsSnapshot as JitStatsSnapshot,
};
pub use vm::entities::{Caller, ExternalFn, FunctionInst};
pub use vm::instance::{Import, ImportValue, Instance};
#[cfg(sf_has_guard_pages)]
use vm::runtime::trap_signal;
pub use vm::runtime::{
    active_runtime_engine, set_reference_backend, set_reference_backend_mode, ReferenceBackendMode,
    RuntimeEngine,
};
pub use vm::value::{RefHandle, Value};

/// Reset process-global native runtime state that does not track module
/// lifetimes on its own. Harnesses that repeatedly construct and drop native
/// modules can call this between independent runs.
pub fn reset_native_runtime_state() {
    #[cfg(sf_has_guard_pages)]
    {
        trap_signal::reset_debug_state();
        trap_signal::clear_registered_jit_ranges();
    }
}
