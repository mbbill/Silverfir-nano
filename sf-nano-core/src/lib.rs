#![no_std]
#![warn(unreachable_pub)]

#[cfg(any(sf_has_std, test))]
extern crate std;

pub(crate) mod collections;
pub mod config;
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
pub use config::{runtime_config, set_runtime_config, ConfigError, RuntimeConfig};
pub use error::WasmError;
pub use module::type_defs::FunctionType;
pub use utils::limits::{Limitable, Limits};
pub use vm::backend::{active_backend, backend_mode, set_backend_mode, BackendKind, BackendMode};
#[cfg(sf_jit)]
pub use vm::build::{
    native_capacity_skips, native_capacity_skips as jit_capacity_skips, native_stats,
    native_stats as jit_stats, native_stats_snapshot, native_stats_snapshot as jit_stats_snapshot,
    NativeStatsSnapshot, NativeStatsSnapshot as JitStatsSnapshot,
};
pub use vm::entities::{Caller, FunctionInst, HostFn};
pub use vm::instance::{
    Import, ImportValue, ImportedTableState, ImportedTagState, Instance, InstanceInstantiationError,
};
#[cfg(sf_has_guard_pages)]
use vm::runtime::trap_signal;
pub use vm::runtime::{active_runtime_engine, RuntimeEngine};
pub use vm::store::LinkRegistry;
pub use vm::tag::TagHandle;
pub use vm::value::{RefHandle, Value};

#[inline]
pub const fn target_has_simd() -> bool {
    cfg!(sf_has_simd)
}

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
