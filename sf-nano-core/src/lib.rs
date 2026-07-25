#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

#[cfg(any(sf_has_std, test))]
extern crate std;

pub(crate) mod collections;
// At least one execution engine has to be compiled in; a crate that can parse
// and validate Wasm but not run it is not a useful build. This is deliberately
// a FEATURE-level check rather than a backend-level one: a target whose
// interpreter backend is not written yet must still `cargo check`, and it
// already fails cleanly at instantiation.
#[cfg(not(any(sf_jit, sf_interp)))]
compile_error!(
    "sf-nano-core needs at least one execution engine: enable the `jit` feature, \
     the `interp` feature, or both (the default)."
);

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
pub use vm::engine::{engine, set_engine, Engine};
pub use vm::entities::{Caller, FunctionInst, HostCallback, HostFn};
pub use vm::instance::{
    Import, ImportValue, ImportedFunction, ImportedTableState, ImportedTagState, Instance,
};
#[cfg(sf_interp)]
pub use vm::interpreter::{
    predecode_function, HostDispatch as InterpHostDispatch, Instr as InterpInstr, InterpInstance,
    Op as InterpOp, PredecodedFunction, FLAG_A_CONST, FLAG_B_CONST,
};
#[cfg(sf_jit)]
pub use vm::jit::arch::active_native_backend_name;
#[cfg(sf_jit)]
pub use vm::jit::build::{
    native_capacity_skips, native_capacity_skips as jit_capacity_skips, native_stats,
    native_stats as jit_stats, native_stats_snapshot, native_stats_snapshot as jit_stats_snapshot,
    NativeStatsSnapshot, NativeStatsSnapshot as JitStatsSnapshot,
};
#[cfg(sf_jit)]
pub use vm::jit::instance::{InstanceInstantiationError, JitInstance};
#[cfg(sf_has_guard_pages)]
use vm::jit::runtime::trap_signal;
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
