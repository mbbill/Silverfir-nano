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

// An ISA the JIT can emit for is one the interpreter must be able to run on.
// The interpreter is the simpler engine and needs no executable memory, so
// coverage can only legitimately err the other way -- an interpreter on a
// target with no JIT backend is fine, the reverse is a hole.
#[cfg(all(
    sf_interp,
    not(sf_interp_engine),
    any(
        sf_backend_arm64,
        sf_backend_x64,
        sf_backend_riscv32,
        sf_backend_riscv64,
        sf_backend_armv7a,
        sf_backend_thumbm
    )
))]
compile_error!(
    "this target has a JIT backend but no interpreter dispatch engine. The \
     interpreter must cover every ISA the JIT does; add the ISA to `interp_gen/` \
     rather than relaxing this check."
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
pub use config::{Config, ConfigError};
pub use error::WasmError;
pub use module::type_defs::FunctionType;
pub use utils::limits::{Limitable, Limits};
pub use vm::engine::{Engine, Tier};
pub use vm::entities::{Caller, FunctionInst, HostCallback, HostFn};
pub use vm::instance::{
    Func, Import, ImportValue, ImportedFunction, ImportedTableState, ImportedTagState, Instance,
};
// The interpreter's instance is public as the counterpart to
// `JitInstance` -- the escape hatch for what only this engine can answer.
// Its predecoded representation (instructions, opcode enum, operand flags)
// is not: it is how the engine stores a function, not something an
// embedder builds against.
#[cfg(sf_interp)]
pub use vm::interpreter::InterpInstance;
#[cfg(sf_jit)]
pub use vm::jit::arch::active_native_backend_name;
// Compile statistics belong to the JIT *engine*, so they carry its name.
// "Native" is this tree's word for the ISA, which is a different axis --
// see `Engine` versus `active_native_backend_name`.
#[cfg(sf_jit)]
pub use vm::jit::build::{jit_stats_snapshot, JitStatsSnapshot};
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
