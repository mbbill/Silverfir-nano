#![no_std]
#![doc = include_str!("../README.md")]
#![warn(unreachable_pub)]

extern crate alloc;

#[cfg(any(sf_has_std, test))]
extern crate std;

#[cfg(all(test, feature = "memprof"))]
#[global_allocator]
static TEST_ALLOCATOR: tracked_alloc::TrackingAllocator<std::alloc::System> =
    tracked_alloc::TrackingAllocator::new(std::alloc::System);

pub(crate) mod collections;
// At least one execution engine has to be compiled in; a crate that can parse
// and validate Wasm but not run it is not a useful build. Whether the engine
// you asked for can exist on this ISA is a separate question, answered in
// build.rs (`require_supported_isa`) before any of this compiles.
#[cfg(not(any(sf_jit, sf_interp)))]
compile_error!(
    "sf-nano-core needs at least one execution engine: enable the `jit` feature, \
     the `interp` feature, or both (the default)."
);

mod config;
pub(crate) mod constants;
mod error;
pub(crate) mod module;
pub(crate) mod op_decoder;
pub(crate) mod opcodes;
pub(crate) mod utils;
mod value;
pub mod value_type;
pub(crate) mod vm;

#[cfg(sf_wasi_host)]
pub mod wasi;

// Public re-exports for ergonomic API
pub use config::{Config, ConfigError};
pub use constants::WASM_PAGE_SIZE;
pub use error::WasmError;
pub use module::type_defs::FunctionType;
pub use module::Module;
pub use utils::limits::Limits;
pub use vm::engine::{Engine, Tier};
pub use vm::entities::{Caller, HostFn};
pub use vm::imports::{Extern, Import};
pub use vm::instance::{Func, Instance, InstanceInstantiationError, RuntimeWorld};
// Diagnostics expose owned data, never instance bodies or instruction enums.
#[cfg(sf_interp)]
pub use vm::interpreter::InterpreterStats;
#[cfg(sf_jit)]
pub use vm::jit::arch::active_native_backend_name;
// Compile statistics belong to the JIT *engine*, so they carry its name.
// "Native" is this tree's word for the ISA, which is a different axis --
// see `Engine` versus `active_native_backend_name`.
pub use value::{RefValue, Value};
#[cfg(sf_jit)]
pub use vm::jit::build::{jit_stats_snapshot, JitStatsSnapshot};
pub use vm::link::{InstanceId, WorldAccess};
pub use vm::memory::{MemoryView, MemoryViewMut};
pub use vm::tag::TagIdentity;

#[inline]
pub const fn target_has_simd() -> bool {
    cfg!(sf_has_simd)
}
