#![no_std]
#![allow(unused)]

extern crate alloc;

#[cfg(any(feature = "wasi", feature = "std"))]
extern crate std;

// No-op log macros (compile to nothing)
macro_rules! log_trace { ($($t:tt)*) => {} }
macro_rules! log_debug { ($($t:tt)*) => {} }
macro_rules! log_info  { ($($t:tt)*) => {} }
macro_rules! log_warn  { ($($t:tt)*) => {} }
macro_rules! log_error { ($($t:tt)*) => {} }

pub mod constants;
pub mod error;
pub mod module;
pub mod op_decoder;
pub mod opcodes;
pub mod value_type;
pub(crate) mod utils;
pub mod vm;

#[cfg(feature = "wasi")]
pub mod wasi;

// Public re-exports for ergonomic API
pub use error::WasmError;
pub use module::type_defs::FunctionType;
pub use utils::limits::Limitable;
pub use vm::instance::{Import, ImportValue, Instance};
pub use vm::entities::ExternalFn;
pub use vm::interp::fast::{active_backend, backend_mode, set_backend_mode, BackendKind, BackendMode};
#[cfg(feature = "micro-jit")]
pub use vm::interp::fast::jit::group::{jit_stats, jit_capacity_skips, jit_stats_snapshot, JitStatsSnapshot};
pub use vm::value::Value;
