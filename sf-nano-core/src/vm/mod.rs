//! VM architecture rebuilt from scratch around `docs/BACKEND_IR_REFACTOR.md`.
//!
//! This tree is intentionally architecture-first. The old implementation lives
//! in `vm_bak/` for reference only.
//!
//! Hard boundary:
//! - `wasm/` may think in semantic / stack-machine terms
//! - `plan/` may think in prepared stack-window / spill-fill / block-shaping terms
//! - `lir/` is the backend boundary
//! - after `lir/`, no backend may reintroduce stack-height or spill-depth logic

pub mod abi;
pub mod backend;
pub mod debug;
pub mod entities;
pub mod expr_eval;
pub mod instance;
#[cfg(feature = "interp")]
pub mod interp;
pub mod lir;
#[cfg(feature = "micro-jit")]
pub mod native;
pub mod plan;
pub mod raw_value;
pub mod runtime;
pub mod stack;
pub mod store;
pub mod value;
pub mod wasm;
