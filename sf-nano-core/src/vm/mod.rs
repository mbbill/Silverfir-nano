//! VM architecture rebuilt from scratch around `docs/BACKEND_IR_REFACTOR.md`.
//!
//! This tree is intentionally architecture-first. The old implementation lives
//! in `vm_bak/` for reference only.
//!
//! Hard boundary:
//! - `wasm/` may think in semantic / stack-machine terms
//! - `plan/` may think in rotating-cache / spill-fill / grouping terms
//! - `lir/` is the backend boundary
//! - after `lir/`, no backend may reintroduce stack-height or spill-depth logic

pub mod abi;
pub mod backend;
pub mod debug;
pub mod entities;
pub mod expr_eval;
pub mod instance;
pub mod interp;
pub mod lir;
pub mod native;
pub mod plan;
pub mod runtime;
pub mod store;
pub mod value;
pub mod wasm;
