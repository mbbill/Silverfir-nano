//! Backend planning inputs shared by the fast/fusion/native family.
//!
//! This layer decides cache width, hot locals, and frame-layout related compile
//! configuration before semantic IR is lowered into backend-specific IR.

pub mod config;
pub mod hot_local;
pub mod plan;
pub mod tos;

pub use config::{CompileConfig, FAST_COMPILE_CONFIG, HOT_LOCAL_COUNT, MAX_HOT_LOCAL_COUNT};
pub use plan::{CompilePlan, HotLocalPlan};
pub use tos::TosState;
