pub mod config;
pub mod context;
pub mod hot_local;
pub mod lowered_ir;
pub mod ir_lower;
pub mod plan;
pub mod semantic_ir;
pub mod stack;

pub use config::{CompileConfig, FAST_COMPILE_CONFIG, HOT_LOCAL_COUNT, MAX_HOT_LOCAL_COUNT};
pub use context::CompileContext;
pub use plan::{CompilePlan, HotLocalPlan};
pub use semantic_ir::{SemanticOp, SemanticOpKind};
pub use stack::{BlockKind, ControlFrame, StackTracker};

pub use lowered_ir as ir;
