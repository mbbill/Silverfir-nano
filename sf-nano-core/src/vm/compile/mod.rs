pub mod common;
pub mod context;
pub mod core_op;
pub mod decode;
pub mod semantic_ir;
pub mod stack;

pub use common::{BrTableEntry, OpIndex, SlotRef};
pub use context::CompileContext;
pub use core_op::CoreOpKind;
pub use semantic_ir::{SemanticOp, SemanticOpKind};
pub use stack::{BlockKind, ControlFrame, StackTracker};
