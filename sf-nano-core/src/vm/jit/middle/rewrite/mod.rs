//! Final SSA rewrite.
//!
//! The rewriter owns mutable lowering state. Planner consultations are routed
//! through `joint_plan::JointPlanner`, while the actual SSA-IR materialization
//! lives under this folder.

mod edge;
mod function;
mod state;

pub(crate) use function::{rewrite_function, RewriteCfg};
