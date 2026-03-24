//! Reference-only native backend skeleton.
//!
//! This module is compiled in debug builds to catch structural regressions,
//! but is not wired into the runtime backend dispatch. All items are
//! intentionally unused from outside.

mod abi;
mod backend;
mod config;
mod control;
mod enc;
mod inst;
mod operands;
mod reg;
mod select;
