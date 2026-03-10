//! Generated IR resolution functions.
//!
//! Maps IrOpKind directly to handler function pointers and encoded immediates.

use crate::vm::lir::{IrOpKind, SlotRef};
use super::encoding;
use super::handler_lookup;
use super::handlers::full_set::*;
use super::handlers::OpHandler as Handler;

include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_ir_resolve.rs"));
