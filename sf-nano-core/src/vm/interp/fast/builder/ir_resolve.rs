//! Generated IR resolution functions.
//!
//! Maps IrOpKind directly to handler function pointers and encoded immediates.

use super::super::handler_lookup;
use super::ir::{IrOpKind, SlotRef};
use super::super::handlers::full_set::*;
use super::super::handlers::OpHandler as Handler;
use super::super::encoding;

include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_ir_resolve.rs"));
