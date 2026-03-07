//! Shared operand encoding for lowered IR immediates.

use crate::vm::lowered::{IrOpKind, SlotRef};
use crate::vm::interp::fast::encoding;

include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_operand_encoding.rs"));
