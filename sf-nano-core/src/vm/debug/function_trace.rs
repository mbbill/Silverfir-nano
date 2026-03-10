//! Sparse function-boundary trace support.
//!
//! Important:
//! - sparse by default
//! - compare only backend-independent semantic state
//! - no runtime materialization solely for trace collection

use alloc::{string::String, vec::Vec};

use crate::vm::{backend::BackendKind, value::Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceEventKind {
    Entry,
    Exit,
    Trap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionTraceId {
    pub ordinal: u64,
    pub func_idx: u32,
    pub call_depth: u32,
    pub kind: TraceEventKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionTraceEvent {
    pub backend: BackendKind,
    pub id: FunctionTraceId,
    pub result_values: Vec<Value>,
    pub globals_hash: u64,
    pub memory_hash: Option<u64>,
    pub trap_text: Option<String>,
}

pub trait FunctionTraceSink {
    fn record(&mut self, event: FunctionTraceEvent);
}
