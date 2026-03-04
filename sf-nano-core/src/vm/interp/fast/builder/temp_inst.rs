//! Intermediate instruction representation.
//!
//! TempInst stores LOGICAL values during compilation.
//! Encoding into imm0/imm1/imm2 happens at finalization time.

use alloc::vec::Vec;
use crate::opcodes::WasmOpcode;
use crate::vm::interp::fast::context::Context;
use crate::vm::interp::fast::encoding::PatternData;
use crate::vm::interp::fast::handlers::NextHandler;
use crate::vm::interp::fast::instruction::Instruction;

/// Handler function type.
/// (ctx, pc, fp, l0, l1, l2, t0, t1, t2, t3, nh) - l0/l1/l2 local cache, 4 TOS, preloaded next handler
pub type Handler = unsafe extern "C" fn(
    *mut Context,
    *mut Instruction,
    *mut u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    NextHandler,
);

/// Entry for br_table: target info for each label.
#[derive(Debug, Clone)]
pub struct BrTableEntry {
    /// Target instruction index (None for forward refs)
    pub target_idx: Option<usize>,
    /// Stack offset for this branch
    pub stack_offset: usize,
    /// Branch arity
    pub arity: usize,
}

/// Temporary instruction during compilation.
///
/// Stores logical field values in `data`, NOT encoded imm values.
/// Encoding happens at finalization time via `finalize_pattern_data()`.
#[derive(Clone)]
pub struct TempInst {
    /// The handler function for this instruction
    pub handler: Handler,
    /// Pattern data - logical field values (NOT encoded)
    pub data: PatternData,
    /// Fallthrough target (next instruction index)
    pub fallthrough_idx: Option<usize>,
    /// Alternate target (for branches, if, else, error paths)
    /// During finalization, this index is converted to a pointer.
    pub alt_idx: Option<usize>,
    /// Whether this instruction encodes a target field that needs pointer patching.
    /// Set for branches (br, br_if, if, else) and fused-with-target patterns.
    pub has_target: bool,
    /// Original opcode (for debugging/patching logic)
    pub wasm_op: WasmOpcode,
    /// BR_TABLE entries - special case, not part of PatternData
    /// because entries need to be converted to inline instruction data
    pub br_table_entries: Option<Vec<BrTableEntry>>,
    /// Stack height before this instruction executes.
    /// Used by JIT group compiler to set initial register window state.
    #[cfg(feature = "micro-jit")]
    pub pre_height: u16,
}

impl TempInst {
    /// Create a new TempInst with the given handler, pattern data, and opcode.
    pub fn new(handler: Handler, data: PatternData, wasm_op: WasmOpcode) -> Self {
        Self {
            handler,
            data,
            fallthrough_idx: None,
            alt_idx: None,
            has_target: false,
            wasm_op,
            br_table_entries: None,
            #[cfg(feature = "micro-jit")]
            pre_height: 0,
        }
    }

    /// Set the fallthrough target index.
    pub fn with_fallthrough(mut self, idx: usize) -> Self {
        self.fallthrough_idx = Some(idx);
        self
    }

    /// Set the alternate target index (for branches).
    pub fn with_alt(mut self, idx: usize) -> Self {
        self.alt_idx = Some(idx);
        self
    }

    /// Mark this instruction as having a target field that needs pointer patching.
    pub fn with_target(mut self) -> Self {
        self.has_target = true;
        self
    }

    /// Set BR_TABLE entries.
    pub fn with_br_table_entries(mut self, entries: Vec<BrTableEntry>) -> Self {
        self.br_table_entries = Some(entries);
        self
    }
}
