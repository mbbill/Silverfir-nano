//! Decoded semantic function-body model.
//!
//! `PrimitiveOpKind` carries the shared leaf-op vocabulary. `SemanticOpKind` is the
//! larger per-function IR that embeds those leaf ops alongside locals, calls,
//! returns, structured control markers, and branch targets.
//!
//! Important:
//! - no frame-slot layout
//! - no spill/fill preparation artifacts
//! - no local-cache decisions
//! - no prepared block shaping or machine placement

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::WasmError;
use crate::value_type::ValueType;

use super::common::{BrTableEntry, SemanticTarget};
use super::primitive_op::PrimitiveOpKind;

/// One semantic Wasm operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticOp {
    pub kind: SemanticOpKind,
}

/// Semantic function-body op kind.
///
/// This owns the parts of Wasm that are not just reusable leaf ops: locals,
/// calls, returns, structured control markers, and branch metadata. Ordinary
/// non-structural ops are represented as `Primitive(PrimitiveOpKind)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticOpKind {
    Primitive(PrimitiveOpKind),
    LocalGet {
        idx: u16,
    },
    LocalSet {
        idx: u16,
    },
    LocalTee {
        idx: u16,
    },
    Block {
        params: u16,
        results: u16,
    },
    Loop {
        params: u16,
        results: u16,
    },
    If {
        params: u16,
        results: u16,
        else_target: SemanticTarget,
    },
    Else {
        end_target: SemanticTarget,
    },
    End,
    Br {
        stack_drop: u32,
        arity: u16,
        target: SemanticTarget,
    },
    BrIf {
        stack_drop: u32,
        arity: u16,
        target: SemanticTarget,
    },
    BrTable {
        entries: Vec<BrTableEntry>,
    },
    CallExternal {
        func_idx: u32,
        params: u16,
        results: u16,
    },
    CallInternal {
        callee: u32,
        params: u16,
        results: u16,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        params: u16,
        results: u16,
    },
    ReturnVoid,
    ReturnOne,
    Return {
        arity: u16,
    },
}

/// Semantic program for one function body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticProgram {
    pub params: u16,
    pub results: u16,
    pub local_count: u16,
    pub max_stack_height: u16,
    pub ops: alloc::vec::Vec<SemanticOp>,
    /// Per-local value types (params ++ non-param locals).
    ///
    /// When non-empty, `local_types.len() == local_count`.
    /// When empty, downstream consumers should treat all values as untyped.
    pub local_types: Vec<ValueType>,
    /// Per-op result types for calls and block-type-producing ops.
    ///
    /// Maps semantic op index to result value types for ops that push
    /// results whose types are not deterministic from the opcode alone
    /// (calls, blocks with multi-value signatures, etc.).
    pub op_result_types: BTreeMap<usize, Vec<ValueType>>,
}

impl SemanticProgram {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self) -> Result<(), WasmError> {
        let len = self.ops.len();

        for (index, op) in self.ops.iter().enumerate() {
            match &op.kind {
                SemanticOpKind::LocalGet { idx }
                | SemanticOpKind::LocalSet { idx }
                | SemanticOpKind::LocalTee { idx } => {
                    if *idx >= self.local_count {
                        return Err(WasmError::internal(alloc::format!(
                            "semantic op {index} uses out-of-range local {idx} (local_count={})",
                            self.local_count,
                        )));
                    }
                }
                SemanticOpKind::If { else_target, .. } => {
                    validate_target(*else_target, len, "semantic if else target")?;
                }
                SemanticOpKind::Else { end_target } => {
                    validate_target(*end_target, len, "semantic else end target")?;
                }
                SemanticOpKind::Br { target, .. } | SemanticOpKind::BrIf { target, .. } => {
                    validate_target(*target, len, "semantic branch target")?;
                }
                SemanticOpKind::BrTable { entries } => {
                    for (entry_index, entry) in entries.iter().enumerate() {
                        validate_target(
                            entry.target,
                            len,
                            alloc::format!("semantic br_table target for entry {entry_index}")
                                .as_str(),
                        )?;
                    }
                }
                SemanticOpKind::ReturnVoid if self.results != 0 => {
                    return Err(WasmError::internal(alloc::format!(
                        "semantic return_void at op {index} does not match function result arity {}",
                        self.results,
                    )));
                }
                SemanticOpKind::ReturnOne if self.results != 1 => {
                    return Err(WasmError::internal(alloc::format!(
                        "semantic return_one at op {index} does not match function result arity {}",
                        self.results,
                    )));
                }
                SemanticOpKind::Return { arity } if *arity != self.results => {
                    return Err(WasmError::internal(alloc::format!(
                        "semantic return at op {index} has arity {arity}, expected {}",
                        self.results,
                    )));
                }
                _ => {}
            }
        }

        // Validate local_types length when present.
        if !self.local_types.is_empty() && self.local_types.len() != self.local_count as usize {
            return Err(WasmError::internal(alloc::format!(
                "semantic local_types length {} does not match local_count {}",
                self.local_types.len(),
                self.local_count,
            )));
        }

        // Validate op_result_types indices are in range and match arity.
        for (op_idx, result_types) in &self.op_result_types {
            if *op_idx >= len {
                return Err(WasmError::internal(alloc::format!(
                    "op_result_types entry for op index {} is out of range (ops len={})",
                    op_idx, len,
                )));
            }
            if let Some(op) = self.ops.get(*op_idx) {
                let expected_results = match &op.kind {
                    SemanticOpKind::CallExternal { results, .. }
                    | SemanticOpKind::CallInternal { results, .. }
                    | SemanticOpKind::CallIndirect { results, .. } => *results as usize,
                    SemanticOpKind::Primitive(kind) => {
                        super::primitive_op::stack_effect(kind).1 as usize
                    }
                    _ => {
                        return Err(WasmError::internal(alloc::format!(
                            "op_result_types entry at op {} attached to a non-producing op",
                            op_idx,
                        )));
                    }
                };
                if result_types.len() != expected_results {
                    return Err(WasmError::internal(alloc::format!(
                        "op_result_types at op {} has {} types but op expects {} results",
                        op_idx,
                        result_types.len(),
                        expected_results,
                    )));
                }
            }
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn validate(&self) -> Result<(), WasmError> {
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    pub fn reachable_ops(&self) -> Vec<bool> {
        if self.ops.is_empty() {
            return Vec::new();
        }

        let mut reachable = alloc::vec![false; self.ops.len()];
        let mut pending = alloc::vec![0usize];

        while let Some(index) = pending.pop() {
            if reachable.get(index).copied().unwrap_or(true) {
                continue;
            }
            reachable[index] = true;

            let Some(op) = self.ops.get(index) else {
                continue;
            };
            for target in semantic_successors(index, self.ops.len(), op) {
                let target_index = target.index().as_usize();
                if target_index < reachable.len() && !reachable[target_index] {
                    pending.push(target_index);
                }
            }
        }

        reachable
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn reachable_ops(&self) -> Vec<bool> {
        Vec::new()
    }
}

impl From<PrimitiveOpKind> for SemanticOpKind {
    #[inline]
    fn from(kind: PrimitiveOpKind) -> Self {
        Self::Primitive(kind)
    }
}

/// Semantic stack effect.
#[inline]
pub fn stack_effect(kind: &SemanticOpKind) -> (u8, u8) {
    match kind {
        SemanticOpKind::Primitive(kind) => super::primitive_op::stack_effect(kind),
        SemanticOpKind::LocalGet { .. } => (0, 1),
        SemanticOpKind::LocalSet { .. } => (1, 0),
        SemanticOpKind::LocalTee { .. } => (0, 0),
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::Else { .. }
        | SemanticOpKind::End => (0, 0),
        SemanticOpKind::If { .. } => (1, 0),
        SemanticOpKind::Br { .. } => (0, 0),
        SemanticOpKind::BrIf { .. } | SemanticOpKind::BrTable { .. } => (1, 0),
        SemanticOpKind::CallExternal { .. } | SemanticOpKind::CallInternal { .. } => (0, 0),
        SemanticOpKind::CallIndirect { .. } => (1, 0),
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => {
            (0, 0)
        }
    }
}

#[cfg(any(debug_assertions, test))]
fn validate_target(target: SemanticTarget, len: usize, label: &str) -> Result<(), WasmError> {
    if target.is_pending() {
        return Err(WasmError::internal(alloc::format!(
            "{label} is still pending at semantic validation time",
        )));
    }
    if target.index().as_usize() >= len {
        return Err(WasmError::internal(alloc::format!(
            "{label} {idx} is out of range for semantic length {len}",
            idx = target.index().as_usize(),
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn semantic_successors(index: usize, len: usize, op: &SemanticOp) -> Vec<SemanticTarget> {
    let mut targets = Vec::new();
    let push_fallthrough = |targets: &mut Vec<SemanticTarget>| {
        if index + 1 < len {
            targets.push(SemanticTarget::new(index + 1));
        }
    };

    match &op.kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {}
        SemanticOpKind::Br { target, .. } => {
            targets.push(*target);
        }
        SemanticOpKind::BrIf { target, .. } => {
            targets.push(*target);
            push_fallthrough(&mut targets);
        }
        SemanticOpKind::BrTable { entries } => {
            for entry in entries {
                targets.push(entry.target);
            }
        }
        SemanticOpKind::If { else_target, .. } => {
            push_fallthrough(&mut targets);
            targets.push(*else_target);
        }
        SemanticOpKind::Else { end_target } => {
            targets.push(*end_target);
        }
        _ => push_fallthrough(&mut targets),
    }

    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_range_local_access() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 1,
            max_stack_height: 0,
            ops: alloc::vec![SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 1 },
            }],
            local_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let error = semantic
            .validate()
            .expect_err("semantic validation should fail");
        assert!(error.message().contains("out-of-range local"));
    }

    #[test]
    fn rejects_out_of_range_branch_target() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 0,
            max_stack_height: 0,
            ops: alloc::vec![SemanticOp {
                kind: SemanticOpKind::Br {
                    stack_drop: 0,
                    arity: 0,
                    target: SemanticTarget::new(3),
                },
            }],
            local_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let error = semantic
            .validate()
            .expect_err("semantic validation should fail");
        assert!(error.message().contains("out of range"));
    }

    #[test]
    fn rejects_mismatched_return_arity() {
        let semantic = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 1,
            ops: alloc::vec![SemanticOp {
                kind: SemanticOpKind::ReturnVoid,
            }],
            local_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let error = semantic
            .validate()
            .expect_err("semantic validation should fail");
        assert!(error
            .message()
            .contains("does not match function result arity"));
    }
}
