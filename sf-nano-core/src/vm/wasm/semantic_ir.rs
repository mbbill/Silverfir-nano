//! Semantic function-body IR before planning.
//!
//! `PrimitiveOpKind` carries the shared leaf-op vocabulary. `SemanticOpKind` is the
//! larger per-function IR that embeds those leaf ops alongside locals, calls,
//! returns, structured control markers, and branch targets.
//!
//! Important:
//! - no backend-facing `variant`
//! - no `pre_height`
//! - no spill/fill planning artifacts
//! - no backend helper-entry specialization

use alloc::vec::Vec;

use crate::error::WasmError;

use super::common::{BrTableEntry, SemanticIndex, SemanticTarget};
use super::primitive_op::PrimitiveOpKind;

/// One semantic Wasm operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticOp {
    pub kind: SemanticOpKind,
    pub next: Option<SemanticIndex>,
    pub alt: Option<SemanticTarget>,
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
    },
    Else,
    End,
    Br {
        stack_drop: u32,
        arity: u16,
    },
    BrIf {
        stack_drop: u32,
        arity: u16,
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
}

impl SemanticProgram {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self) -> Result<(), WasmError> {
        let len = self.ops.len();

        for (index, op) in self.ops.iter().enumerate() {
            validate_optional_index(op.next, len, "semantic next target")?;
            validate_optional_target(op.alt, len, "semantic alt target")?;

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
                SemanticOpKind::If { .. } => {
                    if op.alt.is_none() {
                        return Err(WasmError::internal(alloc::format!(
                            "semantic if at op {index} is missing else/end target",
                        )));
                    }
                }
                SemanticOpKind::Else => {
                    if op.next.is_none() || op.alt.is_none() {
                        return Err(WasmError::internal(alloc::format!(
                            "semantic else at op {index} is missing body/end target",
                        )));
                    }
                }
                SemanticOpKind::Br { .. } | SemanticOpKind::BrIf { .. } => {
                    if op.alt.is_none() {
                        return Err(WasmError::internal(alloc::format!(
                            "semantic branch at op {index} is missing target",
                        )));
                    }
                }
                SemanticOpKind::BrTable { entries } => {
                    for (entry_index, entry) in entries.iter().enumerate() {
                        let Some(target) = entry.target else {
                            return Err(WasmError::internal(alloc::format!(
                                "semantic br_table at op {index} has missing target for entry {entry_index}",
                            )));
                        };
                        validate_target(target, len, "semantic br_table target")?;
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
            for target in semantic_successors(op) {
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
        | SemanticOpKind::Else
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
fn validate_optional_index(
    index: Option<SemanticIndex>,
    len: usize,
    label: &str,
) -> Result<(), WasmError> {
    if let Some(index) = index {
        validate_index(index, len, label)?;
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_optional_target(
    target: Option<SemanticTarget>,
    len: usize,
    label: &str,
) -> Result<(), WasmError> {
    if let Some(target) = target {
        validate_target(target, len, label)?;
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_index(index: SemanticIndex, len: usize, label: &str) -> Result<(), WasmError> {
    if index.as_usize() >= len {
        return Err(WasmError::internal(alloc::format!(
            "{label} {idx} is out of range for semantic length {len}",
            idx = index.as_usize(),
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_target(target: SemanticTarget, len: usize, label: &str) -> Result<(), WasmError> {
    if target.index().as_usize() >= len {
        return Err(WasmError::internal(alloc::format!(
            "{label} {idx} is out of range for semantic length {len}",
            idx = target.index().as_usize(),
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn semantic_successors(op: &SemanticOp) -> Vec<SemanticTarget> {
    let mut targets = Vec::new();
    let push_next = |targets: &mut Vec<SemanticTarget>, next: Option<SemanticIndex>| {
        if let Some(next) = next {
            targets.push(SemanticTarget::new(next.as_usize()));
        }
    };

    match &op.kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => {}
        SemanticOpKind::Br { .. } => {
            if let Some(target) = op.alt {
                targets.push(target);
            }
        }
        SemanticOpKind::BrIf { .. } => {
            if let Some(target) = op.alt {
                targets.push(target);
            }
            push_next(&mut targets, op.next);
        }
        SemanticOpKind::BrTable { entries } => {
            for entry in entries {
                if let Some(target) = entry.target {
                    targets.push(target);
                }
            }
        }
        SemanticOpKind::If { .. } => {
            push_next(&mut targets, op.next);
            if let Some(target) = op.alt {
                targets.push(target);
            }
        }
        SemanticOpKind::Else => {
            if let Some(target) = op.alt {
                targets.push(target);
            }
        }
        _ => push_next(&mut targets, op.next),
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
                next: None,
                alt: None,
            }],
        };

        let error = semantic.validate().expect_err("semantic validation should fail");
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
                },
                next: None,
                alt: Some(SemanticTarget::new(3)),
            }],
        };

        let error = semantic.validate().expect_err("semantic validation should fail");
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
                next: None,
                alt: None,
            }],
        };

        let error = semantic.validate().expect_err("semantic validation should fail");
        assert!(error.message().contains("does not match function result arity"));
    }
}
