//! Planning-stage IR and public planning inputs/outputs.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::wasm::{common::SemanticTarget, primitive_op::PrimitiveOpKind, semantic_ir::SemanticProgram},
};

use super::{
    config::PlanConfig,
    frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
    group::GroupPlan,
    hot_local::HotLocalPlan,
    policy::PlanPolicy,
    tos::{SpillArtifact, TosRotation},
};

/// Planned local placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedLocal {
    Hot(u8),
    Frame(FrameSlot),
}

/// Planning-stage op vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedOpKind {
    Primitive(PrimitiveOpKind),
    InitHotLocals {
        inits: Vec<PlannedHotLocalInit>,
    },
    Spill(SpillArtifact),
    LocalGet {
        local: PlannedLocal,
    },
    LocalSet {
        local: PlannedLocal,
    },
    LocalTee {
        local: PlannedLocal,
    },
    Marker(PlannedMarkerKind),
    Branch {
        kind: PlannedBranchKind,
        condition_slot: Option<FrameSlot>,
        payload: Option<FrameSpan>,
        target: Option<SemanticTarget>,
    },
    BrTable {
        index_slot: FrameSlot,
        entries: Vec<PlannedBrTableEntry>,
    },
    CallExternal {
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    CallInternal {
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
    },
    Return {
        results: Option<FrameSpan>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedHotLocalInit {
    pub reg: u8,
    pub frame_slot: FrameSlot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedMarkerKind {
    Block { params: u16, results: u16 },
    Loop { params: u16, results: u16 },
    If { params: u16, results: u16 },
    Else,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedBranchKind {
    Br,
    BrIf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedBrTableEntry {
    pub target: Option<SemanticTarget>,
    pub payload: Option<FrameSpan>,
}

/// One planned op. This is still stack-aware, but the stack-aware concepts live
/// here instead of leaking into LIR/backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedOp {
    pub kind: PlannedOpKind,
    pub rotation: TosRotation,
    pub height: u16,
    pub alt: Option<SemanticTarget>,
}

/// Planned program bundle.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlannedProgram {
    pub frame: FrameLayoutPlan,
    pub hot_locals: Option<HotLocalPlan>,
    pub ops: Vec<PlannedOp>,
    pub groups: GroupPlan,
}

/// Planning input bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningInput {
    pub config: PlanConfig,
    pub policy: PlanPolicy,
}

impl PlannedProgram {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self, semantic: &SemanticProgram, config: PlanConfig) -> Result<(), WasmError> {
        validate_frame_layout(self.frame, semantic, config)?;
        validate_groups(&self.groups, semantic.ops.len())?;

        if let Some(hot_locals) = &self.hot_locals {
            hot_locals.validate(semantic.local_count, config.hot_local_count)?;
        }

        let expected_init_count = self
            .hot_locals
            .as_ref()
            .map_or(0usize, |plan| plan.mapping().iter().flatten().count());
        let mut seen_init = 0usize;
        let reachable = semantic.reachable_ops();
        let mut semantic_index = 0usize;

        for (index, op) in self.ops.iter().enumerate() {
            let op_reachable = reachable.get(semantic_index).copied().unwrap_or(false);
            validate_rotation(op.rotation, config.tos_register_count, index)?;
            validate_optional_target(op.alt, semantic.ops.len(), "planned op alt target")?;

            match &op.kind {
                PlannedOpKind::Primitive(_) | PlannedOpKind::Marker(_) => {}
                PlannedOpKind::InitHotLocals { inits } => {
                    if index != 0 {
                        return Err(WasmError::internal(
                            "InitHotLocals must be the first planned op".into(),
                        ));
                    }
                    seen_init += 1;
                    for init in inits {
                        if let Some(hot_locals) = &self.hot_locals {
                            if init.reg as usize >= hot_locals.slot_count() {
                                return Err(WasmError::internal(alloc::format!(
                                    "hot-local init uses out-of-range reg {}",
                                    init.reg,
                                )));
                            }
                        } else {
                            return Err(WasmError::internal(
                                "planned InitHotLocals has no hot-local mapping".into(),
                            ));
                        }

                        if !slot_in_span(init.frame_slot, self.frame.locals) {
                            return Err(WasmError::internal(
                                "hot-local init must source from a frame local slot".into(),
                            ));
                        }
                    }
                }
                PlannedOpKind::Spill(artifact) => match artifact {
                    SpillArtifact::TosTransfer(transfer) => {
                        if op_reachable && !span_in_span(transfer.frame, self.frame.operands) {
                            return Err(WasmError::internal(
                                "spill/fill artifact must use operand slots".into(),
                            ));
                        }
                    }
                },
                PlannedOpKind::LocalGet { local }
                | PlannedOpKind::LocalSet { local }
                | PlannedOpKind::LocalTee { local } => {
                    validate_local(local, self, semantic, config)?;
                }
                PlannedOpKind::Branch {
                    condition_slot,
                    payload,
                    target,
                    ..
                } => {
                    let target = target.ok_or_else(|| {
                        WasmError::internal("planned branch is missing target".into())
                    })?;
                    validate_target(target, semantic.ops.len(), "planned branch target")?;

                    if op_reachable {
                        if let Some(condition_slot) = condition_slot {
                            if !slot_in_span(*condition_slot, self.frame.operands) {
                                return Err(WasmError::internal(
                                    "planned branch condition must use an operand slot".into(),
                                ));
                            }
                        }
                    }
                    if op_reachable {
                        if let Some(payload) = payload {
                            if !span_in_span(*payload, self.frame.operands) {
                                return Err(WasmError::internal(
                                    "planned branch payload must use operand slots".into(),
                                ));
                            }
                        }
                    }
                }
                PlannedOpKind::BrTable { index_slot, entries } => {
                    if op_reachable && !slot_in_span(*index_slot, self.frame.operands) {
                        return Err(WasmError::internal(
                            "planned br_table index must use an operand slot".into(),
                        ));
                    }
                    for entry in entries {
                        let target = entry.target.ok_or_else(|| {
                            WasmError::internal("planned br_table entry is missing target".into())
                        })?;
                        validate_target(target, semantic.ops.len(), "planned br_table target")?;
                        if op_reachable {
                            if let Some(payload) = entry.payload {
                                if !span_in_span(payload, self.frame.operands) {
                                    return Err(WasmError::internal(
                                        "planned br_table payload must use operand slots".into(),
                                    ));
                                }
                            }
                        }
                    }
                }
                PlannedOpKind::CallExternal { args, results, .. }
                | PlannedOpKind::CallInternal { args, results, .. } => {
                    if op_reachable {
                        validate_call_span(*args, self.frame.operands, "planned call args")?;
                        validate_call_span(*results, self.frame.operands, "planned call results")?;
                    }
                }
                PlannedOpKind::CallIndirect {
                    index_slot,
                    args,
                    results,
                    ..
                } => {
                    if op_reachable && !slot_in_span(*index_slot, self.frame.operands) {
                        return Err(WasmError::internal(
                            "planned call_indirect index must use an operand slot".into(),
                        ));
                    }
                    if op_reachable {
                        validate_call_span(*args, self.frame.operands, "planned call args")?;
                        validate_call_span(*results, self.frame.operands, "planned call results")?;
                    }
                }
                PlannedOpKind::Return { results } => {
                    if op_reachable {
                        if let Some(results) = results {
                            validate_call_span(
                                *results,
                                self.frame.operands,
                                "planned return values",
                            )?;
                        }
                    }
                }
            }

            if !matches!(op.kind, PlannedOpKind::InitHotLocals { .. } | PlannedOpKind::Spill(_)) {
                semantic_index += 1;
            }
        }

        if semantic_index != semantic.ops.len() {
            return Err(WasmError::internal(alloc::format!(
                "planned op stream covers {} semantic ops, expected {}",
                semantic_index,
                semantic.ops.len(),
            )));
        }

        if expected_init_count == 0 {
            if seen_init != 0 {
                return Err(WasmError::internal(
                    "planned InitHotLocals exists without mapped hot locals".into(),
                ));
            }
        } else if seen_init != 1 {
            return Err(WasmError::internal(alloc::format!(
                "expected one InitHotLocals op, found {seen_init}",
            )));
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn validate(
        &self,
        _semantic: &SemanticProgram,
        _config: PlanConfig,
    ) -> Result<(), WasmError> {
        Ok(())
    }
}

#[cfg(any(debug_assertions, test))]
fn validate_frame_layout(
    frame: FrameLayoutPlan,
    semantic: &SemanticProgram,
    config: PlanConfig,
) -> Result<(), WasmError> {
    if frame.locals.count != semantic.local_count {
        return Err(WasmError::internal(alloc::format!(
            "planned frame local span {} does not match semantic local count {}",
            frame.locals.count,
            semantic.local_count,
        )));
    }

    if frame.operands.count != semantic.max_stack_height {
        return Err(WasmError::internal(alloc::format!(
            "planned frame operand span {} does not match semantic max stack height {}",
            frame.operands.count,
            semantic.max_stack_height,
        )));
    }

    let backend_reserved = frame.backend_reserved.map_or(0, |span| span.count);
    let expected_backend_reserved =
        u16::from(config.hot_local_count).saturating_sub(semantic.local_count);
    if backend_reserved != expected_backend_reserved {
        return Err(WasmError::internal(alloc::format!(
            "planned backend-reserved span {} does not match expected {}",
            backend_reserved,
            expected_backend_reserved,
        )));
    }

    if let Some(span) = frame.backend_reserved {
        if span.start != frame.locals.end() {
            return Err(WasmError::internal(
                "planned backend-reserved span must start after locals".into(),
            ));
        }
    }

    if frame.operand_base != frame.locals.end().advance(backend_reserved).0 {
        return Err(WasmError::internal(
            "planned operand base does not match locals/backend-reserved layout".into(),
        ));
    }

    if frame.frame_size != frame.operand_base {
        return Err(WasmError::internal(
            "planned frame_size must equal operand_base in the current design".into(),
        ));
    }

    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_groups(groups: &GroupPlan, semantic_len: usize) -> Result<(), WasmError> {
    let mut next_start = 0usize;
    for (index, group) in groups.groups.iter().enumerate() {
        if group.id.as_u32() != index as u32 {
            return Err(WasmError::internal(alloc::format!(
                "group {} has non-sequential id {}",
                index,
                group.id.as_u32(),
            )));
        }
        if group.start >= group.end {
            return Err(WasmError::internal(alloc::format!(
                "group {} has empty or inverted range {}..{}",
                index,
                group.start,
                group.end,
            )));
        }
        if group.end as usize > semantic_len {
            return Err(WasmError::internal(alloc::format!(
                "group {} ends past semantic op count {}",
                index,
                semantic_len,
            )));
        }
        if (group.start as usize) < next_start {
            return Err(WasmError::internal(
                "group ranges must be monotonically increasing".into(),
            ));
        }
        next_start = group.end as usize;
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_rotation(
    rotation: TosRotation,
    tos_register_count: u8,
    op_index: usize,
) -> Result<(), WasmError> {
    if tos_register_count == 0 {
        if rotation.0 != 0 {
            return Err(WasmError::internal(alloc::format!(
                "planned op {op_index} has non-zero rotation without TOS registers",
            )));
        }
    } else if rotation.0 >= tos_register_count {
        return Err(WasmError::internal(alloc::format!(
            "planned op {op_index} has out-of-range rotation {} for TOS width {}",
            rotation.0,
            tos_register_count,
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_local(
    local: &PlannedLocal,
    program: &PlannedProgram,
    semantic: &SemanticProgram,
    config: PlanConfig,
) -> Result<(), WasmError> {
    match local {
        PlannedLocal::Hot(reg) => {
            if *reg >= config.hot_local_count {
                return Err(WasmError::internal(alloc::format!(
                    "planned hot local reg {} exceeds configured hot-local count {}",
                    reg,
                    config.hot_local_count,
                )));
            }
            let Some(hot_locals) = &program.hot_locals else {
                return Err(WasmError::internal(
                    "planned hot-local use is missing hot-local mapping".into(),
                ));
            };
            if *reg as usize >= hot_locals.slot_count() {
                return Err(WasmError::internal(alloc::format!(
                    "planned hot local reg {} exceeds mapping width {}",
                    reg,
                    hot_locals.slot_count(),
                )));
            }
            if hot_locals.mapping()[*reg as usize].is_none() {
                return Err(WasmError::internal(alloc::format!(
                    "planned hot local reg {} has no mapped semantic local",
                    reg,
                )));
            }
        }
        PlannedLocal::Frame(slot) => {
            if !slot_in_span(*slot, program.frame.locals) {
                return Err(WasmError::internal(alloc::format!(
                    "planned frame local slot {} is outside local span 0..{}",
                    slot.0,
                    program.frame.locals.end().0,
                )));
            }
            if slot.0 >= semantic.local_count {
                return Err(WasmError::internal(alloc::format!(
                    "planned frame local slot {} exceeds semantic local count {}",
                    slot.0,
                    semantic.local_count,
                )));
            }
        }
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_call_span(span: FrameSpan, operands: FrameSpan, label: &str) -> Result<(), WasmError> {
    if !span_in_span(span, operands) {
        return Err(WasmError::internal(alloc::format!(
            "{label} must live inside the operand span",
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_optional_target(
    target: Option<SemanticTarget>,
    semantic_len: usize,
    label: &str,
) -> Result<(), WasmError> {
    if let Some(target) = target {
        validate_target(target, semantic_len, label)?;
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_target(
    target: SemanticTarget,
    semantic_len: usize,
    label: &str,
) -> Result<(), WasmError> {
    if target.index().as_usize() >= semantic_len {
        return Err(WasmError::internal(alloc::format!(
            "{label} {} is out of range for semantic length {semantic_len}",
            target.index().as_usize(),
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn slot_in_span(slot: FrameSlot, span: FrameSpan) -> bool {
    slot.0 >= span.start.0 && slot.0 < span.end().0
}

#[cfg(any(debug_assertions, test))]
fn span_in_span(span: FrameSpan, outer: FrameSpan) -> bool {
    span.start.0 >= outer.start.0 && span.end().0 <= outer.end().0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::{
        backend::BackendKind,
        plan::{
            frame::FramePlanner,
            group::{GroupPlan, GroupRange, GroupTerminator},
            tos::TosRotation,
        },
        wasm::semantic_ir::{SemanticOp, SemanticOpKind},
    };

    #[test]
    fn rejects_hot_local_use_without_mapping() {
        let frame = FramePlanner::new(1).finish();
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 1,
            max_stack_height: 0,
            ops: alloc::vec![SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 0 },
                next: None,
                alt: None,
            }],
        };
        let config = PlanConfig::for_backend(
            BackendKind::Base,
            crate::vm::backend::BackendConfig {
                ctx_register_count: 1,
                fp_register_count: 1,
                tmp_register_count: 0,
                hot_local_count: 1,
                tos_register_count: 0,
            },
        );
        let planned = PlannedProgram {
            frame,
            hot_locals: None,
            ops: alloc::vec![PlannedOp {
                kind: PlannedOpKind::LocalGet {
                    local: PlannedLocal::Hot(0),
                },
                rotation: TosRotation::new(0),
                height: 0,
                alt: None,
            }],
            groups: GroupPlan::default(),
        };

        let error = planned
            .validate(&semantic, config)
            .expect_err("planned validation should fail");
        assert!(error.message().contains("missing hot-local mapping"));
    }

    #[test]
    fn rejects_out_of_range_group() {
        let frame = FramePlanner::new(0).finish();
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 0,
            max_stack_height: 0,
            ops: alloc::vec![SemanticOp {
                kind: SemanticOpKind::End,
                next: None,
                alt: None,
            }],
        };
        let config = PlanConfig::for_backend(
            BackendKind::Base,
            crate::vm::backend::BackendConfig {
                ctx_register_count: 1,
                fp_register_count: 1,
                tmp_register_count: 0,
                hot_local_count: 0,
                tos_register_count: 0,
            },
        );
        let planned = PlannedProgram {
            frame,
            hot_locals: None,
            ops: alloc::vec![PlannedOp {
                kind: PlannedOpKind::Marker(PlannedMarkerKind::End),
                rotation: TosRotation::new(0),
                height: 0,
                alt: None,
            }],
            groups: GroupPlan {
                groups: alloc::vec![GroupRange {
                    id: super::super::group::GroupId(0),
                    start: 0,
                    end: 2,
                    terminator: Some(GroupTerminator::ReturnVoid),
                }],
            },
        };

        let error = planned
            .validate(&semantic, config)
            .expect_err("planned validation should fail");
        assert!(error.message().contains("ends past semantic op count"));
    }
}
