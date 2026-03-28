//! Structural validation for prepared SSA-IR.

#[cfg(any(debug_assertions, test))]
use alloc::collections::BTreeMap;

use crate::error::WasmError;
#[cfg(any(debug_assertions, test))]
use crate::value_type::ValueType;
#[cfg(any(debug_assertions, test))]
use crate::vm::middle::ssa_ir::ir::SsaOperand;

use super::ir::SsaProgram;
#[cfg(any(debug_assertions, test))]
use super::ir::{SsaBinding, SsaBlock, SsaEdge, SsaValue};
#[cfg(any(debug_assertions, test))]
use super::ir::{SsaInstKind, SsaTerminator};
#[cfg(any(debug_assertions, test))]
use crate::vm::middle::frame::FrameSlot;

#[cfg(any(debug_assertions, test))]
pub(crate) fn validate_program(program: &SsaProgram) -> Result<(), WasmError> {
    if program.blocks.is_empty() {
        if program.entry.as_usize() != 0 {
            return Err(WasmError::internal(
                "empty SSA-IR program must use entry block 0".into(),
            ));
        }
        return Ok(());
    }

    if program.entry.as_usize() >= program.blocks.len() {
        return Err(WasmError::internal(alloc::format!(
            "SSA-IR entry block {} is out of range for {} blocks",
            program.entry.as_usize(),
            program.blocks.len(),
        )));
    }

    for (index, block) in program.blocks.iter().enumerate() {
        validate_block_id(block, index)?;
        validate_params(&block.params, alloc::format!("block b{index} params"))?;

        match &block.terminator {
            SsaTerminator::Goto(edge) => validate_edge(program, edge, index)?,
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                validate_edge(program, then_edge, index)?;
                validate_edge(program, else_edge, index)?;
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    validate_edge(program, edge, index)?;
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }

    for (index, slot) in program.local_cache.gp_preferred_slots.iter().enumerate() {
        if program.local_cache.gp_preferred_slots[..index].contains(slot) {
            return Err(WasmError::internal(alloc::format!(
                "SSA-IR local-cache preferences contain duplicate slot {:?}",
                slot,
            )));
        }
    }
    if program.local_cache.gp_preferred_slots.len() != program.local_cache.gp_preferred_types.len()
    {
        return Err(WasmError::internal(alloc::format!(
            "SSA-IR GP local-cache preferences contain {} slots but {} type entries",
            program.local_cache.gp_preferred_slots.len(),
            program.local_cache.gp_preferred_types.len(),
        )));
    }
    if program.local_cache.fp_preferred_slots.len() != program.local_cache.fp_preferred_types.len()
    {
        return Err(WasmError::internal(alloc::format!(
            "SSA-IR FP local-cache preferences contain {} slots but {} type entries",
            program.local_cache.fp_preferred_slots.len(),
            program.local_cache.fp_preferred_types.len(),
        )));
    }
    for (index, slot) in program.local_cache.fp_preferred_slots.iter().enumerate() {
        if program.local_cache.fp_preferred_slots[..index].contains(slot) {
            return Err(WasmError::internal(alloc::format!(
                "SSA-IR FP local-cache preferences contain duplicate slot {:?}",
                slot,
            )));
        }
    }
    for ty in &program.local_cache.gp_preferred_types {
        if matches!(
            ty,
            ValueType::F32
                | ValueType::F64
        ) {
            return Err(WasmError::internal(
                "SSA-IR GP local-cache preferences must not contain float types".into(),
            ));
        }
    }
    for ty in &program.local_cache.fp_preferred_types {
        if !matches!(
            ty,
            ValueType::F32
                | ValueType::F64
        ) {
            return Err(WasmError::internal(
                "SSA-IR FP local-cache preferences must contain only float types".into(),
            ));
        }
    }

    // Validate value-type side table coverage when present.
    if !program.value_types.is_empty() {
        validate_value_type_coverage(program)?;
        validate_cached_local_slot_types(program)?;
    }

    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_value_type_coverage(program: &SsaProgram) -> Result<(), WasmError> {
    let type_count = program.value_types.len();
    let check = |value: SsaValue, ctx: &str| -> Result<(), WasmError> {
        if value.0 as usize >= type_count {
            return Err(WasmError::internal(alloc::format!(
                "{ctx}: SsaValue({}) is out of range for value_types table (len={type_count})",
                value.0,
            )));
        }
        Ok(())
    };

    for (block_idx, block) in program.blocks.iter().enumerate() {
        let bctx = alloc::format!("b{block_idx}");
        for param in &block.params {
            check(*param, &alloc::format!("{bctx} param"))?;
        }
        for inst in &block.ops {
            match &inst.kind {
                SsaInstKind::Value { args, results, .. } => {
                    for a in args {
                        if let SsaOperand::Value(v) = a {
                            check(*v, &alloc::format!("{bctx} Value arg"))?;
                        }
                    }
                    for r in results {
                        check(*r, &alloc::format!("{bctx} Value result"))?;
                    }
                }
                SsaInstKind::LocalGet { dst, .. } | SsaInstKind::Fill { dst, .. } => {
                    check(*dst, &alloc::format!("{bctx} LocalGet/Fill dst"))?;
                }
                SsaInstKind::LocalSet { src, .. } | SsaInstKind::Spill { src, .. } => {
                    check(*src, &alloc::format!("{bctx} LocalSet/Spill src"))?;
                }
                SsaInstKind::Boundary(_) => {}
            }
        }
        match &block.terminator {
            SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                check(*cond, &alloc::format!("{bctx} Branch cond"))?;
                validate_edge_values(then_edge, &bctx, &check)?;
                validate_edge_values(else_edge, &bctx, &check)?;
            }
            SsaTerminator::Goto(edge) => {
                validate_edge_values(edge, &bctx, &check)?;
            }
            SsaTerminator::BrTable { index, entries } => {
                check(*index, &alloc::format!("{bctx} BrTable index"))?;
                for edge in entries {
                    validate_edge_values(edge, &bctx, &check)?;
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_edge_values(
    edge: &SsaEdge,
    bctx: &str,
    check: &dyn Fn(SsaValue, &str) -> Result<(), WasmError>,
) -> Result<(), WasmError> {
    for binding in &edge.bindings {
        check(binding.param, &alloc::format!("{bctx} edge binding param"))?;
        check(binding.value, &alloc::format!("{bctx} edge binding value"))?;
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_cached_local_slot_types(program: &SsaProgram) -> Result<(), WasmError> {
    let mut cached_slot_types = BTreeMap::<FrameSlot, ValueType>::new();
    for (slot, ty) in program
        .local_cache
        .gp_preferred_slots
        .iter()
        .copied()
        .zip(program.local_cache.gp_preferred_types.iter().copied())
        .chain(
            program
                .local_cache
                .fp_preferred_slots
                .iter()
                .copied()
                .zip(program.local_cache.fp_preferred_types.iter().copied()),
        )
    {
        cached_slot_types.insert(slot, ty);
    }

    if cached_slot_types.is_empty() {
        return Ok(());
    }

    for (block_idx, block) in program.blocks.iter().enumerate() {
        for (op_idx, inst) in block.ops.iter().enumerate() {
            match inst.kind {
                SsaInstKind::LocalGet { slot, dst } => {
                    validate_cached_slot_value_type(
                        program,
                        &cached_slot_types,
                        slot,
                        dst,
                        block_idx,
                        op_idx,
                        "LocalGet dst",
                    )?;
                }
                SsaInstKind::LocalSet { slot, src, .. } => {
                    validate_cached_slot_value_type(
                        program,
                        &cached_slot_types,
                        slot,
                        src,
                        block_idx,
                        op_idx,
                        "LocalSet src",
                    )?;
                }
                SsaInstKind::Fill { .. }
                | SsaInstKind::Spill { .. }
                | SsaInstKind::Value { .. }
                | SsaInstKind::Boundary(_) => {}
            }
        }
    }

    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_cached_slot_value_type(
    program: &SsaProgram,
    cached_slot_types: &BTreeMap<FrameSlot, ValueType>,
    slot: FrameSlot,
    value: SsaValue,
    block_idx: usize,
    op_idx: usize,
    role: &str,
) -> Result<(), WasmError> {
    let Some(cached_ty) = cached_slot_types.get(&slot).copied() else {
        return Ok(());
    };
    let Some(value_ty) = program.value_types.get(value.0 as usize).copied() else {
        return Err(WasmError::internal(alloc::format!(
            "b{block_idx} op {op_idx} {role}: SsaValue({}) is out of range for value_types table",
            value.0,
        )));
    };
    if !cached_slot_value_type_matches(role, value_ty, cached_ty) {
        return Err(WasmError::internal(alloc::format!(
            "b{block_idx} op {op_idx} {role} for cached local slot {:?} uses value {:?} ({:?}), but cache metadata says {:?}",
            slot,
            value,
            value_ty,
            cached_ty,
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn cached_slot_value_type_matches(role: &str, value_ty: ValueType, cached_ty: ValueType) -> bool {
    match role {
        "LocalSet src" => value_ty.is_compatible_with(&cached_ty),
        "LocalGet dst" => cached_ty.is_compatible_with(&value_ty),
        _ => value_ty == cached_ty,
    }
}

#[cfg(not(any(debug_assertions, test)))]
#[inline]
pub(crate) fn validate_program(_program: &SsaProgram) -> Result<(), WasmError> {
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_block_id(block: &SsaBlock, index: usize) -> Result<(), WasmError> {
    if block.id.as_usize() != index {
        return Err(WasmError::internal(alloc::format!(
            "SSA-IR block {} has mismatched id {}",
            index,
            block.id.as_usize(),
        )));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_params(params: &[SsaValue], label: alloc::string::String) -> Result<(), WasmError> {
    for (index, value) in params.iter().enumerate() {
        if params[..index].contains(value) {
            return Err(WasmError::internal(alloc::format!(
                "{label} contains duplicate param {:?}",
                value,
            )));
        }
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_edge(
    program: &SsaProgram,
    edge: &SsaEdge,
    source_block: usize,
) -> Result<(), WasmError> {
    let Some(target) = program.blocks.get(edge.target.as_usize()) else {
        return Err(WasmError::internal(alloc::format!(
            "SSA-IR block {} has edge to out-of-range target {}",
            source_block,
            edge.target.as_usize(),
        )));
    };

    let mut seen_params = alloc::vec::Vec::with_capacity(edge.bindings.len());
    for binding in &edge.bindings {
        validate_binding(
            program,
            binding,
            target,
            source_block,
            edge.target.as_usize(),
        )?;
        if seen_params.contains(&binding.param) {
            return Err(WasmError::internal(alloc::format!(
                "SSA-IR edge b{} -> b{} binds param {:?} more than once",
                source_block,
                edge.target.as_usize(),
                binding.param,
            )));
        }
        seen_params.push(binding.param);
    }

    if edge.bindings.len() != target.params.len() {
        return Err(WasmError::internal(alloc::format!(
            "SSA-IR edge b{} -> b{} has {} bindings, but target expects {} params",
            source_block,
            edge.target.as_usize(),
            edge.bindings.len(),
            target.params.len(),
        )));
    }

    for param in &target.params {
        if !seen_params.contains(param) {
            return Err(WasmError::internal(alloc::format!(
                "SSA-IR edge b{} -> b{} does not bind target param {:?}",
                source_block,
                edge.target.as_usize(),
                param,
            )));
        }
    }

    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_binding(
    program: &SsaProgram,
    binding: &SsaBinding,
    target: &SsaBlock,
    source_block: usize,
    target_block: usize,
) -> Result<(), WasmError> {
    if !target.params.contains(&binding.param) {
        return Err(WasmError::internal(alloc::format!(
            "SSA-IR edge b{} -> b{} binds unknown target param {:?}",
            source_block,
            target_block,
            binding.param,
        )));
    }
    if !program.value_types.is_empty() {
        let param_ty = program.value_types.get(binding.param.0 as usize).copied();
        let value_ty = program.value_types.get(binding.value.0 as usize).copied();
        if let (Some(param_ty), Some(value_ty)) = (param_ty, value_ty) {
            if param_ty != value_ty {
                return Err(WasmError::internal(alloc::format!(
                    "SSA-IR edge b{} -> b{} binds param {:?} ({:?}) from value {:?} ({:?})",
                    source_block,
                    target_block,
                    binding.param,
                    param_ty,
                    binding.value,
                    value_ty,
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::ssa_ir::{
        ir::{SsaInst, SsaLocalCachePrefs},
        target::SsaTarget,
    };
    use alloc::vec::Vec;

    #[test]
    fn rejects_missing_target_binding() {
        let program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs::default(),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(
                        SsaEdge {
                            target: SsaTarget(1),
                            bindings: Vec::new(),
                        }
                    ),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: alloc::vec![SsaValue(0)],
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            value_types: alloc::vec![],
            value_homes: alloc::vec![],
            value_sink_local: alloc::vec![],
            block_local_demand: None,
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error
            .message()
            .contains("has 0 bindings, but target expects 1 params"));
    }

    #[test]
    fn rejects_duplicate_param_binding() {
        let param0 = SsaValue(0);
        let program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs::default(),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(
                        SsaEdge {
                            target: SsaTarget(1),
                            bindings: alloc::vec![
                                SsaBinding {
                                    param: param0,
                                    value: SsaValue(1),
                                },
                                SsaBinding {
                                    param: param0,
                                    value: SsaValue(2),
                                },
                            ],
                        }
                    ),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: alloc::vec![param0],
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            value_types: alloc::vec![],
            value_homes: alloc::vec![],
            value_sink_local: alloc::vec![],
            block_local_demand: None,
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error.message().contains("more than once"));
    }

    #[test]
    fn rejects_edge_binding_type_mismatch() {
        let param0 = SsaValue(0);
        let value1 = SsaValue(1);
        let program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs::default(),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(
                        SsaEdge {
                            target: SsaTarget(1),
                            bindings: alloc::vec![SsaBinding {
                                param: param0,
                                value: value1,
                            }],
                        }
                    ),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: alloc::vec![param0],
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            value_types: alloc::vec![
                ValueType::I64,
                ValueType::I32,
            ],
            value_homes: alloc::vec![],
            value_sink_local: alloc::vec![],
            block_local_demand: None,
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error
            .message()
            .contains("binds param SsaValue(0) (I64) from value SsaValue(1) (I32)"));
    }

    #[test]
    fn rejects_cached_local_slot_type_mismatch() {
        let value0 = SsaValue(0);
        let program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs {
                gp_preferred_slots: alloc::vec![FrameSlot(0)],
                gp_preferred_types: alloc::vec![ValueType::I32],
                fp_preferred_slots: Vec::new(),
                fp_preferred_types: Vec::new(),
                gp_local_info: Vec::new(),
                fp_local_info: Vec::new(),
            },
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: alloc::vec![SsaInst {
                    kind: SsaInstKind::LocalGet {
                        slot: FrameSlot(0),
                        dst: value0,
                    },
                }],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: alloc::vec![ValueType::I64],
            value_homes: alloc::vec![],
            value_sink_local: alloc::vec![],
            block_local_demand: None,
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error
            .message()
            .contains("LocalGet dst for cached local slot FrameSlot(0) uses value SsaValue(0) (I64), but cache metadata says I32"));
    }

    #[test]
    fn accepts_cached_local_ref_subtype_storage_match() {
        let value0 = SsaValue(0);
        let program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs {
                gp_preferred_slots: alloc::vec![FrameSlot(0)],
                gp_preferred_types: alloc::vec![ValueType::funcref()],
                fp_preferred_slots: Vec::new(),
                fp_preferred_types: Vec::new(),
                gp_local_info: Vec::new(),
                fp_local_info: Vec::new(),
            },
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: alloc::vec![SsaInst {
                    kind: SsaInstKind::LocalSet {
                        slot: FrameSlot(0),
                        src: value0,
                        version: 0,
                    },
                }],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: alloc::vec![ValueType::Ref(
                crate::value_type::RefType::non_nullable_concrete(1),
            )],
            value_homes: alloc::vec![],
            value_sink_local: alloc::vec![],
            block_local_demand: None,
        };

        validate_program(&program).expect("ref subtypes share the same GP-word cached-local class");
    }

    #[test]
    fn rejects_cached_local_gp_word_type_mismatch_between_ref_and_i32() {
        let value0 = SsaValue(0);
        let program = SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs {
                gp_preferred_slots: alloc::vec![FrameSlot(0)],
                gp_preferred_types: alloc::vec![ValueType::funcref()],
                fp_preferred_slots: Vec::new(),
                fp_preferred_types: Vec::new(),
                gp_local_info: Vec::new(),
                fp_local_info: Vec::new(),
            },
            blocks: alloc::vec![SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: alloc::vec![SsaInst {
                    kind: SsaInstKind::LocalSet {
                        slot: FrameSlot(0),
                        src: value0,
                        version: 0,
                    },
                }],
                terminator: SsaTerminator::Return { results: None },
            }],
            value_types: alloc::vec![ValueType::I32],
            value_homes: alloc::vec![],
            value_sink_local: alloc::vec![],
            block_local_demand: None,
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error.message().contains(
            "LocalSet src for cached local slot FrameSlot(0) uses value SsaValue(0) (I32), but cache metadata says Ref"
        ));
    }
}
