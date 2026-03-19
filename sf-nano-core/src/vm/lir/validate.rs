//! Structural validation for prepared LIR.

use alloc::collections::BTreeMap;

use crate::{error::WasmError, value_type::ValueType};

use super::ir::{LirBinding, LirBlock, LirEdge, LirProgram, LirTerminator, LirValue};
use crate::vm::plan::frame::FrameSlot;

#[cfg(any(debug_assertions, test))]
pub fn validate_program(program: &LirProgram) -> Result<(), WasmError> {
    if program.blocks.is_empty() {
        if program.entry.as_usize() != 0 {
            return Err(WasmError::internal(
                "empty LIR program must use entry block 0".into(),
            ));
        }
        return Ok(());
    }

    if program.entry.as_usize() >= program.blocks.len() {
        return Err(WasmError::internal(alloc::format!(
            "LIR entry block {} is out of range for {} blocks",
            program.entry.as_usize(),
            program.blocks.len(),
        )));
    }

    for (index, block) in program.blocks.iter().enumerate() {
        validate_block_id(block, index)?;
        validate_params(&block.params, alloc::format!("block b{index} params"))?;

        match &block.terminator {
            LirTerminator::Goto(edge) => validate_edge(program, edge, index)?,
            LirTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                validate_edge(program, then_edge, index)?;
                validate_edge(program, else_edge, index)?;
            }
            LirTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    validate_edge(program, edge, index)?;
                }
            }
            LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
        }
    }

    for (index, slot) in program.local_cache.gp_preferred_slots.iter().enumerate() {
        if program.local_cache.gp_preferred_slots[..index].contains(slot) {
            return Err(WasmError::internal(alloc::format!(
                "LIR local-cache preferences contain duplicate slot {:?}",
                slot,
            )));
        }
    }
    if program.local_cache.gp_preferred_slots.len() != program.local_cache.gp_preferred_types.len()
    {
        return Err(WasmError::internal(alloc::format!(
            "LIR GP local-cache preferences contain {} slots but {} type entries",
            program.local_cache.gp_preferred_slots.len(),
            program.local_cache.gp_preferred_types.len(),
        )));
    }
    if program.local_cache.fp_preferred_slots.len() != program.local_cache.fp_preferred_types.len()
    {
        return Err(WasmError::internal(alloc::format!(
            "LIR FP local-cache preferences contain {} slots but {} type entries",
            program.local_cache.fp_preferred_slots.len(),
            program.local_cache.fp_preferred_types.len(),
        )));
    }
    for (index, slot) in program.local_cache.fp_preferred_slots.iter().enumerate() {
        if program.local_cache.fp_preferred_slots[..index].contains(slot) {
            return Err(WasmError::internal(alloc::format!(
                "LIR FP local-cache preferences contain duplicate slot {:?}",
                slot,
            )));
        }
    }
    for ty in &program.local_cache.gp_preferred_types {
        if matches!(
            ty,
            super::super::super::value_type::ValueType::F32
                | super::super::super::value_type::ValueType::F64
        ) {
            return Err(WasmError::internal(
                "LIR GP local-cache preferences must not contain float types".into(),
            ));
        }
    }
    for ty in &program.local_cache.fp_preferred_types {
        if !matches!(
            ty,
            super::super::super::value_type::ValueType::F32
                | super::super::super::value_type::ValueType::F64
        ) {
            return Err(WasmError::internal(
                "LIR FP local-cache preferences must contain only float types".into(),
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
fn validate_value_type_coverage(program: &LirProgram) -> Result<(), WasmError> {
    use super::ir::LirInstKind;

    let type_count = program.value_types.len();
    let check = |value: LirValue, ctx: &str| -> Result<(), WasmError> {
        if value.0 as usize >= type_count {
            return Err(WasmError::internal(alloc::format!(
                "{ctx}: LirValue({}) is out of range for value_types table (len={type_count})",
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
                LirInstKind::Value { args, results, .. } => {
                    for a in args {
                        check(*a, &alloc::format!("{bctx} Value arg"))?;
                    }
                    for r in results {
                        check(*r, &alloc::format!("{bctx} Value result"))?;
                    }
                }
                LirInstKind::LoadSlot { dst, .. } => {
                    check(*dst, &alloc::format!("{bctx} LoadSlot dst"))?;
                }
                LirInstKind::StoreSlot { src, .. } => {
                    check(*src, &alloc::format!("{bctx} StoreSlot src"))?;
                }
                LirInstKind::Boundary(_) => {}
            }
        }
        match &block.terminator {
            LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                check(*cond, &alloc::format!("{bctx} Branch cond"))?;
                validate_edge_values(then_edge, &bctx, &check)?;
                validate_edge_values(else_edge, &bctx, &check)?;
            }
            LirTerminator::Goto(edge) => {
                validate_edge_values(edge, &bctx, &check)?;
            }
            LirTerminator::BrTable { index, entries } => {
                check(*index, &alloc::format!("{bctx} BrTable index"))?;
                for edge in entries {
                    validate_edge_values(edge, &bctx, &check)?;
                }
            }
            LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
        }
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_edge_values(
    edge: &LirEdge,
    bctx: &str,
    check: &dyn Fn(LirValue, &str) -> Result<(), WasmError>,
) -> Result<(), WasmError> {
    for binding in &edge.bindings {
        check(binding.param, &alloc::format!("{bctx} edge binding param"))?;
        check(binding.value, &alloc::format!("{bctx} edge binding value"))?;
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_cached_local_slot_types(program: &LirProgram) -> Result<(), WasmError> {
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
                super::ir::LirInstKind::LoadSlot { slot, dst } => {
                    validate_cached_slot_value_type(
                        program,
                        &cached_slot_types,
                        slot,
                        dst,
                        block_idx,
                        op_idx,
                        "LoadSlot dst",
                    )?;
                }
                super::ir::LirInstKind::StoreSlot { slot, src } => {
                    validate_cached_slot_value_type(
                        program,
                        &cached_slot_types,
                        slot,
                        src,
                        block_idx,
                        op_idx,
                        "StoreSlot src",
                    )?;
                }
                super::ir::LirInstKind::Value { .. } | super::ir::LirInstKind::Boundary(_) => {}
            }
        }
    }

    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_cached_slot_value_type(
    program: &LirProgram,
    cached_slot_types: &BTreeMap<FrameSlot, ValueType>,
    slot: FrameSlot,
    value: LirValue,
    block_idx: usize,
    op_idx: usize,
    role: &str,
) -> Result<(), WasmError> {
    let Some(cached_ty) = cached_slot_types.get(&slot).copied() else {
        return Ok(());
    };
    let Some(value_ty) = program.value_types.get(value.0 as usize).copied() else {
        return Err(WasmError::internal(alloc::format!(
            "b{block_idx} op {op_idx} {role}: LirValue({}) is out of range for value_types table",
            value.0,
        )));
    };
    if value_type_storage_class(cached_ty) != value_type_storage_class(value_ty) {
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
fn value_type_storage_class(ty: ValueType) -> u8 {
    match ty {
        ValueType::F32 => 1,
        ValueType::F64 => 2,
        ValueType::I64 => 3,
        _ => 4,
    }
}

#[cfg(not(any(debug_assertions, test)))]
#[inline]
pub fn validate_program(_program: &LirProgram) -> Result<(), WasmError> {
    Ok(())
}

fn validate_block_id(block: &LirBlock, index: usize) -> Result<(), WasmError> {
    if block.id.as_usize() != index {
        return Err(WasmError::internal(alloc::format!(
            "LIR block {} has mismatched id {}",
            index,
            block.id.as_usize(),
        )));
    }
    Ok(())
}

fn validate_params(params: &[LirValue], label: alloc::string::String) -> Result<(), WasmError> {
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

fn validate_edge(
    program: &LirProgram,
    edge: &LirEdge,
    source_block: usize,
) -> Result<(), WasmError> {
    let Some(target) = program.blocks.get(edge.target.as_usize()) else {
        return Err(WasmError::internal(alloc::format!(
            "LIR block {} has edge to out-of-range target {}",
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
                "LIR edge b{} -> b{} binds param {:?} more than once",
                source_block,
                edge.target.as_usize(),
                binding.param,
            )));
        }
        seen_params.push(binding.param);
    }

    if edge.bindings.len() != target.params.len() {
        return Err(WasmError::internal(alloc::format!(
            "LIR edge b{} -> b{} has {} bindings, but target expects {} params",
            source_block,
            edge.target.as_usize(),
            edge.bindings.len(),
            target.params.len(),
        )));
    }

    for param in &target.params {
        if !seen_params.contains(param) {
            return Err(WasmError::internal(alloc::format!(
                "LIR edge b{} -> b{} does not bind target param {:?}",
                source_block,
                edge.target.as_usize(),
                param,
            )));
        }
    }

    Ok(())
}

fn validate_binding(
    program: &LirProgram,
    binding: &LirBinding,
    target: &LirBlock,
    source_block: usize,
    target_block: usize,
) -> Result<(), WasmError> {
    if !target.params.contains(&binding.param) {
        return Err(WasmError::internal(alloc::format!(
            "LIR edge b{} -> b{} binds unknown target param {:?}",
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
                    "LIR edge b{} -> b{} binds param {:?} ({:?}) from value {:?} ({:?})",
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
    use crate::vm::lir::{
        ir::{LirInst, LirInstKind, LirLocalCachePrefs, LirValue},
        target::LirTarget,
    };
    use alloc::vec::Vec;

    #[test]
    fn rejects_missing_target_binding() {
        let program = LirProgram {
            entry: LirTarget(0),
            local_cache: LirLocalCachePrefs::default(),
            blocks: alloc::vec![
                crate::vm::lir::ir::LirBlock {
                    id: LirTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: crate::vm::lir::ir::LirTerminator::Goto(
                        crate::vm::lir::ir::LirEdge {
                            target: LirTarget(1),
                            bindings: Vec::new(),
                        }
                    ),
                },
                crate::vm::lir::ir::LirBlock {
                    id: LirTarget(1),
                    params: alloc::vec![LirValue(0)],
                    ops: Vec::new(),
                    terminator: crate::vm::lir::ir::LirTerminator::Return { results: None },
                },
            ],
            value_types: alloc::vec![],
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error
            .message()
            .contains("has 0 bindings, but target expects 1 params"));
    }

    #[test]
    fn rejects_duplicate_param_binding() {
        let param0 = LirValue(0);
        let program = LirProgram {
            entry: LirTarget(0),
            local_cache: LirLocalCachePrefs::default(),
            blocks: alloc::vec![
                crate::vm::lir::ir::LirBlock {
                    id: LirTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: crate::vm::lir::ir::LirTerminator::Goto(
                        crate::vm::lir::ir::LirEdge {
                            target: LirTarget(1),
                            bindings: alloc::vec![
                                LirBinding {
                                    param: param0,
                                    value: LirValue(1),
                                },
                                LirBinding {
                                    param: param0,
                                    value: LirValue(2),
                                },
                            ],
                        }
                    ),
                },
                crate::vm::lir::ir::LirBlock {
                    id: LirTarget(1),
                    params: alloc::vec![param0],
                    ops: Vec::new(),
                    terminator: crate::vm::lir::ir::LirTerminator::Return { results: None },
                },
            ],
            value_types: alloc::vec![],
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error.message().contains("more than once"));
    }

    #[test]
    fn rejects_edge_binding_type_mismatch() {
        let param0 = LirValue(0);
        let value1 = LirValue(1);
        let program = LirProgram {
            entry: LirTarget(0),
            local_cache: LirLocalCachePrefs::default(),
            blocks: alloc::vec![
                crate::vm::lir::ir::LirBlock {
                    id: LirTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: crate::vm::lir::ir::LirTerminator::Goto(
                        crate::vm::lir::ir::LirEdge {
                            target: LirTarget(1),
                            bindings: alloc::vec![LirBinding {
                                param: param0,
                                value: value1,
                            }],
                        }
                    ),
                },
                crate::vm::lir::ir::LirBlock {
                    id: LirTarget(1),
                    params: alloc::vec![param0],
                    ops: Vec::new(),
                    terminator: crate::vm::lir::ir::LirTerminator::Return { results: None },
                },
            ],
            value_types: alloc::vec![
                crate::value_type::ValueType::I64,
                crate::value_type::ValueType::I32,
            ],
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error
            .message()
            .contains("binds param LirValue(0) (I64) from value LirValue(1) (I32)"));
    }

    #[test]
    fn rejects_cached_local_slot_type_mismatch() {
        let value0 = LirValue(0);
        let program = LirProgram {
            entry: LirTarget(0),
            local_cache: LirLocalCachePrefs {
                gp_preferred_slots: alloc::vec![FrameSlot(0)],
                gp_preferred_types: alloc::vec![crate::value_type::ValueType::I32],
                fp_preferred_slots: Vec::new(),
                fp_preferred_types: Vec::new(),
                gp_local_info: Vec::new(),
                fp_local_info: Vec::new(),
            },
            blocks: alloc::vec![crate::vm::lir::ir::LirBlock {
                id: LirTarget(0),
                params: Vec::new(),
                ops: alloc::vec![LirInst {
                    kind: LirInstKind::LoadSlot {
                        slot: FrameSlot(0),
                        dst: value0,
                    },
                }],
                terminator: crate::vm::lir::ir::LirTerminator::Return { results: None },
            }],
            value_types: alloc::vec![crate::value_type::ValueType::I64],
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error
            .message()
            .contains("LoadSlot dst for cached local slot FrameSlot(0) uses value LirValue(0) (I64), but cache metadata says I32"));
    }

    #[test]
    fn accepts_cached_local_ref_subtype_storage_match() {
        let value0 = LirValue(0);
        let program = LirProgram {
            entry: LirTarget(0),
            local_cache: LirLocalCachePrefs {
                gp_preferred_slots: alloc::vec![FrameSlot(0)],
                gp_preferred_types: alloc::vec![crate::value_type::ValueType::funcref()],
                fp_preferred_slots: Vec::new(),
                fp_preferred_types: Vec::new(),
                gp_local_info: Vec::new(),
                fp_local_info: Vec::new(),
            },
            blocks: alloc::vec![crate::vm::lir::ir::LirBlock {
                id: LirTarget(0),
                params: Vec::new(),
                ops: alloc::vec![LirInst {
                    kind: LirInstKind::StoreSlot {
                        slot: FrameSlot(0),
                        src: value0,
                    },
                }],
                terminator: crate::vm::lir::ir::LirTerminator::Return { results: None },
            }],
            value_types: alloc::vec![crate::value_type::ValueType::Ref(
                crate::value_type::RefType::non_nullable_concrete(1),
            )],
        };

        validate_program(&program).expect("ref subtypes share the same GP-word cached-local class");
    }
}
