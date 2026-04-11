//! Structural validation for prepared SSA-IR.

#[cfg(any(debug_assertions, test))]
use crate::collections;

#[cfg(any(debug_assertions, test))]
use tracked_alloc::collections::BTreeMap;

use crate::error::WasmError;
#[cfg(any(debug_assertions, test))]
use crate::value_type::ValueType;
#[cfg(any(debug_assertions, test))]
use crate::vm::middle::frame::FrameSlot;

use super::ir::SsaProgram;
#[cfg(any(debug_assertions, test))]
use super::ir::{SsaBinding, SsaBlock, SsaEdge, SsaInstKind, SsaOperand, SsaTerminator, SsaValue};

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
        return Err(WasmError::internal(
            "SSA-IR entry block is out of range for blocks",
        ));
    }

    if program.local_slot_types.len() != program.local_slot_info.len() {
        return Err(WasmError::internal(
            "SSA-IR local slot facts contain types but info entries",
        ));
    }

    if !program.block_entry_cached_slots.is_empty()
        && program.blocks.len() != program.block_entry_cached_slots.len()
    {
        return Err(WasmError::internal(
            "SSA-IR has blocks but block-entry cache rows",
        ));
    }

    if !program.block_cfg_origins.is_empty()
        && program.blocks.len() != program.block_cfg_origins.len()
    {
        return Err(WasmError::internal("SSA-IR has blocks but CFG-origin rows"));
    }

    for (index, block) in program.blocks.iter().enumerate() {
        validate_block_id(block, index)?;
        validate_params(&block.params)?;

        match &block.terminator {
            SsaTerminator::Goto(edge) => validate_edge(program, edge)?,
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                validate_edge(program, then_edge)?;
                validate_edge(program, else_edge)?;
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    validate_edge(program, edge)?;
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }

    if !program.value_types.is_empty() {
        validate_value_type_coverage(program)?;
        validate_cached_local_slot_types(program)?;
    }

    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_value_type_coverage(program: &SsaProgram) -> Result<(), WasmError> {
    let type_count = program.value_types.len();
    let check = |value: SsaValue| -> Result<(), WasmError> {
        if value.0 as usize >= type_count {
            return Err(WasmError::internal(
                "SSA-IR value is out of range for value_types table",
            ));
        }
        Ok(())
    };

    for block in &program.blocks {
        for param in &block.params {
            check(*param)?;
        }
        for inst in &block.ops {
            match &inst.kind {
                SsaInstKind::Value { args, results, .. } => {
                    for arg in args {
                        if let SsaOperand::Value(value) = arg {
                            check(*value)?;
                        }
                    }
                    for result in results {
                        check(*result)?;
                    }
                }
                SsaInstKind::LocalGetSlot { dst, .. }
                | SsaInstKind::LocalGetCache { dst, .. }
                | SsaInstKind::Fill { dst, .. } => {
                    check(*dst)?;
                }
                SsaInstKind::LocalSetSlot { src, .. }
                | SsaInstKind::LocalSetCache { src, .. }
                | SsaInstKind::Spill { src, .. } => {
                    check(*src)?;
                }
                SsaInstKind::LocalEnsureCache { .. }
                | SsaInstKind::LocalReserveCache { .. }
                | SsaInstKind::LocalDropCache { .. }
                | SsaInstKind::Call(_) => {}
            }
        }
        match &block.terminator {
            SsaTerminator::Goto(edge) => validate_edge_values(edge, &check)?,
            SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                check(*cond)?;
                validate_edge_values(then_edge, &check)?;
                validate_edge_values(else_edge, &check)?;
            }
            SsaTerminator::BrTable { index, entries } => {
                check(*index)?;
                for edge in entries {
                    validate_edge_values(edge, &check)?;
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
    check: &dyn Fn(SsaValue) -> Result<(), WasmError>,
) -> Result<(), WasmError> {
    for binding in &edge.bindings {
        check(binding.param)?;
        check(binding.value)?;
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_cached_local_slot_types(program: &SsaProgram) -> Result<(), WasmError> {
    let mut cached_slot_types = BTreeMap::<FrameSlot, ValueType>::new();
    for block in &program.blocks {
        for inst in &block.ops {
            match inst.kind {
                SsaInstKind::LocalGetCache { slot, dst } => {
                    if let Some(ty) = program.value_types.get(dst.0 as usize).copied() {
                        cached_slot_types.entry(slot).or_insert(ty);
                    }
                }
                SsaInstKind::LocalSetCache { slot, src } => {
                    if let Some(ty) = program.value_types.get(src.0 as usize).copied() {
                        cached_slot_types.entry(slot).or_insert(ty);
                    }
                }
                _ => {}
            }
        }
    }

    for (block_idx, block) in program.blocks.iter().enumerate() {
        for (op_idx, inst) in block.ops.iter().enumerate() {
            match inst.kind {
                SsaInstKind::LocalGetCache { slot, dst } => validate_cached_slot_value_type(
                    program,
                    &cached_slot_types,
                    slot,
                    dst,
                    block_idx,
                    op_idx,
                    "LocalGetCache dst",
                )?,
                SsaInstKind::LocalSetCache { slot, src } => validate_cached_slot_value_type(
                    program,
                    &cached_slot_types,
                    slot,
                    src,
                    block_idx,
                    op_idx,
                    "LocalSetCache src",
                )?,
                _ => {}
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
    _block_idx: usize,
    _op_idx: usize,
    role: &str,
) -> Result<(), WasmError> {
    let Some(cached_ty) = cached_slot_types.get(&slot).copied() else {
        return Ok(());
    };
    let Some(value_ty) = program.value_types.get(value.0 as usize).copied() else {
        return Err(WasmError::internal(
            "SSA-IR cached local value is out of range for value_types table",
        ));
    };
    if !cached_slot_value_type_matches(role, value_ty, cached_ty) {
        return Err(WasmError::internal(
            "cached local slot uses incompatible value type",
        ));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn cached_slot_value_type_matches(role: &str, value_ty: ValueType, cached_ty: ValueType) -> bool {
    if matches!(
        (value_ty, cached_ty),
        (ValueType::Ref(_), ValueType::Ref(_))
    ) {
        return true;
    }
    match role {
        "LocalSetCache src" => value_ty.is_compatible_with(&cached_ty),
        "LocalGetCache dst" => cached_ty.is_compatible_with(&value_ty),
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
        return Err(WasmError::internal("SSA-IR block has mismatched id"));
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_params(params: &[SsaValue]) -> Result<(), WasmError> {
    for (index, value) in params.iter().enumerate() {
        if params[..index].contains(value) {
            return Err(WasmError::internal("SSA-IR block contains duplicate param"));
        }
    }
    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_edge(program: &SsaProgram, edge: &SsaEdge) -> Result<(), WasmError> {
    let Some(target) = program.blocks.get(edge.target.as_usize()) else {
        return Err(WasmError::internal(
            "SSA-IR block has edge to out-of-range target",
        ));
    };

    let mut seen_params = collections::Vec::with_capacity(edge.bindings.len());
    for binding in &edge.bindings {
        validate_binding(program, binding, target)?;
        if seen_params.contains(&binding.param) {
            return Err(WasmError::internal(
                "SSA-IR edge binds target param more than once",
            ));
        }
        seen_params.push(binding.param);
    }

    if edge.bindings.len() != target.params.len() {
        return Err(WasmError::internal(
            "SSA-IR edge binding count does not match target params",
        ));
    }

    for param in &target.params {
        if !seen_params.contains(param) {
            return Err(WasmError::internal(
                "SSA-IR edge does not bind target param",
            ));
        }
    }

    Ok(())
}

#[cfg(any(debug_assertions, test))]
fn validate_binding(
    program: &SsaProgram,
    binding: &SsaBinding,
    target: &SsaBlock,
) -> Result<(), WasmError> {
    if !target.params.contains(&binding.param) {
        return Err(WasmError::internal(
            "SSA-IR edge binds unknown target param",
        ));
    }
    if !program.value_types.is_empty() {
        let param_ty = program.value_types.get(binding.param.0 as usize).copied();
        let value_ty = program.value_types.get(binding.value.0 as usize).copied();
        if let (Some(param_ty), Some(value_ty)) = (param_ty, value_ty) {
            if param_ty != value_ty {
                return Err(WasmError::internal(
                    "SSA-IR edge binds target param from value with mismatched type",
                ));
            }
        }
    }
    Ok(())
}
