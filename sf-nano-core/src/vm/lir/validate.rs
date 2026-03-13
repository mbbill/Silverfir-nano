//! Structural validation for prepared LIR.

use crate::error::WasmError;

use super::ir::{LirBinding, LirBlock, LirEdge, LirProgram, LirTerminator, LirValue};

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

    for (index, slot) in program.local_cache.preferred_slots.iter().enumerate() {
        if program.local_cache.preferred_slots[..index].contains(slot) {
            return Err(WasmError::internal(alloc::format!(
                "LIR local-cache preferences contain duplicate slot {:?}",
                slot,
            )));
        }
    }

    Ok(())
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
        validate_binding(binding, target, source_block, edge.target.as_usize())?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::lir::{
        ir::{LirLocalCachePrefs, LirValue},
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
        };

        let error = validate_program(&program).expect_err("validation should fail");
        assert!(error.message().contains("more than once"));
    }
}
