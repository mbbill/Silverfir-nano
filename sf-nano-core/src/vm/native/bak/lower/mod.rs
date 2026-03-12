//! Lower shared CFG + SSA LIR into target-independent native IR.
//!
//! This directory is the native-owned semantic lowering step after LIR.
//!
//! Long-term structure:
//! - `select`: choose inline/native/cold-helper forms from LIR
//! - `place`: assign values onto the native VM ABI
//! - `peephole`: target-independent native cleanups/fusions
//! - `state`: shared lowering/placement state

mod peephole;
mod place;
mod select;
mod state;

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        lir::ir::LirProgram,
        native::compile::build::LocalCallContract,
        plan::PlannedProgram,
    },
};

use super::ir::NativeProgram;

pub fn lower_native(
    lir: &LirProgram,
    planned: &PlannedProgram,
    backend_config: BackendConfig,
    local_call_contracts: &[Option<LocalCallContract>],
) -> Result<NativeProgram, WasmError> {
    let selected = select::select_program(lir, local_call_contracts)?;
    let mut program = place::place_program(&selected, planned, backend_config);
    peephole::run_native_peepholes(&mut program);
    Ok(program)
}
