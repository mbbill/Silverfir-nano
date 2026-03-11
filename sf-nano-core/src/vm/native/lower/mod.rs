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

use crate::vm::{
    backend::BackendConfig,
    lir::ir::LirProgram,
    plan::PlannedProgram,
};

use super::ir::NativeProgram;

pub fn lower_native(
    lir: &LirProgram,
    planned: &PlannedProgram,
    backend_config: BackendConfig,
) -> NativeProgram {
    let selected = select::select_program(lir);
    let mut program = place::place_program(&selected, planned, backend_config);
    peephole::run_native_peepholes(&mut program);
    program
}
