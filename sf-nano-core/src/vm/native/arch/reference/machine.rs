//! Reference machine for finalized native IR.
//!
//! This module is intentionally the target-independent analogue of
//! `arch/arm64`: it should consume the same post-placement `NativeProgram`
//! contract and nothing higher-level. The implementation is left empty on
//! purpose until the final machine-shaped native IR lands.

use crate::{error::WasmError, vm::native::{context::NativeContext, ir::NativeProgram}};

/// Reference machine for the finalized native machine IR.
#[derive(Debug)]
pub struct ReferenceMachine<'a> {
    ctx: &'a mut NativeContext,
    program: &'a NativeProgram,
    fp: *mut u64,
}

impl<'a> ReferenceMachine<'a> {
    #[inline]
    pub fn new(ctx: &'a mut NativeContext, program: &'a NativeProgram, fp: *mut u64) -> Self {
        Self { ctx, program, fp }
    }

    pub fn run(self) -> Result<(), WasmError> {
        let _ = (self.ctx, self.program, self.fp);
        Err(WasmError::internal(
            "arch/reference machine is not implemented for the finalized native IR yet".into(),
        ))
    }
}

#[inline]
pub fn execute_program(
    ctx: &mut NativeContext,
    program: &NativeProgram,
    fp: *mut u64,
) -> Result<(), WasmError> {
    ReferenceMachine::new(ctx, program, fp).run()
}
