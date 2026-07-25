use crate::{
    error::WasmError,
    vm::{
        jit::arch::common::template::{TemplateBackend, TemplateBranchSense},
        jit::machine::machine_ir::MachineBranchCond,
    },
};

use super::backend::Arm64Backend;

impl<'a> TemplateBackend<'a> for Arm64Backend<'a> {
    const TEMPLATE_JUMP_BYTES: usize = 4;

    fn emit_template_skip_unless(
        &mut self,
        cond: &MachineBranchCond,
        jump_when: TemplateBranchSense,
        skip_bytes: usize,
    ) -> Result<(), WasmError> {
        Arm64Backend::emit_template_skip_unless(self, cond, jump_when, skip_bytes)
    }

    fn emit_template_jump_placeholder(&mut self, next: usize) -> Result<usize, WasmError> {
        Arm64Backend::emit_template_jump_placeholder(self, next)
    }

    fn read_template_jump_next(&self, site: usize) -> Result<usize, WasmError> {
        Arm64Backend::read_template_jump_next(self, site)
    }

    fn patch_template_jump(&mut self, site: usize, target: usize) -> Result<(), WasmError> {
        Arm64Backend::patch_template_jump(self, site, target)
    }

    fn emit_template_jump_to_offset(&mut self, target: usize) -> Result<(), WasmError> {
        Arm64Backend::emit_template_jump_to_offset(self, target)
    }
}
