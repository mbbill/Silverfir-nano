//! Control-flow lowering: branches, calls, traps.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineEdge, MachineTerminator,
        MachineTrapKind, MachineValue,
    },
};

use super::{enc, reg::GpReg, select};

impl<'a> super::backend::ExampleBackend<'a> {

    // ── Branch primitives ────────────────────────────────────────────────

    pub(super) fn lower_b(&mut self, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::b(0));
        self.fixups.push(super::backend::BranchFixup {
            inst_offset, label,
            kind: super::backend::BranchFixupKind::B,
        });
    }

    pub(super) fn lower_b_cond(&mut self, cond: enc::Cond, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::b_cond(cond, 0));
        self.fixups.push(super::backend::BranchFixup {
            inst_offset, label,
            kind: super::backend::BranchFixupKind::BCond(cond),
        });
    }

    pub(super) fn lower_cbz(&mut self, reg: GpReg, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::cbz_64(reg, 0));
        self.fixups.push(super::backend::BranchFixup {
            inst_offset, label,
            kind: super::backend::BranchFixupKind::Cbz(reg),
        });
    }

    pub(super) fn lower_cbnz(&mut self, reg: GpReg, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::cbnz_64(reg, 0));
        self.fixups.push(super::backend::BranchFixup {
            inst_offset, label,
            kind: super::backend::BranchFixupKind::Cbnz(reg),
        });
    }

    // ── Terminator dispatch ──────────────────────────────────────────────

    pub(super) fn lower_terminator(
        &mut self,
        term: &MachineTerminator,
        _fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => {
                let label = self.core.emit_edge(edge.target, &edge.args)?;
                self.lower_b(label);
                Ok(())
            }
            MachineTerminator::Branch { cond, then_edge, else_edge } => {
                self.lower_branch_edges(cond, then_edge, else_edge)
            }
            MachineTerminator::Return => {
                self.lower_return();
                Ok(())
            }
            _ => todo!("example backend: calls, jump tables, traps"),
        }
    }

    // ── Branch ───────────────────────────────────────────────────────────

    fn lower_branch_edges(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &MachineEdge,
        else_edge: &MachineEdge,
    ) -> Result<(), WasmError> {
        let then_label = self.core.emit_edge(then_edge.target, &then_edge.args)?;
        let else_label = self.core.emit_edge(else_edge.target, &else_edge.args)?;
        self.lower_branch(cond, then_label, else_label)
    }

    fn lower_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_label: usize,
        else_label: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(MachineValue::Imm64(0)) => {
                self.lower_b(else_label);
            }
            MachineBranchCond::Value(MachineValue::Imm64(_)) => {
                self.lower_b(then_label);
            }
            MachineBranchCond::Value(MachineValue::Reg(reg)) => {
                let reg = self.map_gp_reg(reg)?;
                self.lower_cbnz(reg, then_label);
                self.lower_b(else_label);
            }
            MachineBranchCond::IntCompare { width, kind, sign, lhs, rhs } => {
                self.lower_cmp(width, lhs, rhs)?;
                self.lower_b_cond(select::map_int_cond(kind, sign), then_label);
                self.lower_b(else_label);
            }
        }
        Ok(())
    }

    // ── Trap-if ──────────────────────────────────────────────────────────

    pub(super) fn lower_trap_if(
        &mut self,
        kind: MachineTrapKind,
        cond: &MachineBranchCond,
    ) -> Result<(), WasmError> {
        let trap_label = self.core.ensure_trap_label(kind);
        match *cond {
            MachineBranchCond::Value(MachineValue::Imm64(0)) => {}
            MachineBranchCond::Value(MachineValue::Imm64(_)) => {
                self.lower_b(trap_label);
            }
            MachineBranchCond::Value(MachineValue::Reg(reg)) => {
                let reg = self.map_gp_reg(reg)?;
                self.lower_cbnz(reg, trap_label);
            }
            MachineBranchCond::IntCompare { width, kind, sign, lhs, rhs } => {
                self.lower_cmp(width, lhs, rhs)?;
                self.lower_b_cond(select::map_int_cond(kind, sign), trap_label);
            }
        }
        Ok(())
    }

    // ── Call helper ──────────────────────────────────────────────────────

    pub(super) fn lower_call_helper(
        &mut self,
        _extern_idx: usize,
        _const_idx: usize,
    ) -> Result<(), WasmError> {
        // A real backend would:
        // 1. MOV helper args into HELPER_ARGS.arg0/arg1/arg2
        // 2. Materialize helper entry into TEMPS.call_target
        // 3. BLR call_target
        // 4. Check return status, branch to error label if nonzero
        todo!("example backend: call helper")
    }

    // ── Local return ──────────────────────────────────────────────────────

    fn lower_return(&mut self) {
        // MachineTerminator::Return is a LOCAL return (callee → caller),
        // not a root return to C. A real backend would:
        // 1. Load continuation address from the call-link area
        // 2. Load caller's frame pointer
        // 3. Copy return values to caller's result slots
        // 4. Restore frame pointer and branch to continuation
        //
        // Root returns go through the pipeline's return_ok_label →
        // emit_return_ok_status() + emit_epilogue() path instead.
        todo!("example backend: local return requires call-link support")
    }
}
