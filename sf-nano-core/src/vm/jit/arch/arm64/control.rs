//! ARM64 terminator emission: branches, calls, traps, jump tables.

use crate::vm::jit::machine::machine_ir::{
    MachineBlockId, MachineBranchCond, MachineCallArgs, MachineCallResults, MachineCallTarget,
    MachineCompareKind, MachineConstId, MachineEdge, MachineFloatWidth, MachineResultDst,
    MachineResultSrc, MachineReturnValue, MachineStorageType, MachineTerminator, MachineTrapKind,
    MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
};
use crate::{collections, error::WasmError};

use super::abi::map_fixed_reg;
use super::fusion::map_int_cond;
use super::inst::{materialize_u64_into, prepare_gp};
use super::reg::{Arm64FpReg, Arm64Reg};
use super::{abi, enc};
use crate::vm::jit::arch::common::helpers::is_fallthrough_edge;
use crate::vm::jit::arch::common::helpers::trap_code;
use crate::vm::jit::arch::common::pipeline::emit_call_arg_lanes;
use crate::vm::jit::arch::common::pipeline::emit_parallel_moves;
use crate::vm::jit::arch::common::template::{
    decode_template_chain_next, encode_template_chain_next, template_i32_delta, TemplateBranchSense,
};
use crate::vm::jit::arch::common::types::{LocalPtrPatch, PendingLocalPtrPatch};
use crate::vm::jit::runtime::{runtime_call::call_runtime_entry_ptr, trap::raise_trap};

const MAX_DIRECT_JUMP_TABLE_RUNS: usize = 2;
const MAX_DIRECT_JUMP_TABLE_CASES: usize = 4096;
const MIN_DIRECT_JUMP_TABLE_SAVINGS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectJumpTableRun {
    start: u32,
    end: u32,
    entry_index: usize,
}

impl DirectJumpTableRun {
    #[inline]
    const fn uses_zero_excluded_prefix(self, zero_is_excluded: bool) -> bool {
        zero_is_excluded && self.start == 1 && self.end < 4095
    }

    #[inline]
    const fn index_cmp_imm(self, zero_is_excluded: bool) -> Option<u32> {
        if self.start == self.end {
            Some(self.start)
        } else if self.start == 0 {
            Some(self.end)
        } else if self.uses_zero_excluded_prefix(zero_is_excluded) {
            Some(self.end + 1)
        } else {
            None
        }
    }

    #[inline]
    const fn needs_range_scratch(self, zero_is_excluded: bool) -> bool {
        self.index_cmp_imm(zero_is_excluded).is_none()
    }

    #[inline]
    const fn standalone_instruction_count(self, zero_is_excluded: bool) -> usize {
        if self.index_cmp_imm(zero_is_excluded).is_some() {
            2
        } else {
            3
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectJumpTablePlan {
    runs: collections::Vec<DirectJumpTableRun>,
    default_index: usize,
    zero_uses_default: bool,
}

impl DirectJumpTablePlan {
    fn instruction_count(&self) -> usize {
        // Every conditional edge has a nearby veneer whose wide-range `b`
        // reaches the real block or edge stub. Case zero shares the final
        // default branch as its veneer, so it adds only the cbz itself.
        let mut count = usize::from(self.zero_uses_default) + 1;
        let mut last_index_cmp = None;
        for run in &self.runs {
            if let Some(imm) = run.index_cmp_imm(self.zero_uses_default) {
                count += usize::from(last_index_cmp != Some(imm)) + 2;
                last_index_cmp = Some(imm);
            } else {
                count += 4;
                last_index_cmp = None;
            }
        }
        count
    }

    fn byte_len(&self) -> usize {
        self.instruction_count() * core::mem::size_of::<u32>()
    }
}

fn dense_jump_table_byte_len(entry_count: usize) -> usize {
    7 * core::mem::size_of::<u32>()
        + core::mem::size_of::<u64>()
        + entry_count * core::mem::size_of::<u64>()
}

/// Select a compact direct-branch lowering for duplicate-heavy tables.
///
/// The final entry is Wasm's default edge. Runs equal only when the complete
/// `MachineEdge` matches, including edge arguments, so compressing them cannot
/// merge distinct parallel-move semantics. Non-default runs are independent
/// unsigned membership tests followed by one final default branch.
fn plan_direct_jump_table(entries: &[MachineEdge]) -> Option<DirectJumpTablePlan> {
    if entries.len() <= 1 {
        return None;
    }
    let default_index = entries.len() - 1;
    let case_count = default_index;
    if case_count > MAX_DIRECT_JUMP_TABLE_CASES {
        return None;
    }

    let default = &entries[default_index];
    let mut zero_uses_default = entries[0] == *default;
    let mut saw_compression = zero_uses_default;
    let mut runs = collections::Vec::new();
    let mut start = 0;
    while start < case_count {
        let mut end = start;
        while end + 1 < case_count && entries[end + 1] == entries[start] {
            end += 1;
        }
        saw_compression |= end != start;
        if entries[start] == *default {
            saw_compression = true;
        } else {
            runs.push(DirectJumpTableRun {
                start: start as u32,
                end: end as u32,
                entry_index: start,
            });
            if runs.len() > MAX_DIRECT_JUMP_TABLE_RUNS {
                return None;
            }
        }
        start = end + 1;
    }
    if !saw_compression {
        return None;
    }
    if runs.is_empty() {
        // Every case is the default edge. No index test or preparation is
        // needed; the complete terminator is one direct branch.
        zero_uses_default = false;
    }

    // Without profile data, prefer the least expensive membership shape;
    // singleton enum/tag cases win ties, then lower indices. Runs are disjoint,
    // so their test order does not affect semantics. The case-zero/default
    // fast path always precedes these tests.
    runs.sort_unstable_by(|lhs, rhs| {
        lhs.standalone_instruction_count(zero_uses_default)
            .cmp(&rhs.standalone_instruction_count(zero_uses_default))
            .then((lhs.start != lhs.end).cmp(&(rhs.start != rhs.end)))
            .then(lhs.start.cmp(&rhs.start))
    });

    let plan = DirectJumpTablePlan {
        runs,
        default_index,
        zero_uses_default,
    };
    let direct_bytes = plan.byte_len();

    // Dense lowering needs at least seven instructions, one 64-bit table-base
    // literal, and one 64-bit target per entry. Requiring both a fixed saving
    // and a 2x reduction keeps this specialization on clearly smaller shapes.
    let dense_bytes = dense_jump_table_byte_len(entries.len());
    if dense_bytes < direct_bytes + MIN_DIRECT_JUMP_TABLE_SAVINGS || direct_bytes > dense_bytes / 2
    {
        return None;
    }

    Some(plan)
}

fn checked_arm64_template_words(
    delta: i32,
    min_words: i32,
    max_words: i32,
    context: &'static str,
) -> Result<i32, WasmError> {
    if delta & 0b11 != 0 {
        return Err(WasmError::internal(context));
    }
    let words = delta / 4;
    if !(min_words..=max_words).contains(&words) {
        return Err(WasmError::internal(context));
    }
    Ok(words)
}

fn arm64_template_skip_words(current: usize, skip_bytes: usize) -> Result<i32, WasmError> {
    let target = current
        .checked_add(4)
        .and_then(|offset| offset.checked_add(skip_bytes))
        .ok_or_else(|| WasmError::internal("arm64 template skip offset overflow"))?;
    let delta = template_i32_delta(current, target)?;
    checked_arm64_template_words(
        delta,
        -(1 << 18),
        (1 << 18) - 1,
        "arm64 template skip branch out of range",
    )
}

fn scalar_fp_width(ty: MachineStorageType) -> Result<MachineFloatWidth, WasmError> {
    ty.float_width()
        .ok_or_else(|| WasmError::internal("arm64 scalar FP return uses non-FP storage"))
}

impl<'a> super::backend::Arm64Backend<'a> {
    // ── Main terminator dispatch ─────────────────────────────────────────────────

    /// Main terminator dispatch -- called by `ArchBackend::emit_terminator`.
    pub(super) fn lower_terminator_dispatch(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => {
                if is_fallthrough_edge(
                    edge.target,
                    &edge.args,
                    fallthrough,
                    self.core.mir_blocks()?,
                ) {
                    return Ok(());
                }
                if !self
                    .core
                    .is_identity_edge(self.core.mir_blocks()?, edge.target, &edge.args)
                {
                    return self.lower_inline_edge(edge, fallthrough);
                }
                let label = self.core.emit_edge(edge.target, &edge.args)?;
                self.lower_b(label);
                Ok(())
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.lower_branch(cond, then_edge, else_edge, fallthrough),
            MachineTerminator::Return => self.lower_return_sequence(None),
            MachineTerminator::ReturnScalar { value } => self.lower_return_sequence(Some(value)),
            MachineTerminator::Trap { kind } => {
                let trap_label = self.core.ensure_trap_label(*kind);
                self.lower_b(trap_label);
                Ok(())
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.lower_jump_table(*index, entries)
            }
            MachineTerminator::Call {
                target,
                frame_delta,
                args,
                results,
                success,
            } => self.lower_call(target, *frame_delta, args, results, success, fallthrough),
            MachineTerminator::TailCall { target, args } => self.lower_tail_call(target, args),
        }
    }

    // ── Branch ───────────────────────────────────────────────────────────────────

    fn lower_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &MachineEdge,
        else_edge: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let blocks = self.core.mir_blocks()?;
        let then_fallthrough =
            is_fallthrough_edge(then_edge.target, &then_edge.args, fallthrough, blocks);
        let else_fallthrough =
            is_fallthrough_edge(else_edge.target, &else_edge.args, fallthrough, blocks);

        // Keep a non-identity taken edge local when the other successor is the
        // natural fallthrough. Sending a one- or two-move hot edge through the
        // function-wide tail stub adds a second taken branch and disrupts
        // I-cache locality in tight loops.
        if else_fallthrough
            && !self
                .core
                .is_identity_edge(blocks, then_edge.target, &then_edge.args)
        {
            let skip = self.core.new_label();
            self.lower_branch_on_false(cond, skip)?;
            self.lower_inline_edge(then_edge, None)?;
            self.core.bind_label(skip);
            return Ok(());
        }

        let then_label = (!then_fallthrough)
            .then(|| self.core.emit_edge(then_edge.target, &then_edge.args))
            .transpose()?;
        let else_label = (!else_fallthrough)
            .then(|| self.core.emit_edge(else_edge.target, &else_edge.args))
            .transpose()?;

        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {
                    if let Some(label) = else_label {
                        self.lower_b(label);
                    }
                }
                MachineValue::Imm64(_) => {
                    if let Some(label) = then_label {
                        self.lower_b(label);
                    }
                }
                MachineValue::Reg(reg) => {
                    // If the block's last instruction was a compare (or ands)
                    // that produced exactly this bool, the NZCV flags still
                    // encode it: branch on the flags directly instead of
                    // re-testing the materialized register. `invert()` is the
                    // exact negation of the bool (NaN semantics of float
                    // compares are baked into the published cond by cset).
                    let fused = match self.select_flags.take() {
                        Some((flags_reg, code)) if flags_reg == reg => Some(code),
                        _ => None,
                    };
                    let reg = self.map_gp_reg(reg)?;
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            match fused {
                                Some(code) => self.lower_b_cond(code, label),
                                None => self.lower_cbnz(reg, label),
                            }
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            match fused {
                                Some(code) => self.lower_b_cond(code.invert(), label),
                                None => self.lower_cbz(reg, label),
                            }
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        match fused {
                            Some(code) => self.lower_b_cond(code, then_label),
                            None => self.lower_cbnz(reg, then_label),
                        }
                        self.lower_b(else_label);
                    }
                }
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm64 branch condition cannot read reserved cache register",
                    ));
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                // Compare-with-zero patterns (e.g. wasm `i32.eqz; br_if`) become
                // single-instruction `cbz`/`cbnz`. Both lhs == 0 and rhs == 0
                // shapes appear depending on whether the wasm operand was
                // hoisted left or right. The 64-bit cbz form is correct for
                // i32 operands too because AArch64 32-bit ops zero the upper
                // 32 bits of the X register.
                let zero_reg = match (kind, lhs, rhs) {
                    (
                        MachineCompareKind::Eq | MachineCompareKind::Ne,
                        MachineValue::Reg(r),
                        MachineValue::Imm64(0),
                    )
                    | (
                        MachineCompareKind::Eq | MachineCompareKind::Ne,
                        MachineValue::Imm64(0),
                        MachineValue::Reg(r),
                    ) => Some(r),
                    _ => None,
                };
                if let Some(reg) = zero_reg {
                    let reg = self.map_gp_reg(reg)?;
                    let is_eq = matches!(kind, MachineCompareKind::Eq);
                    let emit = |this: &mut Self, take_label: usize, take_eq: bool| {
                        if take_eq {
                            this.lower_cbz(reg, take_label);
                        } else {
                            this.lower_cbnz(reg, take_label);
                        }
                    };
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            emit(self, label, is_eq);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            emit(self, label, !is_eq);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        emit(self, then_label, is_eq);
                        self.lower_b(else_label);
                    }
                } else {
                    self.lower_cmp_values(width, lhs, rhs)?;
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            self.lower_b_cond(map_int_cond(kind, sign), label);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            self.lower_b_cond(map_int_cond(kind, sign).invert(), label);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        self.lower_b_cond(map_int_cond(kind, sign), then_label);
                        self.lower_b(else_label);
                    }
                }
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let cond = match kind {
                    MachineCompareKind::Eq => enc::Cond::Eq,
                    MachineCompareKind::Ne => enc::Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch: unsupported compare kind",
                        ))
                    }
                };
                if else_fallthrough {
                    if let Some(label) = then_label {
                        self.lower_b_cond(cond, label);
                    }
                } else if then_fallthrough {
                    if let Some(label) = else_label {
                        self.lower_b_cond(cond.invert(), label);
                    }
                } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                    self.lower_b_cond(cond, then_label);
                    self.lower_b(else_label);
                }
            }
        }
        Ok(())
    }

    fn lower_branch_on_false(
        &mut self,
        cond: &MachineBranchCond,
        label: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(MachineValue::Imm64(0)) => self.lower_b(label),
            MachineBranchCond::Value(MachineValue::Imm64(_)) => {}
            MachineBranchCond::Value(MachineValue::Reg(reg)) => {
                let fused = match self.select_flags.take() {
                    Some((flags_reg, code)) if flags_reg == reg => Some(code),
                    _ => None,
                };
                match fused {
                    Some(code) => self.lower_b_cond(code.invert(), label),
                    None => {
                        let reg = self.map_gp_reg(reg)?;
                        self.lower_cbz(reg, label);
                    }
                }
            }
            MachineBranchCond::Value(MachineValue::ReservedReg(_)) => {
                return Err(WasmError::internal(
                    "arm64 branch condition cannot read reserved cache register",
                ));
            }
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                let zero_reg = match (kind, lhs, rhs) {
                    (
                        MachineCompareKind::Eq | MachineCompareKind::Ne,
                        MachineValue::Reg(reg),
                        MachineValue::Imm64(0),
                    )
                    | (
                        MachineCompareKind::Eq | MachineCompareKind::Ne,
                        MachineValue::Imm64(0),
                        MachineValue::Reg(reg),
                    ) => Some(reg),
                    _ => None,
                };
                if let Some(reg) = zero_reg {
                    let reg = self.map_gp_reg(reg)?;
                    if kind == MachineCompareKind::Eq {
                        self.lower_cbnz(reg, label);
                    } else {
                        self.lower_cbz(reg, label);
                    }
                } else {
                    self.lower_cmp_values(width, lhs, rhs)?;
                    self.lower_b_cond(map_int_cond(kind, sign).invert(), label);
                }
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let code = match kind {
                    MachineCompareKind::Eq => enc::Cond::Ne,
                    MachineCompareKind::Ne => enc::Cond::Eq,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch: unsupported compare kind",
                        ));
                    }
                };
                self.lower_b_cond(code, label);
            }
        }
        Ok(())
    }

    fn lower_inline_edge(
        &mut self,
        edge: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let (params, arg_float_widths) = {
            let block = self
                .core
                .mir_blocks()?
                .get(edge.target.as_usize())
                .ok_or_else(|| WasmError::internal("edge target block is out of range"))?;
            let widths = edge
                .args
                .iter()
                .map(|arg| match arg {
                    MachineValue::Reg(reg) if self.core.is_fp_reg(*reg) => {
                        self.core.fp_reg_width(*reg).map(Some)
                    }
                    MachineValue::ReservedReg(_)
                    | MachineValue::Reg(_)
                    | MachineValue::Imm64(_) => Ok(None),
                })
                .collect::<Result<crate::collections::Vec<_>, _>>()?;
            (block.params.clone(), widths)
        };
        self.core.current_edge_target = Some(edge.target);
        let result = emit_parallel_moves(self, &params, &edge.args, &arg_float_widths);
        self.core.current_edge_target = None;
        result?;
        if fallthrough != Some(edge.target) {
            let target = self.core.block_label(edge.target)?;
            self.lower_b(target);
        }
        Ok(())
    }

    // ── Trap-if ──────────────────────────────────────────────────────────────────

    pub(super) fn lower_trap_if(
        &mut self,
        kind: MachineTrapKind,
        cond: &MachineBranchCond,
    ) -> Result<(), WasmError> {
        let trap_label = self.core.ensure_trap_label(kind);
        self.lower_branch_if(cond, trap_label)
    }

    pub(crate) fn lower_branch_if(
        &mut self,
        cond: &MachineBranchCond,
        trap_label: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {}
                MachineValue::Imm64(_) => self.lower_b(trap_label),
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    self.lower_cbnz(reg, trap_label);
                }
                MachineValue::ReservedReg(_reg) => {
                    return Err(WasmError::internal(
                        "arm64 trap-if cannot read reserved cache register",
                    ));
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.lower_cmp_values(width, lhs, rhs)?;
                self.lower_b_cond(map_int_cond(kind, sign), trap_label);
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let cond = match kind {
                    MachineCompareKind::Eq => enc::Cond::Eq,
                    MachineCompareKind::Ne => enc::Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch_if: unsupported compare kind",
                        ))
                    }
                };
                self.lower_b_cond(cond, trap_label);
            }
        }
        Ok(())
    }

    pub(crate) fn emit_template_skip_unless(
        &mut self,
        cond: &MachineBranchCond,
        jump_when: TemplateBranchSense,
        skip_bytes: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(value) => {
                    let cond_true = value != 0;
                    let jump_on_true = jump_when == TemplateBranchSense::IfTrue;
                    if cond_true != jump_on_true {
                        let skip_words =
                            arm64_template_skip_words(self.core.text.len(), skip_bytes)?;
                        self.core.text.emit_u32(enc::b(skip_words));
                    }
                }
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    let skip_words = arm64_template_skip_words(self.core.text.len(), skip_bytes)?;
                    let inst = match jump_when {
                        TemplateBranchSense::IfTrue => enc::cbz_64(reg, skip_words),
                        TemplateBranchSense::IfFalse => enc::cbnz_64(reg, skip_words),
                    };
                    self.core.text.emit_u32(inst);
                }
                MachineValue::ReservedReg(_) => {
                    return Err(WasmError::internal(
                        "arm64 template branch cannot read reserved cache register",
                    ));
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.lower_cmp_values(width, lhs, rhs)?;
                let skip_words = arm64_template_skip_words(self.core.text.len(), skip_bytes)?;
                let cond = match jump_when {
                    TemplateBranchSense::IfTrue => map_int_cond(kind, sign).invert(),
                    TemplateBranchSense::IfFalse => map_int_cond(kind, sign),
                };
                self.core.text.emit_u32(enc::b_cond(cond, skip_words));
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                self.lower_tst_values(width, src, mask)?;
                let skip_words = arm64_template_skip_words(self.core.text.len(), skip_bytes)?;
                let cond = match kind {
                    MachineCompareKind::Eq => enc::Cond::Eq,
                    MachineCompareKind::Ne => enc::Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "arm64 template TestBits branch uses unsupported compare kind",
                        ))
                    }
                };
                let cond = match jump_when {
                    TemplateBranchSense::IfTrue => cond.invert(),
                    TemplateBranchSense::IfFalse => cond,
                };
                self.core.text.emit_u32(enc::b_cond(cond, skip_words));
            }
        }
        Ok(())
    }

    pub(crate) fn emit_template_jump_placeholder(
        &mut self,
        next: usize,
    ) -> Result<usize, WasmError> {
        Ok(self.core.text.emit_u32(encode_template_chain_next(next)?))
    }

    pub(crate) fn read_template_jump_next(&self, site: usize) -> Result<usize, WasmError> {
        Ok(decode_template_chain_next(self.core.text.read_u32(site)))
    }

    pub(crate) fn patch_template_jump(
        &mut self,
        site: usize,
        target: usize,
    ) -> Result<(), WasmError> {
        let delta = template_i32_delta(site, target)?;
        let words = checked_arm64_template_words(
            delta,
            -(1 << 25),
            (1 << 25) - 1,
            "arm64 template jump branch out of range",
        )?;
        self.core.text.patch_u32(site, enc::b(words));
        Ok(())
    }

    pub(crate) fn emit_template_jump_to_offset(&mut self, target: usize) -> Result<(), WasmError> {
        let site = self.core.text.len();
        let delta = template_i32_delta(site, target)?;
        let words = checked_arm64_template_words(
            delta,
            -(1 << 25),
            (1 << 25) - 1,
            "arm64 template jump branch out of range",
        )?;
        self.core.text.emit_u32(enc::b(words));
        Ok(())
    }

    // ── Compiled call ────────────────────────────────────────────────────────────

    fn lower_call(
        &mut self,
        target: &MachineCallTarget,
        frame_delta: u32,
        args: &MachineCallArgs,
        results: &MachineCallResults,
        success: &MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let body_local_error_label = self.core.body_local_error_label;
        let continuation_label = self.core.block_label(success.target)?;
        let continuation_is_fallthrough = fallthrough == Some(success.target);

        emit_call_arg_lanes::<Self>(self, args)?;

        // Switch MACHINE_FP_REG to the callee frame. The callee returns with
        // fp still pointing at its own frame; the caller restores it below
        // before observing status or fallback results.
        self.adjust_frame_pointer_by_delta(frame_delta, true);

        let scratch_idx = match target {
            MachineCallTarget::Direct(callee) => {
                let scratch_idx = self.gp_scratch.alloc();
                let scratch = self.gp_scratch.reg(scratch_idx);
                let inst_offset = self.core.text.emit_u32(enc::bl(0));
                self.pending_direct_calls
                    .push(super::backend::PendingDirectCall {
                        inst_offset,
                        fallback_scratch_reg: scratch,
                        callee: *callee,
                        link: true,
                    });
                Some(scratch_idx)
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                self.emit_body_returning_blr(callee_entry)?;
                None
            }
        };

        if let Some(scratch_idx) = scratch_idx {
            self.gp_scratch.free_index(scratch_idx);
        }

        // --- callee returns here. Restore the caller frame before status
        // propagation; the caller's error tail expects its own frame.
        self.adjust_frame_pointer_by_delta(frame_delta, false);

        // C_RET0 holds 0 (success) or trap kind (error). Status check
        // propagates to body_local_error_label.
        self.lower_cbnz(abi::C_RET0, body_local_error_label);
        self.lower_call_result_placement(frame_delta, results)?;

        // Continuation: branch only if the next emitted block is not the
        // continuation block. The block layout pass already prefers placing
        // the continuation immediately after the call (see
        // `CompilerCore::extend_block_trace`), so adjacency is the common
        // case and the explicit `b` is dead code we can elide.
        if !continuation_is_fallthrough {
            self.lower_b(continuation_label);
        }
        Ok(())
    }

    fn lower_tail_call(
        &mut self,
        target: &MachineCallTarget,
        args: &MachineCallArgs,
    ) -> Result<(), WasmError> {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();

        emit_call_arg_lanes::<Self>(self, args)?;

        if self.has_body_host_frame() {
            self.lower_preserved_dynamic_body_restore();
            self.core
                .text
                .emit_u32(enc::ldp_64_post_index(x29, x30, abi::stack_reg(), 16));
        }
        // Tail-call arguments have already been repacked to the current frame
        // prefix, so the callee reuses the current MACHINE_FP_REG value.
        let _ = fp_reg;

        match target {
            MachineCallTarget::Direct(callee) => {
                let scratch_idx = self.gp_scratch.alloc();
                let scratch = self.gp_scratch.reg(scratch_idx);
                let inst_offset = self.core.text.emit_u32(enc::b(0));
                self.pending_direct_calls
                    .push(super::backend::PendingDirectCall {
                        inst_offset,
                        fallback_scratch_reg: scratch,
                        callee: *callee,
                        link: false,
                    });
                self.gp_scratch.free_index(scratch_idx);
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                self.core.text.emit_u32(enc::br(callee_entry));
            }
        }
        Ok(())
    }

    fn materialize_frame_addr(&mut self, dst: Arm64Reg, base: Arm64Reg, delta: u32) {
        if delta == 0 {
            if dst != base {
                self.core.text.emit_u32(enc::mov_reg_64(dst, base));
            }
        } else if delta < 4096 {
            self.core.text.emit_u32(enc::add_imm_64(dst, base, delta));
        } else {
            materialize_u64_into(&mut self.core.text, dst, u64::from(delta));
            self.core.text.emit_u32(enc::add_reg_64(dst, base, dst));
        }
    }

    fn adjust_frame_pointer_by_delta(&mut self, delta: u32, add: bool) {
        if delta == 0 {
            return;
        }
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        if delta < 4096 {
            let inst = if add {
                enc::add_imm_64(fp_reg, fp_reg, delta)
            } else {
                enc::sub_imm_64(fp_reg, fp_reg, delta)
            };
            self.core.text.emit_u32(inst);
            return;
        }

        let scratch_idx = self.gp_scratch.alloc();
        let scratch = self.gp_scratch.reg(scratch_idx);
        materialize_u64_into(&mut self.core.text, scratch, u64::from(delta));
        let inst = if add {
            enc::add_reg_64(fp_reg, fp_reg, scratch)
        } else {
            enc::sub_reg_64(fp_reg, fp_reg, scratch)
        };
        self.core.text.emit_u32(inst);
        self.gp_scratch.free_index(scratch_idx);
    }

    fn lower_call_result_placement(
        &mut self,
        frame_delta: u32,
        results: &MachineCallResults,
    ) -> Result<(), WasmError> {
        match results {
            MachineCallResults::None => return Ok(()),
            MachineCallResults::ScalarGp { dst, .. } => {
                self.place_gp_call_result(*dst, abi::W2W_GP_RET0)?;
                return Ok(());
            }
            MachineCallResults::ScalarFp { dst, ty } => {
                self.place_fp_call_result(*dst, abi::fp_zero_reg(), scalar_fp_width(*ty)?)?;
                return Ok(());
            }
            MachineCallResults::ScalarGpPair { .. } => {
                return Err(WasmError::internal(
                    "arm64 cannot lower 32-bit GP pair scalar call result".into(),
                ));
            }
            MachineCallResults::FrameFallback {
                callee_results,
                caller_results,
            } => {
                if callee_results.slots != caller_results.slots {
                    return Err(WasmError::internal(
                        "arm64 frame-fallback result slot count mismatch",
                    ));
                }

                let fp_reg = map_fixed_reg(MACHINE_FP_REG);
                let value_idx = self.gp_scratch.alloc();
                let value = self.gp_scratch.reg(value_idx);
                for index in 0..callee_results.slots as u32 {
                    let src_offset = frame_delta
                        .checked_add(u32::from(callee_results.base_slot).saturating_mul(8))
                        .and_then(|base| base.checked_add(index.saturating_mul(8)))
                        .ok_or_else(|| {
                            WasmError::internal("arm64 result source offset overflow")
                        })?;
                    let dst_offset = u32::from(caller_results.base_slot)
                        .saturating_mul(8)
                        .checked_add(index.saturating_mul(8))
                        .ok_or_else(|| {
                            WasmError::internal("arm64 result destination offset overflow")
                        })?;
                    self.emit_frame_load_by_byte_offset(value, fp_reg, src_offset);
                    self.emit_frame_store_by_byte_offset(value, fp_reg, dst_offset);
                }
                self.gp_scratch.free_index(value_idx);
                Ok(())
            }
        }
    }

    fn place_gp_call_result(
        &mut self,
        dst: MachineResultDst,
        src: Arm64Reg,
    ) -> Result<(), WasmError> {
        match dst {
            MachineResultDst::Reg(reg) => {
                let dst = self.map_gp_reg(reg)?;
                if dst != src {
                    self.core.text.emit_u32(enc::mov_reg_64(dst, src));
                }
            }
            MachineResultDst::FrameSlot(slot) => {
                self.emit_frame_store_by_byte_offset(
                    src,
                    map_fixed_reg(MACHINE_FP_REG),
                    u32::from(slot.0).saturating_mul(8),
                );
            }
        }
        Ok(())
    }

    fn place_fp_call_result(
        &mut self,
        dst: MachineResultDst,
        src: Arm64FpReg,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        match dst {
            MachineResultDst::Reg(reg) => {
                let dst = self.map_fp_reg(reg)?;
                if dst != src {
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s(dst, src),
                        MachineFloatWidth::F64 => enc::fmov_d(dst, src),
                    });
                }
            }
            MachineResultDst::FrameSlot(slot) => {
                self.emit_frame_store_fp_by_byte_offset(
                    src,
                    map_fixed_reg(MACHINE_FP_REG),
                    u32::from(slot.0).saturating_mul(8),
                    width,
                );
            }
        }
        Ok(())
    }

    fn emit_frame_load_by_byte_offset(&mut self, dst: Arm64Reg, base: Arm64Reg, offset: u32) {
        if offset % 8 == 0 {
            let slot = offset / 8;
            if slot < 4096 {
                self.core.text.emit_u32(enc::ldr_64(dst, base, slot));
                return;
            }
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_frame_addr(addr, base, offset);
        self.core.text.emit_u32(enc::ldr_64(dst, addr, 0));
        self.gp_scratch.free_index(addr_idx);
    }

    fn emit_frame_store_by_byte_offset(&mut self, src: Arm64Reg, base: Arm64Reg, offset: u32) {
        if offset % 8 == 0 {
            let slot = offset / 8;
            if slot < 4096 {
                self.core.text.emit_u32(enc::str_64(src, base, slot));
                return;
            }
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_frame_addr(addr, base, offset);
        self.core.text.emit_u32(enc::str_64(src, addr, 0));
        self.gp_scratch.free_index(addr_idx);
    }

    fn lower_return_value_to_lanes(&mut self, value: &MachineReturnValue) -> Result<(), WasmError> {
        match value {
            MachineReturnValue::ScalarGp { src, .. } => {
                self.move_gp_result_src_to_reg(src, abi::W2W_GP_RET0)?;
            }
            MachineReturnValue::ScalarFp { src, ty } => {
                let width = scalar_fp_width(*ty)?;
                self.move_fp_result_src_to_reg(src, abi::fp_zero_reg(), width)?;
            }
            MachineReturnValue::ScalarGpPair { .. } => {
                return Err(WasmError::internal(
                    "arm64 cannot lower 32-bit GP pair scalar return".into(),
                ));
            }
        }
        Ok(())
    }

    fn move_gp_result_src_to_reg(
        &mut self,
        src: &MachineResultSrc,
        dst: Arm64Reg,
    ) -> Result<(), WasmError> {
        match *src {
            MachineResultSrc::Reg(reg) => {
                let src = self.map_gp_reg(reg)?;
                if src != dst {
                    self.core.text.emit_u32(enc::mov_reg_64(dst, src));
                }
            }
            MachineResultSrc::FrameSlot(slot) => {
                self.emit_frame_load_by_byte_offset(
                    dst,
                    map_fixed_reg(MACHINE_FP_REG),
                    u32::from(slot.0).saturating_mul(8),
                );
            }
            MachineResultSrc::FrameSlotOffset { slot, byte_offset } => {
                let offset = u32::from(slot.0)
                    .saturating_mul(8)
                    .saturating_add(byte_offset as u32);
                self.emit_frame_load_by_byte_offset(dst, map_fixed_reg(MACHINE_FP_REG), offset);
            }
        }
        Ok(())
    }

    fn move_fp_result_src_to_reg(
        &mut self,
        src: &MachineResultSrc,
        dst: Arm64FpReg,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        match *src {
            MachineResultSrc::Reg(reg) => {
                let src = self.map_fp_reg(reg)?;
                if src != dst {
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s(dst, src),
                        MachineFloatWidth::F64 => enc::fmov_d(dst, src),
                    });
                }
            }
            MachineResultSrc::FrameSlot(slot) => {
                self.emit_frame_load_fp_by_byte_offset(
                    dst,
                    map_fixed_reg(MACHINE_FP_REG),
                    u32::from(slot.0).saturating_mul(8),
                    width,
                );
            }
            MachineResultSrc::FrameSlotOffset { slot, byte_offset } => {
                let offset = u32::from(slot.0)
                    .saturating_mul(8)
                    .saturating_add(byte_offset as u32);
                self.emit_frame_load_fp_by_byte_offset(
                    dst,
                    map_fixed_reg(MACHINE_FP_REG),
                    offset,
                    width,
                );
            }
        }
        Ok(())
    }

    fn emit_frame_load_fp_by_byte_offset(
        &mut self,
        dst: Arm64FpReg,
        base: Arm64Reg,
        offset: u32,
        width: MachineFloatWidth,
    ) {
        match width {
            MachineFloatWidth::F32 if offset % 4 == 0 => {
                let scaled = offset / 4;
                if scaled < 4096 {
                    self.core.text.emit_u32(enc::ldr_s(dst, base, scaled));
                    return;
                }
            }
            MachineFloatWidth::F64 if offset % 8 == 0 => {
                let scaled = offset / 8;
                if scaled < 4096 {
                    self.core.text.emit_u32(enc::ldr_d(dst, base, scaled));
                    return;
                }
            }
            _ => {}
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_frame_addr(addr, base, offset);
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::ldr_s(dst, addr, 0),
            MachineFloatWidth::F64 => enc::ldr_d(dst, addr, 0),
        });
        self.gp_scratch.free_index(addr_idx);
    }

    fn emit_frame_store_fp_by_byte_offset(
        &mut self,
        src: Arm64FpReg,
        base: Arm64Reg,
        offset: u32,
        width: MachineFloatWidth,
    ) {
        match width {
            MachineFloatWidth::F32 if offset % 4 == 0 => {
                let scaled = offset / 4;
                if scaled < 4096 {
                    self.core.text.emit_u32(enc::str_s(src, base, scaled));
                    return;
                }
            }
            MachineFloatWidth::F64 if offset % 8 == 0 => {
                let scaled = offset / 8;
                if scaled < 4096 {
                    self.core.text.emit_u32(enc::str_d(src, base, scaled));
                    return;
                }
            }
            _ => {}
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_frame_addr(addr, base, offset);
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::str_s(src, addr, 0),
            MachineFloatWidth::F64 => enc::str_d(src, addr, 0),
        });
        self.gp_scratch.free_index(addr_idx);
    }

    // ── Jump table (br_table) ────────────────────────────────────────────────────

    fn lower_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[MachineEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal(
                "arm64 MachineIR jump table requires at least one entry".into(),
            ));
        }
        if entries.len() == 1 {
            let label = self.core.emit_edge(entries[0].target, &entries[0].args)?;
            self.lower_b(label);
            return Ok(());
        }
        if let Some(plan) = plan_direct_jump_table(entries) {
            return self.lower_direct_jump_table(index, entries, &plan);
        }

        let s0 = self.gp_scratch.scoped_alloc().detach();
        let s1 = self.gp_scratch.scoped_alloc().detach();
        let index_reg = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            index,
        )?
        .detach();
        // Keep C-ABI argument registers out of normal control lowering. `s1`
        // holds the clamped jump-table index first, then the scaled byte offset.
        self.materialize_u64(*s1, (entries.len() - 1) as u64);
        self.core.text.emit_u32(enc::cmp_reg_64(*index_reg, *s1));
        self.core
            .text
            .emit_u32(enc::csel_64(*s1, *index_reg, *s1, enc::Cond::Ls));

        let table_base_load = self.core.text.emit_u32(enc::ldr_lit_64(*s0, 0));
        self.core.text.emit_u32(enc::lsl_imm_64(*s1, *s1, 3));
        self.core.text.emit_u32(enc::ldr_reg_64(*s0, *s0, *s1));
        self.core.text.emit_u32(enc::br(*s0));

        let table_base_literal = self.core.text.emit_u64(0);
        let table_offset = self.core.text.len();
        let table_base_delta =
            ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
        self.core
            .text
            .patch_u32(table_base_load, enc::ldr_lit_64(*s0, table_base_delta));
        self.core.resolved_ptr_patches.push(LocalPtrPatch {
            literal_offset: table_base_literal,
            target_offset: table_offset,
        });

        for entry in entries {
            let label = self.core.emit_edge(entry.target, &entry.args)?;
            let literal_offset = self.core.text.emit_u64(0);
            self.core.local_ptr_patches.push(PendingLocalPtrPatch {
                literal_offset,
                target_label: label,
            });
        }
        Ok(())
    }

    fn lower_direct_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[MachineEdge],
        plan: &DirectJumpTablePlan,
    ) -> Result<(), WasmError> {
        let default = &entries[plan.default_index];
        let default_label = self.core.emit_edge(default.target, &default.args)?;
        if plan.runs.is_empty() {
            self.lower_b(default_label);
            return Ok(());
        }

        // MachineIR br_table indices are canonical Wasm i32 values. Truncate
        // a synthetic immediate here as well so the cbz fast path has the same
        // semantics as the 32-bit comparisons below.
        let index = match index {
            MachineValue::Imm64(value) => MachineValue::Imm64(u64::from(value as u32)),
            other => other,
        };
        let index_reg = prepare_gp(
            self.core.compiled.backend(),
            &self.core.fp_reg_widths,
            &mut self.core.text,
            &self.gp_scratch,
            index,
        )?
        .detach();

        let mut conditional_veneers = collections::Vec::with_capacity(plan.runs.len());
        let zero_default_veneer = if plan.zero_uses_default {
            // MachineIR provides a canonical i32 index here, so the existing
            // 64-bit cbz helper is an exact zero test. Keep its target local:
            // the real edge may be a non-identity stub after a >1 MiB body.
            let veneer = self.core.new_label();
            self.lower_cbz(*index_reg, veneer);
            Some(veneer)
        } else {
            None
        };

        let range_scratch = plan
            .runs
            .iter()
            .any(|run| run.needs_range_scratch(plan.zero_uses_default))
            .then(|| self.gp_scratch.scoped_alloc().detach());

        let mut last_index_cmp = None;
        for run in &plan.runs {
            let entry = &entries[run.entry_index];
            let label = self.core.emit_edge(entry.target, &entry.args)?;
            if let Some(imm) = run.index_cmp_imm(plan.zero_uses_default) {
                if last_index_cmp != Some(imm) {
                    self.core.text.emit_u32(enc::cmp_imm_32(*index_reg, imm));
                }
                let cond = if run.start == run.end {
                    enc::Cond::Eq
                } else if run.start == 0 {
                    enc::Cond::Ls
                } else {
                    enc::Cond::Lo
                };
                let veneer = self.core.new_label();
                self.lower_b_cond(cond, veneer);
                conditional_veneers.push((veneer, label));
                last_index_cmp = Some(imm);
            } else {
                let scratch = range_scratch.as_ref().ok_or_else(|| {
                    WasmError::internal("arm64 direct jump-table range is missing scratch")
                })?;
                self.core
                    .text
                    .emit_u32(enc::sub_imm_32(**scratch, *index_reg, run.start));
                self.core
                    .text
                    .emit_u32(enc::cmp_imm_32(**scratch, run.end - run.start));
                let veneer = self.core.new_label();
                self.lower_b_cond(enc::Cond::Ls, veneer);
                conditional_veneers.push((veneer, label));
                last_index_cmp = None;
            }
        }
        if let Some(veneer) = zero_default_veneer {
            self.core.bind_label(veneer);
        }
        self.lower_b(default_label);
        for (veneer, target) in conditional_veneers {
            self.core.bind_label(veneer);
            self.lower_b(target);
        }
        Ok(())
    }

    // ── Return sequence ──────────────────────────────────────────────────────────

    /// Unified Return lowering. Pops the body prelude link save, sets
    /// `C_RET0 = 0` (the success status), and executes the platform `ret`.
    /// The callee leaves `MACHINE_FP_REG` pointing at its own frame; a
    /// returning caller restores its frame after the native call.
    fn lower_return_sequence(
        &mut self,
        value: Option<&MachineReturnValue>,
    ) -> Result<(), WasmError> {
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();

        if let Some(value) = value {
            self.lower_return_value_to_lanes(value)?;
        }

        // Restore any JIT-ABI preserved dynamic regs, then pop the body
        // prelude link save if this body emitted one.
        if self.has_body_host_frame() {
            self.lower_preserved_dynamic_body_restore();
            self.core
                .text
                .emit_u32(enc::ldp_64_post_index(x29, x30, abi::stack_reg(), 16));
        }

        // Success status: C_RET0 = 0.
        self.core.text.emit_u32(enc::mov_zero_64(abi::C_RET0));

        // Native return (uses LR, which the body prelude's ldp restored).
        self.core.text.emit_u32(enc::ret());
        Ok(())
    }

    // ── Call runtime ─────────────────────────────────────────────────────────────

    pub(super) fn lower_call_runtime(&mut self, const_idx: usize) -> Result<(), WasmError> {
        let metadata = self
            .core
            .compiled
            .const_ptr(MachineConstId(const_idx as u32))
            .ok_or_else(|| WasmError::internal("arm64 runtime-call metadata is out of range"))?;

        // Runtime calls are inline runtime calls, not CFG terminators. Pass the
        // current context, the active Wasm frame pointer, and the constant-pool
        // metadata record that describes where args/results live in that frame.
        self.core
            .text
            .emit_u32(enc::mov_reg_64(abi::C_ARG0, map_fixed_reg(MACHINE_CTX_REG)));
        self.core
            .text
            .emit_u32(enc::mov_reg_64(abi::C_ARG1, map_fixed_reg(MACHINE_FP_REG)));
        self.materialize_u64(abi::C_ARG2, metadata as u64);
        let call_scratch_idx = self.gp_scratch.alloc();
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);
        self.materialize_u64(call_scratch, call_runtime_entry_ptr() as usize as u64);
        self.emit_body_returning_blr(call_scratch)?;
        self.gp_scratch.free_index(call_scratch_idx);

        // Nonzero helper status means the runtime stored a WasmError in
        // the NativeContext. Branch to the body-local error tail, which
        // pops the call record and propagates upward via the unified
        // Return mechanism (the caller's post-BL `cbnz w0` does the rest).
        // C_RET0 is already set by raise_trap to NativeCallStatus::Error
        // (= 1), so the propagation status flows through automatically.
        let body_local_error_label = self.core.body_local_error_label;
        self.lower_cbnz(abi::C_RET0, body_local_error_label);
        Ok(())
    }

    // ── Trap stub ────────────────────────────────────────────────────────────────

    /// Emit a trap stub -- called by `ArchBackend::emit_trap`.
    pub(super) fn lower_trap_dispatch(&mut self, kind: MachineTrapKind) {
        let local_link_save = !self.has_body_host_frame();
        if local_link_save {
            self.core.text.emit_u32(enc::stp_64_pre_index(
                abi::host_fp_reg(),
                abi::host_lr_reg(),
                abi::stack_reg(),
                -16,
            ));
        }
        // Set up arguments: x0 = ctx, x1 = trap code
        self.core
            .text
            .emit_u32(enc::mov_reg_64(abi::C_ARG0, map_fixed_reg(MACHINE_CTX_REG)));
        materialize_u64_into(&mut self.core.text, abi::C_ARG1, trap_code(kind));
        let call_scratch_idx = self.gp_scratch.alloc();
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);
        let raise_trap_fn: unsafe extern "C" fn(_, _) -> _ = raise_trap;
        materialize_u64_into(
            &mut self.core.text,
            call_scratch,
            raise_trap_fn as usize as u64,
        );
        self.core.text.emit_u32(enc::blr(call_scratch));
        self.gp_scratch.free_index(call_scratch_idx);
        if local_link_save {
            self.core.text.emit_u32(enc::ldp_64_post_index(
                abi::host_fp_reg(),
                abi::host_lr_reg(),
                abi::stack_reg(),
                16,
            ));
        }
        // raise_trap returned with C_RET0 = NativeCallStatus::Error (= 1).
        // Branch to body_local_error_label, which preserves C_RET0 and
        // propagates upward to the caller through the unified Return tail.
        let body_local_error_label = self.core.body_local_error_label;
        self.lower_b(body_local_error_label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::{
        arch::common::pipeline,
        backend::BackendConfig,
        machine::machine_ir::{
            MachineBlock, MachineBlockParam, MachineFunction, MachineInst, MachineInstKind,
            MachineProgram, MachineReg, MachineRegOwner,
        },
        runtime::code::CodegenModuleView,
    };

    #[derive(Debug)]
    struct TestCodegenModule {
        backend: BackendConfig,
    }

    impl CodegenModuleView for TestCodegenModule {
        fn backend(&self) -> BackendConfig {
            self.backend
        }

        fn runtime_for(
            &self,
            _id: crate::vm::jit::machine::machine_ir::MachineFuncId,
        ) -> Option<&crate::vm::jit::machine::machine_ir::MachineFunctionAbi> {
            None
        }

        fn const_ptr(
            &self,
            _id: crate::vm::jit::machine::machine_ir::MachineConstId,
        ) -> Option<*const u8> {
            None
        }
    }

    fn edge(target: u32, args: &[u16]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args
                .iter()
                .map(|reg| MachineValue::Reg(MachineReg(*reg)))
                .collect(),
        }
    }

    fn selected_edge<'a>(
        entries: &'a [MachineEdge],
        plan: &DirectJumpTablePlan,
        index: u32,
    ) -> &'a MachineEdge {
        if plan.zero_uses_default && index == 0 {
            return &entries[plan.default_index];
        }
        for run in &plan.runs {
            if index.wrapping_sub(run.start) <= run.end - run.start {
                return &entries[run.entry_index];
            }
        }
        &entries[plan.default_index]
    }

    #[test]
    fn duplicate_heavy_type_switch_uses_zero_fast_path_and_two_runs() {
        let default = edge(759, &[]);
        let middle = edge(762, &[]);
        let case_16 = edge(760, &[]);
        let mut entries = collections::vec![default.clone()];
        entries.extend(core::iter::repeat_n(middle.clone(), 15));
        entries.push(case_16.clone());
        entries.push(default.clone());

        let plan = plan_direct_jump_table(&entries).expect("duplicate-heavy table should compress");
        assert!(plan.zero_uses_default);
        assert_eq!(plan.default_index, 17);
        assert_eq!(
            plan.runs,
            collections::vec![
                DirectJumpTableRun {
                    start: 16,
                    end: 16,
                    entry_index: 16,
                },
                DirectJumpTableRun {
                    start: 1,
                    end: 15,
                    entry_index: 1,
                },
            ]
        );
        // The singleton and prefix range both compare against 16, so the
        // second test reuses NZCV. Each non-default conditional has one local
        // wide-branch veneer; case zero shares the final default branch.
        assert_eq!(plan.instruction_count(), 7);
        assert_eq!(plan.byte_len(), 28);
        assert_eq!(dense_jump_table_byte_len(entries.len()), 180);

        assert_eq!(selected_edge(&entries, &plan, 0), &default);
        assert_eq!(selected_edge(&entries, &plan, 1), &middle);
        assert_eq!(selected_edge(&entries, &plan, 15), &middle);
        assert_eq!(selected_edge(&entries, &plan, 16), &case_16);
        assert_eq!(selected_edge(&entries, &plan, 17), &default);
        assert_eq!(selected_edge(&entries, &plan, u32::MAX), &default);
    }

    #[test]
    fn complete_edge_identity_controls_run_compression() {
        let default = edge(7, &[4]);
        let different_args = edge(7, &[5]);
        let entries = collections::vec![
            default.clone(),
            different_args.clone(),
            different_args.clone(),
            default.clone(),
        ];

        let plan = plan_direct_jump_table(&entries).expect("equal full edges should compress");
        assert!(plan.zero_uses_default);
        assert_eq!(
            plan.runs,
            collections::vec![DirectJumpTableRun {
                start: 1,
                end: 2,
                entry_index: 1,
            }]
        );
        assert_eq!(selected_edge(&entries, &plan, 0).args, default.args);
        assert_eq!(selected_edge(&entries, &plan, 1).args, different_args.args);
        assert_eq!(selected_edge(&entries, &plan, 2).args, different_args.args);
        assert_eq!(selected_edge(&entries, &plan, 3).args, default.args);
    }

    #[test]
    fn same_target_with_different_args_is_not_the_zero_default_edge() {
        let case_zero = edge(7, &[4]);
        let repeated = edge(8, &[6]);
        let default = edge(7, &[5]);
        let entries = collections::vec![
            case_zero.clone(),
            repeated.clone(),
            repeated,
            default.clone(),
        ];

        let plan =
            plan_direct_jump_table(&entries).expect("repeated non-default run should compress");
        assert!(!plan.zero_uses_default);
        assert_eq!(selected_edge(&entries, &plan, 0), &case_zero);
        assert_eq!(selected_edge(&entries, &plan, u32::MAX), &default);
    }

    #[test]
    fn unique_edges_keep_dense_lowering() {
        let entries = collections::vec![edge(1, &[]), edge(2, &[]), edge(3, &[]), edge(4, &[]),];
        assert_eq!(plan_direct_jump_table(&entries), None);
    }

    #[test]
    fn isolated_default_case_is_a_compression_opportunity() {
        let default = edge(9, &[]);
        let entries = collections::vec![edge(1, &[]), default.clone(), edge(2, &[]), default];
        let plan = plan_direct_jump_table(&entries).expect("default case removes one direct test");
        assert!(!plan.zero_uses_default);
        assert_eq!(plan.runs.len(), 2);
        assert_eq!(selected_edge(&entries, &plan, 1), &entries[3]);
    }

    #[test]
    fn all_edges_equal_need_only_the_default_branch() {
        let same = edge(7, &[4]);
        let entries = collections::vec![same.clone(), same.clone(), same.clone()];
        let plan = plan_direct_jump_table(&entries).expect("equal edges should collapse");
        assert!(plan.runs.is_empty());
        assert!(!plan.zero_uses_default);
        assert_eq!(plan.instruction_count(), 1);
        assert_eq!(plan.byte_len(), 4);
        assert_eq!(selected_edge(&entries, &plan, 0), &same);
        assert_eq!(selected_edge(&entries, &plan, u32::MAX), &same);
    }

    #[test]
    fn more_than_two_non_default_runs_keep_dense_lowering() {
        let mut entries = collections::Vec::new();
        for target in 1..=3 {
            entries.push(edge(target, &[]));
            entries.push(edge(target, &[]));
        }
        entries.push(edge(99, &[]));
        assert_eq!(plan_direct_jump_table(&entries), None);
    }

    #[test]
    fn immediate_range_limit_preserves_unsigned_default_boundary() {
        let repeated = edge(1, &[]);
        let default = edge(2, &[]);
        let mut entries = collections::vec![repeated.clone(); MAX_DIRECT_JUMP_TABLE_CASES];
        entries.push(default.clone());
        let plan = plan_direct_jump_table(&entries).expect("4096 cases fit ARM64 imm12 bounds");
        assert_eq!(selected_edge(&entries, &plan, 4095), &repeated);
        assert_eq!(selected_edge(&entries, &plan, 4096), &default);
        assert_eq!(selected_edge(&entries, &plan, u32::MAX), &default);

        let mut too_large = collections::vec![repeated; MAX_DIRECT_JUMP_TABLE_CASES + 1];
        too_large.push(default);
        assert_eq!(plan_direct_jump_table(&too_large), None);
    }

    #[test]
    fn far_non_identity_edges_compile_through_local_veneers() {
        let backend = abi::compile_backend_config();
        let view = TestCodegenModule { backend };
        let index = MachineReg(4);
        let default_param = MachineReg(5);
        let repeated_param = MachineReg(6);
        let padding_dst = MachineReg(7);
        let default_edge = MachineEdge {
            target: MachineBlockId(1),
            args: collections::vec![MachineValue::Reg(index)],
        };
        let repeated_edge = MachineEdge {
            target: MachineBlockId(2),
            args: collections::vec![MachineValue::Reg(index)],
        };
        let mut padding = collections::Vec::with_capacity(70_000);
        for _ in 0..70_000 {
            padding.push(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpI64,
                    dst: padding_dst,
                    src: MachineValue::Imm64(0x1234_5678_9abc_def0),
                },
            });
        }
        let function = MachineFunction {
            id: crate::vm::jit::machine::machine_ir::MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                fp_reg_init_widths: collections::vec![
                    None;
                    usize::from(backend.fp_dynamic_budget)
                ],
                blocks: collections::vec![
                    MachineBlock {
                        id: MachineBlockId(0),
                        params: collections::Vec::new(),
                        ops: collections::vec![MachineInst {
                            kind: MachineInstKind::Move {
                                owner: MachineRegOwner::LinearValue,
                                ty: MachineStorageType::GpWord,
                                dst: index,
                                src: MachineValue::Imm64(0),
                            },
                        }],
                        terminator: MachineTerminator::JumpTable {
                            index: MachineValue::Reg(index),
                            entries: collections::vec![
                                default_edge.clone(),
                                repeated_edge.clone(),
                                repeated_edge,
                                default_edge,
                            ],
                        },
                    },
                    MachineBlock {
                        id: MachineBlockId(1),
                        params: collections::vec![MachineBlockParam::gp_word(default_param)],
                        ops: collections::Vec::new(),
                        terminator: MachineTerminator::Jump(MachineEdge {
                            target: MachineBlockId(3),
                            args: collections::Vec::new(),
                        }),
                    },
                    MachineBlock {
                        id: MachineBlockId(2),
                        params: collections::vec![MachineBlockParam::gp_word(repeated_param)],
                        ops: collections::Vec::new(),
                        terminator: MachineTerminator::Jump(MachineEdge {
                            target: MachineBlockId(3),
                            args: collections::Vec::new(),
                        }),
                    },
                    MachineBlock {
                        id: MachineBlockId(3),
                        params: collections::Vec::new(),
                        ops: padding,
                        terminator: MachineTerminator::Return,
                    },
                ],
            },
            preserved_clobbers: collections::Vec::new(),
        };

        let artifact =
            pipeline::compile_function::<super::super::backend::Arm64Backend>(&view, &function)
                .expect("local jump-table veneers must keep far edge stubs reachable");
        assert!(artifact.text.len() > (1 << 20));
    }
}
