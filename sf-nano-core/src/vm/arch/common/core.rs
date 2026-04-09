use alloc::{format, string::String, vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        machine::machine_ir::{
            fp_reg_index, is_fp_reg, MachineBlock, MachineBlockId, MachineFloatWidth,
            MachineFuncId, MachineFunction, MachineFunctionAbi, MachineReg, MachineTerminator,
            MachineTrapKind, MachineValue, MACHINE_FIXED_REG_COUNT,
        },
    },
};

use crate::vm::runtime::code::CompiledNativeModule;

use super::helpers::{trap_kind_index, MACHINE_TRAP_KIND_COUNT};
use super::text_emitter::TextEmitter;
#[cfg(sf_has_debug_regions)]
use super::types::DebugRegion;
use super::types::{
    DirectCallPatch, EdgeStub, FunctionArtifact, LocalPtrPatch, PendingLocalPtrPatch,
};

/// Shared compilation state for all arch backends.
///
/// Owns all mutable bookkeeping (text buffer, labels, edges, patches).
/// Borrows the immutable inputs (`compiled`, `function`).
/// Each arch backend embeds this as `pub core: CompilerCore<'a>`.
#[derive(Debug)]
pub(crate) struct CompilerCore<'a> {
    pub compiled: &'a CompiledNativeModule,
    pub function: &'a MachineFunction,
    pub text: TextEmitter,
    pub labels: Vec<Option<usize>>,
    pub block_labels: Vec<usize>,
    pub edge_stubs: Vec<EdgeStub>,
    pub resolved_ptr_patches: Vec<LocalPtrPatch>,
    pub local_ptr_patches: Vec<PendingLocalPtrPatch>,
    pub direct_call_patches: Vec<DirectCallPatch>,
    pub deferred_traps: Vec<(usize, MachineTrapKind)>,
    pub fp_reg_widths: Vec<Option<MachineFloatWidth>>,
    pub current_block: Option<MachineBlockId>,
    pub current_op_index: Option<usize>,
    pub current_edge_target: Option<MachineBlockId>,
    pub stack_overflow_label: usize,
    /// Trap propagation label inside a function body. Reached from trap
    /// stubs (`lower_trap_dispatch`), `CallExternal` post-helper status
    /// checks, and post-BL status checks at every local-call site. Lowered
    /// as the body-local error tail (pop link save, pop call record,
    /// restore fp_reg, native return) — does NOT touch `C_RET0` so the
    /// caller's status check sees the propagated error code.
    ///
    /// Replaces the old function-wide `return_error_label` whose body was
    /// `lower_epilogue()` + `ret`. That old label assumed the C-ABI
    /// prologue had run, which is no longer true for locally-entered
    /// callees. See `docs/ABI_PLAN.md` §9.
    pub body_local_error_label: usize,
    /// Label bound to the function's internal entry point — i.e. the first
    /// instruction of the body prelude that local SF→SF calls patch
    /// against. Distinct from "the byte right after `lower_prologue()`"
    /// because the public entry contains a caller stub between the
    /// prologue and the body.
    pub internal_entry_label: usize,
    pub shared_trap_labels: [Option<usize>; MACHINE_TRAP_KIND_COUNT],
}

impl<'a> CompilerCore<'a> {
    /// Create a new `CompilerCore`.
    ///
    /// `fp_capacity` is the maximum number of FP machine registers the backend
    /// supports (e.g. `FP_MACHINE_REG_COUNT`).
    pub(crate) fn new(
        compiled: &'a CompiledNativeModule,
        function: &'a MachineFunction,
        fp_capacity: usize,
    ) -> Self {
        let block_cap = function
            .program
            .blocks
            .iter()
            .map(|block| block.id.0 as usize)
            .max()
            .unwrap_or(0)
            + 1;
        let mut labels = Vec::new();
        let mut block_labels = vec![usize::MAX; block_cap];
        for block in &function.program.blocks {
            let label = labels.len();
            labels.push(None);
            block_labels[block.id.0 as usize] = label;
        }
        let stack_overflow_label = labels.len();
        labels.push(None);
        let body_local_error_label = labels.len();
        labels.push(None);
        let internal_entry_label = labels.len();
        labels.push(None);
        let mut shared_trap_labels = [None; MACHINE_TRAP_KIND_COUNT];
        shared_trap_labels[trap_kind_index(MachineTrapKind::StackOverflow)] =
            Some(stack_overflow_label);

        let fp_reg_widths = Self::init_fp_widths(function, compiled.backend(), fp_capacity);

        Self {
            compiled,
            function,
            text: TextEmitter::new(),
            labels,
            block_labels,
            edge_stubs: Vec::new(),
            resolved_ptr_patches: Vec::new(),
            local_ptr_patches: Vec::new(),
            direct_call_patches: Vec::new(),
            deferred_traps: Vec::new(),
            fp_reg_widths,
            current_block: None,
            current_op_index: None,
            current_edge_target: None,
            stack_overflow_label,
            body_local_error_label,
            internal_entry_label,
            shared_trap_labels,
        }
    }

    fn init_fp_widths(
        function: &MachineFunction,
        _config: BackendConfig,
        fp_capacity: usize,
    ) -> Vec<Option<MachineFloatWidth>> {
        let mut widths = vec![None; fp_capacity];
        if function.program.fp_reg_init_widths.is_empty() {
            // Unified dynamic FP banks no longer let us infer ownership or
            // width defaults from register number. Until the lowering path
            // publishes explicit init widths, every dynamic FP reg starts
            // unknown and is typed on first materialization.
        } else {
            for (i, width) in function
                .program
                .fp_reg_init_widths
                .iter()
                .copied()
                .enumerate()
            {
                if i < fp_capacity {
                    widths[i] = width;
                }
            }
        }
        widths
    }

    // ── Label management ─────────────────────────────────────────────────

    pub(crate) fn new_label(&mut self) -> usize {
        let label = self.labels.len();
        self.labels.push(None);
        label
    }

    pub(crate) fn bind_label(&mut self, label: usize) {
        self.labels[label] = Some(self.text.len());
    }

    pub(crate) fn block_label(&self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.block_labels
            .get(target.0 as usize)
            .copied()
            .filter(|label| *label != usize::MAX)
            .ok_or_else(|| WasmError::internal("block label is out of range".into()))
    }

    // ── Trap label management ────────────────────────────────────────────

    pub(crate) fn ensure_trap_label(&mut self, kind: MachineTrapKind) -> usize {
        let slot = trap_kind_index(kind);
        if let Some(label) = self.shared_trap_labels[slot] {
            return label;
        }
        let label = self.new_label();
        self.shared_trap_labels[slot] = Some(label);
        self.deferred_traps.push((label, kind));
        label
    }

    // ── Runtime metadata ─────────────────────────────────────────────────

    pub(crate) fn runtime_for(
        &self,
        func_id: MachineFuncId,
    ) -> Result<&MachineFunctionAbi, WasmError> {
        self.compiled
            .abi()
            .functions
            .get(func_id.0 as usize)
            .ok_or_else(|| {
                WasmError::internal(format!(
                    "runtime metadata missing for machine function {}",
                    func_id.0
                ))
            })
    }

    // ── FP register tracking ─────────────────────────────────────────────

    #[inline]
    pub(crate) fn is_fp_reg(&self, reg: MachineReg) -> bool {
        is_fp_reg(reg, self.compiled.backend())
    }

    /// Return the FP bank index for a machine register.
    pub(crate) fn fp_reg_index(&self, reg: MachineReg) -> Result<usize, WasmError> {
        fp_reg_index(reg, self.compiled.backend()).ok_or_else(|| {
            WasmError::invalid(format!("expected FP register, got machine reg {}", reg.0))
        })
    }

    pub(crate) fn set_fp_reg_width(
        &mut self,
        reg: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        let index = self.fp_reg_index(reg)?;
        let slot = self.fp_reg_widths.get_mut(index).ok_or_else(|| {
            WasmError::invalid(format!("no tracked FP slot for machine reg {}", reg.0))
        })?;
        *slot = Some(width);
        Ok(())
    }

    pub(crate) fn fp_reg_width(&self, reg: MachineReg) -> Result<MachineFloatWidth, WasmError> {
        let index = self.fp_reg_index(reg)?;
        self.fp_reg_widths
            .get(index)
            .and_then(|width| *width)
            .ok_or_else(|| {
                WasmError::invalid(format!(
                    "missing float-width tracking for machine reg {} in function {} at {}",
                    reg.0,
                    self.function.id.0,
                    self.current_location(),
                ))
            })
    }

    pub(crate) fn reset_block_fp_state(&mut self, block: &MachineBlock) -> Result<(), WasmError> {
        for slot in &mut self.fp_reg_widths {
            *slot = None;
        }
        for param in &block.params {
            if let Some(width) = param.ty.float_width() {
                self.set_fp_reg_width(param.reg, width)?;
            }
        }
        Ok(())
    }

    // ── Location tracking ────────────────────────────────────────────────

    pub(crate) fn current_location(&self) -> String {
        if let Some(target) = self.current_edge_target {
            return format!("edge stub to b{}", target.0);
        }
        if let Some(block) = self.current_block {
            if let Some(op_index) = self.current_op_index {
                return format!("b{} op{}", block.0, op_index);
            }
            return format!("b{} terminator", block.0);
        }
        format!("unknown location")
    }

    // ── Block layout ─────────────────────────────────────────────────────

    pub(crate) fn block_layout(&self) -> Vec<MachineBlockId> {
        let mut order = Vec::with_capacity(self.function.program.blocks.len());
        let mut seen = vec![false; self.function.program.blocks.len()];
        let mut worklist = vec![self.function.program.entry];

        while let Some(start) = worklist.pop() {
            self.extend_block_trace(start, &mut seen, &mut order, &mut worklist);
        }

        for block in &self.function.program.blocks {
            if seen[block.id.as_usize()] {
                continue;
            }
            worklist.push(block.id);
            while let Some(start) = worklist.pop() {
                self.extend_block_trace(start, &mut seen, &mut order, &mut worklist);
            }
        }

        order
    }

    fn extend_block_trace(
        &self,
        start: MachineBlockId,
        seen: &mut [bool],
        order: &mut Vec<MachineBlockId>,
        worklist: &mut Vec<MachineBlockId>,
    ) {
        let mut current = Some(start);
        while let Some(block_id) = current {
            let Some(block) = self.function.program.blocks.get(block_id.as_usize()) else {
                break;
            };
            if seen[block_id.as_usize()] {
                break;
            }
            seen[block_id.as_usize()] = true;
            order.push(block_id);

            let mut fallthrough = None;
            match &block.terminator {
                MachineTerminator::Jump(edge) => {
                    if self.is_identity_edge(edge.target, &edge.args) {
                        fallthrough = Some(edge.target);
                    } else {
                        worklist.push(edge.target);
                    }
                }
                MachineTerminator::Branch {
                    then_edge,
                    else_edge,
                    ..
                } => {
                    if self.is_identity_edge(else_edge.target, &else_edge.args) {
                        fallthrough = Some(else_edge.target);
                        worklist.push(then_edge.target);
                    } else {
                        worklist.push(else_edge.target);
                        worklist.push(then_edge.target);
                    }
                }
                MachineTerminator::JumpTable { entries, .. } => {
                    for edge in entries.iter().rev() {
                        worklist.push(edge.target);
                    }
                }
                MachineTerminator::CallDirect { continuation, .. }
                | MachineTerminator::CallIndirect { continuation, .. } => {
                    fallthrough = Some(*continuation);
                }
                MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
            }

            current = fallthrough.filter(|target| !seen[target.as_usize()]);
        }
    }

    // ── Edge management ──────────────────────────────────────────────────

    pub(crate) fn is_identity_edge(&self, target: MachineBlockId, args: &[MachineValue]) -> bool {
        let Some(block) = self.function.program.blocks.get(target.as_usize()) else {
            return false;
        };
        if block.params.len() != args.len() {
            return false;
        }
        block
            .params
            .iter()
            .zip(args.iter())
            .all(|(param, arg)| match arg {
                MachineValue::Reg(r) | MachineValue::ReservedReg(r) => *r == param.reg,
                MachineValue::Imm64(_) => false,
            })
    }

    /// Returns a label for the edge. If the edge is an identity mapping
    /// (no copies needed), returns the target block's label directly.
    /// Otherwise, creates an edge stub.
    pub(crate) fn emit_edge(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        if self.is_identity_edge(target, args) {
            return self.block_label(target);
        }
        self.add_edge_stub(target, args)
    }

    fn add_edge_stub(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        let block = self
            .function
            .program
            .blocks
            .get(target.as_usize())
            .ok_or_else(|| WasmError::internal("edge target block is out of range".into()))?;
        let label = self.new_label();
        let arg_float_widths = args
            .iter()
            .map(|arg| match arg {
                MachineValue::Reg(reg) if self.is_fp_reg(*reg) => self.fp_reg_width(*reg).map(Some),
                MachineValue::ReservedReg(_) | MachineValue::Reg(_) | MachineValue::Imm64(_) => {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.edge_stubs.push(EdgeStub {
            label,
            target,
            params: block.params.clone(),
            args: args.to_vec(),
            arg_float_widths,
        });
        Ok(label)
    }

    // ── Validation ───────────────────────────────────────────────────────

    /// Validate that a MachineFunction's register counts are within the
    /// backend's capacity. Called at the start of compile_function.
    pub(crate) fn validate_function(
        arch_name: &str,
        function: &MachineFunction,
        config: BackendConfig,
        max_total_regs: usize,
        max_fp_regs: usize,
    ) -> Result<(), WasmError> {
        let reg_count = config.total_reg_count();
        let first_fp = config.first_fp_reg();
        if reg_count as usize > max_total_regs {
            return Err(WasmError::invalid(format!(
                "{} backend supports at most {} machine regs, got {} in function {}",
                arch_name, max_total_regs, reg_count, function.id.0
            )));
        }
        if first_fp < MACHINE_FIXED_REG_COUNT || first_fp > reg_count {
            return Err(WasmError::invalid(format!(
                "{} backend received invalid first_fp_reg {} for function {}",
                arch_name, first_fp, function.id.0,
            )));
        }
        if (reg_count - first_fp) as usize > max_fp_regs {
            return Err(WasmError::invalid(format!(
                "{} backend supports at most {} FP machine regs, got {} in function {}",
                arch_name,
                max_fp_regs,
                reg_count - first_fp,
                function.id.0,
            )));
        }
        if function
            .program
            .blocks
            .iter()
            .find(|block| block.id == function.program.entry)
            .map(|block| !block.params.is_empty())
            .unwrap_or(false)
        {
            return Err(WasmError::invalid(format!(
                "{} backend does not support entry block params yet",
                arch_name
            )));
        }
        Ok(())
    }

    // ── Artifact extraction ──────────────────────────────────────────────

    /// Resolve pending local-ptr patches and build the final artifact.
    pub(crate) fn finish_artifact(
        self,
        internal_entry_offset: usize,
        #[cfg(sf_has_guard_pages)] body_local_error_offset: usize,
        #[cfg(sf_has_debug_regions)] debug_regions: Vec<DebugRegion>,
    ) -> Result<FunctionArtifact, WasmError> {
        let mut local_ptr_patches = self.resolved_ptr_patches;
        local_ptr_patches.reserve(self.local_ptr_patches.len());
        for patch in &self.local_ptr_patches {
            let target_offset = self
                .labels
                .get(patch.target_label)
                .and_then(|offset| *offset)
                .ok_or_else(|| {
                    WasmError::internal("local continuation label is unresolved".into())
                })?;
            local_ptr_patches.push(LocalPtrPatch {
                literal_offset: patch.literal_offset,
                target_offset,
            });
        }
        Ok(FunctionArtifact {
            text: self.text,
            local_ptr_patches,
            direct_call_patches: self.direct_call_patches,
            #[cfg(sf_has_guard_pages)]
            body_local_error_offset,
            internal_entry_offset,
            #[cfg(sf_has_debug_regions)]
            debug_regions,
        })
    }
}
