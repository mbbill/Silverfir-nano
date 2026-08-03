use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        jit::backend::BackendConfig,
        jit::machine::machine_ir::{
            fp_reg_index, is_fp_reg, MachineBlock, MachineBlockId, MachineFloatWidth,
            MachineFuncId, MachineFunction, MachineFunctionAbi, MachineParamLoc, MachineReg,
            MachineRegOwner, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
            MACHINE_FIXED_REG_COUNT,
        },
    },
};

#[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
use crate::vm::jit::machine::low32_liveness::Low32DeadHiDefs;
use crate::vm::jit::runtime::code::CodegenModuleView;

use super::helpers::{trap_kind_index, MACHINE_TRAP_KIND_COUNT};
use super::text_emitter::TextEmitter;
#[cfg(sf_has_debug_regions)]
use super::types::DebugRegion;
use super::types::{
    DirectCallPatch, EdgeStub, FunctionArtifact, LocalPtrPatch, PendingLocalPtrPatch,
};

/// Function body source for a backend emission pass.
///
/// The optimized compiler owns a full `MachineFunction`; the template JIT
/// streams directly from wasm and only needs the function id for ABI metadata,
/// patches, and runtime lookups.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FunctionBody<'a> {
    Mir(&'a MachineFunction),
    Template { func_id: MachineFuncId },
}

impl<'a> FunctionBody<'a> {
    pub(crate) fn func_id(self) -> MachineFuncId {
        match self {
            Self::Mir(function) => function.id,
            Self::Template { func_id } => func_id,
        }
    }

    fn mir_function(self) -> Option<&'a MachineFunction> {
        match self {
            Self::Mir(function) => Some(function),
            Self::Template { .. } => None,
        }
    }
}

/// Shared compilation state for all arch backends.
///
/// Owns all mutable bookkeeping (text buffer, labels, edges, patches).
/// Borrows immutable module metadata and records where the function body
/// events come from.
/// Each arch backend embeds this as `pub core: CompilerCore<'a>`.
#[derive(Debug)]
pub(crate) struct CompilerCore<'a> {
    pub compiled: &'a dyn CodegenModuleView,
    pub body: FunctionBody<'a>,
    pub func_id: MachineFuncId,
    pub text: TextEmitter,
    pub labels: collections::Vec<Option<usize>>,
    pub block_labels: collections::Vec<usize>,
    pub edge_stubs: collections::Vec<EdgeStub>,
    pub resolved_ptr_patches: collections::Vec<LocalPtrPatch>,
    pub local_ptr_patches: collections::Vec<PendingLocalPtrPatch>,
    pub direct_call_patches: collections::Vec<DirectCallPatch>,
    pub deferred_traps: collections::Vec<(usize, MachineTrapKind)>,
    pub fp_reg_widths: collections::Vec<Option<MachineFloatWidth>>,
    #[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
    pub low32_dead_hi_defs: Low32DeadHiDefs,
    pub current_block: Option<MachineBlockId>,
    pub current_op_index: Option<usize>,
    pub current_edge_target: Option<MachineBlockId>,
    pub stack_overflow_label: usize,
    /// Trap propagation label inside a function body. Reached from trap
    /// stubs (`lower_trap_dispatch`), `CallRuntime` post-helper status
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
    pub(crate) fn new(compiled: &'a dyn CodegenModuleView, body: FunctionBody<'a>) -> Self {
        let config = compiled.backend();
        let func_id = body.func_id();
        let block_cap = body
            .mir_function()
            .map(|function| {
                function
                    .program
                    .blocks
                    .iter()
                    .map(|block| block.id.0 as usize)
                    .max()
                    .unwrap_or(0)
                    + 1
            })
            .unwrap_or(0);
        let mut labels = collections::Vec::new();
        let mut block_labels = collections::vec![usize::MAX; block_cap];
        if let Some(function) = body.mir_function() {
            for block in &function.program.blocks {
                let label = labels.len();
                labels.push(None);
                block_labels[block.id.0 as usize] = label;
            }
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

        let fp_reg_widths = body
            .mir_function()
            .map(|function| Self::init_fp_widths(function, config))
            .unwrap_or_else(|| collections::vec![None; usize::from(config.fp_dynamic_budget)]);
        #[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
        let low32_dead_hi_defs = body
            .mir_function()
            .map(|function| {
                Low32DeadHiDefs::compute(function, usize::from(config.total_reg_count()))
            })
            .unwrap_or_default();

        Self {
            compiled,
            body,
            func_id,
            text: TextEmitter::new(),
            labels,
            block_labels,
            edge_stubs: collections::Vec::new(),
            resolved_ptr_patches: collections::Vec::new(),
            local_ptr_patches: collections::Vec::new(),
            direct_call_patches: collections::Vec::new(),
            deferred_traps: collections::Vec::new(),
            fp_reg_widths,
            #[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
            low32_dead_hi_defs,
            current_block: None,
            current_op_index: None,
            current_edge_target: None,
            stack_overflow_label,
            body_local_error_label,
            internal_entry_label,
            shared_trap_labels,
        }
    }

    #[inline]
    pub(crate) fn current_runtime(&self) -> Option<&MachineFunctionAbi> {
        self.compiled.runtime_for(self.func_id)
    }

    #[inline]
    pub(crate) fn gp_arg_lane_reg(&self, lane: u8) -> MachineReg {
        MachineReg(MACHINE_FIXED_REG_COUNT + u16::from(lane))
    }

    #[inline]
    pub(crate) fn fp_arg_lane_reg(&self, lane: u8) -> MachineReg {
        MachineReg(
            MACHINE_FIXED_REG_COUNT
                + u16::from(self.compiled.backend().gp_dynamic_budget)
                + u16::from(lane),
        )
    }

    pub(crate) fn mir_function(&self) -> Result<&'a MachineFunction, WasmError> {
        self.body
            .mir_function()
            .ok_or_else(|| WasmError::internal("template emission does not have a MachineFunction"))
    }

    #[cfg(any(sf_backend_arm64, sf_backend_armv7a, sf_backend_thumbm, sf_backend_x64))]
    #[inline]
    pub(crate) fn preserved_clobbers(&self) -> &[MachineReg] {
        self.body
            .mir_function()
            .map(|function| function.preserved_clobbers.as_slice())
            .unwrap_or(&[])
    }

    pub(crate) fn mir_blocks(&self) -> Result<&'a [MachineBlock], WasmError> {
        Ok(&self.mir_function()?.program.blocks)
    }

    #[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
    pub(crate) fn current_pair_hi_dead(&self) -> bool {
        self.low32_dead_hi_defs
            .is_dead_at(self.current_block, self.current_op_index)
    }

    fn init_fp_widths(
        function: &MachineFunction,
        config: BackendConfig,
    ) -> collections::Vec<Option<MachineFloatWidth>> {
        let mut widths = collections::vec![None; usize::from(config.fp_dynamic_budget)];
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
                if i < widths.len() {
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
            .ok_or_else(|| WasmError::internal("block label is out of range"))
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

    #[cfg(not(sf_backend_arm64))]
    pub(crate) fn runtime_for(
        &self,
        func_id: MachineFuncId,
    ) -> Result<&MachineFunctionAbi, WasmError> {
        self.compiled
            .runtime_for(func_id)
            .ok_or_else(|| WasmError::internal("runtime metadata missing for machine function"))
    }

    // ── FP register tracking ─────────────────────────────────────────────

    #[inline]
    pub(crate) fn is_fp_reg(&self, reg: MachineReg) -> bool {
        is_fp_reg(reg, self.compiled.backend())
    }

    /// Return the FP bank index for a machine register.
    pub(crate) fn fp_reg_index(&self, reg: MachineReg) -> Result<usize, WasmError> {
        fp_reg_index(reg, self.compiled.backend())
            .ok_or_else(|| WasmError::invalid("expected FP register, got machine reg"))
    }

    pub(crate) fn set_fp_reg_width(
        &mut self,
        reg: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        let index = self.fp_reg_index(reg)?;
        let slot = self
            .fp_reg_widths
            .get_mut(index)
            .ok_or_else(|| WasmError::invalid("no tracked FP slot for machine reg"))?;
        *slot = Some(width);
        Ok(())
    }

    pub(crate) fn fp_reg_width(&self, reg: MachineReg) -> Result<MachineFloatWidth, WasmError> {
        let index = self.fp_reg_index(reg)?;
        self.fp_reg_widths
            .get(index)
            .and_then(|width| *width)
            .ok_or_else(|| {
                WasmError::invalid("missing float-width tracking for machine reg in function at")
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
        if self
            .body
            .mir_function()
            .map(|function| function.program.entry == block.id)
            .unwrap_or(false)
        {
            let param_locs = self
                .current_runtime()
                .map(|runtime| runtime.param_locs.clone())
                .unwrap_or_default();
            for loc in param_locs {
                if let MachineParamLoc::FpArg { lane, ty, .. } = loc {
                    if let Some(width) = ty.float_width() {
                        self.set_fp_reg_width(self.fp_arg_lane_reg(lane), width)?;
                    }
                }
            }
        }
        Ok(())
    }

    // ── Block layout ─────────────────────────────────────────────────────

    pub(crate) fn block_layout(&self) -> Result<collections::Vec<MachineBlockId>, WasmError> {
        let function = self.mir_function()?;
        let blocks = &function.program.blocks;
        let mut order = collections::Vec::with_capacity(blocks.len());
        let mut seen = collections::vec![false; blocks.len()];
        let mut worklist = collections::vec![function.program.entry];

        while let Some(start) = worklist.pop() {
            self.extend_block_trace(start, blocks, &mut seen, &mut order, &mut worklist);
        }

        for block in blocks {
            if seen[block.id.as_usize()] {
                continue;
            }
            worklist.push(block.id);
            while let Some(start) = worklist.pop() {
                self.extend_block_trace(start, blocks, &mut seen, &mut order, &mut worklist);
            }
        }

        Ok(order)
    }

    fn extend_block_trace(
        &self,
        start: MachineBlockId,
        blocks: &[MachineBlock],
        seen: &mut [bool],
        order: &mut collections::Vec<MachineBlockId>,
        worklist: &mut collections::Vec<MachineBlockId>,
    ) {
        let mut current = Some(start);
        while let Some(block_id) = current {
            let Some(block) = blocks.get(block_id.as_usize()) else {
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
                    if self.is_identity_edge(blocks, edge.target, &edge.args) {
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
                    if self.is_identity_edge(blocks, else_edge.target, &else_edge.args) {
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
                MachineTerminator::Call { success, .. } => {
                    fallthrough = Some(success.target);
                }
                MachineTerminator::TailCall { .. }
                | MachineTerminator::Return
                | MachineTerminator::ReturnScalar { .. }
                | MachineTerminator::Trap { .. } => {}
            }

            current = fallthrough.filter(|target| !seen[target.as_usize()]);
        }
    }

    // ── Edge management ──────────────────────────────────────────────────

    pub(crate) fn is_identity_edge(
        &self,
        blocks: &[MachineBlock],
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> bool {
        let Some(block) = blocks.get(target.as_usize()) else {
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
        let blocks = self.mir_blocks()?;
        if self.is_identity_edge(blocks, target, args) {
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
            .mir_blocks()?
            .get(target.as_usize())
            .ok_or_else(|| WasmError::internal("edge target block is out of range"))?;
        let label = self.new_label();
        let arg_float_widths = args
            .iter()
            .map(|arg| match arg {
                MachineValue::Reg(reg) if self.is_fp_reg(*reg) => self.fp_reg_width(*reg).map(Some),
                MachineValue::ReservedReg(_) | MachineValue::Reg(_) | MachineValue::Imm64(_) => {
                    Ok(None)
                }
            })
            .collect::<Result<collections::Vec<_>, _>>()?;
        self.edge_stubs.push(EdgeStub {
            label,
            target,
            params: block.params.clone(),
            args: args.to_vec().into(),
            arg_float_widths,
        });
        Ok(label)
    }

    // ── Validation ───────────────────────────────────────────────────────

    /// Validate that a MachineFunction's register counts are within the
    /// backend's capacity. Called at the start of compile_function.
    pub(crate) fn validate_function(
        _arch_name: &str,
        function: &MachineFunction,
        runtime: Option<&MachineFunctionAbi>,
        config: BackendConfig,
        max_total_regs: usize,
        max_fp_regs: usize,
    ) -> Result<(), WasmError> {
        let reg_count = config.total_reg_count();
        let first_fp = config.first_fp_reg();
        if reg_count as usize > max_total_regs {
            return Err(WasmError::invalid(
                "backend supports at most machine regs, got in function",
            ));
        }
        if first_fp < MACHINE_FIXED_REG_COUNT || first_fp > reg_count {
            return Err(WasmError::invalid(
                "backend received invalid first_fp_reg for function",
            ));
        }
        if (reg_count - first_fp) as usize > max_fp_regs {
            return Err(WasmError::invalid(
                "backend supports at most FP machine regs, got in function",
            ));
        }
        let entry_params = function
            .program
            .blocks
            .iter()
            .find(|block| block.id == function.program.entry)
            .map(|block| block.params.as_slice())
            .unwrap_or_default();
        if !entry_params.is_empty() {
            let runtime = runtime.ok_or_else(|| {
                WasmError::invalid("entry block params require function ABI metadata")
            })?;
            let mut expected = collections::Vec::new();
            for loc in &runtime.param_locs {
                match *loc {
                    MachineParamLoc::Frame { .. } => {}
                    MachineParamLoc::GpArg { lane, ty, .. } => {
                        expected.push((MachineReg(MACHINE_FIXED_REG_COUNT + u16::from(lane)), ty))
                    }
                    MachineParamLoc::GpArgPair {
                        lo_lane, hi_lane, ..
                    } => {
                        expected.push((
                            MachineReg(MACHINE_FIXED_REG_COUNT + u16::from(lo_lane)),
                            MachineStorageType::GpWord,
                        ));
                        expected.push((
                            MachineReg(MACHINE_FIXED_REG_COUNT + u16::from(hi_lane)),
                            MachineStorageType::GpWord,
                        ));
                    }
                    MachineParamLoc::FpArg { lane, ty, .. } => expected.push((
                        MachineReg(
                            MACHINE_FIXED_REG_COUNT
                                + u16::from(config.gp_dynamic_budget)
                                + u16::from(lane),
                        ),
                        ty,
                    )),
                }
            }
            let matches_incoming_abi = entry_params.len() == expected.len()
                && entry_params
                    .iter()
                    .zip(expected.iter())
                    .all(|(param, &(reg, ty))| {
                        param.reg == reg
                            && param.ty == ty
                            && param.owner == MachineRegOwner::CachedCell
                    });
            if !matches_incoming_abi {
                return Err(WasmError::invalid(
                    "entry block params must exactly match incoming ABI registers",
                ));
            }
        }
        Ok(())
    }

    // ── Artifact extraction ──────────────────────────────────────────────

    /// Resolve pending local-ptr patches and build the final artifact.
    pub(crate) fn finish_artifact(
        self,
        internal_entry_offset: usize,
        #[cfg(sf_has_guard_pages)] body_local_error_offset: usize,
        #[cfg(sf_has_debug_regions)] debug_regions: collections::Vec<DebugRegion>,
    ) -> Result<FunctionArtifact, WasmError> {
        let mut local_ptr_patches = self.resolved_ptr_patches;
        local_ptr_patches.reserve(self.local_ptr_patches.len());
        for patch in &self.local_ptr_patches {
            let target_offset = self
                .labels
                .get(patch.target_label)
                .and_then(|offset| *offset)
                .ok_or_else(|| WasmError::internal("local continuation label is unresolved"))?;
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
