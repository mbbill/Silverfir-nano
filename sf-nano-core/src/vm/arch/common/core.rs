use crate::collections;

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

#[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
use crate::vm::machine::low32_liveness::Low32DeadHiDefs;
use crate::vm::runtime::code::CodegenModuleView;

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
    pub compiled: &'a dyn CodegenModuleView,
    pub function: MachineFunction,
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
    pub(crate) fn new(compiled: &'a dyn CodegenModuleView, function: MachineFunction) -> Self {
        let config = compiled.backend();
        let block_cap = function
            .program
            .blocks
            .iter()
            .map(|block| block.id.0 as usize)
            .max()
            .unwrap_or(0)
            + 1;
        let mut labels = collections::Vec::new();
        let mut block_labels = collections::vec![usize::MAX; block_cap];
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

        let fp_reg_widths = Self::init_fp_widths(&function, config);
        #[cfg(any(sf_backend_armv7a, sf_backend_thumbm, sf_backend_riscv32))]
        let low32_dead_hi_defs =
            Low32DeadHiDefs::compute(&function, usize::from(config.total_reg_count()));

        Self {
            compiled,
            function,
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

    /// Streaming-mode block intake: register an arriving MIR block by
    /// (1) inserting a params-only placeholder at `function.program.blocks[id]`
    /// so later edge-stub drains can read `target.params`, and
    /// (2) ensuring a label exists for the block id so callers can bind
    /// it before lowering ops.
    ///
    /// Forward references — earlier blocks whose terminators name a
    /// not-yet-arrived target id — are handled via `ensure_block_label`,
    /// which lazily allocates labels for ids beyond the current high-water
    /// mark. When the target finally arrives, `add_streaming_block`
    /// replaces the placeholder and `bind_label` resolves the prior
    /// allocation against the current text offset.
    pub(crate) fn add_streaming_block(
        &mut self,
        id: MachineBlockId,
        params: collections::Vec<crate::vm::machine::machine_ir::MachineBlockParam>,
    ) -> Result<usize, WasmError> {
        let id_idx = id.as_usize();
        let blocks = &mut self.function.program.blocks;
        if id_idx >= blocks.len() {
            // Forward-reference padding: any ids in the gap are filled
            // with empty placeholders. They will be replaced when those
            // blocks themselves arrive via add_streaming_block.
            while blocks.len() < id_idx {
                let pad_id = MachineBlockId(blocks.len() as u32);
                blocks.push(MachineBlock {
                    id: pad_id,
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Return,
                });
            }
            blocks.push(MachineBlock {
                id,
                params,
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            });
        } else {
            // Slot already present — either a forward-padded placeholder
            // or the streaming front-end re-emitted with the same id (a
            // bug). Replace with the real params; ops/terminator stay
            // empty since the caller drives begin/emit/end against the
            // block they hand to the emitter, not against this slot.
            blocks[id_idx] = MachineBlock {
                id,
                params,
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            };
        }
        self.ensure_block_label(id)
    }

    /// Streaming-mode label allocation. Returns the existing label if
    /// one was already allocated (e.g. by an earlier forward reference),
    /// otherwise allocates a fresh one and stores it in `block_labels[id]`.
    pub(crate) fn ensure_block_label(&mut self, id: MachineBlockId) -> Result<usize, WasmError> {
        let id_idx = id.as_usize();
        if id_idx >= self.block_labels.len() {
            self.block_labels.resize(id_idx + 1, usize::MAX);
        }
        if self.block_labels[id_idx] == usize::MAX {
            let label = self.labels.len();
            self.labels.push(None);
            self.block_labels[id_idx] = label;
        }
        Ok(self.block_labels[id_idx])
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

    /// Move block `block_idx` out of the owned function and replace the
    /// slot with a stripped placeholder. Used by the streaming pipeline
    /// to drive begin/emit/end against an owned local block, then drop
    pub(crate) fn new_label(&mut self) -> usize {
        let label = self.labels.len();
        self.labels.push(None);
        label
    }

    pub(crate) fn bind_label(&mut self, label: usize) {
        self.labels[label] = Some(self.text.len());
    }

    /// Resolve a target block id to its label. Allocates lazily on first
    /// reference: this lets callers issue branches to forward-reference
    /// blocks (whose `add_streaming_block` call has not yet happened) and
    /// still get back a valid label that the eventual `add_streaming_block`
    /// + `bind_label` resolves at emission time.
    ///
    /// In the buffered (full-opt) pipeline, every block label is
    /// pre-allocated by the constructor, so the lazy path is a no-op.
    pub(crate) fn block_label(&mut self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.ensure_block_label(target)
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
        Ok(())
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
        // Target params are looked up at drain time, not here. That lets
        // the streaming arch-emit driver issue edges to forward-target
        // blocks that have not yet been added to the owned function.
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
        if function
            .program
            .blocks
            .iter()
            .find(|block| block.id == function.program.entry)
            .map(|block| !block.params.is_empty())
            .unwrap_or(false)
        {
            return Err(WasmError::invalid(
                "backend does not support entry block params yet",
            ));
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
