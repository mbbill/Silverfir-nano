//! x86_64 backend: struct definitions and `ArchBackend` trait implementation.
//!
//! This file is the bridge between the common pipeline and the x86_64-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs` as inherent methods.

use crate::collections;

#[cfg(sf_has_guard_pages)]
use crate::vm::jit::machine::machine_ir::MachineAddr;
use crate::{
    error::WasmError,
    vm::{
        jit::machine::machine_ir::{
            MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineInst, MachineIntWidth,
            MachineReg, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
            MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        jit::runtime::{code::NativeRootEntry, code_buf::CodeBuffer, context::ctx_offset},
    },
};

use super::{
    abi::{
        self, fp_machine_reg, map_fixed_reg, map_reg, max_fp_machine_regs, max_total_machine_regs,
        C_ARG0, C_ARG1,
    },
    callconv,
    enc::{self, Cc},
    gp_scratch::GpScratchPool,
    reg::X86Reg,
};
#[cfg(sf_ir_dump)]
use crate::vm::jit::arch::common::types::DebugRegion;
use crate::vm::jit::arch::common::{
    backend::ArchBackend, core::CompilerCore, scratch_pool::ScratchPool, types::ParallelSource,
};

// ── Frame layout ────────────────────────────────────────────────────────────

/// Extra bytes subtracted from RSP after the GP saves. Owned by the active
/// calling convention because the exact shape depends on whether Win64 XMM
/// spill and shadow space are required. See `callconv::sysv::STACK_PADDING`
/// / `callconv::win64::STACK_PADDING`.
const STACK_PADDING: u32 = callconv::STACK_PADDING;

// ── Branch fixup types ───────────────────────────────────────────────────────

/// x86_64 branch fixup: we emit a JMP/Jcc with a placeholder rel32, then
/// patch it once the target label is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BranchFixup {
    /// Byte offset of the rel32 field in the text.
    pub rel32_offset: usize,
    /// Label index to resolve.
    pub label: usize,
}

// ── Compiled entry ───────────────────────────────────────────────────────────

/// Result of compiling one function to x86_64 machine code.
#[derive(Clone, Debug)]
pub(crate) struct CompiledX86_64Entry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    #[cfg(sf_ir_dump)]
    pub debug_regions: collections::Vec<DebugRegion>,
}

// ── X86_64Backend ────────────────────────────────────────────────────────────

#[derive(Debug)]
/// One pending RIP-relative reference into the function's FP literal pool.
pub(super) struct FpLiteralFixup {
    /// Text offset of the load's disp32 field.
    pub disp_offset: usize,
    /// Index into `fp_literals`.
    pub literal_index: usize,
}

pub(crate) struct X86_64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: collections::Vec<BranchFixup>,
    pub(super) gp_scratch: GpScratchPool,
    pub(super) fp_scratch: ScratchPool<u32, 2>,
    /// Deduplicated scalar FP immediates, flushed after the function body as
    /// its literal pool and loaded RIP-relatively.
    pub(super) fp_literals: collections::Vec<u64>,
    pub(super) fp_literal_fixups: collections::Vec<FpLiteralFixup>,
    /// Peephole state: EFLAGS still reflects the 32-bit result of this
    /// register's most recent ALU write, valid only while the text cursor
    /// sits at the recorded position. Any emission moves the cursor and
    /// invalidates the entry implicitly; nothing ever needs to clear it.
    pub(super) flags32: Option<(X86Reg, usize)>,
    /// Jump tables pending emission. Entry words flush after the function
    /// body next to the FP literal pool so table data never sits in the
    /// instruction stream between a dispatch and its handlers.
    pub(super) pending_jump_tables: collections::Vec<PendingJumpTable>,
}

/// One deferred jump table: the movabs immediate to patch with the
/// table's address, and the resolved edge label for every entry.
pub(super) struct PendingJumpTable {
    pub(super) base_imm_offset: usize,
    pub(super) entry_labels: collections::Vec<usize>,
}

impl X86_64Backend<'_> {
    /// Record that EFLAGS now reflects `reg`'s 32-bit result.
    pub(super) fn note_flags32(&mut self, reg: X86Reg) {
        self.flags32 = Some((reg, self.core.text.len()));
    }

    /// True while EFLAGS still reflects `reg`'s 32-bit value — a
    /// `test reg, reg` here would be redundant and would break
    /// ALU/branch macro-fusion.
    pub(super) fn flags32_current(&self, reg: X86Reg) -> bool {
        self.flags32 == Some((reg, self.core.text.len()))
    }

    pub(super) fn intern_fp_literal(&mut self, bits: u64) -> usize {
        if let Some(index) = self
            .fp_literals
            .iter()
            .position(|&existing| existing == bits)
        {
            return index;
        }
        self.fp_literals.push(bits);
        self.fp_literals.len() - 1
    }
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for X86_64Backend<'a> {
    const NAME: &'static str = "x86_64";

    fn max_total_regs() -> usize {
        max_total_machine_regs()
    }
    fn max_fp_regs() -> usize {
        max_fp_machine_regs()
    }

    fn new(core: CompilerCore<'a>) -> Self {
        Self {
            core,
            fixups: collections::Vec::new(),
            gp_scratch: GpScratchPool::new(abi::gp_backend_owned_regs()),
            fp_scratch: abi::new_fp_scratch_pool(),
            fp_literals: collections::Vec::new(),
            fp_literal_fixups: collections::Vec::new(),
            flags32: None,
            pending_jump_tables: collections::Vec::new(),
        }
    }

    fn core(&self) -> &CompilerCore<'a> {
        &self.core
    }
    fn core_mut(&mut self) -> &mut CompilerCore<'a> {
        &mut self.core
    }
    fn into_core(self) -> CompilerCore<'a> {
        self.core
    }

    /// 16-byte-align loop headers so a tight loop body never straddles a
    /// fetch window: a straddled two-instruction loop measured 2 cycles
    /// per iteration where the aligned form sustains 1. Padding executes
    /// only on fallthrough entry, as a few multi-byte NOPs.
    fn align_loop_header(&mut self) {
        const LOOP_HEADER_ALIGN: usize = 16;
        let misalign = self.core.text.next_addr_for_alignment() % LOOP_HEADER_ALIGN;
        if misalign != 0 {
            enc::emit_nops(&mut self.core.text, LOOP_HEADER_ALIGN - misalign);
        }
    }

    fn lower_function_literal_pool(&mut self) -> Result<(), WasmError> {
        if !self.pending_jump_tables.is_empty() {
            while self.core.text.len() % 8 != 0 {
                self.core.text.emit_u8(0);
            }
            for table in core::mem::take(&mut self.pending_jump_tables) {
                let table_offset = self.core.text.len();
                self.core.resolved_ptr_patches.push(
                    crate::vm::jit::arch::common::types::LocalPtrPatch {
                        literal_offset: table.base_imm_offset,
                        target_offset: table_offset,
                    },
                );
                for label in table.entry_labels {
                    let literal_offset = self.core.text.emit_u64(0);
                    self.core.local_ptr_patches.push(
                        crate::vm::jit::arch::common::types::PendingLocalPtrPatch {
                            literal_offset,
                            target_label: label,
                        },
                    );
                }
            }
        }
        if self.fp_literals.is_empty() {
            return Ok(());
        }
        while self.core.text.len() % 8 != 0 {
            self.core.text.emit_u8(0);
        }
        let mut literal_offsets = collections::Vec::with_capacity(self.fp_literals.len());
        for &bits in &self.fp_literals {
            literal_offsets.push(self.core.text.emit_u64(bits));
        }
        for fixup in &self.fp_literal_fixups {
            let literal_offset = literal_offsets[fixup.literal_index];
            // disp32 is relative to the end of the load instruction, whose
            // last field is the displacement itself.
            let disp = literal_offset as i64 - (fixup.disp_offset as i64 + 4);
            let disp = i32::try_from(disp).map_err(|_| {
                WasmError::internal("x86_64 fp literal pool displacement out of range")
            })?;
            self.core.text.patch_i32(fixup.disp_offset, disp);
        }
        Ok(())
    }

    fn lower_prologue(&mut self) {
        for &reg in abi::callee_saved_gp_regs() {
            enc::push(&mut self.core.text, reg);
        }
        if STACK_PADDING > 0 {
            if STACK_PADDING <= 127 {
                enc::sub_rsp_imm8(&mut self.core.text, STACK_PADDING as u8);
            } else {
                enc::sub_rsp_imm32(&mut self.core.text, STACK_PADDING);
            }
        }
        // ABI-specific spills (e.g. XMM6..XMM15 on Win64).
        callconv::emit_prologue_extra(self);
        enc::mov_rr_64(&mut self.core.text, map_fixed_reg(MACHINE_CTX_REG), C_ARG0);
        enc::mov_rr_64(&mut self.core.text, map_fixed_reg(MACHINE_FP_REG), C_ARG1);
        // Load mem0 base/size from ctx
        enc::load_64(
            &mut self.core.text,
            map_fixed_reg(MACHINE_MEM0_BASE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_BASE as i32,
        );
        enc::load_64(
            &mut self.core.text,
            map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_SIZE as i32,
        );
    }

    fn lower_epilogue(&mut self) {
        // ABI-specific restores (mirror of `callconv::emit_prologue_extra`).
        callconv::emit_epilogue_extra(self);
        if STACK_PADDING > 0 {
            if STACK_PADDING <= 127 {
                enc::add_rsp_imm8(&mut self.core.text, STACK_PADDING as u8);
            } else {
                enc::add_rsp_imm32(&mut self.core.text, STACK_PADDING);
            }
        }
        for &reg in abi::callee_saved_gp_regs().iter().rev() {
            enc::pop(&mut self.core.text, reg);
        }
        enc::ret(&mut self.core.text);
    }

    /// Public-entry caller stub. Pushes a root call record onto the host
    /// stack — two 8-byte slots, low = caller_result_base, high = caller_fp,
    /// both = `fp_reg` (= MACHINE_FP_REG) for the root call — then `call`s
    /// the internal entry. After the body's unified Return rets, control
    /// falls through to `lower_epilogue`.
    fn lower_root_caller_stub(&mut self) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        // push caller_fp (→ higher slot after the next push)
        enc::push(&mut self.core.text, fp_reg);
        // push caller_result_base (→ lower slot)
        enc::push(&mut self.core.text, fp_reg);
        self.lower_root_param_lanes_from_frame();
        // call internal_entry_label (patched by patch_fixups)
        let internal_entry_label = self.core.internal_entry_label;
        self.emit_call_rel32(internal_entry_label);
    }

    /// Body entry prelude. On x86_64, `call` already pushed the return
    /// address onto the host stack when the caller stub entered, leaving SP
    /// misaligned by 8 (relative to the 16-byte requirement). The body
    /// prelude subtracts 8 more to restore 16-byte alignment so the body
    /// can make nested C helper / preserved-helper calls without tripping
    /// the ABI's stack alignment check.
    ///
    /// The body's unified Return and `body_local_error_label` both `add
    /// rsp, 8` before popping the call record, mirroring this shim.
    fn lower_body_prelude(&mut self) {
        enc::sub_rsp_imm8(&mut self.core.text, 8);
    }

    /// Body-local error tail (`body_local_error_label`). Reached from trap
    /// stubs and post-call status checks. Undoes the body prelude, loads
    /// the caller's `fp_reg` back from the call record, and `ret 16`s —
    /// releasing the 16-byte call record sitting above the return
    /// address. `C_RET0` is untouched, preserving the error code set by
    /// the trap stub or inherited from a trapped descendant's call.
    ///
    /// Stack layout at entry:
    ///
    /// ```text
    ///   [rsp +  0] = alignment shim (body prelude)
    ///   [rsp +  8] = return address
    ///   [rsp + 16] = caller_result_base (unused in the error path)
    ///   [rsp + 24] = caller_fp
    /// ```
    fn lower_body_local_error_tail(&mut self) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        // Undo body prelude alignment shim.
        enc::add_rsp_imm8(&mut self.core.text, 8);
        // Load caller_fp from [rsp+16] into fp_reg.
        enc::load_64(&mut self.core.text, fp_reg, X86Reg::RSP, 16);
        // ret 16 — pops return address, then releases the 16-byte call
        // record. C_RET0 untouched — preserves the error code.
        enc::ret_imm16(&mut self.core.text, 16);
    }

    #[cfg(sf_has_guard_pages)]
    fn lower_stack_probe(&mut self, addr: MachineAddr) -> Result<(), WasmError> {
        self.emit_stack_probe(addr)
    }

    fn lower_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        self.lower_inst_dispatch(inst)
    }

    fn emit_inst_at(&mut self, inst: &'a MachineInst, index: usize) -> Result<(), WasmError> {
        self.core.current_op_index = Some(index);
        self.lower_inst(inst)?;
        self.gp_scratch.assert_all_free();
        self.fp_scratch.assert_all_free();
        Ok(())
    }

    fn end_block(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core.current_op_index = None;
        let result = self.lower_terminator(term, fallthrough);
        self.gp_scratch.assert_all_free();
        self.fp_scratch.assert_all_free();
        self.core.current_block = None;
        result
    }

    fn lower_terminator(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.lower_terminator_dispatch(term, fallthrough)
    }

    fn lower_trap(&mut self, kind: MachineTrapKind) {
        self.lower_trap_dispatch(kind);
    }

    fn lower_unconditional_branch(&mut self, label: usize) {
        self.emit_jmp(label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target = self
                .core
                .labels
                .get(fixup.label)
                .and_then(|value| *value)
                .ok_or_else(|| WasmError::internal("x86_64 branch target label is unresolved"))?;
            enc::patch_rel32(&mut self.core.text, fixup.rel32_offset, target);
        }
        Ok(())
    }

    fn alloc_gp_scratch(&mut self) -> u8 {
        self.gp_scratch.alloc()
    }
    fn free_gp_scratch(&mut self, id: u8) {
        self.gp_scratch.free_index(id)
    }
    fn alloc_fp_scratch(&mut self) -> u8 {
        self.fp_scratch.alloc()
    }
    fn free_fp_scratch(&mut self, id: u8) {
        self.fp_scratch.free_index(id)
    }

    fn lower_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        self.lower_source_move_dispatch(dst, src)
    }

    fn lower_gp_cycle_break(
        &mut self,
        dst: MachineReg,
        src: MachineReg,
        scratch_id: u8,
    ) -> Result<(), WasmError> {
        let temp = self.gp_scratch.reg(scratch_id);
        let dst_gp = self.map_gp_reg(dst)?;
        let src_gp = self.map_gp_reg(src)?;
        enc::mov_rr_64(&mut self.core.text, temp, dst_gp);
        enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
        Ok(())
    }

    fn lower_fp_cycle_break(
        &mut self,
        dst: MachineBlockParam,
        src: MachineReg,
        _float_width: Option<MachineFloatWidth>,
        scratch_id: u8,
    ) -> Result<(), WasmError> {
        let temp = self.fp_scratch.reg(scratch_id);
        let dst_fp = self.map_fp_reg(dst.reg)? as u8;
        let width = dst.ty.float_width().expect("FP param width");
        match width {
            MachineFloatWidth::F32 => enc::movaps_rr(&mut self.core.text, temp as u8, dst_fp),
            MachineFloatWidth::F64 => enc::movaps_rr(&mut self.core.text, temp as u8, dst_fp),
        };
        let src_fp = self.map_fp_reg(src)? as u8;
        match width {
            MachineFloatWidth::F32 => {
                enc::movaps_rr(&mut self.core.text, dst_fp, src_fp);
                self.core
                    .set_fp_reg_width(dst.reg, MachineFloatWidth::F32)?;
            }
            MachineFloatWidth::F64 => {
                enc::movaps_rr(&mut self.core.text, dst_fp, src_fp);
                self.core
                    .set_fp_reg_width(dst.reg, MachineFloatWidth::F64)?;
            }
        }
        Ok(())
    }

    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
        const INT3: u8 = 0xCC;
        for _ in 0..bytes {
            buf.emit_bytes(&[INT3]);
        }
    }
}

impl<'a> crate::vm::jit::arch::shared_64::ModuleLinkBackend64<'a> for X86_64Backend<'a> {
    type CompiledEntry = CompiledX86_64Entry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::jit::arch::shared_64::EmittedFunction64,
    ) -> Self::CompiledEntry {
        let entry = unsafe {
            buf.fn_ptr::<crate::vm::jit::runtime::code::NativeRootEntry>(emitted.text_offset)
        };
        CompiledX86_64Entry {
            entry,
            text_len: emitted.text_len,
            #[cfg(sf_ir_dump)]
            debug_regions: emitted.debug_regions.clone(),
        }
    }
}

// ── Inherent helper methods ──────────────────────────────────────────────────

impl<'a> X86_64Backend<'a> {
    #[inline]
    pub(super) fn emit_gp_move_width(&mut self, width: MachineIntWidth, dst: X86Reg, src: X86Reg) {
        match width {
            MachineIntWidth::I32 => enc::mov_rr_32(&mut self.core.text, dst, src),
            MachineIntWidth::I64 => enc::mov_rr_64(&mut self.core.text, dst, src),
        }
    }

    #[inline]
    pub(super) fn emit_gp_move_ty(
        &mut self,
        ty: MachineStorageType,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        match ty {
            // GpWord carries both i32 values and references. Preserve the full
            // 64-bit carrier here so ref null sentinels survive plain moves.
            MachineStorageType::GpWord => enc::mov_rr_64(&mut self.core.text, dst, src),
            MachineStorageType::GpI64 => enc::mov_rr_64(&mut self.core.text, dst, src),
            MachineStorageType::Fp32 | MachineStorageType::Fp64 | MachineStorageType::V128 => {
                return Err(WasmError::internal(
                    "x86_64 GP move requested for FP storage type".into(),
                ))
            }
        }
        Ok(())
    }

    #[inline]
    pub(super) fn emit_gp_cmov_ty(
        &mut self,
        ty: MachineStorageType,
        cc: Cc,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        match ty {
            // GpWord covers references as well as i32 carriers; keep the full
            // 64-bit payload when selecting between them.
            MachineStorageType::GpWord => enc::cmovcc_rr_64(&mut self.core.text, cc, dst, src),
            MachineStorageType::GpI64 => enc::cmovcc_rr_64(&mut self.core.text, cc, dst, src),
            MachineStorageType::Fp32 | MachineStorageType::Fp64 | MachineStorageType::V128 => {
                return Err(WasmError::internal(
                    "x86_64 GP cmov requested for FP storage type".into(),
                ))
            }
        }
        Ok(())
    }

    // ── Branch fixup helpers ─────────────────────────────────────────────

    /// Emit JMP rel32 with a fixup to be patched later.
    pub(super) fn emit_jmp(&mut self, label: usize) {
        let rel32_offset = enc::jmp_rel32(&mut self.core.text);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    /// Emit Jcc rel32 with a fixup to be patched later.
    pub(super) fn emit_jcc(&mut self, cc: Cc, label: usize) {
        let rel32_offset = enc::jcc_rel32(&mut self.core.text, cc);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    /// Emit CALL rel32 with a fixup to be patched later.
    pub(super) fn emit_call_rel32(&mut self, label: usize) {
        let rel32_offset = enc::call_rel32(&mut self.core.text);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    // ── Register mapping ─────────────────────────────────────────────────

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<X86Reg, WasmError> {
        crate::vm::jit::arch::common::helpers::validate_gp_reg(self, reg)?;
        map_reg(reg)
    }

    pub(super) fn map_fp_reg(&self, reg: MachineReg) -> Result<u32, WasmError> {
        let index = self.core.fp_reg_index(reg)?;
        fp_machine_reg(index).ok_or_else(|| {
            WasmError::invalid("x86_64 backend has no physical FP mapping for machine reg")
        })
    }

    // ── Value materialization ────────────────────────────────────────────

    pub(super) fn materialize_u64(&mut self, dst: X86Reg, value: u64) {
        if value == 0 {
            enc::xor_rr_32(&mut self.core.text, dst, dst);
        } else if value <= u32::MAX as u64 {
            enc::mov_ri_32(&mut self.core.text, dst, value as u32);
        } else {
            enc::mov_ri_64(&mut self.core.text, dst, value);
        }
    }

    pub(super) fn materialize_value(
        &mut self,
        scratch: X86Reg,
        value: MachineValue,
    ) -> Result<X86Reg, WasmError> {
        match value {
            MachineValue::Reg(reg) if self.core.is_fp_reg(reg) => {
                let src_fp = self.map_fp_reg(reg)?;
                match self.core.fp_reg_width(reg)? {
                    MachineFloatWidth::F32 => {
                        enc::movd_r32_xmm(&mut self.core.text, scratch, src_fp as u8);
                    }
                    MachineFloatWidth::F64 => {
                        enc::movq_r64_xmm(&mut self.core.text, scratch, src_fp as u8);
                    }
                };
                Ok(scratch)
            }
            MachineValue::Reg(reg) => self.map_gp_reg(reg),
            MachineValue::Imm64(value) => {
                self.materialize_u64(scratch, value);
                Ok(scratch)
            }
            MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
                "x86_64 cannot materialize reserved cache register",
            )),
        }
    }

    pub(super) fn prepare_float_operand(
        &mut self,
        width: MachineFloatWidth,
        value: MachineValue,
        gp_scratch: X86Reg,
        fp_scratch: u32,
    ) -> Result<u32, WasmError> {
        if let MachineValue::Reg(reg) = value {
            if self.core.is_fp_reg(reg) {
                return Ok(self.map_fp_reg(reg)?);
            }
        }
        if let MachineValue::Imm64(bits) = value {
            // Load FP immediates from the function's literal pool: one
            // FP-domain load the core can hoist, instead of a GP
            // materialization plus a domain-crossing move on the consumer's
            // critical path.
            let imm = match width {
                MachineFloatWidth::F32 => u64::from(bits as u32),
                MachineFloatWidth::F64 => bits,
            };
            if imm == 0 {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::xorps(&mut self.core.text, fp_scratch as u8, fp_scratch as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::xorpd(&mut self.core.text, fp_scratch as u8, fp_scratch as u8)
                    }
                };
                return Ok(fp_scratch);
            }
            let literal_index = self.intern_fp_literal(imm);
            let disp_offset = match width {
                MachineFloatWidth::F32 => enc::movss_rip(&mut self.core.text, fp_scratch as u8),
                MachineFloatWidth::F64 => enc::movsd_rip(&mut self.core.text, fp_scratch as u8),
            };
            self.fp_literal_fixups.push(FpLiteralFixup {
                disp_offset,
                literal_index,
            });
            return Ok(fp_scratch);
        }
        let gp = self.materialize_value(gp_scratch, value)?;
        match width {
            MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.core.text, fp_scratch as u8, gp),
            MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.core.text, fp_scratch as u8, gp),
        };
        Ok(fp_scratch)
    }

    // ── Source move dispatch ─────────────────────────────────────────────

    fn lower_source_move_dispatch(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        match src {
            ParallelSource::Reg {
                reg: src_reg,
                float_width: src_float_width,
            } => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    if self.core.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, src_fp);
                                self.core
                                    .set_fp_reg_width(dst.reg, MachineFloatWidth::F32)?;
                            }
                            MachineFloatWidth::F64 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, src_fp);
                                self.core
                                    .set_fp_reg_width(dst.reg, MachineFloatWidth::F64)?;
                            }
                        }
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp);
                                self.core
                                    .set_fp_reg_width(dst.reg, MachineFloatWidth::F32)?;
                            }
                            MachineFloatWidth::F64 => {
                                enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp);
                                self.core
                                    .set_fp_reg_width(dst.reg, MachineFloatWidth::F64)?;
                            }
                        }
                    }
                } else {
                    let dst_gp = self.map_gp_reg(dst.reg)?;
                    if self.core.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match src_float_width.ok_or_else(|| {
                            WasmError::internal(
                                "x86_64 edge move is missing float-width metadata for an FP source reg",
                            )
                        })? {
                            MachineFloatWidth::F32 => {
                                enc::movd_r32_xmm(&mut self.core.text, dst_gp, src_fp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movq_r64_xmm(&mut self.core.text, dst_gp, src_fp)
                            }
                        }
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
                    }
                }
            }
            ParallelSource::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "x86_64 received non-identity reserved cache edge move into from",
                ));
            }
            ParallelSource::Imm(value) => {
                if dst.ty.is_fp() {
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    match dst.ty {
                        MachineStorageType::Fp32 => {
                            let scratch = self.gp_scratch.scoped_alloc().detach();
                            self.materialize_u64(*scratch, value);
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch);
                            self.core
                                .set_fp_reg_width(dst.reg, MachineFloatWidth::F32)?;
                        }
                        MachineStorageType::Fp64 => {
                            let scratch = self.gp_scratch.scoped_alloc().detach();
                            self.materialize_u64(*scratch, value);
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, *scratch);
                            self.core
                                .set_fp_reg_width(dst.reg, MachineFloatWidth::F64)?;
                        }
                        MachineStorageType::V128 => {
                            return Err(WasmError::internal(
                                "x86_64 edge move cannot materialize a v128 immediate",
                            ))
                        }
                        MachineStorageType::GpWord | MachineStorageType::GpI64 => unreachable!(),
                    }
                } else {
                    let dst_gp = self.map_gp_reg(dst.reg)?;
                    self.materialize_u64(dst_gp, value);
                }
            }
            ParallelSource::GpTemp(id) => {
                let temp = self.gp_scratch.reg(id);
                let dst_gp = self.map_gp_reg(dst.reg)?;
                enc::mov_rr_64(&mut self.core.text, dst_gp, temp);
            }
            ParallelSource::FpTemp(id, width) => {
                let temp = self.fp_scratch.reg(id);
                let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                match width {
                    MachineFloatWidth::F32 => {
                        enc::movaps_rr(&mut self.core.text, dst_fp, temp as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::movaps_rr(&mut self.core.text, dst_fp, temp as u8)
                    }
                };
                self.core.set_fp_reg_width(dst.reg, width)?;
            }
        }
        Ok(())
    }
}
