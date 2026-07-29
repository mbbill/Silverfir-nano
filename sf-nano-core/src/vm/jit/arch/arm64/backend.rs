//! ARM64 backend: struct definitions and `ArchBackend` trait implementation.
//!
//! This file is the bridge between the common pipeline and the arm64-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs` as inherent methods.

use crate::collections;

#[cfg(sf_has_guard_pages)]
use crate::vm::jit::machine::machine_ir::MachineAddr;
use crate::{
    error::WasmError,
    vm::{
        jit::machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId,
            MachineInst, MachineInstKind, MachineReg, MachineReturnAbi, MachineTerminator,
            MachineTrapKind, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG,
            MACHINE_MEM0_SIZE_REG,
        },
        jit::runtime::{code::NativeRootEntry, code_buf::CodeBuffer, context::ctx_offset},
    },
};

use super::abi::{max_fp_machine_regs, max_total_machine_regs};
use super::{
    abi, enc,
    reg::{Arm64FpReg, Arm64Reg},
};
#[cfg(sf_ir_dump)]
use crate::vm::jit::arch::common::types::DebugRegion;
use crate::vm::jit::arch::common::{
    backend::ArchBackend,
    core::{CompilerCore, FunctionBody},
    scratch_pool::ScratchPool,
    types::ParallelSource,
};

unsafe extern "C" {
    fn memset(dest: *mut core::ffi::c_void, value: i32, len: usize) -> *mut core::ffi::c_void;
    fn memmove(
        dest: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        len: usize,
    ) -> *mut core::ffi::c_void;
}

// ── Frame layout constants ───────────────────────────────────────────────────

const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;
#[cfg(sf_has_simd)]
const PRESERVED_DYNAMIC_FP_SLOT_BYTES: u32 = 16;
#[cfg(not(sf_has_simd))]
const PRESERVED_DYNAMIC_FP_SLOT_BYTES: u32 = STACK_SLOT_BYTES;
const CALLEE_SAVED_GP_FRAME_SIZE: u32 =
    abi::callee_saved_gp_pair_count() as u32 * (2 * STACK_SLOT_BYTES);
const CALLEE_SAVED_FP_FRAME_OFFSET: u32 = CALLEE_SAVED_GP_FRAME_SIZE;
// Match the preserved-helper ABI: scalar builds only need the low 64-bit D
// payload of callee-saved FP regs, while SIMD builds must preserve the full
// 128-bit Q contents because the FP bank also carries v128 values.
#[cfg(sf_has_simd)]
const CALLEE_SAVED_FP_SLOT_BYTES: u32 = 16;
#[cfg(not(sf_has_simd))]
const CALLEE_SAVED_FP_SLOT_BYTES: u32 = STACK_SLOT_BYTES;
const CALLEE_SAVED_FP_FRAME_SIZE: u32 =
    abi::callee_saved_fp_count() as u32 * CALLEE_SAVED_FP_SLOT_BYTES;
const CALLEE_SAVED_FRAME_SIZE: u32 = {
    let total = CALLEE_SAVED_FP_FRAME_OFFSET + CALLEE_SAVED_FP_FRAME_SIZE;
    total.div_ceil(abi::stack_alignment_bytes()) * abi::stack_alignment_bytes()
};

#[cfg(not(sf_has_simd))]
const fn stack_u64_slot(offset_bytes: u32) -> u32 {
    offset_bytes / STACK_SLOT_BYTES
}

const fn stack_pair_imm(offset_bytes: u32) -> i32 {
    (offset_bytes / STACK_SLOT_BYTES) as i32
}

// ── Branch fixup types ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BranchFixupKind {
    B,
    BCond(enc::Cond),
    Cbz(Arm64Reg),
    Cbnz(Arm64Reg),
    /// `bl <label>` — branch with link, populates LR.
    Bl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BranchFixup {
    pub inst_offset: usize,
    pub label: usize,
    pub kind: BranchFixupKind,
}

// ── Compiled entry ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct CompiledArm64Entry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    #[cfg(sf_ir_dump)]
    pub debug_regions: collections::Vec<DebugRegion>,
}

// ── Arm64Backend ─────────────────────────────────────────────────────────────

/// Deferred direct-call patch site. The hot path is a single `bl` patched at
/// module link time. A local veneer is emitted in the end-of-body pool as a
/// correctness fallback for modules whose callee lands outside arm64's ±128MiB
/// branch range.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingDirectCall {
    pub inst_offset: usize,
    pub fallback_scratch_reg: Arm64Reg,
    pub callee: MachineFuncId,
    pub link: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BulkBoundsFact {
    pub offset: MachineValue,
    pub len: MachineValue,
    pub op_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct DirectCallFallbackVeneer {
    callee: MachineFuncId,
    scratch_reg: Arm64Reg,
    veneer_offset: usize,
    literal_offset: usize,
}

struct Arm64BodyAnalysis {
    helper_gp_candidates: collections::Vec<(MachineReg, Arm64Reg)>,
    helper_fp_candidates: collections::Vec<(MachineReg, Arm64FpReg)>,
    bulk_memset_target: Option<Arm64Reg>,
    bulk_memmove_target: Option<Arm64Reg>,
    has_body_host_frame: bool,
}

#[derive(Debug)]
pub(crate) struct Arm64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: collections::Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<Arm64Reg, 2>,
    pub(super) fp_scratch: ScratchPool<Arm64FpReg, 2>,
    /// Caller-clobbered dynamic lanes that this function can actually use.
    /// Preserved helper calls only spill these lanes, instead of the entire
    /// architectural volatile bank.
    pub(super) helper_gp_candidates: collections::Vec<(MachineReg, Arm64Reg)>,
    pub(super) helper_fp_candidates: collections::Vec<(MachineReg, Arm64FpReg)>,
    pub(super) helper_saved_gp: collections::Vec<Arm64Reg>,
    pub(super) helper_saved_fp: collections::Vec<Arm64FpReg>,
    /// Deferred direct-call veneers. Flushed by
    /// `lower_function_literal_pool` (which the pipeline calls between
    /// edge stubs and the body-local error tail).
    pub(super) pending_direct_calls: collections::Vec<PendingDirectCall>,
    /// 1-slot peephole lookahead buffer. Holds the previously seen op so
    /// `emit_inst_at` can attempt a 2-op fusion (zero-store pair, FP
    /// store/load pair) when the next op arrives. Drained on `end_block`.
    /// `usize` is the block-local op index at the time it was buffered.
    pending_op: Option<(&'a MachineInst, usize)>,
    /// Select-condition flags fusion: `Some((reg, cond))` means the NZCV
    /// flags currently encode `reg != 0` as `cond`, set by the immediately
    /// preceding And (emitted as `ands`) or comparison. Consumed by a Select
    /// or terminator on exactly that register.
    pub(super) select_flags: Option<(MachineReg, enc::Cond)>,
    /// Ring of recently dispatched `Store` ops, as `(base, offset, bytes,
    /// seq)`. Consulted by the FP load-pair fusion: an `ldp` that partially
    /// overlaps a recent store defeats store-to-load forwarding on Apple
    /// Silicon (~2x load-latency penalty), so pairing is suppressed and the
    /// exact-width `ldr`s forward cleanly. Deliberately kept across block
    /// boundaries: the hazard follows emission (= fallthrough) order, and a
    /// false positive only costs one extra `ldr`.
    pub(super) recent_stores: [(MachineReg, i32, u8, u32); RECENT_STORE_SLOTS],
    pub(super) recent_store_len: usize,
    pub(super) dispatch_seq: u32,
    /// The two most recent mem0 bulk ranges checked in this basic block.
    /// Keeping two facts lets an adjacent `memory.fill` + `memory.copy`
    /// share the common `[0, len)` source range.
    pub(super) bulk_bounds_facts: [Option<BulkBoundsFact>; 2],
    pub(super) bulk_bounds_next: usize,
    /// Invariant libc entry addresses cached in otherwise-unused
    /// callee-saved registers for bulk-memory-heavy bodies.
    pub(super) bulk_memset_target: Option<Arm64Reg>,
    pub(super) bulk_memmove_target: Option<Arm64Reg>,
    /// Cached ARM64 lowering effect. This is derived with the other
    /// whole-body register facts when the backend is constructed.
    has_body_host_frame: bool,
}

/// Capacity of the recent-store ring.
pub(super) const RECENT_STORE_SLOTS: usize = 8;
/// A recorded store older than this many dispatched ops has retired well
/// past the store buffer and no longer inhibits load pairing.
pub(super) const RECENT_STORE_MAX_AGE: u32 = 24;

impl<'a> Arm64Backend<'a> {
    fn analyze_body(core: &CompilerCore<'a>) -> Arm64BodyAnalysis {
        let FunctionBody::Mir(function) = core.body else {
            return Arm64BodyAnalysis {
                helper_gp_candidates: collections::Vec::new(),
                helper_fp_candidates: collections::Vec::new(),
                bulk_memset_target: None,
                bulk_memmove_target: None,
                has_body_host_frame: true,
            };
        };
        let config = core.compiled.backend();
        let mut defined = collections::vec![false; config.total_reg_count() as usize];
        let mut needs_memset = false;
        let mut needs_memmove = false;
        let mut used_phys = [false; 32];
        let mut has_body_host_frame = !function.preserved_clobbers.is_empty();
        for block in &function.program.blocks {
            for param in &block.params {
                if let Some(slot) = defined.get_mut(param.reg.0 as usize) {
                    *slot = true;
                }
                if let Ok(reg) = abi::map_reg(param.reg) {
                    used_phys[reg.index() as usize] = true;
                    has_body_host_frame |= !param.ty.is_fp() && reg == abi::host_lr_reg();
                }
            }
            for inst in &block.ops {
                match inst.kind {
                    MachineInstKind::MemoryFill { mem_idx: 0, .. } => needs_memset = true,
                    MachineInstKind::MemoryCopy {
                        dst_mem: 0,
                        src_mem: 0,
                        ..
                    } => needs_memmove = true,
                    _ => {}
                }
                has_body_host_frame |= arm64_inst_lowers_to_inline_native_call(&inst.kind);
                inst.kind.for_each_defined_reg(|machine| {
                    if let Some(slot) = defined.get_mut(machine.0 as usize) {
                        *slot = true;
                    }
                    if let Ok(reg) = abi::map_reg(machine) {
                        used_phys[reg.index() as usize] = true;
                        has_body_host_frame |= reg == abi::host_lr_reg();
                    }
                });
                crate::vm::jit::machine::peephole::helpers::visit_source_values(
                    &inst.kind,
                    |value| {
                        if let MachineValue::Reg(machine) = *value {
                            if let Ok(reg) = abi::map_reg(machine) {
                                used_phys[reg.index() as usize] = true;
                            }
                        }
                    },
                );
            }
            crate::vm::jit::machine::peephole::helpers::visit_terminator_source_regs(
                &block.terminator,
                |machine| {
                    if let Ok(reg) = abi::map_reg(machine) {
                        used_phys[reg.index() as usize] = true;
                    }
                },
            );
            has_body_host_frame |= matches!(block.terminator, MachineTerminator::Call { .. });
        }

        // x29 is already preserved with LR by every body that calls a host
        // helper. x28..x23 are the remaining dynamic callee-saved lanes. Only
        // claim lanes the MachineIR body never defines or binds.
        let mut available = [29_u8, 28, 27, 26, 25, 24, 23]
            .into_iter()
            .filter(|index| !used_phys[*index as usize])
            .map(Arm64Reg::from_raw);
        // Give the no-extra-save x29 lane to memmove, normally the more
        // expensive and frequent bulk helper.
        let memmove_target = needs_memmove.then(|| available.next()).flatten();
        let memset_target = needs_memset.then(|| available.next()).flatten();

        let mut gp = collections::Vec::new();
        let mut fp = collections::Vec::new();
        for (index, is_defined) in defined.into_iter().enumerate() {
            if !is_defined {
                continue;
            }
            let reg = MachineReg(index as u16);
            if core.is_fp_reg(reg) {
                let Some(fp_index) = crate::vm::jit::machine::machine_ir::fp_reg_index(reg, config)
                else {
                    continue;
                };
                let Some(mapped) = abi::fp_machine_reg(fp_index) else {
                    continue;
                };
                if abi::fp_dynamic_caller_saved_regs().contains(&mapped) {
                    fp.push((reg, mapped));
                }
            } else if let Ok(mapped) = abi::map_reg(reg) {
                if abi::gp_dynamic_caller_saved_regs().contains(&mapped) {
                    gp.push((reg, mapped));
                }
            }
        }
        Arm64BodyAnalysis {
            helper_gp_candidates: gp,
            helper_fp_candidates: fp,
            bulk_memset_target: memset_target,
            bulk_memmove_target: memmove_target,
            has_body_host_frame,
        }
    }

    /// Record a dispatched `Store` in the recent-store ring.
    fn record_store(&mut self, base: MachineReg, offset: i32, bytes: u8) {
        let slot = self.recent_store_len % RECENT_STORE_SLOTS;
        self.recent_stores[slot] = (base, offset, bytes, self.dispatch_seq);
        self.recent_store_len += 1;
    }

    /// True if a recent store partially overlaps `[offset, offset + bytes)`
    /// at `base`. An exact-width match (same range) forwards fine and does
    /// not count; only a store covering part of the range defeats forwarding.
    pub(super) fn recent_store_partially_overlaps(
        &self,
        base: MachineReg,
        offset: i32,
        bytes: u8,
    ) -> bool {
        let lo = i64::from(offset);
        let hi = lo + i64::from(bytes);
        let tracked = self.recent_store_len.min(RECENT_STORE_SLOTS);
        self.recent_stores[..tracked]
            .iter()
            .any(|&(b, off, w, seq)| {
                b == base && self.dispatch_seq.wrapping_sub(seq) <= RECENT_STORE_MAX_AGE && {
                    let s_lo = i64::from(off);
                    let s_hi = s_lo + i64::from(w);
                    s_lo < hi && lo < s_hi && !(s_lo == lo && s_hi == hi)
                }
            })
    }

    pub(super) fn has_body_host_frame(&self) -> bool {
        self.has_body_host_frame
    }

    pub(super) fn emit_body_returning_blr(&mut self, target: Arm64Reg) -> Result<(), WasmError> {
        if !self.has_body_host_frame() {
            return Err(WasmError::internal(
                "arm64 body-returning native call requires a body link save",
            ));
        }
        self.core.text.emit_u32(enc::blr(target));
        Ok(())
    }

    fn preserved_dynamic_body_regs(
        &self,
    ) -> (collections::Vec<Arm64Reg>, collections::Vec<Arm64FpReg>) {
        let mut gp_regs = collections::Vec::new();
        let mut fp_regs = collections::Vec::new();
        for reg in self.core.preserved_clobbers() {
            if self.core.is_fp_reg(*reg) {
                fp_regs.push(
                    self.map_fp_reg(*reg)
                        .expect("validated arm64 preserved FP clobber must map"),
                );
            } else {
                let mapped =
                    abi::map_reg(*reg).expect("validated arm64 preserved GP clobber must map");
                // x29 is already saved with LR by `lower_body_prelude` and
                // restored by every body return path. Do not allocate a second
                // dynamic-save slot for the same physical register.
                if mapped != abi::host_fp_reg() {
                    gp_regs.push(mapped);
                }
            }
        }
        for target in [self.bulk_memset_target, self.bulk_memmove_target]
            .into_iter()
            .flatten()
        {
            // x29 is already saved beside LR. Other cached target lanes must
            // join the body's lazy preserved save exactly once.
            if target != abi::host_fp_reg() && !gp_regs.contains(&target) {
                gp_regs.push(target);
            }
        }
        (gp_regs, fp_regs)
    }

    fn lower_root_result_copy_to_public_slots(&mut self) {
        let Some(runtime) = self.core.current_runtime() else {
            return;
        };
        match &runtime.return_abi {
            MachineReturnAbi::ScalarGp { .. } => {
                self.emit_root_frame_store_slot(
                    abi::W2W_GP_RET0,
                    abi::map_fixed_reg(MACHINE_FP_REG),
                    0,
                );
                return;
            }
            MachineReturnAbi::ScalarFp { ty } => {
                let Some(width) = ty.float_width() else {
                    return;
                };
                self.emit_root_frame_store_fp_slot(
                    abi::fp_zero_reg(),
                    abi::map_fixed_reg(MACHINE_FP_REG),
                    0,
                    width,
                );
                return;
            }
            MachineReturnAbi::ScalarGpPair => {
                return;
            }
            MachineReturnAbi::None | MachineReturnAbi::FrameFallback { .. } => {}
        }
        let Some(results) = runtime.return_results else {
            return;
        };
        if results.slots == 0 || results.base_slot == 0 {
            return;
        }

        let fp_reg = abi::map_fixed_reg(MACHINE_FP_REG);
        let value_idx = self.gp_scratch.alloc();
        let value = self.gp_scratch.reg(value_idx);
        for index in 0..u32::from(results.slots) {
            let src_slot = u32::from(results.base_slot).saturating_add(index);
            self.emit_root_frame_load_slot(value, fp_reg, src_slot);
            self.emit_root_frame_store_slot(value, fp_reg, index);
        }
        self.gp_scratch.free_index(value_idx);
    }

    fn emit_root_frame_load_slot(&mut self, dst: Arm64Reg, base: Arm64Reg, slot: u32) {
        if slot < 4096 {
            self.core.text.emit_u32(enc::ldr_64(dst, base, slot));
            return;
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_u64(addr, u64::from(slot) * u64::from(STACK_SLOT_BYTES));
        self.core.text.emit_u32(enc::add_reg_64(addr, base, addr));
        self.core.text.emit_u32(enc::ldr_64(dst, addr, 0));
        self.gp_scratch.free_index(addr_idx);
    }

    fn emit_root_frame_store_slot(&mut self, src: Arm64Reg, base: Arm64Reg, slot: u32) {
        if slot < 4096 {
            self.core.text.emit_u32(enc::str_64(src, base, slot));
            return;
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_u64(addr, u64::from(slot) * u64::from(STACK_SLOT_BYTES));
        self.core.text.emit_u32(enc::add_reg_64(addr, base, addr));
        self.core.text.emit_u32(enc::str_64(src, addr, 0));
        self.gp_scratch.free_index(addr_idx);
    }

    fn emit_root_frame_store_fp_slot(
        &mut self,
        src: Arm64FpReg,
        base: Arm64Reg,
        slot: u32,
        width: MachineFloatWidth,
    ) {
        match width {
            MachineFloatWidth::F32 => {
                let scaled = slot.saturating_mul(2);
                if scaled < 4096 {
                    self.core.text.emit_u32(enc::str_s(src, base, scaled));
                    return;
                }
            }
            MachineFloatWidth::F64 => {
                if slot < 4096 {
                    self.core.text.emit_u32(enc::str_d(src, base, slot));
                    return;
                }
            }
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_u64(addr, u64::from(slot) * u64::from(STACK_SLOT_BYTES));
        self.core.text.emit_u32(enc::add_reg_64(addr, base, addr));
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::str_s(src, addr, 0),
            MachineFloatWidth::F64 => enc::str_d(src, addr, 0),
        });
        self.gp_scratch.free_index(addr_idx);
    }

    fn lower_preserved_dynamic_body_save(&mut self) {
        let (gp_regs, fp_regs) = self.preserved_dynamic_body_regs();
        let gp_bytes = gp_regs.len() as u32 * STACK_SLOT_BYTES;
        let fp_offset =
            gp_bytes.div_ceil(PRESERVED_DYNAMIC_FP_SLOT_BYTES) * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let used = fp_offset + fp_regs.len() as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let total = used.div_ceil(abi::stack_alignment_bytes()) * abi::stack_alignment_bytes();
        if total == 0 {
            return;
        }
        self.core
            .text
            .emit_u32(enc::sub_imm_64(abi::stack_reg(), abi::stack_reg(), total));
        let gp_pairs = gp_regs.len() / 2;
        for pair in 0..gp_pairs {
            let index = pair * 2;
            self.core.text.emit_u32(enc::stp_64(
                gp_regs[index],
                gp_regs[index + 1],
                abi::stack_reg(),
                index as i32,
            ));
        }
        if gp_regs.len() & 1 != 0 {
            let index = gp_regs.len() - 1;
            self.core
                .text
                .emit_u32(enc::str_64(gp_regs[index], abi::stack_reg(), index as u32));
        }
        #[cfg(not(sf_has_simd))]
        for pair in 0..fp_regs.len() / 2 {
            let index = pair * 2;
            let byte_offset = fp_offset + index as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
            self.core.text.emit_u32(enc::stp_d(
                fp_regs[index],
                fp_regs[index + 1],
                abi::stack_reg(),
                (byte_offset / STACK_SLOT_BYTES) as i32,
            ));
        }
        for (index, reg) in fp_regs.iter().copied().enumerate() {
            let byte_offset = fp_offset + index as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::str_q(reg, abi::stack_reg(), byte_offset / 16));
            #[cfg(not(sf_has_simd))]
            if fp_regs.len() & 1 != 0 && index + 1 == fp_regs.len() {
                self.core.text.emit_u32(enc::str_d(
                    reg,
                    abi::stack_reg(),
                    byte_offset / STACK_SLOT_BYTES,
                ));
            }
        }
    }

    pub(super) fn lower_preserved_dynamic_body_restore(&mut self) {
        let (gp_regs, fp_regs) = self.preserved_dynamic_body_regs();
        let gp_bytes = gp_regs.len() as u32 * STACK_SLOT_BYTES;
        let fp_offset =
            gp_bytes.div_ceil(PRESERVED_DYNAMIC_FP_SLOT_BYTES) * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let used = fp_offset + fp_regs.len() as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let total = used.div_ceil(abi::stack_alignment_bytes()) * abi::stack_alignment_bytes();
        if total == 0 {
            return;
        }
        let gp_pairs = gp_regs.len() / 2;
        for pair in 0..gp_pairs {
            let index = pair * 2;
            self.core.text.emit_u32(enc::ldp_64(
                gp_regs[index],
                gp_regs[index + 1],
                abi::stack_reg(),
                index as i32,
            ));
        }
        if gp_regs.len() & 1 != 0 {
            let index = gp_regs.len() - 1;
            self.core
                .text
                .emit_u32(enc::ldr_64(gp_regs[index], abi::stack_reg(), index as u32));
        }
        #[cfg(not(sf_has_simd))]
        for pair in 0..fp_regs.len() / 2 {
            let index = pair * 2;
            let byte_offset = fp_offset + index as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
            self.core.text.emit_u32(enc::ldp_d(
                fp_regs[index],
                fp_regs[index + 1],
                abi::stack_reg(),
                (byte_offset / STACK_SLOT_BYTES) as i32,
            ));
        }
        for (index, reg) in fp_regs.iter().copied().enumerate() {
            let byte_offset = fp_offset + index as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::ldr_q(reg, abi::stack_reg(), byte_offset / 16));
            #[cfg(not(sf_has_simd))]
            if fp_regs.len() & 1 != 0 && index + 1 == fp_regs.len() {
                self.core.text.emit_u32(enc::ldr_d(
                    reg,
                    abi::stack_reg(),
                    byte_offset / STACK_SLOT_BYTES,
                ));
            }
        }
        self.core
            .text
            .emit_u32(enc::add_imm_64(abi::stack_reg(), abi::stack_reg(), total));
    }

    fn lower_direct_call_fallback_veneer(
        &mut self,
        scratch_reg: Arm64Reg,
    ) -> Result<(usize, usize), WasmError> {
        let veneer_offset = self.core.text.len();
        let ldr_offset = self.core.text.emit_u32(enc::ldr_lit_64(scratch_reg, 0));
        self.core.text.emit_u32(enc::br(scratch_reg));
        let literal_offset = self.core.text.emit_u64(0);
        let delta_bytes = literal_offset as isize - ldr_offset as isize;
        // ldr_lit_64 encodes a signed 19-bit instruction-word offset.
        // Word range: [-2^18, 2^18 - 1]. Byte range: ±1 MiB.
        // The literal pool always sits after the LDR here, but the bound is
        // checked symmetric because the encoding itself is signed.
        const LDR_LIT_BYTE_MAX: isize = (1 << 20) - 4;
        if delta_bytes & 0b11 != 0 {
            return Err(WasmError::internal(
                "arm64 deferred call literal is not 4-byte aligned",
            ));
        }
        if !(-LDR_LIT_BYTE_MAX..=LDR_LIT_BYTE_MAX).contains(&delta_bytes) {
            return Err(WasmError::internal(
                "arm64 deferred call literal pool is out of range",
            ));
        }
        let delta_words = (delta_bytes / 4) as i32;
        self.core
            .text
            .patch_u32(ldr_offset, enc::ldr_lit_64(scratch_reg, delta_words));
        Ok((veneer_offset, literal_offset))
    }
}

/// ARM64-only lowering effect: true when this MachineIR instruction emits a
/// native call and then resumes inside the same wasm body. Shared MachineIR
/// deliberately does not own this predicate because other backends may lower
/// the same instruction without clobbering a link register.
fn arm64_inst_lowers_to_inline_native_call(kind: &MachineInstKind) -> bool {
    match kind {
        MachineInstKind::CallRuntime(_)
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. }
        | MachineInstKind::EhAllocExnRef { .. }
        | MachineInstKind::MemoryGrow { .. }
        | MachineInstKind::MemoryFill { .. }
        | MachineInstKind::MemoryCopy { .. }
        | MachineInstKind::MemoryInit { .. }
        | MachineInstKind::DataDrop { .. }
        | MachineInstKind::TableGrow { .. }
        | MachineInstKind::TableFill { .. }
        | MachineInstKind::TableCopy { .. }
        | MachineInstKind::TableInit { .. }
        | MachineInstKind::ElemDrop { .. }
        | MachineInstKind::RefFunc { .. }
        | MachineInstKind::RefAsNonNull { .. }
        | MachineInstKind::RefAbsolutize { .. }
        | MachineInstKind::RefEq { .. }
        | MachineInstKind::RefI31 { .. }
        | MachineInstKind::I31GetS { .. }
        | MachineInstKind::I31GetU { .. }
        | MachineInstKind::AnyConvertExtern { .. }
        | MachineInstKind::ExternConvertAny { .. }
        | MachineInstKind::RefTest { .. }
        | MachineInstKind::RefCast { .. }
        | MachineInstKind::StructNew { .. }
        | MachineInstKind::StructNewDefault { .. }
        | MachineInstKind::StructGet { .. }
        | MachineInstKind::StructSet { .. }
        | MachineInstKind::ArrayNew { .. }
        | MachineInstKind::ArrayNewDefault { .. }
        | MachineInstKind::ArrayNewFixed { .. }
        | MachineInstKind::ArrayNewData { .. }
        | MachineInstKind::ArrayNewElem { .. }
        | MachineInstKind::ArrayGet { .. }
        | MachineInstKind::ArraySet { .. }
        | MachineInstKind::ArrayFill { .. }
        | MachineInstKind::ArrayCopy { .. }
        | MachineInstKind::ArrayInitData { .. }
        | MachineInstKind::ArrayInitElem { .. }
        | MachineInstKind::ArrayLen { .. } => true,
        #[cfg(sf_has_simd)]
        MachineInstKind::V128FromRaw { .. } | MachineInstKind::V128ToRaw { .. } => true,
        _ => false,
    }
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for Arm64Backend<'a> {
    const NAME: &'static str = "arm64";

    fn max_total_regs() -> usize {
        max_total_machine_regs()
    }
    fn max_fp_regs() -> usize {
        max_fp_machine_regs()
    }

    fn new(core: CompilerCore<'a>) -> Self {
        let analysis = Self::analyze_body(&core);
        Self {
            core,
            fixups: collections::Vec::new(),
            gp_scratch: abi::new_gp_scratch_pool(),
            fp_scratch: abi::new_fp_scratch_pool(),
            helper_saved_gp: collections::Vec::with_capacity(analysis.helper_gp_candidates.len()),
            helper_saved_fp: collections::Vec::with_capacity(analysis.helper_fp_candidates.len()),
            helper_gp_candidates: analysis.helper_gp_candidates,
            helper_fp_candidates: analysis.helper_fp_candidates,
            pending_direct_calls: collections::Vec::new(),
            pending_op: None,
            select_flags: None,
            recent_stores: [(MachineReg(0), 0, 0, 0); RECENT_STORE_SLOTS],
            recent_store_len: 0,
            dispatch_seq: 0,
            bulk_bounds_facts: [None; 2],
            bulk_bounds_next: 0,
            bulk_memset_target: analysis.bulk_memset_target,
            bulk_memmove_target: analysis.bulk_memmove_target,
            has_body_host_frame: analysis.has_body_host_frame,
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

    fn lower_prologue(&mut self) {
        // Allocate frame and save callee-saved registers.
        self.core.text.emit_u32(enc::sub_imm_64(
            abi::stack_reg(),
            abi::stack_reg(),
            CALLEE_SAVED_FRAME_SIZE,
        ));
        for (index, (lhs, rhs)) in abi::callee_saved_gp_pairs().iter().copied().enumerate() {
            self.core.text.emit_u32(enc::stp_64(
                lhs,
                rhs,
                abi::stack_reg(),
                stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
            ));
        }
        // Save callee-saved FP regs.
        let fp_regs = abi::callee_saved_fp_regs();
        let mut fp_idx = 0usize;
        while fp_idx < fp_regs.len() {
            let byte_off =
                CALLEE_SAVED_FP_FRAME_OFFSET + (fp_idx as u32) * CALLEE_SAVED_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::str_q(fp_regs[fp_idx], abi::stack_reg(), byte_off / 16));
            #[cfg(not(sf_has_simd))]
            self.core.text.emit_u32(enc::str_d(
                fp_regs[fp_idx],
                abi::stack_reg(),
                stack_u64_slot(byte_off),
            ));
            fp_idx += 1;
        }

        // Move entry arguments into pinned roles.
        let ctx = abi::map_fixed_reg(MACHINE_CTX_REG);
        let frame = abi::map_fixed_reg(MACHINE_FP_REG);
        self.core.text.emit_u32(enc::mov_reg_64(ctx, abi::C_ARG0));
        self.core.text.emit_u32(enc::mov_reg_64(frame, abi::C_ARG1));
        self.core.text.emit_u32(enc::ldr_64(
            abi::map_fixed_reg(MACHINE_MEM0_BASE_REG),
            ctx,
            (ctx_offset::MEM0_BASE / 8) as u32,
        ));
        self.core.text.emit_u32(enc::ldr_64(
            abi::map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            ctx,
            (ctx_offset::MEM0_SIZE / 8) as u32,
        ));
    }

    fn lower_epilogue(&mut self) {
        // Restore callee-saved FP registers.
        let fp_regs = abi::callee_saved_fp_regs();
        let mut fp_idx = 0usize;
        while fp_idx < fp_regs.len() {
            let byte_off =
                CALLEE_SAVED_FP_FRAME_OFFSET + (fp_idx as u32) * CALLEE_SAVED_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::ldr_q(fp_regs[fp_idx], abi::stack_reg(), byte_off / 16));
            #[cfg(not(sf_has_simd))]
            self.core.text.emit_u32(enc::ldr_d(
                fp_regs[fp_idx],
                abi::stack_reg(),
                stack_u64_slot(byte_off),
            ));
            fp_idx += 1;
        }
        // Restore callee-saved GP registers and deallocate frame.
        for (index, (lhs, rhs)) in abi::callee_saved_gp_pairs().iter().copied().enumerate() {
            self.core.text.emit_u32(enc::ldp_64(
                lhs,
                rhs,
                abi::stack_reg(),
                stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
            ));
        }
        self.core.text.emit_u32(enc::add_imm_64(
            abi::stack_reg(),
            abi::stack_reg(),
            CALLEE_SAVED_FRAME_SIZE,
        ));
        self.core.text.emit_u32(enc::ret());
    }

    /// Public-entry caller stub. Loads root register parameters and `bl`s the
    /// internal entry. The root function uses the root wasm frame directly, so
    /// successful root returns copy their return-result frame region into the
    /// public result slots at `stack_base[0..]`. `C_RET0` holds 0 or a trap
    /// kind on return.
    fn lower_root_caller_stub(&mut self) {
        self.lower_root_param_lanes_from_frame();
        // bl internal_entry_label  (resolved by patch_fixups)
        let internal_entry_label = self.core.internal_entry_label;
        self.lower_bl(internal_entry_label);

        let done = self.core.new_label();
        self.lower_cbnz(abi::C_RET0, done);
        self.lower_root_result_copy_to_public_slots();
        self.core.bind_label(done);
    }

    /// Body entry prelude. Non-leaf bodies push x29/x30 onto the host stack so
    /// nested native calls can freely clobber LR. Leaf bodies with no preserved
    /// dynamic save area skip this prelude and return using the incoming LR.
    fn lower_body_prelude(&mut self) {
        if !self.has_body_host_frame() {
            return;
        }
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();
        // stp x29, x30, [sp, #-16]!
        self.core
            .text
            .emit_u32(enc::stp_64_pre_index(x29, x30, abi::stack_reg(), -16));
        self.lower_preserved_dynamic_body_save();
        if let Some(target) = self.bulk_memset_target {
            super::inst::materialize_u64_into(
                &mut self.core.text,
                target,
                memset as *const () as usize as u64,
            );
        }
        if let Some(target) = self.bulk_memmove_target {
            super::inst::materialize_u64_into(
                &mut self.core.text,
                target,
                memmove as *const () as usize as u64,
            );
        }
    }

    /// Flush direct-call fallback veneers into the end-of-body pool. The
    /// normal direct-call patch rewrites the call-site instruction to
    /// `bl callee`; when the module linker finds that target out of range, it
    /// patches the same call-site instruction to `bl veneer` instead. The
    /// veneer loads the absolute callee address and tail-branches to it,
    /// preserving the LR produced by the original `bl`.
    fn lower_function_literal_pool(&mut self) -> Result<(), WasmError> {
        let pending = core::mem::take(&mut self.pending_direct_calls);
        let mut veneers: collections::Vec<DirectCallFallbackVeneer> = collections::Vec::new();
        for call in pending {
            let (veneer_offset, literal_offset) = if let Some(veneer) =
                veneers.iter().copied().find(|veneer| {
                    veneer.callee == call.callee && veneer.scratch_reg == call.fallback_scratch_reg
                }) {
                (veneer.veneer_offset, veneer.literal_offset)
            } else {
                let (veneer_offset, literal_offset) =
                    self.lower_direct_call_fallback_veneer(call.fallback_scratch_reg)?;
                veneers.push(DirectCallFallbackVeneer {
                    callee: call.callee,
                    scratch_reg: call.fallback_scratch_reg,
                    veneer_offset,
                    literal_offset,
                });
                (veneer_offset, literal_offset)
            };
            let patch = if call.link {
                crate::vm::jit::arch::common::types::DirectCallPatch::arm64_bl(
                    call.inst_offset,
                    veneer_offset,
                    literal_offset,
                    call.callee,
                )
            } else {
                crate::vm::jit::arch::common::types::DirectCallPatch::arm64_b(
                    call.inst_offset,
                    veneer_offset,
                    literal_offset,
                    call.callee,
                )
            };
            self.core.direct_call_patches.push(patch);
        }

        Ok(())
    }

    fn lower_body_local_error_tail(&mut self) {
        if !self.has_body_host_frame() {
            self.core.text.emit_u32(enc::ret());
            return;
        }
        self.lower_preserved_dynamic_body_restore();
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();
        // ldp x29, x30, [sp], #16    ; pop body prelude link save
        self.core
            .text
            .emit_u32(enc::ldp_64_post_index(x29, x30, abi::stack_reg(), 16));
        // C_RET0 untouched — preserves the error code.
        self.core.text.emit_u32(enc::ret());
    }

    #[cfg(sf_has_guard_pages)]
    fn lower_stack_probe(&mut self, addr: MachineAddr) -> Result<(), WasmError> {
        self.emit_stack_probe(addr)
    }

    fn begin_block(&mut self, block: &MachineBlock) -> Result<(), WasmError> {
        // Streaming entry: clear the lookahead and reset block state.
        self.pending_op = None;
        self.core.current_block = Some(block.id);
        self.core.current_edge_target = None;
        self.core.reset_block_fp_state(block)?;
        self.select_flags = None;
        self.bulk_bounds_facts = [None; 2];
        self.bulk_bounds_next = 0;
        Ok(())
    }

    // Note on dropped burst fusion: a previous experiment grouped consecutive
    // IndexedLoad/IndexedStore ops sharing (base, index, extend) into one
    // shared `add Xs, Xb, Wi, UXTW` plus N immediate-offset loads/stores.
    // It looks like an obvious win but is measurably slower on Apple Silicon
    // (M-series) because that core macro-fuses `add x, x, #imm` with the
    // following `ldr w, [base, x]` into a single AGU op, and `mov w, w` is a
    // zero-latency rename. The 3-instruction sequence `mov + add + ldr-reg`
    // effectively executes as a single load there; the burst form's
    // `add x, base, w_idx, UXTW` is not AGU-fusable so its N dependent loads
    // pay an extra cycle of critical-path latency. The full experiment
    // (`try_lower_indexed_burst` / `try_emit_burst_pair` in inst.rs) and its
    // measurements were removed as dead code; recover them from git history
    // if the tradeoff is ever revisited on a non-Apple core.
    fn emit_inst_at(&mut self, inst: &'a MachineInst, index: usize) -> Result<(), WasmError> {
        self.dispatch_seq = self.dispatch_seq.wrapping_add(1);
        if let MachineInstKind::Store { addr, width, .. } = &inst.kind {
            self.record_store(addr.base, addr.offset, width.bytes() as u8);
        }
        // Try to fuse the previously buffered op with this incoming op. On a
        // hit, both are consumed by a single emitted instruction. On a miss,
        // emit the buffered op solo and buffer the new one for the next call.
        if let Some((prev, prev_index)) = self.pending_op.take() {
            self.core.current_op_index = Some(prev_index);
            if let Some(fusion) = super::fusion::int_compare_select_fusion(prev, inst) {
                let bool_is_dead = self
                    .core
                    .current_block
                    .and_then(|id| self.core.mir_blocks().ok()?.get(id.as_usize()))
                    .is_some_and(|block| {
                        fusion.bool_reg == fusion.select_result
                            || !crate::vm::jit::machine::peephole::helpers::reg_live_after(
                                &block.ops[index + 1..],
                                &block.terminator,
                                fusion.bool_reg,
                            )
                    });
                if bool_is_dead {
                    self.select_flags = None;
                    self.lower_cmp_values(fusion.width, fusion.lhs, fusion.rhs)?;
                    let cond = super::fusion::map_int_cond(fusion.kind, fusion.sign);
                    let MachineInstKind::Select {
                        ty,
                        dst,
                        on_true,
                        on_false,
                        cond: _,
                    } = inst.kind
                    else {
                        unreachable!("validated compare-select fusion");
                    };
                    self.core.current_op_index = Some(index);
                    self.lower_select(
                        ty,
                        dst,
                        on_true,
                        on_false,
                        crate::vm::jit::machine::machine_ir::MachineValue::Reg(fusion.bool_reg),
                        Some((fusion.bool_reg, cond)),
                    )?;
                    self.gp_scratch.assert_all_free();
                    self.fp_scratch.assert_all_free();
                    return Ok(());
                }
            }
            if let Some((base, imm7)) = super::fusion::zero_store_pair_fusion(prev, inst) {
                let base_reg = self.map_gp_reg(base)?;
                self.core.text.emit_u32(enc::stp_zero_64(base_reg, imm7));
                self.gp_scratch.assert_all_free();
                self.fp_scratch.assert_all_free();
                return Ok(());
            }
            if let Some(fusion) = super::fusion::bit_clear_fusion(prev, inst) {
                let temporary_is_dead = fusion.not_result == fusion.dst
                    || self
                        .core
                        .current_block
                        .and_then(|id| self.core.mir_blocks().ok()?.get(id.as_usize()))
                        .is_some_and(|block| {
                            !crate::vm::jit::machine::peephole::helpers::reg_live_after(
                                &block.ops[index + 1..],
                                &block.terminator,
                                fusion.not_result,
                            )
                        });
                if temporary_is_dead {
                    let dst = self.map_gp_reg(fusion.dst)?;
                    let lhs = self.map_gp_reg(fusion.lhs)?;
                    let not_rhs = self.map_gp_reg(fusion.not_rhs)?;
                    let encoded = match fusion.width {
                        crate::vm::jit::machine::machine_ir::MachineIntWidth::I32 => {
                            enc::bics_reg_32(dst, lhs, not_rhs)
                        }
                        crate::vm::jit::machine::machine_ir::MachineIntWidth::I64 => {
                            enc::bics_reg_64(dst, lhs, not_rhs)
                        }
                    };
                    self.core.text.emit_u32(encoded);
                    self.select_flags = Some((fusion.dst, enc::Cond::Ne));
                    self.gp_scratch.assert_all_free();
                    self.fp_scratch.assert_all_free();
                    return Ok(());
                }
            }
            if self.try_lower_fp_pair(prev, inst)? {
                self.gp_scratch.assert_all_free();
                self.fp_scratch.assert_all_free();
                return Ok(());
            }
            // No fusion — emit prev solo before buffering the new op.
            self.lower_inst(prev)?;
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
        }
        self.pending_op = Some((inst, index));
        Ok(())
    }

    fn end_block(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        // Drain any final buffered op before the terminator; nothing else can
        // arrive to fuse with it.
        if let Some((prev, prev_index)) = self.pending_op.take() {
            self.core.current_op_index = Some(prev_index);
            self.lower_inst(prev)?;
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
        }
        self.core.current_op_index = None;
        // Flags published by the block's last instruction flow into the
        // terminator: a Branch on that same bool consumes them as a b.cond
        // (edge-arg moves and cache repairs emit no flag-setting
        // instructions). Other terminator kinds ignore them, and
        // `begin_block` clears before the next block.
        let result = self.lower_terminator(term, fallthrough);
        self.core.current_block = None;
        result
    }

    fn lower_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        self.lower_inst_dispatch(inst)
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
        self.lower_b(label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        // arm64 PC-relative branch reach in instruction words:
        //   b / bl              imm26 signed → ±2^25 words = ±128 MiB bytes
        //   b.cond / cbz / cbnz imm19 signed → ±2^18 words = ±1 MiB bytes
        // The encoders mask the immediate silently (`& 0x...`), so we
        // must validate the delta here or out-of-range targets become
        // wrong addresses at runtime. None of the current fixups should
        // ever exceed these bounds (functions are far smaller than
        // 1 MiB), but failing fast is the only safe option until a
        // veneer/island pass is added.
        const IMM19_WORD_MIN: isize = -(1 << 18);
        const IMM19_WORD_MAX: isize = (1 << 18) - 1;
        const IMM26_WORD_MIN: isize = -(1 << 25);
        const IMM26_WORD_MAX: isize = (1 << 25) - 1;
        for fixup in &self.fixups {
            let target = self
                .core
                .labels
                .get(fixup.label)
                .and_then(|v| *v)
                .ok_or_else(|| WasmError::internal("arm64 branch target label is unresolved"))?;
            let delta_bytes = (target as isize) - (fixup.inst_offset as isize);
            if delta_bytes & 0b11 != 0 {
                return Err(WasmError::internal(
                    "arm64 branch fixup target is not 4-byte aligned (inst at )",
                ));
            }
            let delta_words = delta_bytes / 4;
            let (_kind_name, in_range) = match fixup.kind {
                BranchFixupKind::B => (
                    "b",
                    (IMM26_WORD_MIN..=IMM26_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::Bl => (
                    "bl",
                    (IMM26_WORD_MIN..=IMM26_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::BCond(_) => (
                    "b.cond",
                    (IMM19_WORD_MIN..=IMM19_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::Cbz(_) => (
                    "cbz",
                    (IMM19_WORD_MIN..=IMM19_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::Cbnz(_) => (
                    "cbnz",
                    (IMM19_WORD_MIN..=IMM19_WORD_MAX).contains(&delta_words),
                ),
            };
            if !in_range {
                return Err(WasmError::internal(
                    "arm64 branch fixup is out of pc-relative range",
                ));
            }
            let delta_i32 = delta_words as i32;
            let patched = match fixup.kind {
                BranchFixupKind::B => enc::b(delta_i32),
                BranchFixupKind::BCond(cond) => enc::b_cond(cond, delta_i32),
                BranchFixupKind::Cbz(reg) => enc::cbz_64(reg, delta_i32),
                BranchFixupKind::Cbnz(reg) => enc::cbnz_64(reg, delta_i32),
                BranchFixupKind::Bl => enc::bl(delta_i32),
            };
            self.core.text.patch_u32(fixup.inst_offset, patched);
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
        self.core.text.emit_u32(enc::mov_reg_64(temp, dst_gp));
        self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
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
        let dst_fp = self.map_fp_reg(dst.reg)?;
        let width = dst.ty.float_width().expect("FP param width");
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmov_s(temp, dst_fp),
            MachineFloatWidth::F64 => enc::fmov_d(temp, dst_fp),
        });
        let src_fp = self.map_fp_reg(src)?;
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmov_s(dst_fp, src_fp),
            MachineFloatWidth::F64 => enc::fmov_d(dst_fp, src_fp),
        });
        self.core.set_fp_reg_width(dst.reg, width)?;
        Ok(())
    }

    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
        debug_assert!(bytes % 4 == 0, "ARM64 NOP padding must be 4-byte aligned");
        const ARM64_NOP: [u8; 4] = 0xd503201f_u32.to_le_bytes();
        for _ in 0..bytes / 4 {
            buf.emit_bytes(&ARM64_NOP);
        }
    }
}

impl<'a> crate::vm::jit::arch::shared_64::ModuleLinkBackend64<'a> for Arm64Backend<'a> {
    type CompiledEntry = CompiledArm64Entry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::jit::arch::shared_64::EmittedFunction64,
    ) -> Self::CompiledEntry {
        let entry = unsafe {
            buf.fn_ptr::<crate::vm::jit::runtime::code::NativeRootEntry>(emitted.text_offset)
        };
        CompiledArm64Entry {
            entry,
            text_len: emitted.text_len,
            #[cfg(sf_ir_dump)]
            debug_regions: emitted.debug_regions.clone(),
        }
    }
}
