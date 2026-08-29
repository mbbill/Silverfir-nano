//! Runtime side of the native dispatch chain.
//!
//! The handlers themselves are generated at BUILD time (`interp_gen/`,
//! driven by `build.rs`) and folded into this binary's own `.text` through
//! `global_asm!`. Nothing here allocates or maps executable memory: an
//! interpreter that needs runtime code generation is a JIT, and the
//! project already has one. On an MCU the engine is flash, not heap.
//!
//! What remains at run time is the LINK: turning each predecoded
//! instruction into a 32-byte dispatch cell whose first word is the
//! address of the handler for its exact operand-residency combination,
//! with slot operands pre-scaled to byte offsets and branch targets
//! resolved to absolute cell addresses.

use tracked_alloc::boxed::Box;

use crate::collections::Vec;

#[cfg(test)]
use super::instr::{operand_is_f32, operand_is_float, result_is_f32, result_is_float};
use super::instr::{
    DCell, Instr, Op, FLAG_ADDR64, FLAG_A_ACC, FLAG_A_CONST, FLAG_B_ACC, FLAG_B_CONST,
    FLAG_DST_ACC, FLAG_FUSED, FLAG_NO_NATIVE, FLAG_SHARED_GLOBAL, FLAG_SHARED_TABLE,
};
#[cfg(test)]
use super::layout::slot_fields;
#[cfg(test)]
use super::layout::stage_draft;
use super::layout::{
    c_is_branch_target, decode_draft, draft_payload_is_prelinked, family, op_slot, total_slots,
    transform_bc, writes_acc, Fam, Pinned,
};
use super::predecode::LinkFunction;
#[cfg(test)]
use super::predecode::PredecodedFunction;

// Build-time facts about the generated engine: which operand classes it
// was built with, and how many packed handler slots the table holds.
include!(concat!(env!("OUT_DIR"), "/interp_engine_cfg.rs"));

/// Target capabilities which affect pinned-local selection. The predecoder
/// records the final choice once, while it already owns every instruction
/// mutation, and the linker consumes that compact result directly.
pub(super) const PIN_CAPS: (bool, bool, bool) =
    (INTERP_HAS_L1, INTERP_HAS_FLOAT_REGS, INTERP_FLOAT_PIN_F32);

core::arch::global_asm!(include_str!(concat!(env!("OUT_DIR"), "/interp_engine.s")));

extern "C" {
    /// First byte of the generated blob; every offset in the tables below
    /// is measured from here.
    static sf_interp_code_base: u8;
    /// `[entry, slow_stub, return_exit, call, call_indirect, code_size,
    /// handler_slots]`.
    static sf_interp_meta: [u32; 7];
    /// Packed handler offsets, indexed by [`op_slot`]. Declared as its
    /// first element and read through a pointer: the length is a function
    /// of the `Op` set, which Rust cannot spell in an `extern` array type
    /// without duplicating it as a literal.
    static sf_interp_handlers: u32;
}

/// Exit reasons written to `EnterState::reason`.
pub(super) const EXIT_SLOW: u64 = 1;
/// A `Return` popped a sentinel record: control goes back to Rust.
pub(super) const EXIT_RETURN: u64 = 2;
pub(super) const EXIT_TRAP_BASE: u64 = 16;
/// Native trap kinds, indexed by `reason - EXIT_TRAP_BASE`. Messages must
/// match `exec_ins` exactly (differential and spectest parity).
pub(super) const TRAP_KINDS: &[&str] = &["out of bounds memory access", "call stack exhausted"];

/// Bytes per native return-stack record:
/// `(ret_pc, frame, code_base | caller_l0off<<48, caller_l1off)`.
pub(super) const RET_RECORD: usize = 32;

/// Communication block between Rust and the native chain. Field offsets
/// are baked into every backend's trampoline; keep in sync with
/// `interp_gen`.
#[repr(C)]
pub(super) struct EnterState {
    pub reason: u64,      // 0
    pub pc: u64,          // 8: cell address (in on entry, out on exit)
    pub frame: u64,       // 16: frame base pointer (in AND out: calls move it)
    pub mem_base: u64,    // 24: memory 0 base (unused when len is 0)
    pub mem_len: u64,     // 32
    pub code_base: u64,   // 40: dispatch-cell base of the CURRENT function
    pub globals: u64,     // 48
    pub ret_cursor: u64,  // 56: native return-stack cursor (in and out)
    pub ret_limit: u64,   // 64: return-stack end (depth exhaustion)
    pub stack_limit: u64, // 72: value-stack end (frame exhaustion)
    pub dispatches: u64,  // 80: handler dispatch count (in and out)
    pub l0_value: u64,    // 88: current function's l0 local value (in only)
    pub l1_value: u64,    // 96: current function's l1 local value (in only)
    pub acc_value: u64,   // 104: the accumulator (in AND out — call results
    // ride it across activation boundaries)
    pub table0_base: u64,   // 112: table 0 entries (in only, RefValue slots)
    pub table0_len: u64,    // 120
    pub indirect_base: u64, // 128: per-function indirect-call info, [u64;3]
    // per function index (in only)
    pub indirect_len: u64, // 136: number of per-function info entries
}

/// Native handlers which can return `EXIT_SLOW` after their cell payload has
/// been transformed. Static slow cells keep raw a/b/c and need no inverse.
const fn native_may_exit_slow(op: Op) -> bool {
    use Op::*;
    matches!(
        op,
        I32_DivS
            | I32_DivU
            | I32_RemS
            | I32_RemU
            | I64_DivS
            | I64_DivU
            | I64_RemS
            | I64_RemU
            | I32_TruncF32S
            | I32_TruncF32U
            | I32_TruncF64S
            | I32_TruncF64U
            | I64_TruncF32S
            | I64_TruncF32U
            | I64_TruncF64S
            | I64_TruncF64U
            | MemoryFill
            | MemoryCopy
            | MemoryFillCopy
            | CallIndirect
    )
}

/// Recreate a native cell's stage-A payload on an `EXIT_SLOW` edge.
/// `CallIndirect` keeps its original module type index in the otherwise-unused
/// `c` word while its native handler uses `a` and `b`.
fn restore_native_slow_instr(head: u32, cell: &DCell) -> Option<Instr> {
    let op = super::instr::op_from_index((head & 0xffff) as usize);
    let flags = (head >> 16) as u16;
    if !native_may_exit_slow(op) {
        return None;
    }
    let unscale = |value: u64| {
        debug_assert_eq!(value & 7, 0);
        value / 8
    };
    let (a, b, c) = match op {
        Op::CallIndirect => (
            unscale(cell.a & ((1u64 << 48) - 1)),
            unscale(cell.b & 0xffff_ffff),
            cell.c,
        ),
        Op::MemoryFill | Op::MemoryCopy => (unscale(cell.a), unscale(cell.b), unscale(cell.c)),
        Op::MemoryFillCopy => (unscale(cell.a), unscale(cell.b), cell.c),
        _ => {
            let a = if flags & FLAG_A_CONST != 0 {
                cell.a
            } else {
                unscale(cell.a)
            };
            let b = if flags & FLAG_B_CONST != 0 {
                cell.b
            } else {
                unscale(cell.b)
            };
            (a, b, unscale(cell.c))
        }
    };
    Some(Instr::from_packed_head(head, a, b, c))
}

/// One linked function's slice in a module-wide [`LinkPlan`]. Dispatch cells
/// mirror `PredecodedFunction::code` index-for-index, followed by one prefetch
/// pad cell. The arena owns storage; this keeps a stable absolute entry address
/// for the activation hot path without storing a self-referential slice.
pub(super) struct LinkedFunction {
    #[cfg(test)]
    cell_start: usize,
    /// Stable absolute address of `cell_start` in the preallocated arena.
    ///
    /// Shared-table funcref calls cross the Rust driver on every activation.
    /// Caching this restores their hot lookup to one field load instead of
    /// reloading the arena pointer and scaling `cell_start` each time.
    cell_base: u64,
    cell_len: usize,
    /// Byte offsets of the function's pinned locals in its frame (0 when
    /// absent — the unconditional reload then reads slot 0, which no cell
    /// consumes as a pinned class).
    pub l0_off: u32,
    pub l1_off: u32,
    /// Whether any pinned slot is float-mode: drives the conditional
    /// float-twin reloads on the call and return paths.
    pub fp_pinned: bool,
}

impl LinkedFunction {
    #[inline]
    pub(super) fn cell_base(&self) -> u64 {
        self.cell_base
    }
}

/// A native call site deferred until every function has a stable arena
/// address. Recording these during the cell-build pass avoids rescanning every
/// instruction in the module for the cross-function fixup.
#[cfg_attr(test, derive(Clone, Copy, Debug, PartialEq, Eq))]
pub(super) enum CallFixup {
    Direct {
        cell: usize,
        caller: usize,
        callee: usize,
        arg_base: u64,
    },
    Indirect {
        cell: usize,
        caller: usize,
        table_slot: u64,
        arg_base: u64,
        expected_type: u32,
    },
}

#[derive(Clone, Copy)]
struct SlowHeadCandidate {
    cell: usize,
    head: u32,
}

/// Module-wide storage planned before linking begins.
///
/// The exact cell and branch-table capacities are computed from all
/// predecoded functions. Neither backing vector can therefore reallocate while
/// absolute pointers are installed into dispatch cells. Besides replacing two
/// allocations per function with two per module, the plan carries one compact
/// list of native call sites for the deferred cross-function fixup.
pub(super) struct LinkPlan {
    cells: Vec<DCell>,
    #[cfg(any(test, feature = "interp-count"))]
    heads: Vec<u32>,
    /// `[mask, head_base]` pairs for every 32 cells, followed by their packed
    /// heads. One allocation holds both the O(1) directory and compact data.
    slow_head_sidecar: Vec<u32>,
    slow_head_candidates: Vec<SlowHeadCandidate>,
    br_flat: Vec<u32>,
    call_fixups: Vec<CallFixup>,
    planned_cells: usize,
    planned_br_entries: usize,
    linked_cells: usize,
    in_place: bool,
}

impl LinkPlan {
    #[cfg(test)]
    pub(super) fn for_functions<'a>(
        functions: impl Iterator<Item = &'a PredecodedFunction>,
    ) -> Self {
        let mut function_count = 0usize;
        let mut planned_cells = 0usize;
        let mut planned_br_entries = 0usize;
        for func in functions {
            function_count += 1;
            planned_cells = planned_cells
                .checked_add(func.code.len() + 1)
                .expect("interpreter dispatch-cell count overflow");
            for table in func.br_tables.iter() {
                planned_br_entries = planned_br_entries
                    .checked_add(table.len())
                    .expect("interpreter branch-table count overflow");
            }
        }
        Self {
            cells: Vec::with_capacity(planned_cells),
            #[cfg(any(test, feature = "interp-count"))]
            heads: Vec::new(),
            slow_head_sidecar: Vec::new(),
            slow_head_candidates: Vec::with_capacity(function_count),
            br_flat: Vec::with_capacity(planned_br_entries),
            // Most real functions have no call. One slot per function avoids
            // early growth without reserving in proportion to instruction
            // count or adding a separate call-counting sweep.
            call_fixups: Vec::with_capacity(function_count),
            planned_cells,
            planned_br_entries,
            linked_cells: 0,
            in_place: false,
        }
    }

    pub(super) fn from_cell_arena(
        cells: Vec<DCell>,
        function_count: usize,
        planned_br_entries: usize,
    ) -> Self {
        #[cfg(any(test, feature = "interp-count"))]
        let heads = cells.iter().copied().map(|cell| cell.h as u32).collect();
        let planned_cells = cells.len();
        Self {
            cells,
            #[cfg(any(test, feature = "interp-count"))]
            heads,
            slow_head_sidecar: Vec::new(),
            slow_head_candidates: Vec::with_capacity(function_count),
            br_flat: Vec::with_capacity(planned_br_entries),
            call_fixups: Vec::with_capacity(function_count),
            planned_cells,
            planned_br_entries,
            linked_cells: 0,
            in_place: true,
        }
    }

    #[inline]
    fn begin_function(&mut self, instruction_len: usize) -> usize {
        let start = self.linked_cells;
        self.linked_cells = self
            .linked_cells
            .checked_add(instruction_len + 1)
            .expect("interpreter linked-cell count overflow");
        debug_assert!(self.linked_cells <= self.planned_cells);
        start
    }

    #[inline]
    fn write_cell(&mut self, index: usize, cell: DCell) {
        if self.in_place {
            self.cells[index] = cell;
        } else {
            debug_assert_eq!(index, self.cells.len());
            self.cells.push(cell);
        }
    }

    #[inline]
    fn raw_instr(&self, index: usize) -> Instr {
        let cell = self.cells[index];
        #[cfg(any(test, feature = "interp-count"))]
        debug_assert_eq!(cell.h as u32, self.heads[index]);
        decode_draft(cell)
    }

    /// Verify that the precomputed storage plan and the completed link agree.
    pub(super) fn finish_layout(&self) {
        debug_assert_eq!(self.cells.len(), self.planned_cells);
        debug_assert_eq!(self.br_flat.len(), self.planned_br_entries);
        debug_assert_eq!(self.linked_cells, self.planned_cells);
        #[cfg(any(test, feature = "interp-count"))]
        debug_assert_eq!(
            self.heads.len(),
            if self.in_place { self.planned_cells } else { 0 }
        );
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn cells(&self, func: &LinkedFunction) -> &[DCell] {
        &self.cells[func.cell_start..func.cell_start + func.cell_len]
    }

    #[cfg(test)]
    pub(super) fn branch_bytes(&self) -> core::ops::Range<u64> {
        let start = self.br_flat.as_ptr() as u64;
        start..start + self.br_flat.len() as u64 * core::mem::size_of::<u32>() as u64
    }

    #[cfg(test)]
    pub(super) fn call_fixups_are_drained(&self) -> bool {
        self.call_fixups.is_empty() && self.call_fixups.capacity() == 0
    }

    #[inline]
    pub(super) fn instruction_len(&self, func: &LinkedFunction) -> usize {
        func.cell_len - 1
    }

    #[cfg(any(test, feature = "interp-count"))]
    pub(super) fn instruction_heads(&self, func: &LinkedFunction) -> &[u32] {
        let arena_base = self.cells.as_ptr() as u64;
        let start =
            ((func.cell_base() - arena_base) / core::mem::size_of::<DCell>() as u64) as usize;
        &self.heads[start..start + self.instruction_len(func)]
    }

    #[inline]
    pub(super) fn cell_mut(&mut self, index: usize) -> &mut DCell {
        &mut self.cells[index]
    }

    pub(super) fn take_call_fixups(&mut self) -> Vec<CallFixup> {
        core::mem::take(&mut self.call_fixups)
    }

    #[inline]
    fn record_slow_head_candidate(&mut self, cell: usize, head: u32) {
        debug_assert!(self
            .slow_head_candidates
            .last()
            .is_none_or(|site| site.cell < cell));
        self.slow_head_candidates
            .push(SlowHeadCandidate { cell, head });
    }

    /// Freeze the compact slow-exit directory after cross-function call
    /// fixups have changed their last cells from static-slow to native.
    pub(super) fn finish_slow_heads(&mut self, slow_stub: u64) {
        debug_assert!(self.slow_head_sidecar.is_empty());
        let mut candidates = core::mem::take(&mut self.slow_head_candidates);
        candidates.retain(|candidate| {
            let cell = &self.cells[candidate.cell];
            let op = super::instr::op_from_index((candidate.head & 0xffff) as usize);
            cell.h == slow_stub || native_may_exit_slow(op)
        });
        let block_count = self.cells.len() / 32 + usize::from(self.cells.len() % 32 != 0);
        let directory_len = block_count
            .checked_mul(2)
            .expect("interpreter slow-head directory overflow");
        let sidecar_len = directory_len
            .checked_add(candidates.len())
            .expect("interpreter slow-head sidecar overflow");
        let _ = u32::try_from(sidecar_len).expect("interpreter slow-head sidecar exceeds u32");
        self.slow_head_sidecar.reserve_exact(sidecar_len);

        let mut candidate_index = 0usize;
        for block_index in 0..block_count {
            let mut mask = 0u32;
            let block_head_start = candidate_index;
            while let Some(candidate) = candidates.get(candidate_index).copied() {
                if candidate.cell / 32 != block_index {
                    break;
                }
                mask |= 1u32 << (candidate.cell % 32);
                candidate_index += 1;
            }
            let head_base = directory_len
                .checked_add(block_head_start)
                .expect("interpreter slow-head index overflow");
            self.slow_head_sidecar.push(mask);
            self.slow_head_sidecar
                .push(u32::try_from(head_base).expect("interpreter slow-head count overflow"));
        }
        debug_assert_eq!(candidate_index, candidates.len());
        self.slow_head_sidecar
            .extend(candidates.into_iter().map(|candidate| candidate.head));
    }

    #[inline]
    fn slow_head(&self, cell: usize) -> Option<u32> {
        let directory = (cell / 32).checked_mul(2)?;
        let mask = *self.slow_head_sidecar.get(directory)?;
        let head_base = *self.slow_head_sidecar.get(directory + 1)? as usize;
        let bit = 1u32 << (cell % 32);
        if mask & bit == 0 {
            return None;
        }
        let rank = (mask & bit.wrapping_sub(1)).count_ones() as usize;
        self.slow_head_sidecar.get(head_base + rank).copied()
    }

    pub(super) fn cell_index(&self, address: u64) -> Option<usize> {
        let base = self.cells.as_ptr() as u64;
        let bytes = address.checked_sub(base)?;
        if bytes % core::mem::size_of::<DCell>() as u64 != 0 {
            return None;
        }
        let index = (bytes / core::mem::size_of::<DCell>() as u64) as usize;
        (index < self.cells.len()).then_some(index)
    }

    pub(super) fn restore_slow_instr(&self, index: usize, slow_stub: u64) -> Option<Instr> {
        let cell = self.cells.get(index)?;
        let head = self.slow_head(index)?;
        if cell.h == slow_stub {
            return Some(Instr::from_packed_head(head, cell.a, cell.b, cell.c));
        }
        restore_native_slow_instr(head, cell)
    }

    #[cfg(test)]
    fn restore_slow_instr_dense_reference(&self, index: usize, slow_stub: u64) -> Option<Instr> {
        let cell = self.cells.get(index)?;
        let head = *self.heads.get(index)?;
        if cell.h == slow_stub {
            return Some(Instr::from_packed_head(head, cell.a, cell.b, cell.c));
        }
        restore_native_slow_instr(head, cell)
    }

    #[cfg(test)]
    pub(super) fn assert_sparse_slow_heads(
        &self,
        functions: &[Option<LinkedFunction>],
        slow_stub: u64,
    ) {
        let fields = |ins: Instr| (ins.op, ins.flags, ins.a, ins.b, ins.c);
        for function in functions.iter().flatten() {
            let start = self
                .cell_index(function.cell_base())
                .expect("linked function belongs to its link plan");
            for index in start..start + self.instruction_len(function) {
                let sparse = self.restore_slow_instr(index, slow_stub).map(fields);
                let dense = self
                    .restore_slow_instr_dense_reference(index, slow_stub)
                    .map(fields);
                assert_eq!(sparse, dense, "sparse slow head at cell {index}");
            }
        }
    }
}

/// Per-function working buffers the linker reuses across a module.
///
/// Every one of these is sized by the function being linked and dead the
/// moment it is done, so a module with 14 k functions otherwise pays 14 k
/// allocation/free pairs for each of them. The linker owns their contents
/// only inside one `link` call; nothing here outlives it.
#[derive(Default)]
pub(super) struct LinkScratch {
    table_byte_off: Vec<u64>,
}

/// One instruction's link state while it waits for its immediate successor.
///
/// Accumulator residency is a span-one contract: whether a producer may skip
/// its slot write is known as soon as the next instruction has been
/// classified. Keeping only this pending state avoids two function-sized
/// scratch arrays and the separate accumulator/defensive sweeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedCell {
    flags: u16,
    /// `handler_for(ins, flags, pin)`, with 0 meaning no native form.
    handler: usize,
}

trait LinkCode {
    fn get(&self, plan: &LinkPlan, cell_start: usize, index: usize) -> Instr;
    fn is_in_place(&self) -> bool;
}

struct InPlaceCode;

impl LinkCode for InPlaceCode {
    #[inline]
    fn get(&self, plan: &LinkPlan, cell_start: usize, index: usize) -> Instr {
        plan.raw_instr(cell_start + index)
    }

    fn is_in_place(&self) -> bool {
        true
    }
}

#[cfg(test)]
struct BorrowedCode<'a>(&'a [Instr]);

#[cfg(test)]
impl LinkCode for BorrowedCode<'_> {
    fn get(&self, _plan: &LinkPlan, _cell_start: usize, index: usize) -> Instr {
        self.0[index]
    }

    fn is_in_place(&self) -> bool {
        false
    }
}

/// What [`select_pinned_reference`] records about one frame slot.
///
/// One array rather than four parallel ones: the pass touches every field
/// of a slot together, and four arrays put each access on its own cache
/// line.
#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct SlotStat {
    /// Static reference count, unweighted.
    count: u32,
    /// Domain set of this slot's WRITERS: bit 0 integer/agnostic, bit 1
    /// float. Writers decide the pinned register file; a mixed set makes
    /// the slot unpinnable, because neither file could stay authoritative.
    wdom: u8,
    /// Domain set of this slot's READERS, same bits. Breaks the tie only
    /// when the slot has no writer.
    rdom: u8,
    /// Whether the slot's float accesses are single-precision. Some
    /// backends cannot hold an f32 in a pinned float register (see
    /// `INTERP_FLOAT_PIN_F32`), so the width has to be known here.
    f32dom: bool,
}

/// Original full-stream linker census, retained as a test-only oracle.
/// Pick the function's pinned locals — the most- and second-most-
/// referenced slots, by UNWEIGHTED static count (`u64::MAX` = none).
///
/// Loop-depth weighting (10^depth over back edges) was tried and measured
/// 11% WORSE on CoreMark: it systematically displaced frequently-WRITTEN
/// locals (loop-carried state, e.g. 33r/17w) with read-mostly ones (base
/// pointers, 39r/3w). Pinning pays if and only if it breaks a binding
/// loop-carried store->load chain, which needs the WRITTEN local;
/// read-mostly slot loads are independent and the out-of-order core hides
/// them anyway. A write-biased score (reads + 2*writes) measured
/// inconclusive on CoreMark, which the traffic model says cannot show the
/// effect — it stays open.
#[cfg(test)]
pub(super) fn select_pinned_reference(code: &[Instr], n_locals: u32) -> Pinned {
    select_pinned_reference_iter(code.iter().copied(), n_locals)
}

#[cfg(test)]
pub(super) fn select_pinned_reference_iter(
    code: impl IntoIterator<Item = Instr>,
    n_locals: u32,
) -> Pinned {
    let n = n_locals as u64;
    if n == 0 {
        return Pinned::NONE;
    }
    let mut slots = Vec::new();
    slots.resize(n_locals as usize, SlotStat::default());
    for ins in code {
        let (a_s, b_s, c_d) = slot_fields(ins.op);
        if a_s && ins.flags & FLAG_A_CONST == 0 && ins.a < n {
            let slot = &mut slots[ins.a as usize];
            slot.count += 1;
            if operand_is_float(ins.op, false) {
                slot.rdom |= 2;
                slot.f32dom |= operand_is_f32(ins.op, false);
            } else {
                slot.rdom |= 1;
            }
        }
        if b_s && ins.flags & FLAG_B_CONST == 0 && ins.b < n {
            let slot = &mut slots[ins.b as usize];
            slot.count += 1;
            if operand_is_float(ins.op, true) {
                slot.rdom |= 2;
                slot.f32dom |= operand_is_f32(ins.op, true);
            } else {
                slot.rdom |= 1;
            }
        }
        if c_d && ins.c < n {
            let slot = &mut slots[ins.c as usize];
            slot.count += 1;
            if result_is_float(ins.op) {
                slot.wdom |= 2;
                slot.f32dom |= result_is_f32(ins.op);
            } else {
                slot.wdom |= 1;
            }
        }
        if matches!(ins.op, Op::I32_SubBrIf | Op::I64_SubBrIf) && ins.a < n {
            // This control-shaped cell is also an in-place integer write to
            // `a`. Count both halves of the read/modify/write so pin
            // selection and register-domain authority match the unfused
            // subtraction it replaces.
            let slot = &mut slots[ins.a as usize];
            slot.count += 1;
            slot.wdom |= 1;
        }
        if ins.op == Op::Select {
            let dslot = ins.c & 0xffff_ffff;
            if dslot < n {
                slots[dslot as usize].wdom |= 1;
            }
        }
        if ins.op == Op::MovPair {
            // `slot_fields` cannot describe two packed destinations, so
            // account for both here. The linker uses the resulting pin
            // choice to select an ordered pair handler that writes through
            // both frame slots and both authoritative registers.
            for dslot in [ins.c >> 32, ins.c & 0xffff_ffff] {
                if dslot < n {
                    let slot = &mut slots[dslot as usize];
                    slot.count += 1;
                    slot.wdom |= 1;
                }
            }
        }
    }
    // Whether slot `i` could live in the float register file at all.
    let float_ok = |i: usize| INTERP_HAS_FLOAT_REGS && (INTERP_FLOAT_PIN_F32 || !slots[i].f32dom);
    let mut best = (usize::MAX, 0u32);
    let mut second = (usize::MAX, 0u32);
    for (i, stat) in slots.iter().enumerate() {
        let (c, wdom) = (stat.count, stat.wdom);
        // A slot is pinnable only when ONE register file can stay
        // authoritative for it. Mixed-domain writers rule that out, and so
        // does a float writer on a backend (or a width) that cannot pin a
        // float: the write would land in the slot alone and leave the
        // integer pinned register stale for the next integer-domain read.
        if wdom == 3 || (wdom & 2 != 0 && !float_ok(i)) {
            continue;
        }
        if c > best.1 {
            second = best;
            best = (i, c);
        } else if c > second.1 {
            second = (i, c);
        }
    }
    // Byte offsets must fit the 16-bit packing in call cells / records.
    let ok = |(i, c): (usize, u32)| c > 0 && i * 8 < 1 << 16;
    // A slot's authoritative register file: float only when it has a
    // float writer (or, with no writer at all, only float readers) AND
    // this backend has float twins for the pinned registers.
    let mode = |i: usize| {
        float_ok(i) && (slots[i].wdom == 2 || (slots[i].wdom == 0 && slots[i].rdom == 2))
    };
    let (l0, l0f) = if ok(best) {
        (best.0 as u64, mode(best.0))
    } else {
        (u64::MAX, false)
    };
    let (l1, l1f) = if INTERP_HAS_L1 && l0 != u64::MAX && ok(second) {
        (second.0 as u64, mode(second.0))
    } else {
        (u64::MAX, false)
    };
    Pinned {
        l0,
        l1,
        l0_float: l0f,
        l1_float: l1f,
    }
}

/// Whether a memory op's static offset fits one 32-bit machine word.
/// Only consulted on 32-bit hosts; the fused store form keeps its offset
/// in the low half of `c` and the predecoder already bounds it there.
fn offset_fits_word(ins: &Instr) -> bool {
    match family(ins.op) {
        Fam::Load => ins.b >> 32 == 0,
        Fam::Store => ins.flags & FLAG_FUSED != 0 || ins.c >> 32 == 0,
        _ => true,
    }
}

/// Whether this target's backend implements the native call protocol.
/// When it does not, the cross-function fixup is skipped and both call
/// flavours link to the slow stub: the driver then owns every activation,
/// which costs two chain crossings per call but is otherwise identical.
pub(super) const NATIVE_CALLS: bool = INTERP_NATIVE_CALLS;

/// The generated handler set plus the entry points around it. Holds no
/// memory of its own beyond one boxed sentinel cell — everything else is
/// a pointer into the binary.
pub(super) struct NativeEngine {
    base: usize,
    handlers: *const u32,
    slow_stub: usize,
    call_handler: usize,
    callindirect_handler: usize,
    code_len: usize,
    entry: usize,
    /// One synthetic cell whose handler word is the `EXIT_RETURN` stub.
    /// Sentinel return-stack records point here, so a native `Return` that
    /// pops a sentinel lands in Rust — the boxed cell must outlive every
    /// record that references it.
    exit_cell: Box<DCell>,
}

impl NativeEngine {
    pub(super) fn new() -> NativeEngine {
        // The generated table is packed by the same `family`/`op_slot`
        // code the linker indexes it with. If those two ever disagree the
        // engine would silently mis-dispatch, so the shapes are checked
        // rather than assumed.
        let meta = unsafe { sf_interp_meta };
        debug_assert_eq!(meta[6] as usize, total_slots());
        debug_assert_eq!(INTERP_HANDLER_SLOTS, total_slots());
        let base = unsafe { &sf_interp_code_base as *const u8 as usize };
        let entry = base + meta[0] as usize;
        let slow_stub = base + meta[1] as usize;
        let return_exit = base + meta[2] as usize;
        NativeEngine {
            base,
            handlers: unsafe { &sf_interp_handlers as *const u32 },
            slow_stub,
            call_handler: base + meta[3] as usize,
            callindirect_handler: base + meta[4] as usize,
            code_len: meta[5] as usize,
            entry,
            exit_cell: Box::new(DCell {
                h: return_exit as u64,
                a: 0,
                b: 0,
                c: 0,
            }),
        }
    }

    /// Generated engine size in bytes. It is linked into the binary, so
    /// this is text-segment cost, not an allocation — and on MCU targets
    /// it is a flash budget.
    pub(super) fn code_len(&self) -> usize {
        self.code_len
    }

    /// Address of the sentinel exit cell (see `exit_cell`).
    pub(super) fn exit_cell_addr(&self) -> u64 {
        &*self.exit_cell as *const DCell as u64
    }

    pub(super) fn call_handler_addr(&self) -> u64 {
        self.call_handler as u64
    }

    pub(super) fn callindirect_handler_addr(&self) -> u64 {
        self.callindirect_handler as u64
    }

    pub(super) fn slow_stub_addr(&self) -> u64 {
        self.slow_stub as u64
    }

    /// Handler address for a packed slot, or `None` when the backend has
    /// no native form (slot offset 0 is the blob base, which is never a
    /// handler, so it doubles as the sentinel).
    fn handler_at(&self, slot: usize) -> Option<usize> {
        debug_assert!(slot < INTERP_HANDLER_SLOTS);
        let off = unsafe { *self.handlers.add(slot) };
        if off == 0 {
            None
        } else {
            Some(self.base + off as usize)
        }
    }

    /// Handler address for one cell under a given pinned-local choice, or
    /// 0 when the cell has no native form and must take the slow stub.
    ///
    /// The blob base is never a handler address, so 0 doubles as the
    /// sentinel — the same convention `handler_at` uses for an unemitted
    /// slot. Predecode caches instruction-local native eligibility in
    /// `FLAG_NO_NATIVE`, so accumulator retries do not re-read and classify
    /// the cell's payload.
    fn handler_for(&self, ins: &Instr, flags: u16, pin: &Pinned) -> usize {
        // The generated handlers zero-extend a 32-bit address, so a 64-bit
        // memory access has no native form and takes the shared executor.
        if flags & (FLAG_ADDR64 | FLAG_SHARED_TABLE | FLAG_SHARED_GLOBAL | FLAG_NO_NATIVE) != 0 {
            return 0;
        }
        match op_slot(ins, flags, pin) {
            Some(slot) => self.handler_at(slot).unwrap_or(0),
            None => 0,
        }
    }

    /// The payload-inspecting implementation replaced by `FLAG_NO_NATIVE`.
    /// Kept only as a differential oracle for the cached production path.
    #[cfg(test)]
    fn handler_for_uncached_reference(&self, ins: &Instr, flags: u16, pin: &Pinned) -> usize {
        if flags & (FLAG_ADDR64 | FLAG_SHARED_TABLE | FLAG_SHARED_GLOBAL) != 0
            || !super::layout::native_guard(ins)
        {
            return 0;
        }
        match op_slot(ins, flags, pin) {
            Some(slot) => self.handler_at(slot).unwrap_or(0),
            None => 0,
        }
    }

    #[inline]
    fn initial_resolution(&self, ins: &Instr, pin: &Pinned) -> ResolvedCell {
        ResolvedCell {
            flags: ins.flags,
            handler: self.handler_for(ins, ins.flags, pin),
        }
    }

    #[inline]
    fn first_resolution(&self, ins: &Instr, pin: &Pinned) -> ResolvedCell {
        let mut state = self.initial_resolution(ins, pin);
        // A malformed first cell has no preceding accumulator producer.
        if state.flags & (FLAG_A_ACC | FLAG_B_ACC) != 0 {
            state.flags &= !(FLAG_A_ACC | FLAG_B_ACC);
            state.handler = self.handler_for(ins, state.flags, pin);
        }
        state
    }

    /// Resolve `next`'s incoming accumulator edge and, with that decision in
    /// hand, finish `prev`'s outgoing edge.
    ///
    /// This is the old two link sweeps expressed as a one-cell delay. A
    /// failed consumer loses both ACC source bits, and its producer loses
    /// `DST_ACC`; every changed instruction is reclassified immediately so
    /// the next pair observes the same handler the reference sweeps did.
    #[inline]
    fn resolve_pair(
        &self,
        prev_ins: &Instr,
        prev: &mut ResolvedCell,
        next_ins: &Instr,
        next: &mut ResolvedCell,
        pin: &Pinned,
    ) {
        if next.flags & (FLAG_A_ACC | FLAG_B_ACC) != 0 {
            let prev_is_call = matches!(prev_ins.op, Op::Call | Op::CallIndirect);
            let ok = (writes_acc(prev_ins.op) || prev_is_call)
                && (prev_is_call || prev.handler != 0)
                && next.handler != 0;
            if !ok {
                next.flags &= !(FLAG_A_ACC | FLAG_B_ACC);
                next.handler = self.handler_for(next_ins, next.flags, pin);
                prev.flags &= !FLAG_DST_ACC;
                prev.handler = self.handler_for(prev_ins, prev.flags, pin);
            }
        } else {
            // Defensive counterpart of the old second sweep: a producer may
            // skip its slot write only when the very next cell consumes ACC.
            if prev.flags & FLAG_DST_ACC != 0 {
                prev.flags &= !FLAG_DST_ACC;
                prev.handler = self.handler_for(prev_ins, prev.flags, pin);
            }
        }
    }

    #[inline]
    fn finish_last(&self, ins: &Instr, state: &mut ResolvedCell, pin: &Pinned) {
        if state.flags & FLAG_DST_ACC != 0 {
            state.flags &= !FLAG_DST_ACC;
            state.handler = self.handler_for(ins, state.flags, pin);
        }
    }

    /// The former whole-function resolver, retained only as a differential
    /// oracle for the streaming implementation.
    #[cfg(test)]
    fn resolve_reference(&self, code: &[Instr], pin: &Pinned) -> Vec<ResolvedCell> {
        let mut flags: Vec<u16> = code.iter().map(|ins| ins.flags).collect();
        let mut handlers: Vec<usize> = code
            .iter()
            .zip(flags.iter())
            .map(|(ins, &fl)| self.handler_for(ins, fl, pin))
            .collect();
        for j in 0..code.len() {
            if flags[j] & (FLAG_A_ACC | FLAG_B_ACC) == 0 {
                continue;
            }
            let prev_is_call = j > 0 && matches!(code[j - 1].op, Op::Call | Op::CallIndirect);
            let ok = j > 0
                && (writes_acc(code[j - 1].op) || prev_is_call)
                && (prev_is_call || handlers[j - 1] != 0)
                && handlers[j] != 0;
            if !ok {
                flags[j] &= !(FLAG_A_ACC | FLAG_B_ACC);
                handlers[j] = self.handler_for(&code[j], flags[j], pin);
                if j > 0 {
                    flags[j - 1] &= !FLAG_DST_ACC;
                    handlers[j - 1] = self.handler_for(&code[j - 1], flags[j - 1], pin);
                }
            }
        }
        for i in 0..code.len() {
            if flags[i] & FLAG_DST_ACC != 0
                && (i + 1 >= code.len() || flags[i + 1] & (FLAG_A_ACC | FLAG_B_ACC) == 0)
            {
                flags[i] &= !FLAG_DST_ACC;
                handlers[i] = self.handler_for(&code[i], flags[i], pin);
            }
        }
        flags
            .into_iter()
            .zip(handlers)
            .map(|(flags, handler)| ResolvedCell { flags, handler })
            .collect()
    }

    #[cfg(test)]
    fn resolve_streaming_for_test(&self, code: &[Instr], pin: &Pinned) -> Vec<ResolvedCell> {
        let mut resolved = Vec::with_capacity(code.len());
        let mut iter = code.iter();
        if let Some(mut prev_ins) = iter.next() {
            let mut prev = self.first_resolution(prev_ins, pin);
            for ins in iter {
                let mut current = self.initial_resolution(ins, pin);
                self.resolve_pair(prev_ins, &mut prev, ins, &mut current, pin);
                resolved.push(prev);
                prev_ins = ins;
                prev = current;
            }
            self.finish_last(prev_ins, &mut prev, pin);
            resolved.push(prev);
        }
        resolved
    }

    /// Build the dispatch cells for one predecoded function.
    ///
    /// Every `call_indirect` type index is passed to
    /// `mark_call_indirect_type` on the way past. The caller marks it directly
    /// in the module-sized canonical-type table, and this pass already reads
    /// every instruction, so no growing side list or second instruction sweep
    /// is needed.
    #[cfg(test)]
    pub(super) fn link(
        &self,
        func: &PredecodedFunction,
        caller_index: usize,
        plan: &mut LinkPlan,
        scratch: &mut LinkScratch,
        mark_call_indirect_type: &mut impl FnMut(u32),
    ) -> LinkedFunction {
        self.link_source(
            func,
            BorrowedCode(&func.code),
            caller_index,
            plan,
            scratch,
            mark_call_indirect_type,
        )
    }

    pub(super) fn link_in_place<F: LinkFunction + ?Sized>(
        &self,
        func: &F,
        caller_index: usize,
        plan: &mut LinkPlan,
        scratch: &mut LinkScratch,
        mark_call_indirect_type: &mut impl FnMut(u32),
    ) -> LinkedFunction {
        self.link_source(
            func,
            InPlaceCode,
            caller_index,
            plan,
            scratch,
            mark_call_indirect_type,
        )
    }

    fn link_source<F: LinkFunction + ?Sized, S: LinkCode>(
        &self,
        func: &F,
        source: S,
        caller_index: usize,
        plan: &mut LinkPlan,
        scratch: &mut LinkScratch,
        mark_call_indirect_type: &mut impl FnMut(u32),
    ) -> LinkedFunction {
        let pin = func.pinned();
        let code_len = func.code_len();

        // Flatten the branch tables; BrTable cells carry a byte offset
        // into the flat buffer until the final address fixup below.
        let table_byte_off = &mut scratch.table_byte_off;
        table_byte_off.clear();
        for table_index in 0..func.br_table_count() {
            let t = func
                .br_table(table_index)
                .expect("predecoded branch-table index");
            table_byte_off.push(plan.br_flat.len() as u64 * 4);
            for &target in t.iter() {
                plan.br_flat.push(target);
            }
        }

        // One cell of slack past the last instruction. Every handler that
        // ends in a dispatch tail prefetches the NEXT cell's handler word
        // at entry, so the last real cell reads eight bytes beyond the
        // instruction stream. A predecoded function always ends in
        // `Return`, which does not prefetch — but that is a predecoder
        // invariant holding up a memory access in six separate backends,
        // and one padding cell removes the dependency. Its handler is the
        // slow stub, so control reaching it exits cleanly rather than
        // jumping through uninitialized memory.
        let off_of = |slot: u64| {
            if slot == u64::MAX {
                0
            } else {
                (slot * 8) as u32
            }
        };
        let cell_start = plan.begin_function(code_len);
        if source.is_in_place() {
            debug_assert_eq!(cell_start, func.code_start());
        }
        // Every linked body contains at least its prefetch pad, so this is
        // always an in-allocation address rather than a one-past pointer.
        let cells_base = unsafe { plan.cells.as_ptr().add(cell_start) as u64 };
        let lf = LinkedFunction {
            #[cfg(test)]
            cell_start,
            cell_base: cells_base,
            cell_len: code_len + 1,
            l0_off: off_of(pin.l0),
            l1_off: off_of(pin.l1),
            fp_pinned: (pin.l0 != u64::MAX && pin.l0_float) || (pin.l1 != u64::MAX && pin.l1_float),
        };
        // Both relocations are applied as each cell is built rather than in
        // a second sweep over the finished block: both buffers have their
        // final allocation from `with_capacity`, and neither is ever grown,
        // so the base addresses here are the ones the handlers will see.
        //
        // A branch target becomes absolute for the same reason the JIT
        // prefers absolute targets: it removes one link from the taken
        // path's dependency chain and lets the target's handler word be
        // loaded at handler entry rather than after the branch resolves —
        // measured -4.80% of CoreMark cycles. A BrTable cell's `b` becomes
        // the absolute base of its slice of the flat table buffer.
        // `LinkPlan::for_functions` reserved the exact module totals before
        // any absolute address is observed. These bases consequently remain
        // stable through every push below and through all later fixups.
        if !plan.in_place {
            debug_assert!(plan.cells.capacity() - plan.cells.len() >= lf.cell_len);
        }
        let br_base = plan.br_flat.as_ptr() as u64;
        if code_len != 0 {
            let mut prev_index = 0usize;
            let mut prev_ins = source.get(plan, cell_start, 0);
            let mut prev = self.first_resolution(&prev_ins, &pin);
            for index in 1..code_len {
                let ins = source.get(plan, cell_start, index);
                let mut current = self.initial_resolution(&ins, &pin);
                self.resolve_pair(&prev_ins, &mut prev, &ins, &mut current, &pin);
                if prev_ins.op == Op::CallIndirect {
                    mark_call_indirect_type(prev_ins.c as u32);
                }
                self.push_cell(
                    plan,
                    cell_start + prev_index,
                    &prev_ins,
                    prev,
                    cells_base,
                    br_base,
                    func,
                    table_byte_off,
                );
                Self::record_call_fixup(
                    plan,
                    func,
                    caller_index,
                    cell_start,
                    prev_index,
                    &prev_ins,
                );
                prev_index = index;
                prev_ins = ins;
                prev = current;
            }
            self.finish_last(&prev_ins, &mut prev, &pin);
            if prev_ins.op == Op::CallIndirect {
                mark_call_indirect_type(prev_ins.c as u32);
            }
            self.push_cell(
                plan,
                cell_start + prev_index,
                &prev_ins,
                prev,
                cells_base,
                br_base,
                func,
                table_byte_off,
            );
            Self::record_call_fixup(plan, func, caller_index, cell_start, prev_index, &prev_ins);
        }

        plan.write_cell(
            cell_start + code_len,
            DCell {
                h: self.slow_stub as u64,
                a: 0,
                b: 0,
                c: 0,
            },
        );
        lf
    }

    #[inline]
    fn record_call_fixup<F: LinkFunction + ?Sized>(
        plan: &mut LinkPlan,
        func: &F,
        caller_index: usize,
        cell_start: usize,
        index: usize,
        ins: &Instr,
    ) {
        if !NATIVE_CALLS {
            return;
        }
        let cell = cell_start + index;
        match ins.op {
            Op::Call if !func.has_exception_handlers_at(index as u32) => {
                plan.call_fixups.push(CallFixup::Direct {
                    cell,
                    caller: caller_index,
                    callee: ins.a as usize,
                    arg_base: ins.b,
                });
            }
            Op::CallIndirect
                if ins.flags & FLAG_A_CONST == 0
                    && ins.c >> 32 == 0
                    && !func.has_exception_handlers_at(index as u32) =>
            {
                plan.call_fixups.push(CallFixup::Indirect {
                    cell,
                    caller: caller_index,
                    table_slot: ins.a,
                    arg_base: ins.b,
                    expected_type: ins.c as u32,
                });
            }
            _ => {}
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn push_cell<F: LinkFunction + ?Sized>(
        &self,
        plan: &mut LinkPlan,
        cell_index: usize,
        ins: &Instr,
        state: ResolvedCell,
        cells_base: u64,
        br_base: u64,
        func: &F,
        table_byte_off: &[u64],
    ) {
        let fl = state.flags;
        let staged = if plan.in_place {
            Some(plan.cells[cell_index])
        } else {
            None
        };
        let mut h = Some(state.handler).filter(|&h| h != 0);
        // A 32-bit host reads a cell's static offset as one machine
        // word, so a wasm offset that does not fit in 32 bits cannot
        // run natively — the handler would silently use the truncated
        // value and turn an out-of-bounds access into an in-bounds one.
        if INTERP_PTR_BYTES == 4 && !offset_fits_word(ins) {
            h = None;
        }
        if let (Some(h), Some(staged)) = (h, staged) {
            if ins.op != Op::BrTable && draft_payload_is_prelinked(&staged) {
                let cell = plan.cell_mut(cell_index);
                cell.h = h as u64;
                if c_is_branch_target(ins.op) {
                    cell.c += cells_base;
                }
                if native_may_exit_slow(ins.op) {
                    plan.record_slow_head_candidate(cell_index, ins.packed_head());
                }
                return;
            }
        }
        let cell = match h {
            Some(h) if ins.op == Op::BrTable => {
                let table = func
                    .br_table(ins.c as usize)
                    .expect("predecoded branch-table index");
                DCell {
                    h: h as u64,
                    a: ins.a * 8,
                    b: br_base + table_byte_off[ins.c as usize],
                    c: (table.len() - 1) as u64,
                }
            }
            Some(h) => {
                let a = if fl & FLAG_A_CONST != 0 {
                    ins.a
                } else {
                    ins.a * 8
                };
                let (b, mut c) = transform_bc(ins, fl);
                if c_is_branch_target(ins.op) {
                    c += cells_base;
                }
                DCell {
                    h: h as u64,
                    a,
                    b,
                    c,
                }
            }
            None => DCell {
                h: self.slow_stub as u64,
                a: ins.a,
                b: ins.b,
                c: ins.c,
            },
        };
        let needs_slow_head = cell.h == self.slow_stub as u64 || native_may_exit_slow(ins.op);
        plan.write_cell(cell_index, cell);
        if needs_slow_head {
            plan.record_slow_head_candidate(cell_index, ins.packed_head());
        }
    }

    /// The entry trampoline as a callable function pointer.
    pub(super) fn entry_fn(&self) -> extern "C" fn(*mut EnterState) {
        unsafe { core::mem::transmute::<usize, extern "C" fn(*mut EnterState)>(self.entry) }
    }
}

#[cfg(test)]
mod tests {
    use super::super::instr::{op_from_index, N_OPS};
    use super::super::layout::native_guard;
    use super::*;
    use crate::collections::{vec, Vec};
    use crate::vm::interpreter::predecode::linker_test_function;

    fn reference_cells(
        engine: &NativeEngine,
        func: &PredecodedFunction,
        resolved: &[ResolvedCell],
        cells_base: u64,
        br_base: u64,
    ) -> Vec<DCell> {
        let mut table_byte_off = Vec::with_capacity(func.br_tables.len());
        let mut flat_len = 0usize;
        for table in &func.br_tables {
            table_byte_off.push(flat_len as u64 * 4);
            flat_len += table.len();
        }

        let mut cells = Vec::with_capacity(func.code.len() + 1);
        for (ins, state) in func.code.iter().zip(resolved) {
            let flags = state.flags;
            let mut handler = Some(state.handler).filter(|&handler| handler != 0);
            if INTERP_PTR_BYTES == 4 && !offset_fits_word(ins) {
                handler = None;
            }
            match handler {
                Some(handler) if ins.op == Op::BrTable => {
                    let table = &func.br_tables[ins.c as usize];
                    cells.push(DCell {
                        h: handler as u64,
                        a: ins.a * 8,
                        b: br_base + table_byte_off[ins.c as usize],
                        c: (table.len() - 1) as u64,
                    });
                }
                Some(handler) => {
                    let a = if flags & FLAG_A_CONST != 0 {
                        ins.a
                    } else {
                        ins.a * 8
                    };
                    let (b, mut c) = transform_bc(ins, flags);
                    if c_is_branch_target(ins.op) {
                        c += cells_base;
                    }
                    cells.push(DCell {
                        h: handler as u64,
                        a,
                        b,
                        c,
                    });
                }
                None => cells.push(DCell {
                    h: engine.slow_stub as u64,
                    a: ins.a,
                    b: ins.b,
                    c: ins.c,
                }),
            }
        }
        cells.push(DCell {
            h: engine.slow_stub as u64,
            a: 0,
            b: 0,
            c: 0,
        });
        cells
    }

    fn assert_reference_equivalence(engine: &NativeEngine, func: &PredecodedFunction) {
        let pin = func.pinned();
        assert_eq!(
            pin,
            select_pinned_reference(&func.code, func.n_locals),
            "predecoded pinned census differs from the full-stream oracle"
        );
        let reference = engine.resolve_reference(&func.code, &pin);
        let streaming = engine.resolve_streaming_for_test(&func.code, &pin);
        assert_eq!(streaming, reference, "resolved flags/handlers differ");

        let mut link_scratch = LinkScratch::default();
        let mut call_indirect_types = Vec::new();
        let mut record_call_indirect_type = |type_index| call_indirect_types.push(type_index);
        let mut plan = LinkPlan::for_functions(core::iter::once(func));
        let linked = engine.link(
            func,
            0,
            &mut plan,
            &mut link_scratch,
            &mut record_call_indirect_type,
        );
        plan.finish_layout();
        plan.finish_slow_heads(engine.slow_stub as u64);

        let expected_flat: Vec<u32> = func
            .br_tables
            .iter()
            .flat_map(|table| table.iter().copied())
            .collect();
        assert_eq!(plan.br_flat, expected_flat, "branch flattening differs");
        let cells = plan.cells(&linked);
        let expected_cells = reference_cells(
            engine,
            func,
            &reference,
            linked.cell_base(),
            plan.br_flat.as_ptr() as u64,
        );
        assert_eq!(cells, expected_cells.as_slice(), "dispatch cells differ");

        let off_of = |slot: u64| {
            if slot == u64::MAX {
                0
            } else {
                (slot * 8) as u32
            }
        };
        assert_eq!(linked.l0_off, off_of(pin.l0));
        assert_eq!(linked.l1_off, off_of(pin.l1));
        assert_eq!(
            linked.fp_pinned,
            (pin.l0 != u64::MAX && pin.l0_float) || (pin.l1 != u64::MAX && pin.l1_float)
        );
        let expected_types: Vec<u32> = func
            .code
            .iter()
            .filter(|ins| ins.op == Op::CallIndirect)
            .map(|ins| ins.c as u32)
            .collect();
        assert_eq!(call_indirect_types, expected_types);
        assert_eq!(cells.len(), func.code.len() + 1);
        assert_eq!(
            cells.last(),
            Some(&DCell {
                h: engine.slow_stub as u64,
                a: 0,
                b: 0,
                c: 0,
            }),
            "prefetch pad changed"
        );

        let mut arena: Vec<DCell> = func.code.iter().copied().map(stage_draft).collect();
        arena.push(stage_draft(Instr::new(Op::Unreachable, 0, 0, 0, 0)));
        let allocation = arena.as_ptr() as usize;
        let expected_heads: Vec<u32> = arena.iter().map(|cell| cell.h as u32).collect();
        let mut in_place = LinkPlan::from_cell_arena(arena, 1, expected_flat.len());
        assert_eq!(in_place.cells.as_ptr() as usize, allocation);
        let mut in_place_scratch = LinkScratch::default();
        let mut in_place_types = Vec::new();
        let mut record_in_place_type = |type_index| in_place_types.push(type_index);
        let in_place_linked = engine.link_in_place(
            func,
            0,
            &mut in_place,
            &mut in_place_scratch,
            &mut record_in_place_type,
        );
        drop(record_in_place_type);
        in_place.finish_layout();
        in_place.finish_slow_heads(engine.slow_stub as u64);

        let normalize = |mut cell: DCell, op: Op, cells_base: u64, br_base: u64| -> DCell {
            if op == Op::BrTable && cell.h != engine.slow_stub as u64 {
                cell.b -= br_base;
            }
            if c_is_branch_target(op) && cell.h != engine.slow_stub as u64 {
                cell.c -= cells_base;
            }
            cell
        };
        let in_place_cells = in_place.cells(&in_place_linked);
        let old_br_base = plan.br_flat.as_ptr() as u64;
        let new_br_base = in_place.br_flat.as_ptr() as u64;
        for (index, ins) in func.code.iter().enumerate() {
            assert_eq!(
                normalize(
                    in_place_cells[index],
                    ins.op,
                    in_place_linked.cell_base(),
                    new_br_base,
                ),
                normalize(cells[index], ins.op, linked.cell_base(), old_br_base),
                "in-place linked cell {index} ({:?})",
                ins.op,
            );
        }
        assert_eq!(in_place_cells.last(), cells.last());
        assert_eq!(in_place.br_flat, plan.br_flat);
        assert_eq!(in_place.call_fixups, plan.call_fixups);
        assert_eq!(in_place_types, call_indirect_types);
        assert_eq!(in_place.heads, expected_heads);
        let fields = |ins: Instr| (ins.op, ins.flags, ins.a, ins.b, ins.c);
        for index in 0..func.code.len() {
            let sparse = in_place
                .restore_slow_instr(index, engine.slow_stub as u64)
                .map(fields);
            let dense = in_place
                .restore_slow_instr_dense_reference(index, engine.slow_stub as u64)
                .map(fields);
            assert_eq!(sparse, dense, "sparse slow head {index}");
            let should_restore = in_place_cells[index].h == engine.slow_stub as u64
                || native_may_exit_slow(func.code[index].op);
            assert_eq!(
                sparse.is_some(),
                should_restore,
                "slow-head membership {index} ({:?})",
                func.code[index].op,
            );
        }
        assert_eq!(in_place_linked.l0_off, linked.l0_off);
        assert_eq!(in_place_linked.l1_off, linked.l1_off);
        assert_eq!(in_place_linked.fp_pinned, linked.fp_pinned);
    }

    #[test]
    fn direct_cell_arena_keeps_one_owner_and_allocation() {
        let instrs = vec![
            Instr::new(Op::I32_Add, FLAG_B_CONST, 3, 41, 7),
            Instr::new(Op::Return, 0, 7, 1, 0),
        ];
        let cells: Vec<DCell> = instrs.iter().copied().map(stage_draft).collect();
        for (cell, expected) in cells.into_iter().zip(instrs) {
            let actual = decode_draft(cell);
            assert_eq!(
                (actual.op, actual.flags, actual.a, actual.b, actual.c),
                (
                    expected.op,
                    expected.flags,
                    expected.a,
                    expected.b,
                    expected.c,
                )
            );
        }
    }

    #[test]
    fn ordinary_native_link_replaces_only_the_draft_head_word() {
        let engine = NativeEngine::new();
        let semantic = Instr::new(Op::I32_Add, 0, 0, 1, 2);
        let func = linker_test_function(vec![semantic], Vec::new(), 3);
        let staged = stage_draft(semantic);
        assert!(draft_payload_is_prelinked(&staged));
        let payload = (staged.a, staged.b, staged.c);
        let mut plan = LinkPlan::from_cell_arena(
            vec![staged, stage_draft(Instr::new(Op::Unreachable, 0, 0, 0, 0))],
            1,
            0,
        );
        let mut scratch = LinkScratch::default();
        let mut mark_indirect_type = |_| {};
        let linked =
            engine.link_in_place(&func, 0, &mut plan, &mut scratch, &mut mark_indirect_type);
        plan.finish_layout();
        let cell = plan.cells(&linked)[0];
        assert_ne!(cell.h, staged.h);
        assert_eq!((cell.a, cell.b, cell.c), payload);
    }

    #[test]
    fn dynamic_native_slow_payloads_round_trip() {
        use Op::*;
        let dynamic = [
            I32_DivS,
            I32_DivU,
            I32_RemS,
            I32_RemU,
            I64_DivS,
            I64_DivU,
            I64_RemS,
            I64_RemU,
            I32_TruncF32S,
            I32_TruncF32U,
            I32_TruncF64S,
            I32_TruncF64U,
            I64_TruncF32S,
            I64_TruncF32U,
            I64_TruncF64S,
            I64_TruncF64U,
        ];
        for op in dynamic {
            for flags in [0, FLAG_A_CONST, FLAG_B_CONST, FLAG_A_CONST | FLAG_B_CONST] {
                let ins = Instr::new(op, flags, 13, 29, 41);
                let a = if flags & FLAG_A_CONST != 0 {
                    ins.a
                } else {
                    ins.a * 8
                };
                let (b, c) = transform_bc(&ins, flags);
                let cell = DCell { h: 1, a, b, c };
                let restored =
                    restore_native_slow_instr(ins.packed_head(), &cell).expect("dynamic slow op");
                assert_eq!(
                    (
                        restored.op,
                        restored.flags,
                        restored.a,
                        restored.b,
                        restored.c
                    ),
                    (ins.op, ins.flags, ins.a, ins.b, ins.c),
                    "{op:?} flags={flags:#x}"
                );
            }
        }

        for op in [MemoryFill, MemoryCopy, MemoryFillCopy] {
            let ins = if op == MemoryFillCopy {
                Instr::new(op, 0, 17, 23, 0)
            } else {
                Instr::new(op, 0, 17, 0, 0)
            };
            let (b, c) = transform_bc(&ins, 0);
            let cell = DCell {
                h: 1,
                a: ins.a * 8,
                b,
                c,
            };
            let restored =
                restore_native_slow_instr(ins.packed_head(), &cell).expect("bulk slow op");
            assert_eq!(
                (restored.op, restored.a, restored.b, restored.c),
                (ins.op, ins.a, ins.b, ins.c)
            );
        }

        let indirect = Instr::new(CallIndirect, 0, 37, 43, 47);
        let linked = DCell {
            h: 1,
            a: 17u64 << 48 | indirect.a * 8,
            b: 13u64 << 48 | 19u64 << 32 | indirect.b * 8,
            c: indirect.c,
        };
        let restored = restore_native_slow_instr(indirect.packed_head(), &linked)
            .expect("call_indirect type payload");
        assert_eq!(
            (restored.op, restored.a, restored.b, restored.c),
            (indirect.op, indirect.a, indirect.b, indirect.c)
        );
    }

    #[test]
    fn delayed_linker_matches_reference_on_acc_edge_cases() {
        let engine = NativeEngine::new();
        let empty = linker_test_function(Vec::new(), Vec::new(), 0);
        assert_reference_equivalence(&engine, &empty);
        let single = linker_test_function(
            vec![Instr::new(Op::I32_Add, FLAG_A_ACC | FLAG_DST_ACC, 0, 1, 2)],
            Vec::new(),
            3,
        );
        assert_reference_equivalence(&engine, &single);

        let code = vec![
            // No predecessor: both incoming hints must be stripped. Its
            // orphan outgoing hint is stripped when the next cell arrives.
            Instr::new(Op::I32_Add, FLAG_A_ACC | FLAG_DST_ACC, 0, 1, 2),
            Instr::new(Op::MovConst, FLAG_A_CONST, 7, 0, 3),
            // A normal producer/consumer chain.
            Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 9, 0, 4),
            Instr::new(Op::I32_Add, FLAG_A_ACC | FLAG_DST_ACC, 4, 1, 5),
            // Permanently slow consumer: strips its hint and the producer's
            // store-skipping hint, then recomputes both handlers.
            Instr::new(Op::TableGet, FLAG_A_ACC | FLAG_SHARED_TABLE, 5, 0, 6),
            // Calls are valid accumulator producers even though their table
            // handler is zero.
            Instr::new(Op::Call, 0, 0, 0, 0),
            Instr::new(Op::I64_Add, FLAG_B_ACC | FLAG_DST_ACC, 0, 6, 7),
            Instr::new(Op::I64_Store, FLAG_B_ACC, 0, 7, 0),
            // A native-guard failure must participate in ACC resolution as
            // the old pass saw it.
            Instr::new(Op::I32_Load, FLAG_ADDR64 | FLAG_DST_ACC, 0, 0, 1),
            Instr::new(Op::I32_Add, FLAG_A_ACC, 1, 2, 3),
            Instr::new(Op::BrTable, 0, 0, 0, 0),
            Instr::new(Op::BrTable, 0, 1, 0, 1),
            Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 1, 0, 0),
            Instr::new(Op::Return, 0, 0, 0, 0),
        ];
        let func = linker_test_function(code, vec![vec![0, 3, 13], vec![2, 5, 8, 13]], 8);
        assert_reference_equivalence(&engine, &func);
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn generated_instruction(rng: &mut Lcg, len: usize) -> Instr {
        let a = rng.below(8);
        let b = rng.below(8);
        let c = rng.below(8);
        let mut ins = match rng.below(16) {
            0 => Instr::new(Op::MovConst, FLAG_A_CONST, rng.next(), 0, c),
            1 => Instr::new(Op::MovSlot, 0, a, 0, c),
            2 => Instr::new(Op::I32_Add, 0, a, b, c),
            3 => Instr::new(Op::I64_Mul, 0, a, b, c),
            4 => Instr::new(Op::F32_Add, 0, a, b, c),
            5 => Instr::new(Op::I32_Load, 0, a, rng.below(256), c),
            6 => Instr::new(Op::I64_Store, 0, a, b, rng.below(256)),
            7 => Instr::new(Op::Select, 0, a, b, (rng.below(8) << 32) | c),
            8 => Instr::new(Op::Br, 0, 0, 0, rng.below(len as u64)),
            9 => Instr::new(Op::BrIf, 0, a, 0, rng.below(len as u64)),
            10 => Instr::new(Op::BrTable, 0, a, 0, 0),
            11 => Instr::new(Op::Call, 0, rng.below(4), b, 0),
            12 => Instr::new(Op::CallIndirect, 0, a, b, rng.below(32)),
            13 => Instr::new(Op::GlobalGet, 0, rng.below(4), 0, c),
            14 => Instr::new(Op::TableGet, 0, a, rng.below(2), c),
            _ => {
                let dst1 = rng.below(8);
                let mut dst2 = rng.below(7);
                if dst2 >= dst1 {
                    dst2 += 1;
                }
                Instr::new(Op::MovPair, 0, a, b, (dst1 << 32) | dst2)
            }
        };

        if rng.next() & 1 != 0 {
            match ins.op {
                Op::I32_Load => {
                    ins.flags |= FLAG_FUSED;
                    ins.c = (rng.below(8) << 32) | c;
                }
                Op::I64_Store => {
                    ins.flags |= FLAG_FUSED;
                    ins.c = (rng.below(8) << 32) | (ins.c & 0xffff_ffff);
                }
                _ => {}
            }
        }

        match rng.below(4) {
            1 => ins.flags |= FLAG_A_ACC,
            2 => ins.flags |= FLAG_B_ACC,
            3 => ins.flags |= FLAG_A_ACC | FLAG_B_ACC,
            _ => {}
        }
        if rng.next() & 1 != 0 {
            ins.flags |= FLAG_DST_ACC;
        }
        if rng.below(13) == 0 {
            ins.flags |= FLAG_ADDR64;
        }
        if rng.below(17) == 0 {
            ins.flags |= FLAG_SHARED_GLOBAL;
        }
        ins
    }

    #[test]
    fn delayed_linker_matches_reference_on_generated_streams() {
        let engine = NativeEngine::new();
        let mut rng = Lcg(0x05ee_d5ee_dd15_ca11);
        for case in 0..256usize {
            let len = case % 64 + 1;
            let mut code = Vec::with_capacity(len + 1);
            for _ in 0..len {
                code.push(generated_instruction(&mut rng, len + 1));
            }
            code.push(Instr::new(Op::Return, 0, 0, 0, 0));
            let last = code.len() as u32 - 1;
            let func = linker_test_function(code, vec![vec![0, last / 2, last]], 8);
            assert_reference_equivalence(&engine, &func);
        }
    }

    /// Cross every opcode with every pre-existing flag combination, every
    /// pinned-register domain/occupancy shape, and payloads which exercise
    /// each memory-guard outcome. The exact handler address must remain the
    /// same when the old payload test is replaced by its cached flag.
    #[test]
    fn cached_native_guard_matches_uncached_handler_addresses() {
        const L0: u64 = 11;
        const L1: u64 = 22;
        const OTHER: u64 = 33;
        let pins = [
            Pinned::NONE,
            Pinned {
                l0: L0,
                l1: L1,
                l0_float: false,
                l1_float: false,
            },
            Pinned {
                l0: L0,
                l1: L1,
                l0_float: true,
                l1_float: true,
            },
            Pinned {
                l0: L0,
                l1: L1,
                l0_float: false,
                l1_float: true,
            },
            Pinned {
                l0: L0,
                l1: L1,
                l0_float: true,
                l1_float: false,
            },
            Pinned {
                l0: L0,
                l1: u64::MAX,
                l0_float: false,
                l1_float: false,
            },
            Pinned {
                l0: u64::MAX,
                l1: L1,
                l0_float: false,
                l1_float: true,
            },
        ];
        let payloads = [
            // Unpinned, L0/L1-aliasing, zero-index, nonzero-index, packed
            // memory-index, and packed static-offset cases.
            (OTHER, OTHER + 1, OTHER + 2),
            (L0, L1, L0),
            (L1, L0, L1),
            (OTHER, 0, 0),
            (OTHER, 1, 1),
            (OTHER, 1u64 << 32, 1u64 << 32),
            (OTHER, 1u64 << 48, 1u64 << 48),
            (OTHER, (1u64 << 48) | 7, (1u64 << 48) | 9),
        ];

        let engine = NativeEngine::new();
        for op_index in 0..N_OPS {
            let op = op_from_index(op_index);
            for (pin_index, pin) in pins.iter().enumerate() {
                // Bits 0..=8 are the complete flag space before the cache
                // bit was introduced.
                for flags in 0..(FLAG_NO_NATIVE) {
                    for &(a, b, c) in &payloads {
                        let ins = Instr::new(op, flags, a, b, c);
                        let cached = flags
                            | if native_guard(&ins) {
                                0
                            } else {
                                FLAG_NO_NATIVE
                            };
                        assert_eq!(
                            engine.handler_for(&ins, cached, pin),
                            engine.handler_for_uncached_reference(&ins, flags, pin),
                            "op={op:?} flags={flags:#x} pin={pin_index} a={a:#x} b={b:#x} c={c:#x}"
                        );
                    }
                }
            }
        }
    }
}
