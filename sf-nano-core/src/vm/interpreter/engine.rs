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

use tracked_alloc::{boxed::Box, retype_vec};

use crate::collections::Vec;

use super::instr::{operand_is_f32, operand_is_float, result_is_f32, result_is_float};
use super::instr::{
    Instr, Op, FLAG_ADDR64, FLAG_A_ACC, FLAG_A_CONST, FLAG_B_ACC, FLAG_B_CONST, FLAG_DST_ACC,
    FLAG_FUSED, FLAG_NO_NATIVE, FLAG_SHARED_GLOBAL, FLAG_SHARED_TABLE,
};
use super::layout::slot_fields;
use super::layout::{
    c_is_branch_target, family, op_slot, total_slots, transform_bc, writes_acc, Fam, Pinned,
};
use super::predecode::LinkFunction;
#[cfg(test)]
use super::predecode::PredecodedFunction;

// Build-time facts about the generated engine: which operand classes it
// was built with, and how many packed handler slots the table holds.
include!(concat!(env!("OUT_DIR"), "/interp_engine_cfg.rs"));

/// Target capabilities used by the independent incremental test oracle.
/// Production reads the generated constants directly in [`PinCensus`].
#[cfg(test)]
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
    /// Target-specific appended handler offsets.
    #[cfg(sf_has_apple_arm64_interp_tuning)]
    static sf_interp_supers: u32;
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

/// One 32-byte dispatch cell: [`Instr`] with the leading word replaced by
/// the handler address.
#[repr(C, align(32))]
#[cfg_attr(test, derive(Clone, Copy, Debug, PartialEq, Eq))]
pub(super) struct DCell {
    pub h: u64,
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

// Stage A owns exactly the same allocation layout as stage B. The explicit
// `Instr::head_pad` makes the leading eight bytes fully initialized before
// ownership is transferred; these assertions make a future field/layout
// drift a compile-time failure rather than allocator or aliasing UB.
const _: () = {
    assert!(core::mem::size_of::<Instr>() == core::mem::size_of::<DCell>());
    assert!(core::mem::align_of::<Instr>() == core::mem::align_of::<DCell>());
    assert!(!core::mem::needs_drop::<Instr>());
    assert!(!core::mem::needs_drop::<DCell>());
    assert!(core::mem::offset_of!(Instr, a) == core::mem::offset_of!(DCell, a));
    assert!(core::mem::offset_of!(Instr, b) == core::mem::offset_of!(DCell, b));
    assert!(core::mem::offset_of!(Instr, c) == core::mem::offset_of!(DCell, c));
};

/// Transfer the module-wide stage-A allocation into its stage-B element
/// type without allocating or copying any instruction payloads.
fn into_dispatch_cells(instrs: Vec<Instr>) -> Vec<DCell> {
    // SAFETY: the compile-time assertions above prove identical element
    // size/alignment and a field-for-field payload layout. `Instr` is Copy
    // and has no drop glue; its explicit head padding is initialized. The
    // tracking-aware transfer retypes the one live allocation record in one
    // operation. There is one owner throughout and no live reference crosses
    // the transfer.
    unsafe { retype_vec(instrs) }
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

/// Module-wide storage planned before linking begins.
///
/// The exact cell and branch-table capacities are computed from all
/// predecoded functions. Neither backing vector can therefore reallocate while
/// absolute pointers are installed into dispatch cells. Besides replacing two
/// allocations per function with two per module, the plan carries one compact
/// list of native call sites for the deferred cross-function fixup.
pub(super) struct LinkPlan {
    cells: Vec<DCell>,
    heads: Vec<u32>,
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
            heads: Vec::new(),
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

    pub(super) fn from_instr_arena(
        instrs: Vec<Instr>,
        function_count: usize,
        planned_br_entries: usize,
    ) -> Self {
        let heads = instrs.iter().copied().map(Instr::packed_head).collect();
        let planned_cells = instrs.len();
        Self {
            cells: into_dispatch_cells(instrs),
            heads,
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
        let cell = &self.cells[index];
        Instr::from_packed_head(self.heads[index], cell.a, cell.b, cell.c)
    }

    /// Verify that the precomputed storage plan and the completed link agree.
    pub(super) fn finish_layout(&self) {
        debug_assert_eq!(self.cells.len(), self.planned_cells);
        debug_assert_eq!(self.br_flat.len(), self.planned_br_entries);
        debug_assert_eq!(self.linked_cells, self.planned_cells);
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

    /// Recover the raw stage-A operands of a natively linked
    /// `CallIndirect` without consulting the dense instruction-head arena.
    /// Only the generated indirect-call handler uses this payload layout.
    pub(super) fn native_call_indirect_payload(
        &self,
        index: usize,
        callindirect_handler: u64,
    ) -> Option<(u64, u64, u32)> {
        let cell = self.cells.get(index)?;
        if cell.h != callindirect_handler {
            return None;
        }
        debug_assert_eq!(cell.a & 7, 0);
        debug_assert_eq!(cell.b & 7, 0);
        Some((
            (cell.a & ((1u64 << 48) - 1)) / 8,
            (cell.b & 0xffff_ffff) / 8,
            cell.c as u32,
        ))
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
        let head = *self.heads.get(index)?;
        if cell.h == slow_stub {
            return Some(Instr::from_packed_head(head, cell.a, cell.b, cell.c));
        }
        restore_native_slow_instr(head, cell)
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

#[cfg(sf_has_apple_arm64_interp_supers)]
#[derive(Clone, Copy)]
struct SuperCell {
    op: Op,
    flags: u16,
    handler: u64,
    a: u64,
    b: u64,
    c: u64,
    index: usize,
}

/// Four-cell, allocation-free look-behind for the small Apple ARM64
/// superhandler bank.
///
/// The linked arena already retains every operand word. Only the resolved
/// op/flag head is transient, so the rolling state is four `u32`s instead of
/// four complete instruction and resolution records. Candidate opcodes read
/// the preceding linked cells directly; all other cells pay one packed-head
/// store and one opcode match.
#[cfg(sf_has_apple_arm64_interp_supers)]
struct SuperWindow {
    heads: [u32; 4],
    cell_start: usize,
    covered_until: usize,
}

#[cfg(sf_has_apple_arm64_interp_supers)]
impl SuperWindow {
    fn new(cell_start: usize) -> Self {
        Self {
            heads: [0; 4],
            cell_start,
            covered_until: cell_start,
        }
    }

    fn cell(&self, plan: &LinkPlan, index: usize) -> SuperCell {
        let head = self.heads[index & 3];
        let cell = &plan.cells[index];
        SuperCell {
            op: super::instr::op_from_index((head & 0xffff) as usize),
            flags: (head >> 16) as u16,
            handler: cell.h,
            a: cell.a,
            b: cell.b,
            c: cell.c,
            index,
        }
    }

    fn push(
        &mut self,
        engine: &NativeEngine,
        plan: &mut LinkPlan,
        pin: &Pinned,
        index: usize,
        op: Op,
        flags: u16,
    ) {
        self.heads[index & 3] = op as u32 | ((flags as u32) << 16);

        let candidate = match op {
            Op::BrIf => self.match_list_step(plan, pin, index, engine.slow_stub as u64),
            Op::I32_Add => self.match_add4(plan, pin, index, engine.slow_stub as u64),
            Op::I32_BrNe => self.match_add_brne(plan, pin, index, engine.slow_stub as u64),
            Op::I32_And => self.match_shru_and(plan, pin, index, engine.slow_stub as u64),
            Op::I32_BrEq => self.match_and_breq(plan, pin, index, engine.slow_stub as u64),
            _ => None,
        };
        let Some((slot, start, cells)) = candidate else {
            return;
        };
        if start < self.covered_until {
            return;
        }
        let Some(handler) = engine.super_handler_at(slot) else {
            return;
        };
        plan.cell_mut(start).h = handler as u64;
        self.covered_until = start + cells;
    }

    fn all_native(cells: &[SuperCell], slow_stub: u64) -> bool {
        cells.iter().all(|cell| cell.handler != slow_stub)
    }

    fn is_pin_offset(offset: u64, slot: u64) -> bool {
        slot != u64::MAX && offset == slot * 8
    }

    fn plain_int_slot(offset: u64, pin: &Pinned) -> bool {
        !Self::is_pin_offset(offset, pin.l0) && !Self::is_pin_offset(offset, pin.l1)
    }

    fn int_pins(pin: &Pinned) -> bool {
        pin.l0 != u64::MAX && pin.l1 != u64::MAX && !pin.l0_float && !pin.l1_float
    }

    fn match_list_step(
        &self,
        plan: &LinkPlan,
        pin: &Pinned,
        index: usize,
        slow_stub: u64,
    ) -> Option<(usize, usize, usize)> {
        if index < self.cell_start + 3 || !Self::int_pins(pin) {
            return None;
        }
        let cells = [
            self.cell(plan, index - 3),
            self.cell(plan, index - 2),
            self.cell(plan, index - 1),
            self.cell(plan, index),
        ];
        let [mov, load, store, branch] = cells;
        let dst1 = mov.c >> 32;
        let dst2 = mov.c & 0xffff_ffff;
        if !Self::all_native(&cells, slow_stub)
            || (mov.op, mov.flags) != (Op::MovPair, 0)
            || (load.op, load.flags) != (Op::I32_Load, FLAG_A_ACC)
            || (store.op, store.flags) != (Op::I32_Store, 0)
            || (branch.op, branch.flags) != (Op::BrIf, 0)
            || !Self::is_pin_offset(mov.a, pin.l0)
            || !Self::is_pin_offset(dst2, pin.l0)
            || !Self::plain_int_slot(mov.b, pin)
            || !Self::plain_int_slot(dst1, pin)
            || mov.b == dst1
            || load.a != dst2
            || load.b != 0
            || load.c != mov.b
            || store.a != load.a
            || store.b != dst1
            || store.c != load.b
            || branch.a != load.c
        {
            return None;
        }
        Some((INTERP_SUPER_MOVPAIR_LOAD_STORE_BRIF, mov.index, 4))
    }

    fn match_add4(
        &self,
        plan: &LinkPlan,
        pin: &Pinned,
        index: usize,
        slow_stub: u64,
    ) -> Option<(usize, usize, usize)> {
        if index < self.cell_start + 3 || !Self::int_pins(pin) {
            return None;
        }
        let cells = [
            self.cell(plan, index - 3),
            self.cell(plan, index - 2),
            self.cell(plan, index - 1),
            self.cell(plan, index),
        ];
        let [a0, a1, a2, a3] = cells;
        if !Self::all_native(&cells, slow_stub)
            || (a0.op, a0.flags) != (Op::I32_Add, FLAG_A_ACC | FLAG_DST_ACC)
            || (a1.op, a1.flags) != (Op::I32_Add, FLAG_B_ACC)
            || (a2.op, a2.flags) != (Op::I32_Add, 0)
            || (a3.op, a3.flags) != (Op::I32_Add, FLAG_B_CONST)
            || a0.a != a0.c
            || !Self::plain_int_slot(a0.a, pin)
            || !Self::is_pin_offset(a0.b, pin.l1)
            || a1.b != a0.c
            || !Self::plain_int_slot(a1.a, pin)
            || !Self::is_pin_offset(a1.c, pin.l1)
            || !Self::is_pin_offset(a2.a, pin.l0)
            || !Self::is_pin_offset(a2.c, pin.l0)
            || !Self::plain_int_slot(a2.b, pin)
            || a3.a != a3.c
            || !Self::plain_int_slot(a3.a, pin)
            || a3.b != 4
        {
            return None;
        }
        Some((INTERP_SUPER_ADD4, a0.index, 4))
    }

    fn match_add_brne(
        &self,
        plan: &LinkPlan,
        pin: &Pinned,
        index: usize,
        slow_stub: u64,
    ) -> Option<(usize, usize, usize)> {
        if index < self.cell_start + 1 {
            return None;
        }
        let cells = [self.cell(plan, index - 1), self.cell(plan, index)];
        let [add, branch] = cells;
        if !Self::all_native(&cells, slow_stub)
            || (add.op, add.flags) != (Op::I32_Add, FLAG_B_CONST)
            || (branch.op, branch.flags) != (Op::I32_BrNe, FLAG_B_ACC)
            || add.a != add.c
            || branch.b != add.c
            || !Self::plain_int_slot(add.a, pin)
            || !Self::plain_int_slot(branch.a, pin)
        {
            return None;
        }
        Some((INTERP_SUPER_ADD_BRNE, add.index, 2))
    }

    fn match_shru_and(
        &self,
        plan: &LinkPlan,
        pin: &Pinned,
        index: usize,
        slow_stub: u64,
    ) -> Option<(usize, usize, usize)> {
        if index < self.cell_start + 1 {
            return None;
        }
        let cells = [self.cell(plan, index - 1), self.cell(plan, index)];
        let [shift, and] = cells;
        if !Self::all_native(&cells, slow_stub)
            || (shift.op, shift.flags) != (Op::I32_ShrU, FLAG_A_ACC | FLAG_B_CONST | FLAG_DST_ACC)
            || (and.op, and.flags) != (Op::I32_And, FLAG_A_ACC | FLAG_B_CONST)
            || and.a != shift.c
            || !Self::plain_int_slot(shift.a, pin)
            || !Self::plain_int_slot(shift.c, pin)
            || !Self::plain_int_slot(and.c, pin)
        {
            return None;
        }
        Some((INTERP_SUPER_SHRU_AND, shift.index, 2))
    }

    fn match_and_breq(
        &self,
        plan: &LinkPlan,
        pin: &Pinned,
        index: usize,
        slow_stub: u64,
    ) -> Option<(usize, usize, usize)> {
        if index < self.cell_start + 1 {
            return None;
        }
        let cells = [self.cell(plan, index - 1), self.cell(plan, index)];
        let [and, branch] = cells;
        if !Self::all_native(&cells, slow_stub)
            || (and.op, and.flags) != (Op::I32_And, FLAG_B_CONST)
            || (branch.op, branch.flags) != (Op::I32_BrEq, FLAG_A_ACC | FLAG_B_CONST)
            || branch.a != and.c
            || !Self::plain_int_slot(and.a, pin)
            || !Self::plain_int_slot(and.c, pin)
        {
            return None;
        }
        Some((INTERP_SUPER_AND_BREQ, and.index, 2))
    }
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

/// What [`PinCensus::select`] records about one frame slot.
///
/// One array rather than four parallel ones: the pass touches every field
/// of a slot together, and four arrays put each access on its own cache
/// line.
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

/// Reusable storage for the one exact pinned-local pass over a finished
/// instruction stream.
///
/// Translation mutates, fuses, and removes instructions heavily. Waiting
/// until the stream is final makes each surviving cell contribute exactly
/// once and keeps this storage module-scoped rather than allocating it for
/// every function.
#[derive(Default)]
pub(super) struct PinCensus {
    slots: Vec<SlotStat>,
}

impl PinCensus {
    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.slots.capacity()
    }

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
    pub(super) fn select(&mut self, code: &[Instr], n_locals: u32) -> Pinned {
        let n = n_locals as u64;
        self.slots.clear();
        if n == 0 {
            return Pinned::NONE;
        }
        self.slots.resize(n_locals as usize, SlotStat::default());
        for ins in code.iter() {
            let (a_s, b_s, c_d) = slot_fields(ins.op);
            if a_s && ins.flags & FLAG_A_CONST == 0 && ins.a < n {
                let slot = &mut self.slots[ins.a as usize];
                slot.count += 1;
                if operand_is_float(ins.op, false) {
                    slot.rdom |= 2;
                    slot.f32dom |= operand_is_f32(ins.op, false);
                } else {
                    slot.rdom |= 1;
                }
            }
            if b_s && ins.flags & FLAG_B_CONST == 0 && ins.b < n {
                let slot = &mut self.slots[ins.b as usize];
                slot.count += 1;
                if operand_is_float(ins.op, true) {
                    slot.rdom |= 2;
                    slot.f32dom |= operand_is_f32(ins.op, true);
                } else {
                    slot.rdom |= 1;
                }
            }
            if c_d && ins.c < n {
                let slot = &mut self.slots[ins.c as usize];
                slot.count += 1;
                if result_is_float(ins.op) {
                    slot.wdom |= 2;
                    slot.f32dom |= result_is_f32(ins.op);
                } else {
                    slot.wdom |= 1;
                }
            }
            if matches!(ins.op, Op::I32_SubBrIf | Op::I64_SubBrIf) && ins.a < n {
                // This control-shaped cell is also an in-place integer write
                // to `a`. Count both halves of the read/modify/write so pin
                // selection matches the unfused subtraction it replaces.
                let slot = &mut self.slots[ins.a as usize];
                slot.count += 1;
                slot.wdom |= 1;
            }
            if ins.op == Op::Select {
                let dslot = ins.c & 0xffff_ffff;
                if dslot < n {
                    self.slots[dslot as usize].wdom |= 1;
                }
            }
            if ins.op == Op::MovPair {
                // `slot_fields` cannot describe two packed destinations, so
                // account for both here.
                for dslot in [ins.c >> 32, ins.c & 0xffff_ffff] {
                    if dslot < n {
                        let slot = &mut self.slots[dslot as usize];
                        slot.count += 1;
                        slot.wdom |= 1;
                    }
                }
            }
        }
        // Whether slot `i` could live in the float register file at all.
        let float_ok =
            |i: usize| INTERP_HAS_FLOAT_REGS && (INTERP_FLOAT_PIN_F32 || !self.slots[i].f32dom);
        let mut best = (usize::MAX, 0u32);
        let mut second = (usize::MAX, 0u32);
        for (i, stat) in self.slots.iter().enumerate() {
            let (c, wdom) = (stat.count, stat.wdom);
            // A slot is pinnable only when one register file can stay
            // authoritative for it.
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
        // float writer, or with no writer at all, only float readers.
        let mode = |i: usize| {
            float_ok(i)
                && (self.slots[i].wdom == 2 || (self.slots[i].wdom == 0 && self.slots[i].rdom == 2))
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
}

/// Fresh-storage wrapper retained for differential tests. Production calls
/// [`PinCensus::select`] on module-scoped reusable storage.
#[cfg(test)]
pub(super) fn select_pinned_reference(code: &[Instr], n_locals: u32) -> Pinned {
    PinCensus::default().select(code, n_locals)
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
    #[cfg(sf_has_apple_arm64_interp_tuning)]
    supers: *const u32,
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
            #[cfg(sf_has_apple_arm64_interp_tuning)]
            supers: unsafe { &sf_interp_supers as *const u32 },
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

    #[cfg(sf_has_apple_arm64_interp_tuning)]
    fn super_handler_at(&self, slot: usize) -> Option<usize> {
        debug_assert!(slot < INTERP_SUPER_HANDLER_SLOTS);
        let off = unsafe { *self.supers.add(slot) };
        (off != 0).then_some(self.base + off as usize)
    }

    #[cfg(all(test, sf_has_apple_arm64_interp_tuning))]
    pub(super) fn is_backedge_branch_handler(&self, op: Op, handler: usize) -> bool {
        let base = if op == Op::BrIf {
            INTERP_SUPER_BACKEDGE_BRIF_REG_BASE
        } else if op == Op::BrIfNot {
            INTERP_SUPER_BACKEDGE_BRIFNOT_REG_BASE
        } else {
            return false;
        };
        (0..3).any(|index| self.super_handler_at(base + index) == Some(handler))
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

    /// Replace a pinned-register-source conditional branch with its appended
    /// Apple ARM64 handler only when the still-raw target index names the same
    /// or an earlier cell. This runs after accumulator resolution, so the
    /// class agrees with the final flags installed in the dispatch cell.
    #[cfg(sf_has_apple_arm64_interp_tuning)]
    fn backedge_branch_handler(
        &self,
        ins: &Instr,
        index: usize,
        ordinary: usize,
        flags: u16,
        pin: &Pinned,
    ) -> usize {
        if ordinary == 0 || !matches!(ins.op, Op::BrIf | Op::BrIfNot) || ins.c > index as u64 {
            return ordinary;
        }
        // Conditional branches vary only over source A. Classify that one
        // operand directly instead of re-entering the generic handler-table
        // indexer during module linking. Keep the same Const > pinned > Acc
        // precedence as `op_slot`.
        let register_index = if flags & FLAG_A_CONST != 0 {
            return ordinary;
        } else if ins.a == pin.l0 && !pin.l0_float {
            1
        } else if ins.a == pin.l1 && !pin.l1_float {
            2
        } else if flags & FLAG_A_ACC != 0 {
            // Accumulator branches are already dependency-optimal in the
            // ordinary bank. Keeping them there also avoids perturbing the
            // common GlobalGet -> BrIf countdown shape.
            return ordinary;
        } else {
            return ordinary;
        };
        let base = if ins.op == Op::BrIf {
            INTERP_SUPER_BACKEDGE_BRIF_REG_BASE
        } else {
            INTERP_SUPER_BACKEDGE_BRIFNOT_REG_BASE
        };
        self.super_handler_at(base + register_index)
            .unwrap_or(ordinary)
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
    /// Every `call_indirect` type index is appended to `call_indirect_types`
    /// on the way past. The caller needs them to number the canonical type
    /// classes for the native indirect-call check, and this pass already
    /// reads every instruction — collecting them separately cost a second
    /// sweep of the whole module's instruction stream.
    #[cfg(test)]
    pub(super) fn link(
        &self,
        func: &PredecodedFunction,
        caller_index: usize,
        plan: &mut LinkPlan,
        scratch: &mut LinkScratch,
        call_indirect_types: &mut Vec<u32>,
    ) -> LinkedFunction {
        self.link_source(
            func,
            BorrowedCode(&func.code),
            caller_index,
            plan,
            scratch,
            call_indirect_types,
        )
    }

    pub(super) fn link_in_place<F: LinkFunction + ?Sized>(
        &self,
        func: &F,
        caller_index: usize,
        plan: &mut LinkPlan,
        scratch: &mut LinkScratch,
        call_indirect_types: &mut Vec<u32>,
    ) -> LinkedFunction {
        self.link_source(
            func,
            InPlaceCode,
            caller_index,
            plan,
            scratch,
            call_indirect_types,
        )
    }

    fn link_source<F: LinkFunction + ?Sized, S: LinkCode>(
        &self,
        func: &F,
        source: S,
        caller_index: usize,
        plan: &mut LinkPlan,
        scratch: &mut LinkScratch,
        call_indirect_types: &mut Vec<u32>,
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
        #[cfg(sf_has_apple_arm64_interp_supers)]
        let mut super_window = SuperWindow::new(cell_start);
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
                    call_indirect_types.push(prev_ins.c as u32);
                }
                self.push_cell(
                    plan,
                    cell_start + prev_index,
                    #[cfg(sf_has_apple_arm64_interp_tuning)]
                    prev_index,
                    &prev_ins,
                    prev,
                    #[cfg(sf_has_apple_arm64_interp_tuning)]
                    &pin,
                    cells_base,
                    br_base,
                    func,
                    table_byte_off,
                );
                #[cfg(sf_has_apple_arm64_interp_supers)]
                super_window.push(
                    self,
                    plan,
                    &pin,
                    cell_start + prev_index,
                    prev_ins.op,
                    prev.flags,
                );
                if call_fixup_op(prev_ins.op) {
                    Self::record_call_fixup(
                        plan,
                        func,
                        caller_index,
                        cell_start,
                        prev_index,
                        &prev_ins,
                    );
                }
                prev_index = index;
                prev_ins = ins;
                prev = current;
            }
            self.finish_last(&prev_ins, &mut prev, &pin);
            if prev_ins.op == Op::CallIndirect {
                call_indirect_types.push(prev_ins.c as u32);
            }
            self.push_cell(
                plan,
                cell_start + prev_index,
                #[cfg(sf_has_apple_arm64_interp_tuning)]
                prev_index,
                &prev_ins,
                prev,
                #[cfg(sf_has_apple_arm64_interp_tuning)]
                &pin,
                cells_base,
                br_base,
                func,
                table_byte_off,
            );
            #[cfg(sf_has_apple_arm64_interp_supers)]
            super_window.push(
                self,
                plan,
                &pin,
                cell_start + prev_index,
                prev_ins.op,
                prev.flags,
            );
            if call_fixup_op(prev_ins.op) {
                Self::record_call_fixup(
                    plan,
                    func,
                    caller_index,
                    cell_start,
                    prev_index,
                    &prev_ins,
                );
            }
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

    // This is the per-cell body of the link loop. Leaving the inliner to its
    // size heuristic turned the whole routine into one out-of-line call per
    // instruction when the rare Apple backedge case was added. Keep the
    // common cell materialization in the caller; the uncommon selector below
    // remains independently outlined.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn push_cell<F: LinkFunction + ?Sized>(
        &self,
        plan: &mut LinkPlan,
        cell_index: usize,
        #[cfg(sf_has_apple_arm64_interp_tuning)] local_index: usize,
        ins: &Instr,
        state: ResolvedCell,
        #[cfg(sf_has_apple_arm64_interp_tuning)] pin: &Pinned,
        cells_base: u64,
        br_base: u64,
        func: &F,
        table_byte_off: &[u64],
    ) {
        let fl = state.flags;
        let mut h = Some(state.handler).filter(|&h| h != 0);
        // A 32-bit host reads a cell's static offset as one machine
        // word, so a wasm offset that does not fit in 32 bits cannot
        // run natively — the handler would silently use the truncated
        // value and turn an out-of-bounds access into an in-bounds one.
        if INTERP_PTR_BYTES == 4 && !offset_fits_word(ins) {
            h = None;
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
                #[cfg(sf_has_apple_arm64_interp_tuning)]
                let mut h = h;
                let a = if fl & FLAG_A_CONST != 0 {
                    ins.a
                } else {
                    ins.a * 8
                };
                let (b, mut c) = transform_bc(ins, fl);
                if c_is_branch_target(ins.op) {
                    #[cfg(sf_has_apple_arm64_interp_tuning)]
                    if matches!(ins.op, Op::BrIf | Op::BrIfNot) && ins.c <= local_index as u64 {
                        h = self.backedge_branch_handler(ins, local_index, h, fl, pin);
                    }
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
        plan.write_cell(cell_index, cell);
    }

    /// The entry trampoline as a callable function pointer.
    pub(super) fn entry_fn(&self) -> extern "C" fn(*mut EnterState) {
        unsafe { core::mem::transmute::<usize, extern "C" fn(*mut EnterState)>(self.entry) }
    }
}

/// Whether a cell can possibly contribute a deferred cross-function fixup.
/// Keep this tiny caller-side gate separate from the metadata-heavy outlined
/// recorder: ordinary cells dominate every real module and must never enter
/// that function merely to fail its opcode match.
#[inline(always)]
const fn call_fixup_op(op: Op) -> bool {
    matches!(op, Op::Call | Op::CallIndirect)
}

#[cfg(test)]
mod tests {
    use super::super::instr::{op_from_index, N_OPS};
    use super::super::layout::native_guard;
    use super::*;
    use crate::collections::{vec, Vec};
    use crate::vm::interpreter::predecode::linker_test_function;

    #[cfg(sf_has_apple_arm64_interp_supers)]
    struct ForcedPinnedLinkFunction {
        code_len: usize,
        pin: Pinned,
    }

    #[cfg(sf_has_apple_arm64_interp_supers)]
    impl LinkFunction for ForcedPinnedLinkFunction {
        fn code_start(&self) -> usize {
            0
        }

        fn code_len(&self) -> usize {
            self.code_len
        }

        fn pinned(&self) -> Pinned {
            self.pin
        }

        fn br_table_count(&self) -> usize {
            0
        }

        fn br_table(&self, _index: usize) -> Option<&[u32]> {
            None
        }

        fn has_exception_handlers_at(&self, _pc: u32) -> bool {
            false
        }
    }

    /// Link `code` after a separately linked function, returning its final
    /// dispatch cells. The prefix makes the tested function start at cell 3,
    /// so SuperWindow's absolute ring indices exercise a non-zero boundary.
    #[cfg(sf_has_apple_arm64_interp_supers)]
    fn link_super_test_cells(
        engine: &NativeEngine,
        prefix: &[Instr],
        code: &[Instr],
        pin: Pinned,
    ) -> Vec<DCell> {
        let planned_cells = prefix.len() + 1 + code.len() + 1;
        let mut plan = LinkPlan {
            cells: Vec::with_capacity(planned_cells),
            heads: Vec::new(),
            br_flat: Vec::new(),
            call_fixups: Vec::with_capacity(2),
            planned_cells,
            planned_br_entries: 0,
            linked_cells: 0,
            in_place: false,
        };
        let mut scratch = LinkScratch::default();
        let mut indirect_types = Vec::new();
        let prefix_func = ForcedPinnedLinkFunction {
            code_len: prefix.len(),
            pin,
        };
        let _ = engine.link_source(
            &prefix_func,
            BorrowedCode(prefix),
            0,
            &mut plan,
            &mut scratch,
            &mut indirect_types,
        );
        let func = ForcedPinnedLinkFunction {
            code_len: code.len(),
            pin,
        };
        let linked = engine.link_source(
            &func,
            BorrowedCode(code),
            1,
            &mut plan,
            &mut scratch,
            &mut indirect_types,
        );
        plan.finish_layout();
        plan.cells(&linked).to_vec().into()
    }

    #[cfg(sf_has_apple_arm64_interp_supers)]
    #[test]
    fn apple_super_fusions_select_linked_handlers_at_nonzero_cell_start() {
        let engine = NativeEngine::new();
        let pin = Pinned {
            l0: 0,
            l1: 1,
            l0_float: false,
            l1_float: false,
        };
        let prefix = [
            Instr::new(Op::MovConst, FLAG_A_CONST, 7, 0, 6),
            Instr::new(Op::Return, 0, 0, 0, 0),
        ];
        let cases = [
            (
                "ordered list step",
                vec![
                    Instr::new(Op::MovPair, 0, 0, 2, (3 << 32) | 0),
                    Instr::new(Op::I32_Load, FLAG_A_ACC, 0, 0, 2),
                    Instr::new(Op::I32_Store, 0, 0, 3, 0),
                    Instr::new(Op::BrIf, 0, 2, 0, 0),
                ],
                0,
                INTERP_SUPER_MOVPAIR_LOAD_STORE_BRIF,
            ),
            (
                "four dependent adds",
                vec![
                    Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 9, 0, 2),
                    Instr::new(Op::I32_Add, FLAG_A_ACC | FLAG_DST_ACC, 2, 1, 2),
                    Instr::new(Op::I32_Add, FLAG_B_ACC, 3, 2, 1),
                    Instr::new(Op::I32_Add, 0, 0, 4, 0),
                    Instr::new(Op::I32_Add, FLAG_B_CONST, 5, 4, 5),
                ],
                1,
                INTERP_SUPER_ADD4,
            ),
            (
                "add then not-equal branch",
                vec![
                    Instr::new(Op::I32_Add, FLAG_B_CONST, 2, 7, 2),
                    Instr::new(Op::I32_BrNe, FLAG_B_ACC, 3, 2, 0),
                ],
                0,
                INTERP_SUPER_ADD_BRNE,
            ),
            (
                "shift then mask",
                vec![
                    Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 9, 0, 2),
                    Instr::new(
                        Op::I32_ShrU,
                        FLAG_A_ACC | FLAG_B_CONST | FLAG_DST_ACC,
                        2,
                        37,
                        3,
                    ),
                    Instr::new(Op::I32_And, FLAG_A_ACC | FLAG_B_CONST, 3, 0xff, 4),
                ],
                1,
                INTERP_SUPER_SHRU_AND,
            ),
            (
                "mask then equal branch",
                vec![
                    Instr::new(Op::I32_And, FLAG_B_CONST, 2, 0xff, 3),
                    Instr::new(Op::I32_BrEq, FLAG_A_ACC | FLAG_B_CONST, 3, 7, 0),
                ],
                0,
                INTERP_SUPER_AND_BREQ,
            ),
        ];

        for (name, code, start, slot) in cases {
            let cells = link_super_test_cells(&engine, &prefix, &code, pin);
            let expected = engine
                .super_handler_at(slot)
                .expect("Apple ARM64 superhandler slot");
            assert_eq!(
                cells[start].h, expected as u64,
                "{name} must replace its first linked handler"
            );
        }
    }

    #[cfg(sf_has_apple_arm64_interp_supers)]
    #[test]
    fn apple_super_fusions_reject_alias_constant_and_function_boundary_changes() {
        let engine = NativeEngine::new();
        let pin = Pinned {
            l0: 0,
            l1: 1,
            l0_float: false,
            l1_float: false,
        };
        let benign_prefix = [
            Instr::new(Op::MovConst, FLAG_A_CONST, 7, 0, 6),
            Instr::new(Op::Return, 0, 0, 0, 0),
        ];

        // MovPair reads its second source after committing destination 1.
        // Aliasing those slots changes the value observed by the load and
        // must keep the ordered ordinary handlers.
        let list_alias = vec![
            Instr::new(Op::MovPair, 0, 0, 3, (3 << 32) | 0),
            Instr::new(Op::I32_Load, FLAG_A_ACC, 0, 0, 3),
            Instr::new(Op::I32_Store, 0, 0, 3, 0),
            Instr::new(Op::BrIf, 0, 3, 0, 0),
        ];
        let cells = link_super_test_cells(&engine, &benign_prefix, &list_alias, pin);
        let ordinary = engine.handler_for(&list_alias[0], list_alias[0].flags, &pin);
        assert_ne!(ordinary, 0);
        assert_eq!(cells[0].h, ordinary as u64);
        assert_ne!(
            cells[0].h,
            engine
                .super_handler_at(INTERP_SUPER_MOVPAIR_LOAD_STORE_BRIF)
                .expect("list-step superhandler") as u64
        );

        // The emitted fourth add uses an immediate #4. No other constant is
        // semantically interchangeable with that handler.
        let add8 = vec![
            Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 9, 0, 2),
            Instr::new(Op::I32_Add, FLAG_A_ACC | FLAG_DST_ACC, 2, 1, 2),
            Instr::new(Op::I32_Add, FLAG_B_ACC, 3, 2, 1),
            Instr::new(Op::I32_Add, 0, 0, 4, 0),
            Instr::new(Op::I32_Add, FLAG_B_CONST, 5, 8, 5),
        ];
        let cells = link_super_test_cells(&engine, &benign_prefix, &add8, pin);
        let ordinary = engine.handler_for(&add8[1], add8[1].flags, &pin);
        assert_ne!(ordinary, 0);
        assert_eq!(cells[1].h, ordinary as u64);
        assert_ne!(
            cells[1].h,
            engine
                .super_handler_at(INTERP_SUPER_ADD4)
                .expect("add4 superhandler") as u64
        );

        // Even if the preceding function ends with the first three add4
        // cells, a one-cell function cannot complete that rolling window.
        let add4_prefix = vec![
            Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 9, 0, 2),
            Instr::new(Op::I32_Add, FLAG_A_ACC | FLAG_DST_ACC, 2, 1, 2),
            Instr::new(Op::I32_Add, FLAG_B_ACC, 3, 2, 1),
            Instr::new(Op::I32_Add, 0, 0, 4, 0),
        ];
        let last_add = [Instr::new(Op::I32_Add, FLAG_B_CONST, 5, 4, 5)];
        let cells = link_super_test_cells(&engine, &add4_prefix, &last_add, pin);
        let ordinary = engine.handler_for(&last_add[0], last_add[0].flags, &pin);
        assert_ne!(ordinary, 0);
        assert_eq!(cells[0].h, ordinary as u64);
        assert_ne!(
            cells[0].h,
            engine
                .super_handler_at(INTERP_SUPER_ADD4)
                .expect("add4 superhandler") as u64
        );
    }

    #[cfg(sf_has_apple_arm64_interp_tuning)]
    #[test]
    fn apple_backedge_branch_bank_selects_only_pinned_sources_and_backward_targets() {
        let engine = NativeEngine::new();
        let pin = Pinned {
            l0: 3,
            l1: 5,
            l0_float: false,
            l1_float: false,
        };
        let sources = [
            ("slot", 0, 7, None),
            ("const", FLAG_A_CONST, 7, None),
            ("acc", FLAG_A_ACC, 7, None),
            ("l0", 0, pin.l0, Some(1usize)),
            ("l1", 0, pin.l1, Some(2usize)),
        ];

        for (op, base) in [
            (Op::BrIf, INTERP_SUPER_BACKEDGE_BRIF_REG_BASE),
            (Op::BrIfNot, INTERP_SUPER_BACKEDGE_BRIFNOT_REG_BASE),
        ] {
            for (name, flags, operand, register_index) in sources {
                for (target, backward) in [(2, true), (8, true), (9, false)] {
                    let ins = Instr::new(op, flags, operand, 0, target);
                    let ordinary = engine.handler_for(&ins, flags, &pin);
                    assert_ne!(ordinary, 0, "{op:?}/{name} must have an ordinary handler");
                    let selected = engine.backedge_branch_handler(&ins, 8, ordinary, flags, &pin);

                    let expected = if backward {
                        register_index
                            .and_then(|index| engine.super_handler_at(base + index))
                            .unwrap_or(ordinary)
                    } else {
                        ordinary
                    };
                    assert_eq!(selected, expected, "{op:?}/{name} backward={backward}");
                    assert_eq!(
                        selected != ordinary,
                        backward && register_index.is_some(),
                        "{op:?}/{name} specialization boundary"
                    );
                }
            }
        }

        // op_slot gives a pinned payload precedence over an ACC hint. The
        // specialized selection must inherit that exact linker rule rather
        // than interpreting flags and payload independently.
        let ins = Instr::new(Op::BrIf, FLAG_A_ACC, pin.l0, 0, 2);
        let ordinary = engine.handler_for(&ins, FLAG_A_ACC, &pin);
        assert_eq!(
            engine.backedge_branch_handler(&ins, 8, ordinary, FLAG_A_ACC, &pin),
            engine
                .super_handler_at(INTERP_SUPER_BACKEDGE_BRIF_REG_BASE + 1)
                .expect("l0 backedge handler")
        );

        let float_pin = Pinned {
            l0_float: true,
            ..pin
        };
        let ins = Instr::new(Op::BrIf, 0, float_pin.l0, 0, 2);
        let ordinary = engine.handler_for(&ins, 0, &float_pin);
        assert_eq!(
            engine.backedge_branch_handler(&ins, 8, ordinary, 0, &float_pin),
            ordinary,
            "an integer branch cannot consume a float-pinned source register"
        );
    }

    #[cfg(sf_has_apple_arm64_interp_tuning)]
    #[test]
    fn apple_backedge_selection_uses_function_local_indices_at_materialization() {
        let engine = NativeEngine::new();
        let mut first_code = Vec::new();
        for slot in 0..16 {
            first_code.push(Instr::new(Op::MovConst, FLAG_A_CONST, slot, 0, slot));
        }
        first_code.push(Instr::new(Op::Return, 0, 0, 0, 0));
        let first = linker_test_function(first_code, Vec::new(), 16);

        let second_code = vec![
            Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 1, 0, 7),
            // Forward in this function, despite its raw target being below
            // the module-global arena index after `first` is linked.
            Instr::new(Op::BrIf, FLAG_A_ACC, 7, 0, 3),
            Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 1, 0, 7),
            // A legal self-backedge exercises the inclusive boundary.
            Instr::new(Op::BrIfNot, FLAG_A_ACC, 7, 0, 3),
            Instr::new(Op::Return, 0, 0, 0, 0),
        ];
        let second = linker_test_function(second_code, Vec::new(), 8);
        let last_code = vec![
            Instr::new(Op::MovConst, FLAG_A_CONST | FLAG_DST_ACC, 1, 0, 7),
            Instr::new(Op::BrIf, FLAG_A_ACC, 7, 0, 0),
        ];
        let last = linker_test_function(last_code, Vec::new(), 8);
        let mut plan = LinkPlan::for_functions([&first, &second, &last].into_iter());
        let mut scratch = LinkScratch::default();
        let mut indirect_types = Vec::new();
        let _ = engine.link(&first, 0, &mut plan, &mut scratch, &mut indirect_types);
        let linked = engine.link(&second, 1, &mut plan, &mut scratch, &mut indirect_types);
        let last_linked = engine.link(&last, 2, &mut plan, &mut scratch, &mut indirect_types);
        plan.finish_layout();

        let pin = second.pinned();
        let resolved = engine.resolve_reference(&second.code, &pin);
        let cells = plan.cells(&linked);
        assert_eq!(
            cells[1].h, resolved[1].handler as u64,
            "a forward target must keep the ordinary handler"
        );
        assert_eq!(
            cells[3].h,
            engine.backedge_branch_handler(
                &second.code[3],
                3,
                resolved[3].handler,
                resolved[3].flags,
                &pin,
            ) as u64,
            "an intermediate push must specialize an inclusive self-backedge"
        );
        assert_ne!(cells[3].h, resolved[3].handler as u64);

        let pin = last.pinned();
        let resolved = engine.resolve_reference(&last.code, &pin);
        let cells = plan.cells(&last_linked);
        assert_eq!(
            cells[1].h,
            engine.backedge_branch_handler(
                &last.code[1],
                1,
                resolved[1].handler,
                resolved[1].flags,
                &pin,
            ) as u64,
            "finish_last and the final push must specialize a backedge"
        );
        assert_ne!(cells[1].h, resolved[1].handler as u64);
    }

    #[test]
    fn caller_call_fixup_gate_matches_the_legacy_match_for_all_ops_and_flags() {
        let mut gated_ops = 0usize;
        for op_index in 0..N_OPS {
            let op = op_from_index(op_index);
            gated_ops += usize::from(call_fixup_op(op));
            for flags in u16::MIN..=u16::MAX {
                // Bits: table0/no-EH, table0/EH, table1/no-EH,
                // table1/EH. This crosses every condition the old callee
                // evaluated after entering for every final cell.
                let legacy = if !NATIVE_CALLS {
                    0
                } else {
                    match op {
                        Op::Call => 0b0101,
                        Op::CallIndirect if flags & FLAG_A_CONST == 0 => 0b0001,
                        _ => 0,
                    }
                };
                let gated = if !call_fixup_op(op) || !NATIVE_CALLS {
                    0
                } else {
                    match op {
                        Op::Call => 0b0101,
                        Op::CallIndirect if flags & FLAG_A_CONST == 0 => 0b0001,
                        _ => 0,
                    }
                };
                assert_eq!(gated, legacy, "op={op:?} flags={flags:#06x}");
            }
        }
        assert_eq!(gated_ops, 2);
        assert_eq!(N_OPS - gated_ops, N_OPS - 2);
    }

    struct FixupTestFunction {
        has_exception_handlers: bool,
    }

    impl LinkFunction for FixupTestFunction {
        fn code_start(&self) -> usize {
            0
        }

        fn code_len(&self) -> usize {
            0
        }

        fn pinned(&self) -> Pinned {
            Pinned::NONE
        }

        fn br_table_count(&self) -> usize {
            0
        }

        fn br_table(&self, _index: usize) -> Option<&[u32]> {
            None
        }

        fn has_exception_handlers_at(&self, _pc: u32) -> bool {
            self.has_exception_handlers
        }
    }

    #[test]
    fn gated_call_fixups_preserve_every_recorded_field_and_slow_condition() {
        let mut plan = LinkPlan::for_functions(core::iter::empty::<&PredecodedFunction>());
        let plain = FixupTestFunction {
            has_exception_handlers: false,
        };
        let trapped = FixupTestFunction {
            has_exception_handlers: true,
        };
        let cases = [
            (Instr::new(Op::Call, 0, 5, 9, 0), &plain),
            (Instr::new(Op::CallIndirect, 0, 4, 8, 21), &plain),
            (Instr::new(Op::Call, 0, 5, 9, 0), &trapped),
            (Instr::new(Op::CallIndirect, FLAG_A_CONST, 4, 8, 21), &plain),
            (
                Instr::new(Op::CallIndirect, 0, 4, 8, (1u64 << 32) | 21),
                &plain,
            ),
            (Instr::new(Op::CallIndirect, 0, 4, 8, 21), &trapped),
            (Instr::new(Op::ReturnCall, 0, 5, 9, 0), &plain),
            (Instr::new(Op::ReturnCallIndirect, 0, 4, 8, 21), &plain),
        ];
        for (index, (ins, func)) in cases.into_iter().enumerate() {
            if call_fixup_op(ins.op) {
                NativeEngine::record_call_fixup(&mut plan, func, 7, 10, index, &ins);
            }
        }

        if NATIVE_CALLS {
            assert_eq!(
                plan.call_fixups,
                vec![
                    CallFixup::Direct {
                        cell: 10,
                        caller: 7,
                        callee: 5,
                        arg_base: 9,
                    },
                    CallFixup::Indirect {
                        cell: 11,
                        caller: 7,
                        table_slot: 4,
                        arg_base: 8,
                        expected_type: 21,
                    },
                ]
            );
        } else {
            assert!(plan.call_fixups.is_empty());
        }
    }

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
        #[cfg(sf_has_apple_arm64_interp_tuning)]
        let pin = func.pinned();
        for (index, (ins, state)) in func.code.iter().zip(resolved).enumerate() {
            let flags = state.flags;
            let mut handler = Some(state.handler).filter(|&handler| handler != 0);
            #[cfg(sf_has_apple_arm64_interp_tuning)]
            if let Some(ordinary) = handler {
                handler = Some(engine.backedge_branch_handler(ins, index, ordinary, flags, &pin));
            }
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
        let mut plan = LinkPlan::for_functions(core::iter::once(func));
        let linked = engine.link(
            func,
            0,
            &mut plan,
            &mut link_scratch,
            &mut call_indirect_types,
        );
        plan.finish_layout();

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

        let mut arena = tracked_alloc::from_alloc_vec(func.code.to_vec());
        arena.push(Instr::new(Op::Unreachable, 0, 0, 0, 0));
        let allocation = arena.as_ptr() as usize;
        let expected_heads: Vec<u32> = arena.iter().copied().map(Instr::packed_head).collect();
        let mut in_place = LinkPlan::from_instr_arena(arena, 1, expected_flat.len());
        assert_eq!(in_place.cells.as_ptr() as usize, allocation);
        let mut in_place_scratch = LinkScratch::default();
        let mut in_place_types = Vec::new();
        let in_place_linked = engine.link_in_place(
            func,
            0,
            &mut in_place,
            &mut in_place_scratch,
            &mut in_place_types,
        );
        in_place.finish_layout();

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
        assert_eq!(in_place_linked.l0_off, linked.l0_off);
        assert_eq!(in_place_linked.l1_off, linked.l1_off);
        assert_eq!(in_place_linked.fp_pinned, linked.fp_pinned);
    }

    #[test]
    fn instruction_arena_transfer_keeps_one_owner_and_allocation() {
        let instrs = vec![
            Instr::new(Op::I32_Add, FLAG_B_CONST, 3, 41, 7),
            Instr::new(Op::Return, 0, 7, 1, 0),
        ];
        let allocation = instrs.as_ptr() as usize;
        let capacity = instrs.capacity();
        let cells = into_dispatch_cells(instrs);
        assert_eq!(cells.as_ptr() as usize, allocation);
        assert_eq!(cells.capacity(), capacity);
        assert_eq!((cells[0].a, cells[0].b, cells[0].c), (3, 41, 7));
        assert_eq!((cells[1].a, cells[1].b, cells[1].c), (7, 1, 0));
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
