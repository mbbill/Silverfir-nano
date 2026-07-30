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

use crate::collections::{vec, Vec};

use super::instr::{
    operand_is_float, result_is_float, Instr, Op, FLAG_ADDR64, FLAG_A_ACC, FLAG_A_CONST,
    FLAG_B_ACC, FLAG_B_CONST, FLAG_DST_ACC, FLAG_FUSED, FLAG_SHARED_GLOBAL, FLAG_SHARED_TABLE,
};
use super::layout::{
    build_op_base, c_is_branch_target, family, native_guard, op_slot, operand_is_f32,
    result_is_f32, slot_fields, total_slots, transform_bc, writes_acc, Fam, Pinned, CELL_NARROW,
    CELL_WIDE, N_OPS,
};
use super::predecode::PredecodedFunction;

// Build-time facts about the generated engine: which operand classes it
// was built with, and how many packed handler slots the table holds.
include!(concat!(env!("OUT_DIR"), "/interp_engine_cfg.rs"));

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
    pub table0_base: u64, // 112: table 0 entries (in only, RefHandle slots)
    pub table0_len: u64,  // 120
    pub indirect_base: u64, // 128: per-function indirect-call info, [u64;3]
                          // per function index (in only)
}

/// One 32-byte dispatch cell: [`Instr`] with the leading word replaced by
/// the handler address.
#[repr(C, align(32))]
pub(super) struct DCell {
    pub h: u64,
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

/// One linked function: dispatch cells mirroring `PredecodedFunction::code`
/// index-for-index, plus the flattened branch tables whose entries
/// `BrTable` cells point into (`b` holds the absolute base address, so the
/// buffer must live exactly as long as the cells).
pub(super) struct LinkedFunction {
    pub cells: Vec<DCell>,
    br_flat: Vec<u32>,
    /// Byte offset of each instruction's cell, indexed by instruction, with
    /// one extra entry for the pad cell so the offset past the last
    /// instruction is addressable. Cells are not all the same size, so the
    /// driver cannot recover this by shifting a pc.
    ///
    /// Built on demand: it is needed only where the driver crosses between
    /// instruction-index space and cell-address space — a slow exit, a trap,
    /// or a call resume — and most functions never do. Walking the stream
    /// per crossing instead would be quadratic for a function whose hot loop
    /// contains a slow op.
    cell_off: Option<Vec<u32>>,
    /// Which cells take the wide form, one bit per instruction. Only two
    /// sizes exist, so recording the choice costs an eighth of a byte per
    /// instruction against the sixteen the narrow form saves — and it has to
    /// be recorded rather than recomputed, because the size follows the
    /// link-resolved flags, which do not outlive linking.
    cell_wide: Vec<u64>,
    /// Byte offsets of the function's pinned locals in its frame (0 when
    /// absent — the unconditional reload then reads slot 0, which no cell
    /// consumes as a pinned class).
    pub l0_off: u32,
    pub l1_off: u32,
    /// Whether any pinned slot is float-mode: drives the conditional
    /// float-twin reloads on the call and return paths.
    pub fp_pinned: bool,
    /// Byte offset of the pad cell, i.e. the length of the real cell
    /// stream. Held eagerly because the pc range map needs it for every
    /// function at link time, which a lazy offset table would defeat; one
    /// word per function is free.
    pub code_bytes: u32,
}

impl LinkedFunction {
    /// Size of instruction `i`'s cell.
    fn cell_size(&self, i: usize) -> u32 {
        if self.cell_wide[i / 64] >> (i % 64) & 1 == 1 {
            CELL_WIDE
        } else {
            CELL_NARROW
        }
    }

    /// Cell byte offsets by instruction index, with one entry past the last
    /// instruction for the pad cell. Materialized on first use.
    fn cell_offsets(&mut self) -> &[u32] {
        if self.cell_off.is_none() {
            let n = self.cells.len();
            let mut off = Vec::with_capacity(n + 1);
            let mut at = 0u32;
            for i in 0..n {
                off.push(at);
                at += self.cell_size(i);
            }
            off.push(at);
            self.cell_off = Some(off);
        }
        self.cell_off.as_deref().expect("materialized above")
    }

    /// Byte offset of instruction `idx`'s cell.
    pub(super) fn offset_of(&mut self, idx: usize) -> Option<u32> {
        self.cell_offsets().get(idx).copied()
    }

    /// The instruction whose cell starts at `off`. `None` when `off` is not a
    /// cell boundary, which is a corrupt pc rather than a recoverable state.
    pub(super) fn index_at(&mut self, off: u32) -> Option<usize> {
        self.cell_offsets().binary_search(&off).ok()
    }
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
fn select_pinned(func: &PredecodedFunction) -> Pinned {
    let n = func.n_locals as u64;
    if n == 0 {
        return Pinned::NONE;
    }
    let mut counts = vec![0u64; func.n_locals as usize];
    // Per-slot domain sets (bit 0 = integer/agnostic, bit 1 = float):
    // writers decide the pinned register file; mixed writers make the slot
    // unpinnable (neither file could stay authoritative). Readers only
    // break the tie when no writer exists.
    let mut wdom = vec![0u8; func.n_locals as usize];
    let mut rdom = vec![0u8; func.n_locals as usize];
    // Whether a slot's float accesses are single-precision. Some backends
    // cannot hold an f32 in a pinned float register (see
    // `INTERP_FLOAT_PIN_F32`), so the width has to be known here.
    let mut f32dom = vec![false; func.n_locals as usize];
    for ins in func.code.iter() {
        let (a_s, b_s, c_d) = slot_fields(ins.op);
        if a_s && ins.flags & FLAG_A_CONST == 0 && ins.a < n {
            counts[ins.a as usize] += 1;
            if operand_is_float(ins.op, false) {
                rdom[ins.a as usize] |= 2;
                f32dom[ins.a as usize] |= operand_is_f32(ins.op, false);
            } else {
                rdom[ins.a as usize] |= 1;
            }
        }
        if b_s && ins.flags & FLAG_B_CONST == 0 && ins.b < n {
            counts[ins.b as usize] += 1;
            if operand_is_float(ins.op, true) {
                rdom[ins.b as usize] |= 2;
                f32dom[ins.b as usize] |= operand_is_f32(ins.op, true);
            } else {
                rdom[ins.b as usize] |= 1;
            }
        }
        if c_d && ins.c < n {
            counts[ins.c as usize] += 1;
            if result_is_float(ins.op) {
                wdom[ins.c as usize] |= 2;
                f32dom[ins.c as usize] |= result_is_f32(ins.op);
            } else {
                wdom[ins.c as usize] |= 1;
            }
        }
        if matches!(ins.op, Op::I32_SubBrIf | Op::I64_SubBrIf) && ins.a < n {
            // This control-shaped cell is also an in-place integer write to
            // `a`. Count both halves of the read/modify/write so pin
            // selection and register-domain authority match the unfused
            // subtraction it replaces.
            counts[ins.a as usize] += 1;
            wdom[ins.a as usize] |= 1;
        }
        if ins.op == Op::Select {
            let dslot = ins.c & 0xffff_ffff;
            if dslot < n {
                wdom[dslot as usize] |= 1;
            }
        }
        if ins.op == Op::MovPair {
            // `slot_fields` cannot describe two packed destinations, so
            // account for both here. The linker uses the resulting pin
            // choice to select an ordered pair handler that writes through
            // both frame slots and both authoritative registers.
            for dslot in [ins.c >> 32, ins.c & 0xffff_ffff] {
                if dslot < n {
                    counts[dslot as usize] += 1;
                    wdom[dslot as usize] |= 1;
                }
            }
        }
    }
    // Whether slot `i` could live in the float register file at all.
    let float_ok = |i: usize| INTERP_HAS_FLOAT_REGS && (INTERP_FLOAT_PIN_F32 || !f32dom[i]);
    let mut best = (usize::MAX, 0u64);
    let mut second = (usize::MAX, 0u64);
    for (i, &c) in counts.iter().enumerate() {
        // A slot is pinnable only when ONE register file can stay
        // authoritative for it. Mixed-domain writers rule that out, and so
        // does a float writer on a backend (or a width) that cannot pin a
        // float: the write would land in the slot alone and leave the
        // integer pinned register stale for the next integer-domain read.
        if wdom[i] == 3 || (wdom[i] & 2 != 0 && !float_ok(i)) {
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
    let ok = |(i, c): (usize, u64)| c > 0 && i * 8 < 1 << 16;
    // A slot's authoritative register file: float only when it has a
    // float writer (or, with no writer at all, only float readers) AND
    // this backend has float twins for the pinned registers.
    let mode = |i: usize| float_ok(i) && (wdom[i] == 2 || (wdom[i] == 0 && rdom[i] == 2));
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
    op_base: [u32; N_OPS],
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
            op_base: build_op_base(),
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

    /// Handler address for one cell under a given pinned-local choice.
    fn handler_for(&self, ins: &Instr, flags: u16, pin: &Pinned) -> Option<usize> {
        // The generated handlers zero-extend a 32-bit address, so a 64-bit
        // memory access has no native form and takes the shared executor.
        if flags & (FLAG_ADDR64 | FLAG_SHARED_TABLE | FLAG_SHARED_GLOBAL) != 0 {
            return None;
        }
        let slot = op_slot(ins, flags, pin, &self.op_base)?;
        self.handler_at(slot)
    }

    /// Build the dispatch cells for one predecoded function.
    pub(super) fn link(&self, func: &PredecodedFunction) -> LinkedFunction {
        let pin = select_pinned(func);

        // Resolve the predecoder's acc hints: an acc consumer's producer
        // is EXACTLY the preceding cell (strict adjacency). Honor the mark
        // only when both sides run natively; otherwise fall back to slots
        // — and a store-skipping producer whose consumer fell back must
        // store again. Lookups are pinning-aware: a pinned operand
        // outranks its acc flag inside `op_slot`, so a stripped flag on
        // such an operand is inert either way.
        let mut flags: Vec<u16> = func.code.iter().map(|i| i.flags).collect();
        for j in 0..func.code.len() {
            if flags[j] & (FLAG_A_ACC | FLAG_B_ACC) == 0 {
                continue;
            }
            // Call/CallIndirect are wired by the fixup pass, never through
            // the handler table; every call flavor (native, slow, host)
            // delivers result 0 through the accumulator relay, so they are
            // valid producers regardless of the table lookup.
            let prev_is_call = j > 0 && matches!(func.code[j - 1].op, Op::Call | Op::CallIndirect);
            let ok = j > 0
                && (writes_acc(func.code[j - 1].op) || prev_is_call)
                && (prev_is_call
                    || self
                        .handler_for(&func.code[j - 1], flags[j - 1], &pin)
                        .is_some())
                && self.handler_for(&func.code[j], flags[j], &pin).is_some()
                && native_guard(&func.code[j - 1])
                && native_guard(&func.code[j]);
            if !ok {
                flags[j] &= !(FLAG_A_ACC | FLAG_B_ACC);
                if j > 0 {
                    flags[j - 1] &= !FLAG_DST_ACC;
                }
            }
        }
        // Defensive: a store-skipping producer requires an acc consumer
        // right behind it (predecode marks them in pairs).
        for i in 0..func.code.len() {
            if flags[i] & FLAG_DST_ACC != 0
                && (i + 1 >= func.code.len() || flags[i + 1] & (FLAG_A_ACC | FLAG_B_ACC) == 0)
            {
                flags[i] &= !FLAG_DST_ACC;
            }
        }

        // Flatten the branch tables; BrTable cells carry a byte offset
        // into the flat buffer until the final address fixup below.
        let total: usize = func.br_tables.iter().map(|t| t.len()).sum();
        let mut br_flat: Vec<u32> = Vec::with_capacity(total);
        let mut table_byte_off: Vec<u64> = Vec::with_capacity(func.br_tables.len());
        for t in func.br_tables.iter() {
            table_byte_off.push(br_flat.len() as u64 * 4);
            for &target in t.iter() {
                br_flat.push(target);
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
        // A handler reads a slot field as a halfword, so a frame whose byte
        // offsets do not fit 16 bits has no native form: the load would
        // silently index the wrong slot. `frame_slots` bounds every slot
        // field, so one check per function covers all of them. Honored on
        // every backend rather than per-ISA — 8,192 slots is three orders of
        // magnitude past the widest frame in the corpus, so the bound costs
        // nothing to apply unconditionally.
        let slots_fit_halfword = (func.frame_slots as u64) * 8 <= 0xffff;

        let mut cells = Vec::with_capacity(func.code.len() + 1);
        let mut brtable_native = vec![false; func.code.len()];
        for (i, ins) in func.code.iter().enumerate() {
            let fl = flags[i];
            let mut h = self.handler_for(ins, fl, &pin);
            if !native_guard(ins) || !slots_fit_halfword {
                h = None;
            }
            // A 32-bit host reads a cell's static offset as one machine
            // word, so a wasm offset that does not fit in 32 bits cannot
            // run natively — the handler would silently use the truncated
            // value and turn an out-of-bounds access into an in-bounds one.
            if INTERP_PTR_BYTES == 4 && !offset_fits_word(ins) {
                h = None;
            }
            match h {
                Some(h) if ins.op == Op::BrTable => {
                    let table = &func.br_tables[ins.c as usize];
                    brtable_native[i] = true;
                    cells.push(DCell {
                        h: h as u64,
                        a: ins.a * 8,
                        b: table_byte_off[ins.c as usize],
                        c: (table.len() - 1) as u64,
                    });
                }
                Some(h) => {
                    let a = if fl & FLAG_A_CONST != 0 {
                        ins.a
                    } else {
                        ins.a * 8
                    };
                    let (b, c) = transform_bc(ins, fl);
                    cells.push(DCell {
                        h: h as u64,
                        a,
                        b,
                        c,
                    });
                }
                None => cells.push(DCell {
                    h: self.slow_stub as u64,
                    a: ins.a,
                    b: ins.b,
                    c: ins.c,
                }),
            }
        }

        cells.push(DCell {
            h: self.slow_stub as u64,
            a: 0,
            b: 0,
            c: 0,
        });

        let off_of = |slot: u64| {
            if slot == u64::MAX {
                0
            } else {
                (slot * 8) as u32
            }
        };
        let mut lf = LinkedFunction {
            cells,
            br_flat,
            cell_off: None,
            // Every cell is wide until the narrow forms land, which makes the
            // offsets this expands to identical to the shift they replace.
            cell_wide: vec![u64::MAX; (func.code.len() + 1).div_ceil(64)],
            l0_off: off_of(pin.l0),
            l1_off: off_of(pin.l1),
            fp_pinned: (pin.l0 != u64::MAX && pin.l0_float) || (pin.l1 != u64::MAX && pin.l1_float),
            code_bytes: 0,
        };
        // Cell offsets, computed here and deliberately NOT stored: the link
        // pass needs them to resolve targets, the driver needs them only on
        // an escape, and most functions never take one.
        let mut off = Vec::with_capacity(func.code.len() + 1);
        let mut at = 0u32;
        for i in 0..=func.code.len() {
            off.push(at);
            at += lf.cell_size(i);
        }
        lf.code_bytes = off[func.code.len()];
        // The flat table held target instruction indices; resolve them to
        // cell byte offsets now that the sizes are known.
        for e in lf.br_flat.iter_mut() {
            *e = off[*e as usize];
        }
        // The flat buffer has its final allocation now; resolve BrTable
        // base offsets to absolute addresses.
        let base = lf.br_flat.as_ptr() as u64;
        for i in 0..func.code.len() {
            if brtable_native[i] {
                lf.cells[i].b += base;
            }
        }
        // Same reasoning for branch targets: an absolute target removes
        // one link from the taken path's dependency chain and lets the
        // target's handler word be loaded at handler entry rather than
        // after the branch resolves — measured -4.80% of CoreMark cycles.
        // Moving the Vec is safe: absolute addresses point into its heap
        // buffer, which does not move with it.
        let cells_base = lf.cells.as_ptr() as u64;
        for i in 0..func.code.len() {
            if c_is_branch_target(func.code[i].op) {
                lf.cells[i].c = cells_base + off[lf.cells[i].c as usize] as u64;
            }
        }
        lf
    }

    /// The entry trampoline as a callable function pointer.
    pub(super) fn entry_fn(&self) -> extern "C" fn(*mut EnterState) {
        unsafe { core::mem::transmute::<usize, extern "C" fn(*mut EnterState)>(self.entry) }
    }
}
