//! Stage-B native dispatch for ARM64: threaded-code handlers emitted into
//! executable memory at instance setup, tail-chained with `ldr; br`
//! (design doc §7). Compiled only where executable memory support exists
//! (`sf_jit`) on an arm64 host; other configurations keep the stage-A loop.
//!
//! Deliberately self-contained: the instruction encoder below is a small
//! arm64 micro-encoder private to the interpreter, NOT the JIT's encoder —
//! interpreter iteration must never touch the JIT pipeline. Only the
//! executable-memory substrate (`CodeBuffer`, the `os` layer under it) is
//! shared runtime infrastructure.
//!
//! Register contract (all handlers, chain-invariant roles):
//!   x19 = pc (current 32-byte dispatch cell)
//!   x20 = frame base (u64 value slots)
//!   x21 = &EnterState (exit protocol)
//!   x22 = memory 0 base pointer
//!   x23 = memory 0 length in bytes
//!   x24 = dispatch-cell array base (branch targets are cell byte offsets)
//!   x25 = globals base pointer
//!   x9-x15 = scratch. sp belongs to the entry trampoline's frame.
//!
//! Dispatch cell ([`DCell`]): `[ handler:u64 | a | b | c ]`, the [`Instr`]
//! layout with the op/flags word replaced by the handler address. At link
//! time frame-slot operands and global indices are pre-scaled to byte
//! offsets and branch targets to cell byte offsets. Ops without a native
//! handler keep their operands RAW and point at the slow-exit stub; the
//! driver executes them with `exec_ins` on the ORIGINAL instruction (looked
//! up by cell index) and re-enters, which uniformly covers calls, returns,
//! traps-with-messages, and every rare op.

use tracked_alloc::boxed::Box;

use crate::collections::{vec, Vec};
use crate::error::WasmError;
use crate::vm::runtime::code_buf::CodeBuffer;

use super::instr::{Instr, Op, FLAG_A_ACC, FLAG_A_CONST, FLAG_B_ACC, FLAG_B_CONST, FLAG_DST_ACC};
use super::predecode::PredecodedFunction;

/// Exit reasons written to `EnterState::reason`.
pub(super) const EXIT_SLOW: u64 = 1;
/// A `Return` popped a sentinel record: control goes back to Rust.
pub(super) const EXIT_RETURN: u64 = 2;
pub(super) const EXIT_TRAP_BASE: u64 = 16;
/// Native trap kinds, indexed by `reason - EXIT_TRAP_BASE`. Messages must
/// match `exec_ins` exactly (differential/spectest parity).
pub(super) const TRAP_KINDS: &[&str] = &["out of bounds memory access", "call stack exhausted"];

/// Bytes per native return-stack record:
/// `(ret_pc, frame, code_base | caller_l0off<<48, caller_l1off)`.
pub(super) const RET_RECORD: usize = 32;

/// Communication block between Rust and the native chain. Field offsets are
/// baked into the trampoline; keep in sync with `emit_engine`.
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
}

/// One 32-byte dispatch cell; `Instr` with the leading word replaced by the
/// handler address.
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
    /// Byte offsets of the function's pinned locals in its frame (0 when
    /// absent — the unconditional reload then reads slot 0, which no
    /// cell consumes as a pinned class).
    pub l0_off: u32,
    pub l1_off: u32,
}

const N_OPS: usize = Op::Unreachable as usize + 1;
/// Variant slots per op: operand classes a×b ∈ {slot, const, acc, l0,
/// l1}², destination ∈ {mem, acc, l0, l1} — densely mapped, 5·5·4 = 100.
const N_VARIANTS: usize = 100;

/// Operand residency class for handler emission and link classification.
#[derive(Clone, Copy, PartialEq)]
enum Cls {
    Slot,
    Const,
    Acc,
    L0,
    L1,
}

/// Destination residency class. `Mem` computes into the accumulator and
/// stores; `Acc` skips the store; `L0`/`L1` compute into the pinned
/// register AND store (write-through — the slot stays authoritative for
/// the slow path and for reloads).
#[derive(Clone, Copy, PartialEq)]
enum DstCls {
    Mem,
    Acc,
    L0,
    L1,
}

const CLASSES: [Cls; 5] = [Cls::Slot, Cls::Const, Cls::Acc, Cls::L0, Cls::L1];
const DSTS: [DstCls; 4] = [DstCls::Mem, DstCls::Acc, DstCls::L0, DstCls::L1];

/// Dense variant index for an (a, b, dst) class combination.
fn variant(a: Cls, b: Cls, d: DstCls) -> usize {
    let ai = match a {
        Cls::Slot => 0,
        Cls::Const => 1,
        Cls::Acc => 2,
        Cls::L0 => 3,
        Cls::L1 => 4,
    };
    let bi = match b {
        Cls::Slot => 0,
        Cls::Const => 1,
        Cls::Acc => 2,
        Cls::L0 => 3,
        Cls::L1 => 4,
    };
    let di = match d {
        DstCls::Mem => 0,
        DstCls::Acc => 1,
        DstCls::L0 => 2,
        DstCls::L1 => 3,
    };
    ai + 5 * bi + 25 * di
}

fn between(op: Op, lo: Op, hi: Op) -> bool {
    (op as u16) >= (lo as u16) && (op as u16) <= (hi as u16)
}

/// Which cell fields are frame-slot references for this op:
/// `(a_is_slot_operand, b_is_slot_operand, c_is_value_dst)`. Exactness
/// matters — a wrong `true` here mis-classes an operand into a variant
/// the emitter never registered and silently demotes the cell to the
/// slow path.
fn slot_fields(op: Op) -> (bool, bool, bool) {
    use Op::*;
    // Binary value ops (b is a real operand).
    let binary = between(op, I32_Add, I32_Rotr)
        || between(op, I32_Eq, I32_GeU)
        || between(op, I64_Add, I64_Rotr)
        || between(op, I64_Eq, I64_GeU)
        || between(op, F32_Add, F32_Copysign)
        || between(op, F32_Eq, F32_Ge)
        || between(op, F64_Add, F64_Copysign)
        || between(op, F64_Eq, F64_Ge);
    if binary {
        return (true, true, true);
    }
    // Unary value ops and conversions.
    if between(op, I32_Clz, I32_Eqz)
        || between(op, I64_Clz, I64_Eqz)
        || between(op, I32_WrapI64, I64_ExtendI32U)
        || between(op, F32_Abs, F32_Sqrt)
        || between(op, F64_Abs, F64_Sqrt)
        || between(op, I32_TruncF32S, F64_ReinterpretI64)
    {
        return (true, false, true);
    }
    // Loads: a = address, b = static offset, c = dst.
    if between(op, I32_Load, I64_Load32U) {
        return (true, false, true);
    }
    // Stores: a = address, b = value, c = static offset.
    if between(op, I32_Store, I64_Store32) {
        return (true, true, false);
    }
    // Fused compare-branches: a, b operands, c = target.
    if between(op, I32_BrEq, I64_BrGeU) {
        return (true, true, false);
    }
    match op {
        MovSlot => (true, false, true),
        MovConst => (false, false, true),
        MemoryGrow => (true, false, true),
        MemorySize => (false, false, true),
        GlobalGet => (false, false, true),
        GlobalSet => (true, false, false),
        // Select's dst/cond are packed in c; handled by a link guard.
        Select => (true, true, false),
        BrIf | BrIfNot | BrTable => (true, false, false),
        RefIsNull | TableGet => (true, false, true),
        _ => (false, false, false),
    }
}

/// Pick the function's two pinned locals — the most- and second-most-
/// referenced slots, by UNWEIGHTED static count (`u64::MAX` = none).
/// Loop-depth weighting (10^depth over back edges) was tried and
/// measured 11% WORSE on CoreMark; the per-function diagnostic showed
/// why: depth weighting systematically displaced frequently-WRITTEN
/// locals (loop-carried state, e.g. 33r/17w) with read-mostly ones
/// (base pointers, 39r/3w). The pinned-local payoff is breaking the
/// loop-carried store→load chain, which needs the WRITTEN local;
/// read-mostly slot loads are independent and the OoO core hides them
/// anyway. A write-biased score (reads + 2·writes) measured
/// inconclusive (flips only cold functions on CoreMark).
fn select_pinned(func: &PredecodedFunction) -> (u64, u64) {
    let n = func.n_locals as u64;
    if n == 0 {
        return (u64::MAX, u64::MAX);
    }
    let mut counts = vec![0u64; func.n_locals as usize];
    for ins in func.code.iter() {
        let (a_s, b_s, c_d) = slot_fields(ins.op);
        if a_s && ins.flags & FLAG_A_CONST == 0 && ins.a < n {
            counts[ins.a as usize] += 1;
        }
        if b_s && ins.flags & FLAG_B_CONST == 0 && ins.b < n {
            counts[ins.b as usize] += 1;
        }
        if c_d && ins.c < n {
            counts[ins.c as usize] += 1;
        }
    }
    let mut best = (usize::MAX, 0u64);
    let mut second = (usize::MAX, 0u64);
    for (i, &c) in counts.iter().enumerate() {
        if c > best.1 {
            second = best;
            best = (i, c);
        } else if c > second.1 {
            second = (i, c);
        }
    }
    // Byte offsets must fit the 16-bit packing in call cells / records.
    let ok = |(i, c): (usize, u64)| c > 0 && i * 8 < 1 << 16;
    let l0 = if ok(best) { best.0 as u64 } else { u64::MAX };
    let l1 = if l0 != u64::MAX && ok(second) {
        second.0 as u64
    } else {
        u64::MAX
    };
    (l0, l1)
}

/// Classify one cell into its dense variant, given link-resolved flags
/// and the function's pinned slots (`u64::MAX` = none). Pinned classes
/// take precedence over acc hints on the same operand; a `const` flag
/// wins over both.
fn op_variant(ins: &Instr, flags: u16, l0_slot: u64, l1_slot: u64) -> usize {
    let (a_s, b_s, c_d) = slot_fields(ins.op);
    let a = if flags & FLAG_A_CONST != 0 {
        Cls::Const
    } else if a_s && ins.a == l0_slot {
        Cls::L0
    } else if a_s && ins.a == l1_slot {
        Cls::L1
    } else if flags & FLAG_A_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    };
    let b = if flags & FLAG_B_CONST != 0 {
        Cls::Const
    } else if b_s && ins.b == l0_slot {
        Cls::L0
    } else if b_s && ins.b == l1_slot {
        Cls::L1
    } else if flags & FLAG_B_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    };
    let d = if ins.op == Op::Select {
        // Select's dst slot is packed in c's low half.
        let dslot = ins.c & 0xffff_ffff;
        if dslot == l0_slot {
            DstCls::L0
        } else if dslot == l1_slot {
            DstCls::L1
        } else if flags & FLAG_DST_ACC != 0 {
            DstCls::Acc
        } else {
            DstCls::Mem
        }
    } else if c_d && ins.c == l0_slot {
        DstCls::L0
    } else if c_d && ins.c == l1_slot {
        DstCls::L1
    } else if flags & FLAG_DST_ACC != 0 {
        DstCls::Acc
    } else {
        DstCls::Mem
    };
    variant(a, b, d)
}

/// Whether an op's native handler leaves a result in the accumulator
/// (every value producer computes into it; see `finish_dst`). Relies on
/// the `Op` enum's contiguous value-op ranges; ops that are never native
/// are harmlessly included — the linker's native check is the real gate.
fn writes_acc(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    (d >= MovSlot as u16 && d <= F64_ReinterpretI64 as u16)
        || (d >= I32_Load as u16 && d <= I64_Load32U as u16)
        || matches!(
            op,
            MemorySize
                | MemoryGrow
                | GlobalGet
                | RefIsNull
                | TableGet
                | TableSize
                | TableGrow
                | Select
        )
}

/// Ops whose static offset packs a memory index in the high bits can only
/// run natively against memory 0.
fn native_guard(ins: &Instr) -> bool {
    use Op::*;
    match ins.op {
        I32_Load | I64_Load | F32_Load | F64_Load | I32_Load8S | I32_Load8U | I32_Load16S
        | I32_Load16U | I64_Load8S | I64_Load8U | I64_Load16S | I64_Load16U | I64_Load32S
        | I64_Load32U => ins.b >> 48 == 0,
        I32_Store | I64_Store | F32_Store | F64_Store | I32_Store8 | I32_Store16 | I64_Store8
        | I64_Store16 | I64_Store32 => ins.c >> 48 == 0,
        _ => true,
    }
}

/// Per-op b/c operand pre-scaling for native handlers (a is handled
/// uniformly at the call site: slot index ×8 unless const). `flags` are
/// the link-resolved flags (acc hints possibly stripped).
fn transform_bc(ins: &Instr, flags: u16) -> (u64, u64) {
    use Op::*;
    match ins.op {
        // control: c = target cell byte offset; b unused
        Br | BrIf | BrIfNot => (ins.b, ins.c * CELL as u64),
        // fused compare-branches: b = compare operand, c = target
        I32_BrEq | I32_BrNe | I32_BrLtS | I32_BrLtU | I32_BrGtS | I32_BrGtU | I32_BrLeS
        | I32_BrLeU | I32_BrGeS | I32_BrGeU | I64_BrEq | I64_BrNe | I64_BrLtS | I64_BrLtU
        | I64_BrGtS | I64_BrGtU | I64_BrLeS | I64_BrLeU | I64_BrGeS | I64_BrGeU => {
            let b = if flags & FLAG_B_CONST != 0 {
                ins.b
            } else {
                ins.b * 8
            };
            (b, ins.c * CELL as u64)
        }
        // loads: b = static offset (stays raw), c = dst slot
        I32_Load | I64_Load | F32_Load | F64_Load | I32_Load8S | I32_Load8U | I32_Load16S
        | I32_Load16U | I64_Load8S | I64_Load8U | I64_Load16S | I64_Load16U | I64_Load32S
        | I64_Load32U => (ins.b, ins.c * 8),
        // stores: b = value operand, c = static offset (stays raw)
        I32_Store | I64_Store | F32_Store | F64_Store | I32_Store8 | I32_Store16 | I64_Store8
        | I64_Store16 | I64_Store32 => {
            let b = if flags & FLAG_B_CONST != 0 {
                ins.b
            } else {
                ins.b * 8
            };
            (b, ins.c)
        }
        // GlobalSet: c = global index
        GlobalSet => (ins.b, ins.c * 8),
        // Return: b = result count, c unused — both stay raw
        Return => (ins.b, ins.c),
        // Select: c = cond_slot << 32 | dst_slot, both scaled
        Select => {
            let b = if flags & FLAG_B_CONST != 0 {
                ins.b
            } else {
                ins.b * 8
            };
            (b, ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8)
        }
        // plain value ops (incl. GlobalGet): b = operand, c = dst slot
        _ => {
            let b = if flags & FLAG_B_CONST != 0 {
                ins.b
            } else {
                ins.b * 8
            };
            (b, ins.c * 8)
        }
    }
}

/// The emitted native engine: handler set + trampoline + exit stubs.
pub(super) struct NativeEngine {
    buf: CodeBuffer,
    entry: usize,
    /// Handler code offsets keyed by [`key`]; `u32::MAX` = no native form.
    handlers: Vec<u32>,
    slow_stub: usize,
    /// The native `Call` handler (wired by the cross-function fixup pass,
    /// not through `handlers`: its cells need callee addresses).
    call_handler: usize,
    /// One synthetic cell whose handler word is the `EXIT_RETURN` stub.
    /// Sentinel return-stack records point here, so a native `Return` that
    /// pops a sentinel lands in Rust — the boxed cell must outlive every
    /// record that references it.
    exit_cell: Box<DCell>,
}

impl NativeEngine {
    pub(super) fn new() -> Result<Self, WasmError> {
        let mut buf = CodeBuffer::with_capacity(512 * 1024).map_err(WasmError::invalid)?;
        buf.begin_write();
        let mut handlers = vec![u32::MAX; N_OPS * N_VARIANTS];
        let out = {
            let mut e = Enc { buf: &mut buf };
            emit_engine(&mut e, &mut handlers)
        };
        let len = buf.len();
        buf.finish_write(0, len);
        let exit_cell = Box::new(DCell {
            h: unsafe { buf.ptr(out.return_exit) } as u64,
            a: 0,
            b: 0,
            c: 0,
        });
        Ok(NativeEngine {
            buf,
            entry: out.entry,
            handlers,
            slow_stub: out.slow_stub,
            call_handler: out.call_handler,
            exit_cell,
        })
    }

    /// Address of the sentinel exit cell (see `exit_cell`).
    pub(super) fn exit_cell_addr(&self) -> u64 {
        &*self.exit_cell as *const DCell as u64
    }

    /// Address of the native `Call` handler, for the fixup pass.
    pub(super) fn call_handler_addr(&self) -> u64 {
        unsafe { self.buf.ptr(self.call_handler) as u64 }
    }

    /// Handler-table lookup by op and dense variant.
    fn h(&self, op: Op, v: usize) -> u32 {
        self.handlers[op as usize * N_VARIANTS + v]
    }

    /// Build the dispatch cells for one predecoded function.
    pub(super) fn link(&self, func: &PredecodedFunction) -> LinkedFunction {
        // Pick the function's l0: the most-referenced local slot. Static
        // unweighted counts; byte offset must fit the 16-bit packing in
        // call cells and return records.
        let (l0_slot, l1_slot) = select_pinned(func);

        // Resolve the predecoder's acc hints: an acc consumer's producer
        // is EXACTLY the preceding cell (strict adjacency). Honor the
        // mark only when both sides run natively; otherwise fall back to
        // slots — and a store-skipping producer whose consumer fell back
        // must store again. Lookups are l0-aware: an l0-classed operand
        // outranks its acc flag inside `op_variant`, so a stripped flag
        // on such an operand is inert either way.
        let mut flags: Vec<u16> = func.code.iter().map(|i| i.flags).collect();
        for j in 0..func.code.len() {
            if flags[j] & (FLAG_A_ACC | FLAG_B_ACC) == 0 {
                continue;
            }
            let ok = j > 0
                && writes_acc(func.code[j - 1].op)
                && self.h(
                    func.code[j - 1].op,
                    op_variant(&func.code[j - 1], flags[j - 1], l0_slot, l1_slot),
                ) != u32::MAX
                && self.h(
                    func.code[j].op,
                    op_variant(&func.code[j], flags[j], l0_slot, l1_slot),
                ) != u32::MAX
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

        let mut cells = Vec::with_capacity(func.code.len());
        let mut brtable_native = vec![false; func.code.len()];
        for (i, ins) in func.code.iter().enumerate() {
            let fl = flags[i];
            let off = self.h(ins.op, op_variant(ins, fl, l0_slot, l1_slot));
            if ins.op == Op::BrTable && off != u32::MAX {
                let table = &func.br_tables[ins.c as usize];
                brtable_native[i] = true;
                cells.push(DCell {
                    h: unsafe { self.buf.ptr(off as usize) } as u64,
                    a: ins.a * 8,
                    b: table_byte_off[ins.c as usize],
                    c: (table.len() - 1) as u64,
                });
            } else if off != u32::MAX && native_guard(ins) {
                let a = if fl & FLAG_A_CONST != 0 {
                    ins.a
                } else {
                    ins.a * 8
                };
                let (b, c) = transform_bc(ins, fl);
                let h = unsafe { self.buf.ptr(off as usize) } as u64;
                cells.push(DCell { h, a, b, c });
            } else {
                let h = unsafe { self.buf.ptr(self.slow_stub) } as u64;
                cells.push(DCell {
                    h,
                    a: ins.a,
                    b: ins.b,
                    c: ins.c,
                });
            }
        }

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
            l0_off: off_of(l0_slot),
            l1_off: off_of(l1_slot),
        };
        // The flat buffer has its final allocation now; resolve BrTable
        // base offsets to absolute addresses.
        let base = lf.br_flat.as_ptr() as u64;
        for i in 0..func.code.len() {
            if brtable_native[i] {
                lf.cells[i].b += base;
            }
        }
        lf
    }

    /// The entry trampoline as a callable function pointer.
    pub(super) fn entry_fn(&self) -> extern "C" fn(*mut EnterState) {
        unsafe { self.buf.fn_ptr(self.entry) }
    }
}

// ---------------------------------------------------------------------------
// arm64 micro-encoder
// ---------------------------------------------------------------------------

/// The accumulator (design doc §8): carries a span-1 temp between an
/// adjacent producer/consumer pair. Never live across an exit, so it is
/// not part of EnterState.
const ACC: u32 = 8;
/// The l0/l1 registers: the function's two hottest locals,
/// register-resident across the whole body (write-through — the frame
/// slots stay current). Reloaded from the slots at every chain entry,
/// call, and return.
const L0R: u32 = 17;
const L1R: u32 = 14;
const X9: u32 = 9;
const X10: u32 = 10;
const X11: u32 = 11;
const X12: u32 = 12;
const X13: u32 = 13;
const PC: u32 = 19;
const FRAME: u32 = 20;
const STATE: u32 = 21;
const MEM: u32 = 22;
const MEMLEN: u32 = 23;
const CODE: u32 = 24;
const GLOB: u32 = 25;
const RETSP: u32 = 26;
const RETLIM: u32 = 27;
const STKLIM: u32 = 28;
/// Dispatch counter (caller-saved scratch, saved through EnterState).
const DCNT: u32 = 15;

// condition codes
const EQ: u32 = 0;
const NE: u32 = 1;
const HS: u32 = 2;
const LO: u32 = 3;
const HI: u32 = 8;
const LS: u32 = 9;
const GE: u32 = 10;
const LT: u32 = 11;
const GT: u32 = 12;
const LE: u32 = 13;

const CELL: u32 = 32;

struct Enc<'b> {
    buf: &'b mut CodeBuffer,
}

impl<'b> Enc<'b> {
    fn i(&mut self, word: u32) {
        self.buf.emit_u32(word);
    }
    fn here(&self) -> usize {
        self.buf.len()
    }

    // ---- loads/stores ----
    /// LDR Xt, [Xn, #imm] (unsigned scaled, imm % 8 == 0)
    fn ldr_x_imm(&mut self, rt: u32, rn: u32, imm: u32) {
        self.i(0xF940_0000 | ((imm / 8) << 10) | (rn << 5) | rt);
    }
    fn str_x_imm(&mut self, rt: u32, rn: u32, imm: u32) {
        self.i(0xF900_0000 | ((imm / 8) << 10) | (rn << 5) | rt);
    }
    // register-offset (extend option LSL #0)
    fn ldr_x_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0xF860_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn str_x_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0xF820_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldr_w_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0xB860_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn str_w_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0xB820_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldrb_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x3860_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldrh_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x7860_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldrsb_w_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x38E0_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldrsh_w_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x78E0_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldrsb_x_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x38A0_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldrsh_x_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x78A0_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn ldrsw_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0xB8A0_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn strb_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x3820_6800 | (rm << 16) | (rn << 5) | rt);
    }
    fn strh_reg(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0x7820_6800 | (rm << 16) | (rn << 5) | rt);
    }

    // ---- arithmetic/logic ----
    fn add_x_imm(&mut self, rd: u32, rn: u32, imm12: u32) {
        self.i(0x9100_0000 | (imm12 << 10) | (rn << 5) | rd);
    }
    /// ADD Xd, Xn, Wm, UXTW
    fn add_x_uxtw(&mut self, rd: u32, rn: u32, rm: u32) {
        self.i(0x8B20_4000 | (rm << 16) | (rn << 5) | rd);
    }
    fn add_x_reg(&mut self, rd: u32, rn: u32, rm: u32) {
        self.i(0x8B00_0000 | (rm << 16) | (rn << 5) | rd);
    }
    /// Three-register form with a caller-supplied opcode base
    /// (`base | rm<<16 | rn<<5 | rd`).
    fn alu_reg(&mut self, base: u32, rd: u32, rn: u32, rm: u32) {
        self.i(base | (rm << 16) | (rn << 5) | rd);
    }
    fn cmp_x(&mut self, rn: u32, rm: u32) {
        self.i(0xEB00_001F | (rm << 16) | (rn << 5));
    }
    fn cmp_w(&mut self, rn: u32, rm: u32) {
        self.i(0x6B00_001F | (rm << 16) | (rn << 5));
    }
    /// CSET Wd, cond (CSINC Wd, WZR, WZR, inv(cond))
    fn cset_w(&mut self, rd: u32, cond: u32) {
        self.i(0x1A9F_07E0 | ((cond ^ 1) << 12) | rd);
    }
    fn neg_w(&mut self, rd: u32, rm: u32) {
        self.i(0x4B00_03E0 | (rm << 16) | rd);
    }
    fn neg_x(&mut self, rd: u32, rm: u32) {
        self.i(0xCB00_03E0 | (rm << 16) | rd);
    }
    /// MOV Wd, Wm — also the canonical 32-bit zero-extend
    fn mov_w(&mut self, rd: u32, rm: u32) {
        self.i(0x2A00_03E0 | (rm << 16) | rd);
    }
    fn sxtb_w(&mut self, rd: u32, rn: u32) {
        self.i(0x1300_1C00 | (rn << 5) | rd);
    }
    fn sxth_w(&mut self, rd: u32, rn: u32) {
        self.i(0x1300_3C00 | (rn << 5) | rd);
    }
    fn sxtb_x(&mut self, rd: u32, rn: u32) {
        self.i(0x9340_1C00 | (rn << 5) | rd);
    }
    fn sxth_x(&mut self, rd: u32, rn: u32) {
        self.i(0x9340_3C00 | (rn << 5) | rd);
    }
    fn sxtw_x(&mut self, rd: u32, rn: u32) {
        self.i(0x9340_7C00 | (rn << 5) | rd);
    }
    /// CSEL Xd, Xn, Xm, cond
    fn csel_x(&mut self, rd: u32, rn: u32, rm: u32, cond: u32) {
        self.i(0x9A80_0000 | (rm << 16) | (cond << 12) | (rn << 5) | rd);
    }
    fn csel_w(&mut self, rd: u32, rn: u32, rm: u32, cond: u32) {
        self.i(0x1A80_0000 | (rm << 16) | (cond << 12) | (rn << 5) | rd);
    }
    fn ubfm_x(&mut self, rd: u32, rn: u32, immr: u32, imms: u32) {
        self.i(0xD340_0000 | (immr << 16) | (imms << 10) | (rn << 5) | rd);
    }
    fn lsr_x(&mut self, rd: u32, rn: u32, shift: u32) {
        self.ubfm_x(rd, rn, shift, 63);
    }
    fn lsl_x(&mut self, rd: u32, rn: u32, shift: u32) {
        self.ubfm_x(rd, rn, 64 - shift, 63 - shift);
    }
    /// LDR Wt, [Xn, Wm, UXTW #2]
    fn ldr_w_uxtw2(&mut self, rt: u32, rn: u32, rm: u32) {
        self.i(0xB860_5800 | (rm << 16) | (rn << 5) | rt);
    }
    /// UBFX Xd, Xn, #lsb, #width
    fn ubfx_x(&mut self, rd: u32, rn: u32, lsb: u32, width: u32) {
        self.ubfm_x(rd, rn, lsb, lsb + width - 1);
    }
    /// ADD Xd, Xn, Xm, LSL #shift
    fn add_x_lsl(&mut self, rd: u32, rn: u32, rm: u32, shift: u32) {
        self.i(0x8B00_0000 | (rm << 16) | (shift << 10) | (rn << 5) | rd);
    }
    fn sub_x_imm(&mut self, rd: u32, rn: u32, imm12: u32) {
        self.i(0xD100_0000 | (imm12 << 10) | (rn << 5) | rd);
    }
    fn mov_x(&mut self, rd: u32, rm: u32) {
        self.i(0xAA00_03E0 | (rm << 16) | rd);
    }
    /// ORR Xd, Xn, Xm, LSL #shift
    fn orr_x_lsl(&mut self, rd: u32, rn: u32, rm: u32, shift: u32) {
        self.i(0xAA00_0000 | (rm << 16) | (shift << 10) | (rn << 5) | rd);
    }
    /// STR Xt, [Xn], #imm (post-index)
    fn str_x_post(&mut self, rt: u32, rn: u32, imm: i32) {
        self.i(0xF800_0400 | (((imm as u32) & 0x1FF) << 12) | (rn << 5) | rt);
    }
    /// LDR Xt, [Xn], #imm (post-index)
    fn ldr_x_post(&mut self, rt: u32, rn: u32, imm: i32) {
        self.i(0xF840_0400 | (((imm as u32) & 0x1FF) << 12) | (rn << 5) | rt);
    }
    /// STP Xt, Xt2, [Xn] (signed offset 0)
    fn stp_x_base(&mut self, rt: u32, rt2: u32, rn: u32) {
        self.i(0xA900_0000 | (rt2 << 10) | (rn << 5) | rt);
    }
    /// LDP Xt, Xt2, [Xn, #imm]! (pre-index)
    fn ldp_x_pre(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.i(0xA9C0_0000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt);
    }
    fn cbz_x(&mut self, rt: u32, delta_insns: i32) {
        self.i(0xB400_0000 | (((delta_insns as u32) & 0x7FFFF) << 5) | rt);
    }
    fn cbnz_x(&mut self, rt: u32, delta_insns: i32) {
        self.i(0xB500_0000 | (((delta_insns as u32) & 0x7FFFF) << 5) | rt);
    }
    fn clz_w(&mut self, rd: u32, rn: u32) {
        self.i(0x5AC0_1000 | (rn << 5) | rd);
    }
    fn clz_x(&mut self, rd: u32, rn: u32) {
        self.i(0xDAC0_1000 | (rn << 5) | rd);
    }
    fn rbit_w(&mut self, rd: u32, rn: u32) {
        self.i(0x5AC0_0000 | (rn << 5) | rd);
    }
    fn rbit_x(&mut self, rd: u32, rn: u32) {
        self.i(0xDAC0_0000 | (rn << 5) | rd);
    }

    // ---- branches ----
    fn cbz_w(&mut self, rt: u32, delta_insns: i32) {
        self.i(0x3400_0000 | (((delta_insns as u32) & 0x7FFFF) << 5) | rt);
    }
    fn cbnz_w(&mut self, rt: u32, delta_insns: i32) {
        self.i(0x3500_0000 | (((delta_insns as u32) & 0x7FFFF) << 5) | rt);
    }
    fn b_cond(&mut self, cond: u32, delta_insns: i32) {
        self.i(0x5400_0000 | (((delta_insns as u32) & 0x7FFFF) << 5) | cond);
    }
    fn b(&mut self, delta_insns: i32) {
        self.i(0x1400_0000 | ((delta_insns as u32) & 0x03FF_FFFF));
    }
    fn br(&mut self, rn: u32) {
        self.i(0xD61F_0000 | (rn << 5));
    }
    fn ret(&mut self) {
        self.i(0xD65F_03C0);
    }

    // ---- constants, prologue/epilogue ----
    fn movz_x(&mut self, rd: u32, imm16: u32) {
        self.i(0xD280_0000 | (imm16 << 5) | rd);
    }
    fn stp_x_pre(&mut self, rt: u32, rt2: u32, imm: i32) {
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.i(0xA980_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt);
    }
    fn ldp_x_post(&mut self, rt: u32, rt2: u32, imm: i32) {
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.i(0xA8C0_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt);
    }
    fn stp_x_off(&mut self, rt: u32, rt2: u32, imm: i32) {
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.i(0xA900_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt);
    }
    fn ldp_x_off(&mut self, rt: u32, rt2: u32, imm: i32) {
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.i(0xA940_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt);
    }
}

// ---------------------------------------------------------------------------
// Engine emission
// ---------------------------------------------------------------------------

/// Diagnostic count-mode for the in-chain dispatch counter:
/// 0 = count every dispatch (production), 1 = count only handlers whose
/// variant involves the L0 class, 2 = only L1-involved. Modes 1/2 turn
/// one CoreMark run into an exact, timing-independent measurement of
/// dynamic pinned-class engagement.
const COUNT_MODE: u8 = 0;

/// Whether a handler of this variant bumps the dispatch counter under
/// the current COUNT_MODE.
fn counted(a: Cls, b: Cls, d: DstCls) -> bool {
    match COUNT_MODE {
        1 => a == Cls::L0 || b == Cls::L0 || d == DstCls::L0,
        2 => a == Cls::L1 || b == Cls::L1 || d == DstCls::L1,
        _ => true,
    }
}

fn bump(e: &mut Enc<'_>, on: bool) {
    if on {
        e.add_x_imm(DCNT, DCNT, 1);
    }
}

/// The `add pc, #32; ldr x9, [pc]; br x9` dispatch tail plus the
/// one-instruction counter. (A pre-index `ldr x9, [pc, #32]!` form
/// measured no better on Apple silicon: the writeback µop serializes
/// against the load, while the separate `add` schedules freely.)
fn tail(e: &mut Enc<'_>, count: bool) {
    bump(e, count);
    e.add_x_imm(PC, PC, CELL);
    e.ldr_x_imm(X9, PC, 0);
    e.br(X9);
}

/// Materialize operand a in a register: the acc IS a register (no code);
/// consts load inline from the cell; slots load through the frame.
/// Returns the register holding the value.
fn src_a(e: &mut Enc<'_>, cls: Cls, tmp: u32) -> u32 {
    match cls {
        Cls::Acc => ACC,
        Cls::L0 => L0R,
        Cls::L1 => L1R,
        Cls::Const => {
            e.ldr_x_imm(tmp, PC, 8);
            tmp
        }
        Cls::Slot => {
            e.ldr_x_imm(tmp, PC, 8);
            e.ldr_x_reg(tmp, FRAME, tmp);
            tmp
        }
    }
}

fn src_b(e: &mut Enc<'_>, cls: Cls, tmp: u32) -> u32 {
    match cls {
        Cls::Acc => ACC,
        Cls::L0 => L0R,
        Cls::L1 => L1R,
        Cls::Const => {
            e.ldr_x_imm(tmp, PC, 16);
            tmp
        }
        Cls::Slot => {
            e.ldr_x_imm(tmp, PC, 16);
            e.ldr_x_reg(tmp, FRAME, tmp);
            tmp
        }
    }
}

/// Store `src` to the dst slot (c = pre-scaled byte offset).
fn store_dst(e: &mut Enc<'_>, src: u32) {
    e.ldr_x_imm(X12, PC, 24);
    e.str_x_reg(src, FRAME, X12);
}

/// The register a value handler computes into: the l0 register for
/// L0-destination variants, the accumulator otherwise. Mem variants
/// store from the accumulator, so every non-L0 value handler leaves its
/// result in the acc as a side effect — which is what makes
/// write-through acc residency free. L0-dst handlers write the l0
/// register AND the slot (write-through) and deliberately leave the acc
/// alone: the linker never pairs an acc consumer with an L0-dst
/// producer (temp slots and locals are disjoint, and l0-slot operands
/// classify as L0, not Acc).
fn dst_target(d: DstCls) -> u32 {
    match d {
        DstCls::L0 => L0R,
        DstCls::L1 => L1R,
        _ => ACC,
    }
}

/// Finish a value-producing handler: Mem and L0 destinations store to
/// the slot; Acc skips the store.
fn finish(e: &mut Enc<'_>, d: DstCls, src: u32) {
    if d != DstCls::Acc {
        store_dst(e, src);
    }
}

fn def(handlers: &mut [u32], op: Op, variant: usize, off: usize) {
    handlers[op as usize * N_VARIANTS + variant] = off as u32;
}

/// Offsets of the non-handler entry points produced by [`emit_engine`].
struct EmitOut {
    entry: usize,
    slow_stub: usize,
    call_handler: usize,
    return_exit: usize,
}

/// Emit stubs, trampoline, and all handlers.
fn emit_engine(e: &mut Enc<'_>, handlers: &mut [u32]) -> EmitOut {
    // ---- common exit path: x9 = reason ----
    let exit_common = e.here();
    e.str_x_imm(X9, STATE, 0); // state.reason
    e.str_x_imm(PC, STATE, 8); // state.pc
    e.str_x_imm(FRAME, STATE, 16); // calls move the frame; write it back
    e.str_x_imm(RETSP, STATE, 56); // ...and the return-stack cursor
    e.str_x_imm(DCNT, STATE, 80); // ...and the dispatch counter
    e.ldp_x_off(19, 20, 16);
    e.ldp_x_off(21, 22, 32);
    e.ldp_x_off(23, 24, 48);
    e.ldp_x_off(25, 26, 64);
    e.ldp_x_off(27, 28, 80);
    e.ldp_x_post(29, 30, 96);
    e.ret();

    fn emit_exit(e: &mut Enc<'_>, reason: u64, exit_common: usize) -> usize {
        let off = e.here();
        e.movz_x(X9, reason as u32);
        let delta = (exit_common as i64 - e.here() as i64) / 4;
        e.b(delta as i32);
        off
    }
    let slow_stub = emit_exit(e, EXIT_SLOW, exit_common);
    let return_exit = emit_exit(e, EXIT_RETURN, exit_common);
    let trap_oob = emit_exit(e, EXIT_TRAP_BASE, exit_common);
    let trap_exhaust = emit_exit(e, EXIT_TRAP_BASE + 1, exit_common);

    // ---- entry trampoline: extern "C" fn(*mut EnterState) ----
    let entry = e.here();
    e.stp_x_pre(29, 30, -96);
    e.stp_x_off(19, 20, 16);
    e.stp_x_off(21, 22, 32);
    e.stp_x_off(23, 24, 48);
    e.stp_x_off(25, 26, 64);
    e.stp_x_off(27, 28, 80);
    e.i(0xAA00_03F5); // mov x21, x0 (state pointer)
    e.ldr_x_imm(PC, STATE, 8);
    e.ldr_x_imm(FRAME, STATE, 16);
    e.ldr_x_imm(MEM, STATE, 24);
    e.ldr_x_imm(MEMLEN, STATE, 32);
    e.ldr_x_imm(CODE, STATE, 40);
    e.ldr_x_imm(GLOB, STATE, 48);
    e.ldr_x_imm(RETSP, STATE, 56);
    e.ldr_x_imm(RETLIM, STATE, 64);
    e.ldr_x_imm(STKLIM, STATE, 72);
    e.ldr_x_imm(DCNT, STATE, 80);
    e.ldr_x_imm(L0R, STATE, 88);
    e.ldr_x_imm(L1R, STATE, 96);
    e.ldr_x_imm(X9, PC, 0);
    e.br(X9);

    // ---- call (rewired by the fixup pass; not in `handlers`) ----
    // cell: a = callee cells base, b = arg_base*8,
    //       c = frame_slots<<32 | n_locals<<16 | n_params
    let call_handler = e.here();
    {
        bump(e, COUNT_MODE == 0);
        // Cell layout: a = caller_l1off<<48 | callee cells (48-bit VA),
        // b = callee_l1off<<48 | callee_l0off<<32 | arg_base*8,
        // c = caller_l0off<<48 | frame_slots<<32 | n_locals<<16 | n_params
        e.ldr_x_imm(X12, PC, 24);
        // return-stack depth
        e.cmp_x(RETSP, RETLIM);
        {
            let delta = (trap_exhaust as i64 - e.here() as i64) / 4;
            e.b_cond(HS, delta as i32);
        }
        e.ldr_x_imm(X11, PC, 16);
        e.ldr_x_imm(X9, PC, 8);
        e.mov_w(X10, X11); // arg_base*8 (low 32, zero-extended)
        e.add_x_reg(X10, FRAME, X10); // new frame base
        e.ubfx_x(X13, X12, 32, 16); // frame_slots
        e.add_x_lsl(X13, X10, X13, 3);
        e.cmp_x(X13, STKLIM);
        {
            let delta = (trap_exhaust as i64 - e.here() as i64) / 4;
            e.b_cond(HI, delta as i32);
        }
        // push (ret_pc, caller frame, code | caller_l0off<<48, caller_l1off)
        e.add_x_imm(X13, PC, CELL);
        e.stp_x_base(X13, FRAME, RETSP);
        e.lsr_x(X13, X12, 48); // caller l0off
        e.orr_x_lsl(X13, CODE, X13, 48);
        e.str_x_imm(X13, RETSP, 16);
        e.lsr_x(X13, X9, 48); // caller l1off
        e.str_x_imm(X13, RETSP, 24);
        e.add_x_imm(RETSP, RETSP, RET_RECORD as u32);
        e.mov_x(FRAME, X10);
        // zero the fresh locals: [n_params*8, n_locals*8)
        e.ubfx_x(X10, X12, 0, 16);
        e.ubfx_x(X13, X12, 16, 16);
        e.add_x_lsl(X10, FRAME, X10, 3);
        e.add_x_lsl(X13, FRAME, X13, 3);
        e.cmp_x(X10, X13); // zl:
        e.b_cond(HS, 3); // -> zdone
        e.str_x_post(31, X10, 8); // str xzr, [x10], #8
        e.b(-3); // -> zl
                 // load the callee's pinned locals (after zeroing) and jump in
        e.ubfx_x(X13, X11, 32, 16); // zdone: callee l0off
        e.ldr_x_reg(L0R, FRAME, X13);
        e.lsr_x(X13, X11, 48); // callee l1off
        e.ldr_x_reg(L1R, FRAME, X13);
        e.ubfx_x(CODE, X9, 0, 48);
        e.mov_x(PC, CODE);
        e.ldr_x_imm(X9, PC, 0);
        e.br(X9);
    }

    // ---- return (a = first-result slot*8, b = result count) ----
    // Copies results to the frame base (the caller's staged-argument area:
    // frames overlap), then pops a return record. Sentinel records route
    // through `exit_cell` into `return_exit`.
    {
        let off = e.here();
        bump(e, COUNT_MODE == 0);
        e.ldr_x_imm(X10, PC, 8);
        e.ldr_x_imm(X11, PC, 16);
        e.add_x_reg(X10, FRAME, X10);
        e.mov_x(X12, FRAME);
        e.cbz_x(X11, 5); // -> pop
        e.ldr_x_post(X13, X10, 8); // cl:
        e.str_x_post(X13, X12, 8);
        e.sub_x_imm(X11, X11, 1);
        e.cbnz_x(X11, -3); // -> cl
        e.ldp_x_pre(X13, FRAME, RETSP, -(RET_RECORD as i32)); // pop:
        e.ldr_x_imm(X12, RETSP, 16); // caller code | caller_l0off<<48
        e.ldr_x_imm(X10, RETSP, 24); // caller l1off
        e.lsr_x(X11, X12, 48);
        e.ubfx_x(CODE, X12, 0, 48);
        // reload the caller's pinned locals (sentinel records carry a
        // readable dummy frame, so these loads are always safe)
        e.ldr_x_reg(L0R, FRAME, X11);
        e.ldr_x_reg(L1R, FRAME, X10);
        e.mov_x(PC, X13);
        e.ldr_x_imm(X9, PC, 0);
        e.br(X9);
        def(handlers, Op::Return, 0, off);
    }

    // ---- data movement (a ∈ {slot, acc, l0} for MovSlot, const for
    // MovConst; dst ∈ {mem, acc, l0}) ----
    let mut mov_variants: Vec<(Op, Cls)> = Vec::new();
    for a in [Cls::Slot, Cls::Acc, Cls::L0, Cls::L1] {
        mov_variants.push((Op::MovSlot, a));
    }
    mov_variants.push((Op::MovConst, Cls::Const));
    for &(op, a) in mov_variants.iter() {
        for d in DSTS {
            let off = e.here();
            let rd = dst_target(d);
            let v = match a {
                Cls::Acc => ACC,
                Cls::L0 => L0R,
                Cls::L1 => L1R,
                Cls::Const => {
                    e.ldr_x_imm(rd, PC, 8);
                    rd
                }
                Cls::Slot => {
                    e.ldr_x_imm(X10, PC, 8);
                    e.ldr_x_reg(rd, FRAME, X10);
                    rd
                }
            };
            if v != rd {
                e.mov_x(rd, v);
            }
            finish(e, d, rd);
            tail(e, counted(a, Cls::Slot, d));
            def(handlers, op, variant(a, Cls::Slot, d), off);
        }
    }

    // ---- binary ALU, all 4 operand-class variants ----
    // kind: 0 = plain 3-reg ALU, 1 = rotl (neg + rorv)
    struct Bin {
        op: Op,
        w: bool,
        enc: u32,
        kind: u8,
    }
    let bins = [
        Bin {
            op: Op::I32_Add,
            w: true,
            enc: 0x0B00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I32_Sub,
            w: true,
            enc: 0x4B00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I32_And,
            w: true,
            enc: 0x0A00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I32_Or,
            w: true,
            enc: 0x2A00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I32_Xor,
            w: true,
            enc: 0x4A00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I32_Mul,
            w: true,
            enc: 0x1B00_7C00,
            kind: 0,
        },
        Bin {
            op: Op::I32_Shl,
            w: true,
            enc: 0x1AC0_2000,
            kind: 0,
        },
        Bin {
            op: Op::I32_ShrU,
            w: true,
            enc: 0x1AC0_2400,
            kind: 0,
        },
        Bin {
            op: Op::I32_ShrS,
            w: true,
            enc: 0x1AC0_2800,
            kind: 0,
        },
        Bin {
            op: Op::I32_Rotr,
            w: true,
            enc: 0x1AC0_2C00,
            kind: 0,
        },
        Bin {
            op: Op::I32_Rotl,
            w: true,
            enc: 0x1AC0_2C00,
            kind: 1,
        },
        Bin {
            op: Op::I64_Add,
            w: false,
            enc: 0x8B00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I64_Sub,
            w: false,
            enc: 0xCB00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I64_And,
            w: false,
            enc: 0x8A00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I64_Or,
            w: false,
            enc: 0xAA00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I64_Xor,
            w: false,
            enc: 0xCA00_0000,
            kind: 0,
        },
        Bin {
            op: Op::I64_Mul,
            w: false,
            enc: 0x9B00_7C00,
            kind: 0,
        },
        Bin {
            op: Op::I64_Shl,
            w: false,
            enc: 0x9AC0_2000,
            kind: 0,
        },
        Bin {
            op: Op::I64_ShrU,
            w: false,
            enc: 0x9AC0_2400,
            kind: 0,
        },
        Bin {
            op: Op::I64_ShrS,
            w: false,
            enc: 0x9AC0_2800,
            kind: 0,
        },
        Bin {
            op: Op::I64_Rotr,
            w: false,
            enc: 0x9AC0_2C00,
            kind: 0,
        },
        Bin {
            op: Op::I64_Rotl,
            w: false,
            enc: 0x9AC0_2C00,
            kind: 1,
        },
    ];
    for b in bins.iter() {
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                for d in DSTS {
                    let off = e.here();
                    let ra = src_a(e, a_cls, X10);
                    let mut rb = src_b(e, b_cls, X11);
                    if b.kind == 1 {
                        // rotl x, n == rotr x, -n (rorv masks the amount)
                        if b.w {
                            e.neg_w(X11, rb);
                        } else {
                            e.neg_x(X11, rb);
                        }
                        rb = X11;
                    }
                    let rd = dst_target(d);
                    e.alu_reg(b.enc, rd, ra, rb);
                    finish(e, d, rd);
                    tail(e, counted(a_cls, b_cls, d));
                    def(handlers, b.op, variant(a_cls, b_cls, d), off);
                }
            }
        }
    }

    // ---- compares (cmp + cset) ----
    struct Cmp {
        op: Op,
        w: bool,
        cond: u32,
    }
    let cmps = [
        Cmp {
            op: Op::I32_Eq,
            w: true,
            cond: EQ,
        },
        Cmp {
            op: Op::I32_Ne,
            w: true,
            cond: NE,
        },
        Cmp {
            op: Op::I32_LtS,
            w: true,
            cond: LT,
        },
        Cmp {
            op: Op::I32_LtU,
            w: true,
            cond: LO,
        },
        Cmp {
            op: Op::I32_GtS,
            w: true,
            cond: GT,
        },
        Cmp {
            op: Op::I32_GtU,
            w: true,
            cond: HI,
        },
        Cmp {
            op: Op::I32_LeS,
            w: true,
            cond: LE,
        },
        Cmp {
            op: Op::I32_LeU,
            w: true,
            cond: LS,
        },
        Cmp {
            op: Op::I32_GeS,
            w: true,
            cond: GE,
        },
        Cmp {
            op: Op::I32_GeU,
            w: true,
            cond: HS,
        },
        Cmp {
            op: Op::I64_Eq,
            w: false,
            cond: EQ,
        },
        Cmp {
            op: Op::I64_Ne,
            w: false,
            cond: NE,
        },
        Cmp {
            op: Op::I64_LtS,
            w: false,
            cond: LT,
        },
        Cmp {
            op: Op::I64_LtU,
            w: false,
            cond: LO,
        },
        Cmp {
            op: Op::I64_GtS,
            w: false,
            cond: GT,
        },
        Cmp {
            op: Op::I64_GtU,
            w: false,
            cond: HI,
        },
        Cmp {
            op: Op::I64_LeS,
            w: false,
            cond: LE,
        },
        Cmp {
            op: Op::I64_LeU,
            w: false,
            cond: LS,
        },
        Cmp {
            op: Op::I64_GeS,
            w: false,
            cond: GE,
        },
        Cmp {
            op: Op::I64_GeU,
            w: false,
            cond: HS,
        },
    ];
    for c in cmps.iter() {
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                for d in DSTS {
                    let off = e.here();
                    let ra = src_a(e, a_cls, X10);
                    let rb = src_b(e, b_cls, X11);
                    if c.w {
                        e.cmp_w(ra, rb);
                    } else {
                        e.cmp_x(ra, rb);
                    }
                    let rd = dst_target(d);
                    e.cset_w(rd, c.cond);
                    finish(e, d, rd);
                    tail(e, counted(a_cls, b_cls, d));
                    def(handlers, c.op, variant(a_cls, b_cls, d), off);
                }
            }
        }
    }
    for (op, w) in [(Op::I32_Eqz, true), (Op::I64_Eqz, false)] {
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                let ra = src_a(e, a_cls, X10);
                if w {
                    e.cmp_w(ra, 31); // wzr
                } else {
                    e.cmp_x(ra, 31);
                }
                let rd = dst_target(d);
                e.cset_w(rd, EQ);
                finish(e, d, rd);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // ---- unary int ops / width conversions ----
    for op in [
        Op::I32_Clz,
        Op::I64_Clz,
        Op::I32_Ctz,
        Op::I64_Ctz,
        Op::I32_Extend8S,
        Op::I32_Extend16S,
        Op::I64_Extend8S,
        Op::I64_Extend16S,
        Op::I64_Extend32S,
        Op::I32_WrapI64,
        Op::I64_ExtendI32S,
        Op::I64_ExtendI32U,
    ] {
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                let ra = src_a(e, a_cls, X10);
                let rd = dst_target(d);
                match op {
                    Op::I32_Clz => e.clz_w(rd, ra),
                    Op::I64_Clz => e.clz_x(rd, ra),
                    Op::I32_Ctz => {
                        e.rbit_w(rd, ra);
                        e.clz_w(rd, rd);
                    }
                    Op::I64_Ctz => {
                        e.rbit_x(rd, ra);
                        e.clz_x(rd, rd);
                    }
                    Op::I32_Extend8S => e.sxtb_w(rd, ra),
                    Op::I32_Extend16S => e.sxth_w(rd, ra),
                    Op::I64_Extend8S => e.sxtb_x(rd, ra),
                    Op::I64_Extend16S => e.sxth_x(rd, ra),
                    Op::I64_Extend32S => e.sxtw_x(rd, ra),
                    Op::I32_WrapI64 => e.mov_w(rd, ra),
                    Op::I64_ExtendI32S => e.sxtw_x(rd, ra),
                    Op::I64_ExtendI32U => e.mov_w(rd, ra),
                    _ => unreachable!(),
                }
                finish(e, d, rd);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // ---- branches ----
    {
        let off = e.here();
        bump(e, COUNT_MODE == 0);
        e.ldr_x_imm(X12, PC, 24);
        e.add_x_reg(PC, CODE, X12);
        e.ldr_x_imm(X9, PC, 0);
        e.br(X9);
        def(handlers, Op::Br, 0, off);
    }
    for (op, taken_on_nonzero) in [(Op::BrIf, true), (Op::BrIfNot, false)] {
        for a_cls in CLASSES {
            let cnt = counted(a_cls, Cls::Slot, DstCls::Mem);
            let off = e.here();
            let ra = src_a(e, a_cls, X10);
            // not-taken: skip the taken path into the tail (its length
            // depends on whether this variant bumps the counter)
            if taken_on_nonzero {
                e.cbz_w(ra, 5 + cnt as i32);
            } else {
                e.cbnz_w(ra, 5 + cnt as i32);
            }
            bump(e, cnt);
            e.ldr_x_imm(X12, PC, 24);
            e.add_x_reg(PC, CODE, X12);
            e.ldr_x_imm(X9, PC, 0);
            e.br(X9);
            tail(e, counted(a_cls, Cls::Slot, DstCls::Mem));
            def(handlers, op, variant(a_cls, Cls::Slot, DstCls::Mem), off);
        }
    }

    // ---- fused compare-and-branch ----
    let cmp_brs = [
        (Op::I32_BrEq, true, EQ),
        (Op::I32_BrNe, true, NE),
        (Op::I32_BrLtS, true, LT),
        (Op::I32_BrLtU, true, LO),
        (Op::I32_BrGtS, true, GT),
        (Op::I32_BrGtU, true, HI),
        (Op::I32_BrLeS, true, LE),
        (Op::I32_BrLeU, true, LS),
        (Op::I32_BrGeS, true, GE),
        (Op::I32_BrGeU, true, HS),
        (Op::I64_BrEq, false, EQ),
        (Op::I64_BrNe, false, NE),
        (Op::I64_BrLtS, false, LT),
        (Op::I64_BrLtU, false, LO),
        (Op::I64_BrGtS, false, GT),
        (Op::I64_BrGtU, false, HI),
        (Op::I64_BrLeS, false, LE),
        (Op::I64_BrLeU, false, LS),
        (Op::I64_BrGeS, false, GE),
        (Op::I64_BrGeU, false, HS),
    ];
    for &(op, w, cond) in cmp_brs.iter() {
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                let off = e.here();
                let ra = src_a(e, a_cls, X10);
                let rb = src_b(e, b_cls, X11);
                if w {
                    e.cmp_w(ra, rb);
                } else {
                    e.cmp_x(ra, rb);
                }
                let cnt = counted(a_cls, b_cls, DstCls::Mem);
                // not-taken: inverted condition skips the taken path
                e.b_cond(cond ^ 1, 5 + cnt as i32);
                bump(e, cnt);
                e.ldr_x_imm(X12, PC, 24);
                e.add_x_reg(PC, CODE, X12);
                e.ldr_x_imm(X9, PC, 0);
                e.br(X9);
                tail(e, counted(a_cls, b_cls, DstCls::Mem));
                def(handlers, op, variant(a_cls, b_cls, DstCls::Mem), off);
            }
        }
    }

    // ---- select (c = cond_byteoff << 32 | dst_byteoff; pinned dsts
    // compute into their register and still store — write-through) ----
    for a_cls in CLASSES {
        for b_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                let ra = src_a(e, a_cls, X10);
                let rb = src_b(e, b_cls, X11);
                e.ldr_x_imm(X12, PC, 24);
                e.lsr_x(X13, X12, 32); // cond slot byte offset
                e.ldr_x_reg(X13, FRAME, X13);
                e.cmp_x(X13, 31);
                let rd = dst_target(d);
                e.csel_x(rd, ra, rb, NE);
                if d != DstCls::Acc {
                    e.mov_w(X12, X12); // dst slot byte offset
                    e.str_x_reg(rd, FRAME, X12);
                }
                tail(e, counted(a_cls, b_cls, d));
                def(handlers, Op::Select, variant(a_cls, b_cls, d), off);
            }
        }
    }

    // ---- br_table (a = index slot, b = flat table ptr, c = len-1) ----
    // The flat table is `[targets..., default]`; clamping the index to
    // len-1 selects the default for any out-of-range value.
    for a_cls in [Cls::Slot, Cls::Acc, Cls::L0, Cls::L1] {
        let off = e.here();
        bump(e, counted(a_cls, Cls::Slot, DstCls::Mem));
        let ra = src_a(e, a_cls, X10);
        e.ldr_x_imm(X11, PC, 16);
        e.ldr_x_imm(X12, PC, 24);
        e.cmp_w(ra, X12);
        e.csel_w(X10, ra, X12, LO);
        e.ldr_w_uxtw2(X10, X11, X10); // target instruction index
        e.lsl_x(X10, X10, 5); // ×32 → cell byte offset
        e.add_x_reg(PC, CODE, X10);
        e.ldr_x_imm(X9, PC, 0);
        e.br(X9);
        def(
            handlers,
            Op::BrTable,
            variant(a_cls, Cls::Slot, DstCls::Mem),
            off,
        );
    }

    // ---- globals (indices pre-scaled ×8 at link) ----
    for d in DSTS {
        let off = e.here();
        e.ldr_x_imm(X10, PC, 8); // a = index*8
        let rd = dst_target(d);
        e.ldr_x_reg(rd, GLOB, X10);
        finish(e, d, rd);
        tail(e, counted(Cls::Slot, Cls::Slot, d));
        def(
            handlers,
            Op::GlobalGet,
            variant(Cls::Slot, Cls::Slot, d),
            off,
        );
    }
    for a_cls in CLASSES {
        let off = e.here();
        let ra = src_a(e, a_cls, X10); // value operand
        e.ldr_x_imm(X12, PC, 24); // c = index*8
        e.str_x_reg(ra, GLOB, X12);
        tail(e, counted(a_cls, Cls::Slot, DstCls::Mem));
        def(
            handlers,
            Op::GlobalSet,
            variant(a_cls, Cls::Slot, DstCls::Mem),
            off,
        );
    }

    // ---- memory 0 loads/stores with inline bounds checks ----
    // access kind: 0 = w, 1 = x, 2 = b, 3 = h, 4 = sb->w, 5 = sh->w,
    // 6 = sb->x, 7 = sh->x, 8 = sw->x
    struct Mem {
        op: Op,
        size: u32,
        load: bool,
        kind: u8,
    }
    let mems = [
        Mem {
            op: Op::I32_Load,
            size: 4,
            load: true,
            kind: 0,
        },
        Mem {
            op: Op::F32_Load,
            size: 4,
            load: true,
            kind: 0,
        },
        Mem {
            op: Op::I64_Load,
            size: 8,
            load: true,
            kind: 1,
        },
        Mem {
            op: Op::F64_Load,
            size: 8,
            load: true,
            kind: 1,
        },
        Mem {
            op: Op::I32_Load8U,
            size: 1,
            load: true,
            kind: 2,
        },
        Mem {
            op: Op::I64_Load8U,
            size: 1,
            load: true,
            kind: 2,
        },
        Mem {
            op: Op::I32_Load16U,
            size: 2,
            load: true,
            kind: 3,
        },
        Mem {
            op: Op::I64_Load16U,
            size: 2,
            load: true,
            kind: 3,
        },
        Mem {
            op: Op::I32_Load8S,
            size: 1,
            load: true,
            kind: 4,
        },
        Mem {
            op: Op::I32_Load16S,
            size: 2,
            load: true,
            kind: 5,
        },
        Mem {
            op: Op::I64_Load8S,
            size: 1,
            load: true,
            kind: 6,
        },
        Mem {
            op: Op::I64_Load16S,
            size: 2,
            load: true,
            kind: 7,
        },
        Mem {
            op: Op::I64_Load32S,
            size: 4,
            load: true,
            kind: 8,
        },
        Mem {
            op: Op::I64_Load32U,
            size: 4,
            load: true,
            kind: 0,
        },
        Mem {
            op: Op::I32_Store,
            size: 4,
            load: false,
            kind: 0,
        },
        Mem {
            op: Op::F32_Store,
            size: 4,
            load: false,
            kind: 0,
        },
        Mem {
            op: Op::I64_Store,
            size: 8,
            load: false,
            kind: 1,
        },
        Mem {
            op: Op::F64_Store,
            size: 8,
            load: false,
            kind: 1,
        },
        Mem {
            op: Op::I32_Store8,
            size: 1,
            load: false,
            kind: 2,
        },
        Mem {
            op: Op::I64_Store8,
            size: 1,
            load: false,
            kind: 2,
        },
        Mem {
            op: Op::I32_Store16,
            size: 2,
            load: false,
            kind: 3,
        },
        Mem {
            op: Op::I64_Store16,
            size: 2,
            load: false,
            kind: 3,
        },
        Mem {
            op: Op::I64_Store32,
            size: 4,
            load: false,
            kind: 0,
        },
    ];
    for m in mems.iter() {
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                for d in DSTS {
                    // A load's b is the static offset, not an operand;
                    // stores produce nothing.
                    if m.load && b_cls != Cls::Slot || !m.load && d != DstCls::Mem {
                        continue;
                    }
                    let off = e.here();
                    let ra = src_a(e, a_cls, X10); // address (u32 low bits)
                    if m.load {
                        e.ldr_x_imm(X11, PC, 16); // static offset
                    } else {
                        e.ldr_x_imm(X11, PC, 24); // stores: offset in c
                    }
                    e.add_x_uxtw(X12, X11, ra); // ea = offset + zext(addr)
                    e.add_x_imm(X13, X12, m.size);
                    e.cmp_x(X13, MEMLEN);
                    {
                        let delta = (trap_oob as i64 - e.here() as i64) / 4;
                        e.b_cond(HI, delta as i32);
                    }
                    if m.load {
                        let rd = dst_target(d);
                        match m.kind {
                            0 => e.ldr_w_reg(rd, MEM, X12),
                            1 => e.ldr_x_reg(rd, MEM, X12),
                            2 => e.ldrb_reg(rd, MEM, X12),
                            3 => e.ldrh_reg(rd, MEM, X12),
                            4 => e.ldrsb_w_reg(rd, MEM, X12),
                            5 => e.ldrsh_w_reg(rd, MEM, X12),
                            6 => e.ldrsb_x_reg(rd, MEM, X12),
                            7 => e.ldrsh_x_reg(rd, MEM, X12),
                            8 => e.ldrsw_reg(rd, MEM, X12),
                            _ => unreachable!(),
                        }
                        finish(e, d, rd);
                    } else {
                        // value operand (the offset in X11 is consumed)
                        let rb = src_b(e, b_cls, X11);
                        match m.kind {
                            0 => e.str_w_reg(rb, MEM, X12),
                            1 => e.str_x_reg(rb, MEM, X12),
                            2 => e.strb_reg(rb, MEM, X12),
                            3 => e.strh_reg(rb, MEM, X12),
                            _ => unreachable!(),
                        }
                    }
                    tail(e, counted(a_cls, b_cls, d));
                    def(handlers, m.op, variant(a_cls, b_cls, d), off);
                }
            }
        }
    }

    EmitOut {
        entry,
        slow_stub,
        call_handler,
        return_exit,
    }
}
