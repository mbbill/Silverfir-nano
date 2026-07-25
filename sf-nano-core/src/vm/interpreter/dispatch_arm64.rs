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

use super::instr::{
    operand_is_float, result_is_float, Instr, Op, FLAG_A_ACC, FLAG_A_CONST, FLAG_B_ACC,
    FLAG_B_CONST, FLAG_DST_ACC, FLAG_FUSED,
};
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
    pub acc_value: u64,   // 104: the accumulator (in AND out — call results
    // ride it across activation boundaries)
    pub table0_base: u64, // 112: funcref table 0 entries (in only, u32 elems)
    pub table0_len: u64,  // 120
    pub indirect_base: u64, // 128: per-function indirect-call info, [u64;3]
                          // per function index (in only)
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
    /// Whether any pinned slot is float-mode: drives the conditional
    /// float-twin reloads on the call and return paths.
    pub fp_pinned: bool,
}

const N_OPS: usize = Op::Unreachable as usize + 1;
/// Variant slots per op: operand classes a×b ∈ {slot, const, acc, l0,
/// l1}², destination ∈ {mem, acc, l0, l1} — densely mapped, 5·5·4 = 100.
/// Address-fused memory ops occupy a second bank at +100.
const N_VARIANTS: usize = 200;

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
    if between(op, I32_BrEq, I32_BrAndNot) {
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
        MovPair => (true, true, false),
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
fn select_pinned(func: &PredecodedFunction) -> (u64, u64, bool, bool) {
    let n = func.n_locals as u64;
    if n == 0 {
        return (u64::MAX, u64::MAX, false, false);
    }
    let mut counts = vec![0u64; func.n_locals as usize];
    // Per-slot domain sets (bit 0 = integer/agnostic, bit 1 = float):
    // writers decide the pinned register file; mixed writers make the
    // slot unpinnable (neither file could stay authoritative). Readers
    // only break the tie when no writer exists.
    let mut wdom = vec![0u8; func.n_locals as usize];
    let mut rdom = vec![0u8; func.n_locals as usize];
    for ins in func.code.iter() {
        let (a_s, b_s, c_d) = slot_fields(ins.op);
        if a_s && ins.flags & FLAG_A_CONST == 0 && ins.a < n {
            counts[ins.a as usize] += 1;
            rdom[ins.a as usize] |= if operand_is_float(ins.op, false) {
                2
            } else {
                1
            };
        }
        if b_s && ins.flags & FLAG_B_CONST == 0 && ins.b < n {
            counts[ins.b as usize] += 1;
            rdom[ins.b as usize] |= if operand_is_float(ins.op, true) { 2 } else { 1 };
        }
        if c_d && ins.c < n {
            counts[ins.c as usize] += 1;
            wdom[ins.c as usize] |= if result_is_float(ins.op) { 2 } else { 1 };
        }
        if ins.op == Op::Select {
            let dslot = ins.c & 0xffff_ffff;
            if dslot < n {
                wdom[dslot as usize] |= 1;
            }
        }
    }
    let mut best = (usize::MAX, 0u64);
    let mut second = (usize::MAX, 0u64);
    for (i, &c) in counts.iter().enumerate() {
        if wdom[i] == 3 {
            continue; // mixed-domain writers: unpinnable
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
    let mode = |i: usize| wdom[i] == 2 || (wdom[i] == 0 && rdom[i] == 2);
    let (l0, l0f) = if ok(best) {
        (best.0 as u64, mode(best.0))
    } else {
        (u64::MAX, false)
    };
    let (l1, l1f) = if l0 != u64::MAX && ok(second) {
        (second.0 as u64, mode(second.0))
    } else {
        (u64::MAX, false)
    };
    (l0, l1, l0f, l1f)
}

/// Classify one cell into its dense variant, given link-resolved flags
/// and the function's pinned slots (`u64::MAX` = none). Pinned classes
/// take precedence over acc hints on the same operand; a `const` flag
/// wins over both.
fn op_variant(ins: &Instr, flags: u16, l0_slot: u64, l1_slot: u64, l0f: bool, l1f: bool) -> usize {
    let (a_s, b_s, c_d) = slot_fields(ins.op);
    // Domain demotion: a pinned class is taken only when the access's
    // value domain matches the slot's pinned register file; otherwise
    // the access falls back to the (write-through, hence current) slot.
    let af = operand_is_float(ins.op, false);
    let bf = operand_is_float(ins.op, true);
    let rf = result_is_float(ins.op);
    let a = if flags & FLAG_A_CONST != 0 {
        Cls::Const
    } else if a_s && ins.a == l0_slot && af == l0f {
        Cls::L0
    } else if a_s && ins.a == l1_slot && af == l1f {
        Cls::L1
    } else if flags & FLAG_A_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    };
    let b = if flags & FLAG_B_CONST != 0 {
        Cls::Const
    } else if b_s && ins.b == l0_slot && bf == l0f {
        Cls::L0
    } else if b_s && ins.b == l1_slot && bf == l1f {
        Cls::L1
    } else if flags & FLAG_B_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    };
    let d = if ins.op == Op::Select {
        // Select's dst slot is packed in c's low half (integer-domain
        // writer: a float-pinned dst slot cannot occur — the writer
        // scan would have made it unpinnable).
        let dslot = ins.c & 0xffff_ffff;
        if dslot == l0_slot && !l0f {
            DstCls::L0
        } else if dslot == l1_slot && !l1f {
            DstCls::L1
        } else if flags & FLAG_DST_ACC != 0 {
            DstCls::Acc
        } else {
            DstCls::Mem
        }
    } else if flags & FLAG_FUSED != 0 && c_d {
        // fused loads pack addr2 in c's high half; the dst is the low
        let dslot = ins.c & 0xffff_ffff;
        if dslot == l0_slot && rf == l0f {
            DstCls::L0
        } else if dslot == l1_slot && rf == l1f {
            DstCls::L1
        } else if flags & FLAG_DST_ACC != 0 {
            DstCls::Acc
        } else {
            DstCls::Mem
        }
    } else if c_d && ins.c == l0_slot && rf == l0f {
        DstCls::L0
    } else if c_d && ins.c == l1_slot && rf == l1f {
        DstCls::L1
    } else if flags & FLAG_DST_ACC != 0 {
        DstCls::Acc
    } else {
        DstCls::Mem
    };
    let v = variant(a, b, d);
    if flags & FLAG_FUSED != 0 {
        v + 100
    } else {
        v
    }
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
        // b packs the memory index (copy: dst<<32 | src)
        MemoryFill | MemoryCopy => ins.b == 0,
        _ => true,
    }
}

/// Per-op b/c operand pre-scaling for native handlers (a is handled
/// uniformly at the call site: slot index ×8 unless const). `flags` are
/// the link-resolved flags (acc hints possibly stripped).
fn transform_bc(ins: &Instr, flags: u16) -> (u64, u64) {
    use Op::*;
    if flags & FLAG_FUSED != 0 {
        return if between(ins.op, I32_Load, I64_Load32U) {
            // loads: b = static offset (raw), c = addr2*8 << 32 | dst*8
            (ins.b, ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8)
        } else {
            // stores: b = value, c = addr2*8 << 32 | static offset (raw)
            let b = if flags & FLAG_B_CONST != 0 {
                ins.b
            } else {
                ins.b * 8
            };
            (b, ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff))
        };
    }
    match ins.op {
        // control: c = target cell byte offset; b unused
        Br | BrIf | BrIfNot => (ins.b, ins.c * CELL as u64),
        // fused compare-branches: b = compare operand, c = target
        I32_BrEq | I32_BrNe | I32_BrLtS | I32_BrLtU | I32_BrGtS | I32_BrGtU | I32_BrLeS
        | I32_BrLeU | I32_BrGeS | I32_BrGeU | I64_BrEq | I64_BrNe | I64_BrLtS | I64_BrLtU
        | I64_BrGtS | I64_BrGtU | I64_BrLeS | I64_BrLeU | I64_BrGeS | I64_BrGeU | I32_BrAnd
        | I32_BrAndNot => {
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
        // MovPair: b = second source slot, c = dst1<<32 | dst2
        MovPair => (
            ins.b * 8,
            ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8,
        ),
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
    callindirect_handler: usize,
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
            callindirect_handler: out.callindirect_handler,
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

    /// Address of the native `CallIndirect` handler, for the fixup pass.
    pub(super) fn callindirect_handler_addr(&self) -> u64 {
        unsafe { self.buf.ptr(self.callindirect_handler) as u64 }
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
        let (l0_slot, l1_slot, l0f, l1f) = select_pinned(func);

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
            // Call/CallIndirect are wired by the fixup pass, never via the
            // handler table; every call flavor (native, slow, host) delivers
            // result 0 through the accumulator relay, so they are valid
            // producers regardless of the table lookup.
            let prev_is_call = j > 0 && matches!(func.code[j - 1].op, Op::Call | Op::CallIndirect);
            let ok = j > 0
                && (writes_acc(func.code[j - 1].op) || prev_is_call)
                && (prev_is_call
                    || self.h(
                        func.code[j - 1].op,
                        op_variant(&func.code[j - 1], flags[j - 1], l0_slot, l1_slot, l0f, l1f),
                    ) != u32::MAX)
                && self.h(
                    func.code[j].op,
                    op_variant(&func.code[j], flags[j], l0_slot, l1_slot, l0f, l1f),
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
            let mut off = self.h(ins.op, op_variant(ins, fl, l0_slot, l1_slot, l0f, l1f));
            // MovPair's packed dsts are never classed: one that writes a
            // pinned slot must run slow so the re-entry reload keeps the
            // pinned register current.
            if ins.op == Op::MovPair
                && (ins.c >> 32 == l0_slot
                    || ins.c >> 32 == l1_slot
                    || ins.c & 0xffff_ffff == l0_slot
                    || ins.c & 0xffff_ffff == l1_slot)
            {
                off = u32::MAX;
            }
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
            fp_pinned: (l0_slot != u64::MAX && l0f) || (l1_slot != u64::MAX && l1f),
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
const X7: u32 = 7;
/// The float accumulator: v16, caller-saved, so the entry/exit paths
/// never touch it. Strict-adjacency pairing means it is never live
/// across a chain exit (float acc producers only bail to traps).
const FACC: u32 = 16;
/// Float views of the pinned-local slots (v17/v18, caller-saved,
/// reloaded at every chain entry / call / return like their integer
/// twins x17/x14). A slot is float-pinned only when every writer is
/// float-domain, so exactly one register file is authoritative per
/// slot; reads from the other domain are demoted to the slot at link.
const FL0R: u32 = 17;
const FL1R: u32 = 18;
const X10: u32 = 10;
/// Extra scratch for the indirect-call handler (safe: handlers never
/// call out, so the platform's IP0 role is irrelevant here).
const X16: u32 = 16;
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
const MI: u32 = 4;
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
    fn tst_w(&mut self, rn: u32, rm: u32) {
        self.i(0x6A00_0000 | rm << 16 | rn << 5 | 31);
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
    fn ldp_x_base_imm(&mut self, rt: u32, rt2: u32, rn: u32, imm: u32) {
        self.i(0xA940_0000 | (imm / 8) << 15 | rt2 << 10 | rn << 5 | rt);
    }
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
    fn ldr_d_reg(&mut self, vt: u32, rn: u32, rm: u32) {
        self.i(0xFC60_6800 | rm << 16 | rn << 5 | vt);
    }
    fn ldr_s_reg(&mut self, vt: u32, rn: u32, rm: u32) {
        self.i(0xBC60_6800 | rm << 16 | rn << 5 | vt);
    }
    fn ldr_d_imm(&mut self, vt: u32, rn: u32, imm: u32) {
        self.i(0xFD40_0000 | (imm / 8) << 10 | rn << 5 | vt);
    }
    fn str_d_reg(&mut self, vt: u32, rn: u32, rm: u32) {
        self.i(0xFC20_6800 | rm << 16 | rn << 5 | vt);
    }
    fn str_s_reg(&mut self, vt: u32, rn: u32, rm: u32) {
        self.i(0xBC20_6800 | rm << 16 | rn << 5 | vt);
    }
    fn ldr_s_imm(&mut self, vt: u32, rn: u32, imm: u32) {
        self.i(0xBD40_0000 | (imm / 4) << 10 | rn << 5 | vt);
    }
    fn fmov_d_x(&mut self, vd: u32, xn: u32) {
        self.i(0x9E67_0000 | xn << 5 | vd);
    }
    fn fmov_x_d(&mut self, xd: u32, vn: u32) {
        self.i(0x9E66_0000 | vn << 5 | xd);
    }
    fn fmov_s_w(&mut self, vd: u32, wn: u32) {
        self.i(0x1E27_0000 | wn << 5 | vd);
    }
    fn fmov_w_s(&mut self, wd: u32, vn: u32) {
        self.i(0x1E26_0000 | vn << 5 | wd);
    }
    /// Generic 3-register FP op (fadd/fsub/fmul/fdiv/fmin/fmax forms).
    fn fp2(&mut self, base: u32, vd: u32, vn: u32, vm: u32) {
        self.i(base | vm << 16 | vn << 5 | vd);
    }
    /// Generic 2-register FP/convert op.
    fn fp1(&mut self, base: u32, rd: u32, rn: u32) {
        self.i(base | rn << 5 | rd);
    }
    fn fcmp(&mut self, f32w: bool, vn: u32, vm: u32) {
        let base = if f32w { 0x1E20_2000 } else { 0x1E60_2000 };
        self.i(base | vm << 16 | vn << 5);
    }
    fn tbz(&mut self, rt: u32, bit: u32, delta_insns: i32) {
        debug_assert!(bit < 32);
        self.i(0x3600_0000 | bit << 19 | ((delta_insns as u32) & 0x3FFF) << 5 | rt);
    }
    fn tbnz(&mut self, rt: u32, bit: u32, delta_insns: i32) {
        debug_assert!(bit < 32);
        self.i(0x3700_0000 | bit << 19 | ((delta_insns as u32) & 0x3FFF) << 5 | rt);
    }
    fn movz_hw(&mut self, rd: u32, imm16: u32, hw: u32) {
        self.i(0xD280_0000 | hw << 21 | imm16 << 5 | rd);
    }
    fn movk_hw(&mut self, rd: u32, imm16: u32, hw: u32) {
        self.i(0xF280_0000 | hw << 21 | imm16 << 5 | rd);
    }
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
/// Prefetch the NEXT cell's handler word at handler entry: the load
/// issues in parallel with the handler body, so the dispatch branch's
/// operand is ready (and a misprediction resolves) sooner. Every
/// handler that ends in [`tail`] must open with this.
fn pre(e: &mut Enc<'_>) {
    e.ldr_x_imm(X7, PC, CELL);
}

fn tail(e: &mut Enc<'_>, count: bool) {
    bump(e, count);
    e.add_x_imm(PC, PC, CELL);
    e.br(X7);
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

/// Fetch both operands; when both live in slots the two cell fields
/// load with a single ldp (one instruction saved on the hottest binop
/// variant).
fn src_ab(e: &mut Enc<'_>, a_cls: Cls, b_cls: Cls) -> (u32, u32) {
    if a_cls == Cls::Slot && b_cls == Cls::Slot {
        e.ldp_x_base_imm(X10, X11, PC, 8);
        e.ldr_x_reg(X10, FRAME, X10);
        e.ldr_x_reg(X11, FRAME, X11);
        (X10, X11)
    } else {
        (src_a(e, a_cls, X10), src_b(e, b_cls, X11))
    }
}

/// Float twin of [`src_ab`].
fn src_fp_ab(e: &mut Enc<'_>, a_cls: Cls, b_cls: Cls, f32w: bool) -> (u32, u32) {
    if a_cls == Cls::Slot && b_cls == Cls::Slot {
        e.ldp_x_base_imm(X10, X11, PC, 8);
        if f32w {
            e.ldr_s_reg(0, FRAME, X10);
            e.ldr_s_reg(1, FRAME, X11);
        } else {
            e.ldr_d_reg(0, FRAME, X10);
            e.ldr_d_reg(1, FRAME, X11);
        }
        (0, 1)
    } else {
        (
            src_fp(e, a_cls, f32w, 0, 8, X10),
            src_fp(e, b_cls, f32w, 1, 16, X11),
        )
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

/// Load a floating operand into FP register `v`. Register classes move
/// raw bits over (f32 values are zero-extended in slots and registers,
/// so an S-view read is always exact).
fn src_fp(e: &mut Enc<'_>, cls: Cls, f32w: bool, v: u32, pcoff: u32, tmp: u32) -> u32 {
    match cls {
        Cls::Slot => {
            e.ldr_x_imm(tmp, PC, pcoff);
            if f32w {
                e.ldr_s_reg(v, FRAME, tmp);
            } else {
                e.ldr_d_reg(v, FRAME, tmp);
            }
            v
        }
        Cls::Const => {
            if f32w {
                e.ldr_s_imm(v, PC, pcoff);
            } else {
                e.ldr_d_imm(v, PC, pcoff);
            }
            v
        }
        Cls::Acc => FACC,
        // A float-domain access classed L0/L1 only links inside a
        // float-pinned function, where the value is register-resident
        // in the NEON file — no cross-domain move.
        Cls::L0 => FL0R,
        Cls::L1 => FL1R,
    }
}

/// The NEON register a float result computes into, per destination
/// class: pinned destinations ARE registers, everything else stages in
/// the float accumulator (which doubles as the Mem-dst mirror).
fn fp_target(d: DstCls) -> u32 {
    match d {
        DstCls::L0 => FL0R,
        DstCls::L1 => FL1R,
        _ => FACC,
    }
}

/// Land a float result: it was computed into `fp_target(d)`. Non-acc
/// destinations store to the slot straight from the NEON register
/// (D-width even for f32 — S-form producers zero the upper bits, so
/// the slot zero-extension convention holds for free); for pinned
/// destinations this is the write-through.
fn finish_fp(e: &mut Enc<'_>, d: DstCls) {
    if d != DstCls::Acc {
        e.ldr_x_imm(X12, PC, 24);
        e.str_d_reg(fp_target(d), FRAME, X12);
    }
}

/// Move an FP result's bits into an integer destination register (the
/// S-form zero-extends, upholding the f32 slot convention).
fn fp_result(e: &mut Enc<'_>, f32w: bool, rd: u32, v: u32) {
    if f32w {
        e.fmov_w_s(rd, v);
    } else {
        e.fmov_x_d(rd, v);
    }
}

/// Materialize an arbitrary 64-bit constant (movz + movk per non-zero
/// halfword).
fn mov_imm64(e: &mut Enc<'_>, rd: u32, val: u64) {
    let mut set = false;
    for hw in 0..4 {
        let part = ((val >> (hw * 16)) & 0xFFFF) as u32;
        if part == 0 {
            continue;
        }
        if set {
            e.movk_hw(rd, part, hw);
        } else {
            e.movz_hw(rd, part, hw);
            set = true;
        }
    }
    if !set {
        e.movz_hw(rd, 0, 0);
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
    callindirect_handler: usize,
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
    e.str_x_imm(ACC, STATE, 104); // ...and the accumulator (call results)
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
    e.ldr_x_imm(ACC, STATE, 104);
    e.ldr_d_imm(FL0R, STATE, 88);
    e.ldr_d_imm(FL1R, STATE, 96);
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
        e.ldr_x_imm(X11, PC, 16);
        e.ldr_x_imm(X9, PC, 8);
    }
    // The shared activation-entry tail: X9/X11/X12 hold the a/b/c-shaped
    // callee description (the indirect handler composes them from its
    // runtime lookup instead of cell fields).
    let call_core = e.here();
    {
        // return-stack depth
        e.cmp_x(RETSP, RETLIM);
        {
            let delta = (trap_exhaust as i64 - e.here() as i64) / 4;
            e.b_cond(HS, delta as i32);
        }
        e.ubfx_x(X10, X11, 0, 31); // arg_base*8 (bit 31 is the fp flag)
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
        // Float twins only when the callee has float-pinned slots
        // (cell b bit 31, stamped by the fixup): int->FP transfers are
        // expensive on this core, and integer-only code predicts the
        // skip perfectly.
        e.tbnz(X11, 31, 5); // -> fp (out of line: int code never jumps)
        e.ubfx_x(CODE, X9, 0, 48); // cont:
        e.mov_x(PC, CODE);
        e.ldr_x_imm(X9, PC, 0);
        e.br(X9);
        e.fmov_d_x(FL0R, L0R); // fp:
        e.fmov_d_x(FL1R, L1R);
        e.b(-6); // -> cont
    }

    // ---- call_indirect (rewired by the fixup pass; not in `handlers`)
    // Cell: a = caller_l1off<<48 | index_slot*8,
    //       b = caller_l0off<<48 | canon_expected<<32 | arg_base*8.
    // Table 0 base/len and the per-function info table come from the
    // entry state (refreshed every chain entry, so table.grow/set need
    // no invalidation). Every guard failure (bounds, null, type
    // mismatch, import callee) bails to the slow stub, which re-executes
    // the cell from its predecoded form and raises the proper trap or
    // routes the host call.
    let callindirect_handler = e.here();
    {
        bump(e, COUNT_MODE == 0);
        e.ldr_x_imm(X9, PC, 8);
        e.ubfx_x(X10, X9, 0, 32); // index_slot*8
        e.ldr_x_reg(X10, FRAME, X10); // t (i32, zero-extended)
        e.ldr_x_imm(X11, STATE, 112); // table 0 entries
        e.ldr_x_imm(X12, STATE, 120); // table 0 len
        e.cmp_w(X10, X12);
        {
            let delta = (slow_stub as i64 - e.here() as i64) / 4;
            e.b_cond(HS, delta as i32); // out of bounds (or no table)
        }
        e.ldr_w_uxtw2(X12, X11, X10); // fi = entries[t]
        e.i(0x3100_0000 | (1 << 10) | (X12 << 5) | 31); // cmn w12, #1
        {
            let delta = (slow_stub as i64 - e.here() as i64) / 4;
            e.b_cond(EQ, delta as i32); // null entry
        }
        e.ldr_x_imm(X13, STATE, 128); // info base
        e.add_x_lsl(X11, X12, X12, 1); // fi*3
        e.add_x_lsl(X13, X13, X11, 3); // entry = info + fi*24
        e.ldr_x_imm(X11, X13, 8); // l1off<<48 | l0off<<32 | canon type
        e.ldr_x_imm(X10, PC, 16); // cell b
        e.ubfx_x(X12, X11, 0, 32); // canonical actual
        e.ubfx_x(X16, X10, 32, 16); // canonical expected
        e.cmp_x(X12, X16);
        {
            let delta = (slow_stub as i64 - e.here() as i64) / 4;
            e.b_cond(NE, delta as i32); // type mismatch
        }
        e.ldr_x_imm(X12, X13, 0); // callee cells | fp flag (0 = slow)
        {
            let delta = (slow_stub as i64 - e.here() as i64) / 4;
            e.cbz_x(X12, delta as i32);
        }
        e.ldr_x_imm(X16, X13, 16); // frame metadata (entry ptr consumed)
                                   // compose the call_core inputs; the callee fp flag moves from
                                   // entry bit 0 (cells are 32-byte aligned, low bits free) to
                                   // the b-equiv bit 31
        e.ubfx_x(X13, X12, 0, 1); // fp flag
        e.lsr_x(X12, X12, 5);
        e.lsl_x(X12, X12, 5); // clean cells address
        e.lsr_x(X9, X9, 48);
        e.orr_x_lsl(X9, X12, X9, 48); // a-equiv: caller_l1off | cells
        e.lsr_x(X12, X10, 48);
        e.orr_x_lsl(X12, X16, X12, 48); // c-equiv: caller_l0off | meta
        e.lsr_x(X11, X11, 32);
        e.lsl_x(X11, X11, 32); // callee l0/l1 offsets, canon cleared
        e.ubfx_x(X16, X10, 0, 32);
        e.orr_x_lsl(X11, X11, X16, 0); // b-equiv: | arg_base*8
        e.orr_x_lsl(X11, X11, X13, 31); // | callee fp flag
        {
            let delta = (call_core as i64 - e.here() as i64) / 4;
            e.b(delta as i32);
        }
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
        e.cbz_x(X11, 5); // -> pop (skipping the copy loop)
                         // The accumulator doubles as the copy scratch: after a
                         // single-result copy it holds result 0, which is the
                         // call-result-in-acc convention at zero extra instructions
                         // (multi-result leaves the last result, a skipped copy leaves
                         // stale acc — neither case is ever marked).
        e.ldr_x_post(ACC, X10, 8); // cl:
        e.str_x_post(ACC, X12, 8);
        e.sub_x_imm(X11, X11, 1);
        e.cbnz_x(X11, -3); // -> cl
        e.ldp_x_pre(X13, FRAME, RETSP, -(RET_RECORD as i32)); // pop:
        e.ldr_x_imm(X12, RETSP, 16); // caller code | caller_l0off<<48
        e.ldr_x_imm(X10, RETSP, 24); // caller l1off
        e.lsr_x(X11, X12, 48);
        e.ubfx_x(CODE, X12, 0, 48);
        // reload the caller's pinned locals (sentinel records carry a
        // readable dummy frame, so these loads are always safe)
        // The caller's float-pinned flag rides bit 0 of its recorded
        // l0 offset (offsets are byte-scaled, so bit 0 is structurally
        // free); integer-only callers take the flagless path.
        e.tbnz(X11, 0, 6); // -> fp (out of line: int callers fall through)
        e.ldr_x_reg(L0R, FRAME, X11);
        e.ldr_x_reg(L1R, FRAME, X10);
        e.mov_x(PC, X13); // join:
        e.ldr_x_imm(X9, PC, 0);
        e.br(X9);
        e.sub_x_imm(X11, X11, 1); // fp:
        e.ldr_x_reg(L0R, FRAME, X11);
        e.ldr_x_reg(L1R, FRAME, X10);
        e.fmov_d_x(FL0R, L0R);
        e.fmov_d_x(FL1R, L1R);
        e.b(-8); // -> join
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
            pre(e);
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
                    pre(e);
                    let (ra, mut rb) = src_ab(e, a_cls, b_cls);
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
                    pre(e);
                    let (ra, rb) = src_ab(e, a_cls, b_cls);
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
                pre(e);
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
                pre(e);
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
            pre(e);
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
        (Op::I32_BrAnd, true, NE),
        (Op::I32_BrAndNot, true, EQ),
    ];
    for &(op, w, cond) in cmp_brs.iter() {
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                let off = e.here();
                pre(e);
                let (ra, rb) = src_ab(e, a_cls, b_cls);
                if matches!(op, Op::I32_BrAnd | Op::I32_BrAndNot) {
                    e.tst_w(ra, rb);
                } else if w {
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
                pre(e);
                let (ra, rb) = src_ab(e, a_cls, b_cls);
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
        pre(e);
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
        pre(e);
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
                    pre(e);
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
                    let fp = matches!(
                        m.op,
                        Op::F32_Load | Op::F64_Load | Op::F32_Store | Op::F64_Store
                    );
                    if m.load {
                        if fp {
                            // float loads land in their destination's
                            // NEON register (acc or pinned) and write
                            // through from it — no integer round trip
                            if m.kind == 0 {
                                e.ldr_s_reg(fp_target(d), MEM, X12);
                            } else {
                                e.ldr_d_reg(fp_target(d), MEM, X12);
                            }
                            finish_fp(e, d);
                        } else {
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
                        }
                    } else if fp && matches!(b_cls, Cls::Acc | Cls::L0 | Cls::L1) {
                        // float store value is register-resident (acc or
                        // pinned NEON register)
                        let v = match b_cls {
                            Cls::L0 => FL0R,
                            Cls::L1 => FL1R,
                            _ => FACC,
                        };
                        if m.kind == 0 {
                            e.str_s_reg(v, MEM, X12);
                        } else {
                            e.str_d_reg(v, MEM, X12);
                        }
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

    // ---- address-fused memory ops (FLAG_FUSED bank at variant+100):
    // ea = zext32(addr1 + addr2) + static offset, the corpus-universal
    // base+index pattern folded into one dispatch. addr2 always loads
    // from its slot (packed in c's high half, like Select's cond);
    // addr1 takes the register classes. ----
    for m in mems.iter() {
        if matches!(
            m.op,
            Op::I64_Load8S
                | Op::I64_Load8U
                | Op::I64_Load16S
                | Op::I64_Load16U
                | Op::I64_Load32S
                | Op::I64_Load32U
        ) {
            continue;
        }
        let fp = matches!(
            m.op,
            Op::F32_Load | Op::F64_Load | Op::F32_Store | Op::F64_Store
        );
        if m.load {
            for a_cls in [Cls::Slot, Cls::Acc, Cls::L0, Cls::L1] {
                for d in DSTS {
                    let off = e.here();
                    pre(e);
                    let ra1 = src_a(e, a_cls, X10);
                    e.ldr_x_imm(X11, PC, 24); // addr2*8 << 32 | dst*8
                    e.lsr_x(X13, X11, 32);
                    e.ubfx_x(X11, X11, 0, 32); // dst byte offset
                    e.ldr_x_reg(X13, FRAME, X13); // addr2
                    e.alu_reg(0x0B00_0000, X13, ra1, X13); // wrapping i32 sum
                    e.ldr_x_imm(X12, PC, 16); // static offset
                    e.add_x_uxtw(X12, X12, X13); // ea
                    e.add_x_imm(X13, X12, m.size);
                    e.cmp_x(X13, MEMLEN);
                    {
                        let delta = (trap_oob as i64 - e.here() as i64) / 4;
                        e.b_cond(HI, delta as i32);
                    }
                    if fp {
                        if m.kind == 0 {
                            e.ldr_s_reg(fp_target(d), MEM, X12);
                        } else {
                            e.ldr_d_reg(fp_target(d), MEM, X12);
                        }
                        if d != DstCls::Acc {
                            e.str_d_reg(fp_target(d), FRAME, X11);
                        }
                    } else {
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
                        if d != DstCls::Acc {
                            e.str_x_reg(rd, FRAME, X11);
                        }
                    }
                    tail(e, counted(a_cls, Cls::Slot, d));
                    def(handlers, m.op, variant(a_cls, Cls::Slot, d) + 100, off);
                }
            }
        } else {
            for a_cls in [Cls::Slot, Cls::Acc, Cls::L0, Cls::L1] {
                for b_cls in CLASSES {
                    let off = e.here();
                    pre(e);
                    let ra1 = src_a(e, a_cls, X10);
                    e.ldr_x_imm(X11, PC, 24); // addr2*8 << 32 | offset
                    e.lsr_x(X13, X11, 32);
                    e.ldr_x_reg(X13, FRAME, X13); // addr2
                    e.alu_reg(0x0B00_0000, X13, ra1, X13); // wrapping i32 sum
                    e.ubfx_x(X12, X11, 0, 32); // static offset
                    e.add_x_uxtw(X12, X12, X13); // ea
                    e.add_x_imm(X13, X12, m.size);
                    e.cmp_x(X13, MEMLEN);
                    {
                        let delta = (trap_oob as i64 - e.here() as i64) / 4;
                        e.b_cond(HI, delta as i32);
                    }
                    if fp && matches!(b_cls, Cls::Acc | Cls::L0 | Cls::L1) {
                        let v = match b_cls {
                            Cls::L0 => FL0R,
                            Cls::L1 => FL1R,
                            _ => FACC,
                        };
                        if m.kind == 0 {
                            e.str_s_reg(v, MEM, X12);
                        } else {
                            e.str_d_reg(v, MEM, X12);
                        }
                    } else {
                        let rb = src_b(e, b_cls, X11);
                        match m.kind {
                            0 => e.str_w_reg(rb, MEM, X12),
                            1 => e.str_x_reg(rb, MEM, X12),
                            2 => e.strb_reg(rb, MEM, X12),
                            3 => e.strh_reg(rb, MEM, X12),
                            _ => unreachable!(),
                        }
                    }
                    tail(e, counted(a_cls, b_cls, DstCls::Mem));
                    def(
                        handlers,
                        m.op,
                        variant(a_cls, b_cls, DstCls::Mem) + 100,
                        off,
                    );
                }
            }
        }
    }

    // ---- integer division and remainder ----
    // The trap edges (div0, and MIN/-1 for div_s) branch to the slow
    // stub: exec_ins re-executes the cell and raises the proper trap.
    // rem_s needs no overflow edge — sdiv wraps MIN/-1 to MIN and msub
    // then yields the correct 0.
    struct DivOp {
        op: Op,
        w: bool,
        signed: bool,
        rem: bool,
    }
    let divs = [
        DivOp {
            op: Op::I32_DivS,
            w: true,
            signed: true,
            rem: false,
        },
        DivOp {
            op: Op::I32_DivU,
            w: true,
            signed: false,
            rem: false,
        },
        DivOp {
            op: Op::I32_RemS,
            w: true,
            signed: true,
            rem: true,
        },
        DivOp {
            op: Op::I32_RemU,
            w: true,
            signed: false,
            rem: true,
        },
        DivOp {
            op: Op::I64_DivS,
            w: false,
            signed: true,
            rem: false,
        },
        DivOp {
            op: Op::I64_DivU,
            w: false,
            signed: false,
            rem: false,
        },
        DivOp {
            op: Op::I64_RemS,
            w: false,
            signed: true,
            rem: true,
        },
        DivOp {
            op: Op::I64_RemU,
            w: false,
            signed: false,
            rem: true,
        },
    ];
    for dv in divs.iter() {
        let div_enc: u32 = match (dv.w, dv.signed) {
            (true, false) => 0x1AC0_0800,  // udiv w
            (true, true) => 0x1AC0_0C00,   // sdiv w
            (false, false) => 0x9AC0_0800, // udiv x
            (false, true) => 0x9AC0_0C00,  // sdiv x
        };
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                for d in DSTS {
                    let off = e.here();
                    pre(e);
                    let (ra, rb) = src_ab(e, a_cls, b_cls);
                    {
                        let delta = (slow_stub as i64 - e.here() as i64) / 4;
                        if dv.w {
                            e.cbz_w(rb, delta as i32);
                        } else {
                            e.cbz_x(rb, delta as i32);
                        }
                    }
                    if dv.signed && !dv.rem {
                        // movn -1 into X12; MIN materializes in X13
                        if dv.w {
                            e.i(0x1280_0000 | X12);
                            e.cmp_w(rb, X12);
                        } else {
                            e.i(0x9280_0000 | X12);
                            e.cmp_x(rb, X12);
                        }
                        e.b_cond(NE, 5); // -> div
                        e.movz_x(X13, 0x8000);
                        e.lsl_x(X13, X13, if dv.w { 16 } else { 48 });
                        if dv.w {
                            e.cmp_w(ra, X13);
                        } else {
                            e.cmp_x(ra, X13);
                        }
                        let delta = (slow_stub as i64 - e.here() as i64) / 4;
                        e.b_cond(EQ, delta as i32);
                    }
                    let rd = dst_target(d);
                    if dv.rem {
                        e.alu_reg(div_enc, X12, ra, rb);
                        // msub rd, X12, rb, ra: rd = ra - quotient*rb
                        let msub: u32 = if dv.w { 0x1B00_8000 } else { 0x9B00_8000 };
                        e.i(msub | (rb << 16) | (ra << 10) | (X12 << 5) | rd);
                    } else {
                        e.alu_reg(div_enc, rd, ra, rb);
                    }
                    finish(e, d, rd);
                    tail(e, counted(a_cls, b_cls, d));
                    def(handlers, dv.op, variant(a_cls, b_cls, d), off);
                }
            }
        }
    }

    // ---- memory.fill / memory.copy (memory 0; a = base slot of the
    // three contiguous operands). 8-byte main loop with a byte tail;
    // bounds failures bail to the slow stub (exec_ins raises the trap).
    // Overlapping-downward copies take the backward block below.
    // (inputs: X10 = d, X11 = s, X12 = n, all raw offsets)
    let copy_backward = e.here();
    {
        e.add_x_reg(X10, MEM, X10);
        e.add_x_reg(X11, MEM, X11);
        e.movz_x(X13, 8);
        e.cmp_x(X12, X13); // b8:
        e.b_cond(LO, 5); // -> bytes
        e.sub_x_imm(X12, X12, 8);
        e.ldr_x_reg(X9, X11, X12);
        e.str_x_reg(X9, X10, X12);
        e.b(-5); // -> b8
        e.cbz_x(X12, 5); // bytes: -> done
        e.sub_x_imm(X12, X12, 1);
        e.ldrb_reg(X9, X11, X12);
        e.strb_reg(X9, X10, X12);
        e.b(-4); // -> bytes
        tail(e, COUNT_MODE == 0); // done:
    }
    {
        let off = e.here();
        pre(e);
        bump(e, COUNT_MODE == 0);
        e.ldr_x_imm(X9, PC, 8);
        e.add_x_reg(X9, FRAME, X9);
        e.ldr_x_imm(X10, X9, 0); // d
        e.ldr_x_imm(X11, X9, 8); // value
        e.ldr_x_imm(X12, X9, 16); // n
        e.add_x_reg(X13, X10, X12);
        e.cmp_x(X13, MEMLEN);
        {
            let delta = (slow_stub as i64 - e.here() as i64) / 4;
            e.b_cond(HI, delta as i32); // out of bounds
        }
        e.ubfx_x(X11, X11, 0, 8); // splat the fill byte
        e.orr_x_lsl(X11, X11, X11, 8);
        e.orr_x_lsl(X11, X11, X11, 16);
        e.orr_x_lsl(X11, X11, X11, 32);
        e.add_x_reg(X10, MEM, X10); // cursor
        e.add_x_reg(X13, MEM, X13); // end
        e.add_x_imm(X9, X10, 8); // l8:
        e.cmp_x(X9, X13);
        e.b_cond(HI, 3); // -> bytes
        e.str_x_post(X11, X10, 8);
        e.b(-4); // -> l8
        e.cmp_x(X10, X13); // bytes:
        e.b_cond(HS, 4); // -> done
        e.strb_reg(X11, X10, 31);
        e.add_x_imm(X10, X10, 1);
        e.b(-4); // -> bytes
        tail(e, COUNT_MODE == 0); // done:
        def(handlers, Op::MemoryFill, 0, off);
    }
    {
        let off = e.here();
        pre(e);
        bump(e, COUNT_MODE == 0);
        e.ldr_x_imm(X9, PC, 8);
        e.add_x_reg(X9, FRAME, X9);
        e.ldr_x_imm(X10, X9, 0); // d
        e.ldr_x_imm(X11, X9, 8); // s
        e.ldr_x_imm(X12, X9, 16); // n
        e.add_x_reg(X13, X10, X12);
        e.cmp_x(X13, MEMLEN);
        {
            let delta = (slow_stub as i64 - e.here() as i64) / 4;
            e.b_cond(HI, delta as i32); // dst out of bounds
        }
        e.add_x_reg(X13, X11, X12);
        e.cmp_x(X13, MEMLEN);
        {
            let delta = (slow_stub as i64 - e.here() as i64) / 4;
            e.b_cond(HI, delta as i32); // src out of bounds
        }
        // A forward copy is wrong only when s < d < s+n; that
        // overlapping-downward move runs the backward block (X13 still
        // holds s+n).
        e.cmp_x(X10, X11);
        e.b_cond(LS, 3); // d <= s -> fwd
        e.cmp_x(X10, X13);
        {
            let delta = (copy_backward as i64 - e.here() as i64) / 4;
            e.b_cond(LO, delta as i32); // overlap -> backward copy
        }
        e.add_x_reg(X10, MEM, X10); // fwd: dst cursor
        e.add_x_reg(X11, MEM, X11); // src cursor
        e.add_x_reg(X13, X10, X12); // dst end
        e.add_x_imm(X9, X10, 8); // l8:
        e.cmp_x(X9, X13);
        e.b_cond(HI, 4); // -> bytes
        e.ldr_x_post(X12, X11, 8);
        e.str_x_post(X12, X10, 8);
        e.b(-5); // -> l8
        e.cmp_x(X10, X13); // bytes:
        e.b_cond(HS, 6); // -> done
        e.ldrb_reg(X12, X11, 31);
        e.strb_reg(X12, X10, 31);
        e.add_x_imm(X10, X10, 1);
        e.add_x_imm(X11, X11, 1);
        e.b(-6); // -> bytes
        tail(e, COUNT_MODE == 0); // done:
        def(handlers, Op::MemoryCopy, 0, off);
    }

    // ---- floating point (S-form encodings = D-form with bit 22
    // cleared). Emitted after the integer families: float handlers must
    // never displace the hot integer ones. Values move between frame
    // slots / pinned registers / the accumulator as raw bits, so every
    // operand and destination class composes exactly like the integer
    // families. ----

    // binary arithmetic with a direct FP instruction (hardware
    // fmin/fmax match wasm NaN and signed-zero semantics)
    let fbins: [(Op, bool, u32); 12] = [
        (Op::F32_Add, true, 0x1E60_2800),
        (Op::F32_Sub, true, 0x1E60_3800),
        (Op::F32_Mul, true, 0x1E60_0800),
        (Op::F32_Div, true, 0x1E60_1800),
        (Op::F32_Min, true, 0x1E60_5800),
        (Op::F32_Max, true, 0x1E60_4800),
        (Op::F64_Add, false, 0x1E60_2800),
        (Op::F64_Sub, false, 0x1E60_3800),
        (Op::F64_Mul, false, 0x1E60_0800),
        (Op::F64_Div, false, 0x1E60_1800),
        (Op::F64_Min, false, 0x1E60_5800),
        (Op::F64_Max, false, 0x1E60_4800),
    ];
    for &(op, f32w, denc) in fbins.iter() {
        let enc = if f32w { denc - 0x0040_0000 } else { denc };
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                for d in DSTS {
                    let off = e.here();
                    pre(e);
                    let (va, vb) = src_fp_ab(e, a_cls, b_cls, f32w);
                    e.fp2(enc, fp_target(d), va, vb);
                    finish_fp(e, d);
                    tail(e, counted(a_cls, b_cls, d));
                    def(handlers, op, variant(a_cls, b_cls, d), off);
                }
            }
        }
    }

    // copysign: pure bit splice, no FP unit involved
    for &(op, f32w) in [(Op::F32_Copysign, true), (Op::F64_Copysign, false)].iter() {
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                for d in DSTS {
                    let off = e.here();
                    pre(e);
                    // float-domain operands arrive in the float acc; the
                    // splice itself runs on integer registers
                    let ra = match a_cls {
                        Cls::Acc | Cls::L0 | Cls::L1 => {
                            let v = src_fp(e, a_cls, f32w, 0, 8, X10);
                            fp_result(e, f32w, X10, v);
                            X10
                        }
                        _ => src_a(e, a_cls, X10),
                    };
                    let rb = match b_cls {
                        Cls::Acc | Cls::L0 | Cls::L1 => {
                            let v = src_fp(e, b_cls, f32w, 1, 16, X11);
                            fp_result(e, f32w, X11, v);
                            X11
                        }
                        _ => src_b(e, b_cls, X11),
                    };
                    if f32w {
                        e.lsl_x(X12, ra, 33);
                        e.lsr_x(X12, X12, 33);
                        e.ubfx_x(X13, rb, 31, 1);
                        e.orr_x_lsl(X12, X12, X13, 31);
                        e.fmov_s_w(fp_target(d), X12);
                    } else {
                        e.lsl_x(X12, ra, 1);
                        e.lsr_x(X12, X12, 1);
                        e.lsr_x(X13, rb, 63);
                        e.orr_x_lsl(X12, X12, X13, 63);
                        e.fmov_d_x(fp_target(d), X12);
                    }
                    finish_fp(e, d);
                    tail(e, counted(a_cls, b_cls, d));
                    def(handlers, op, variant(a_cls, b_cls, d), off);
                }
            }
        }
    }

    // compares (fcmp + cset with unordered-false condition mapping;
    // Ne is unordered-true via NE)
    let fcmps: [(Op, bool, u32); 12] = [
        (Op::F32_Eq, true, EQ),
        (Op::F32_Ne, true, NE),
        (Op::F32_Lt, true, MI),
        (Op::F32_Gt, true, GT),
        (Op::F32_Le, true, LS),
        (Op::F32_Ge, true, GE),
        (Op::F64_Eq, false, EQ),
        (Op::F64_Ne, false, NE),
        (Op::F64_Lt, false, MI),
        (Op::F64_Gt, false, GT),
        (Op::F64_Le, false, LS),
        (Op::F64_Ge, false, GE),
    ];
    for &(op, f32w, cond) in fcmps.iter() {
        for a_cls in CLASSES {
            for b_cls in CLASSES {
                for d in DSTS {
                    let off = e.here();
                    pre(e);
                    let (va, vb) = src_fp_ab(e, a_cls, b_cls, f32w);
                    e.fcmp(f32w, va, vb);
                    let rd = dst_target(d);
                    e.cset_w(rd, cond);
                    finish(e, d, rd);
                    tail(e, counted(a_cls, b_cls, d));
                    def(handlers, op, variant(a_cls, b_cls, d), off);
                }
            }
        }
    }

    // unary FP ops
    let funs: [(Op, bool, u32); 14] = [
        (Op::F32_Abs, true, 0x1E60_C000),
        (Op::F32_Neg, true, 0x1E61_4000),
        (Op::F32_Sqrt, true, 0x1E61_C000),
        (Op::F32_Ceil, true, 0x1E64_C000),
        (Op::F32_Floor, true, 0x1E65_4000),
        (Op::F32_Trunc, true, 0x1E65_C000),
        (Op::F32_Nearest, true, 0x1E64_4000),
        (Op::F64_Abs, false, 0x1E60_C000),
        (Op::F64_Neg, false, 0x1E61_4000),
        (Op::F64_Sqrt, false, 0x1E61_C000),
        (Op::F64_Ceil, false, 0x1E64_C000),
        (Op::F64_Floor, false, 0x1E65_4000),
        (Op::F64_Trunc, false, 0x1E65_C000),
        (Op::F64_Nearest, false, 0x1E64_4000),
    ];
    for &(op, f32w, denc) in funs.iter() {
        let enc = if f32w { denc - 0x0040_0000 } else { denc };
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let va = src_fp(e, a_cls, f32w, 0, 8, X10);
                e.fp1(enc, fp_target(d), va);
                finish_fp(e, d);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // int -> float conversions (scvtf/ucvtf: base = signed w->d;
    // +0x1_0000 unsigned, +0x8000_0000 64-bit source, -0x40_0000 f32
    // destination); the integer operand comes through the int classes
    let cvts: [(Op, bool, bool, bool); 8] = [
        // (op, dst_f32, src64, unsigned)
        (Op::F32_ConvertI32S, true, false, false),
        (Op::F32_ConvertI32U, true, false, true),
        (Op::F32_ConvertI64S, true, true, false),
        (Op::F32_ConvertI64U, true, true, true),
        (Op::F64_ConvertI32S, false, false, false),
        (Op::F64_ConvertI32U, false, false, true),
        (Op::F64_ConvertI64S, false, true, false),
        (Op::F64_ConvertI64U, false, true, true),
    ];
    for &(op, dst32, src64, uns) in cvts.iter() {
        let mut enc: u32 = 0x1E62_0000;
        if uns {
            enc += 0x0001_0000;
        }
        if src64 {
            enc += 0x8000_0000;
        }
        if dst32 {
            enc -= 0x0040_0000;
        }
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let ra = src_a(e, a_cls, X10);
                e.fp1(enc, fp_target(d), ra);
                finish_fp(e, d);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // demote / promote
    for &(op, to32) in [(Op::F32_DemoteF64, true), (Op::F64_PromoteF32, false)].iter() {
        let enc: u32 = if to32 { 0x1E62_4000 } else { 0x1E22_C000 };
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let va = src_fp(e, a_cls, !to32, 0, 8, X10);
                e.fp1(enc, fp_target(d), va);
                finish_fp(e, d);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // saturating float -> int (fcvtzs/fcvtzu implement wasm trunc_sat
    // exactly: clamp at the bounds, NaN -> 0); base = signed d->w
    let sats: [(Op, bool, bool, bool); 8] = [
        // (op, src_f32, to64, unsigned)
        (Op::I32_TruncSatF32S, true, false, false),
        (Op::I32_TruncSatF32U, true, false, true),
        (Op::I32_TruncSatF64S, false, false, false),
        (Op::I32_TruncSatF64U, false, false, true),
        (Op::I64_TruncSatF32S, true, true, false),
        (Op::I64_TruncSatF32U, true, true, true),
        (Op::I64_TruncSatF64S, false, true, false),
        (Op::I64_TruncSatF64U, false, true, true),
    ];
    let sat_enc = |src32: bool, to64: bool, uns: bool| -> u32 {
        let mut enc: u32 = 0x1E78_0000;
        if uns {
            enc += 0x0001_0000;
        }
        if to64 {
            enc += 0x8000_0000;
        }
        if src32 {
            enc -= 0x0040_0000;
        }
        enc
    };
    for &(op, src32, to64, uns) in sats.iter() {
        let enc = sat_enc(src32, to64, uns);
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let va = src_fp(e, a_cls, src32, 0, 8, X10);
                let rd = dst_target(d);
                e.fp1(enc, rd, va);
                finish(e, d, rd);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // trapping float -> int: precise range pre-check so a bail always
    // means a trap (never a valid value computed on the slow path — a
    // dynamically-successful bail would break the accumulator pairing
    // and stale the mirrored result). Bounds are the exclusive trap
    // boundaries on the UNtruncated operand; the low compare's LE is
    // unordered-true, so it also catches NaN.
    let traps: [(Op, bool, bool, bool, u64, u64); 8] = [
        // (op, src_f32, to64, unsigned, lo_bits, hi_bits)
        (
            Op::I32_TruncF32S,
            true,
            false,
            false,
            0xCF00_0001,
            0x4F00_0000,
        ),
        (
            Op::I32_TruncF32U,
            true,
            false,
            true,
            0xBF80_0000,
            0x4F80_0000,
        ),
        (
            Op::I32_TruncF64S,
            false,
            false,
            false,
            0xC1E0_0000_0020_0000,
            0x41E0_0000_0000_0000,
        ),
        (
            Op::I32_TruncF64U,
            false,
            false,
            true,
            0xBFF0_0000_0000_0000,
            0x41F0_0000_0000_0000,
        ),
        (
            Op::I64_TruncF32S,
            true,
            true,
            false,
            0xDF00_0001,
            0x5F00_0000,
        ),
        (
            Op::I64_TruncF32U,
            true,
            true,
            true,
            0xBF80_0000,
            0x5F80_0000,
        ),
        (
            Op::I64_TruncF64S,
            false,
            true,
            false,
            0xC3E0_0000_0000_0001,
            0x43E0_0000_0000_0000,
        ),
        (
            Op::I64_TruncF64U,
            false,
            true,
            true,
            0xBFF0_0000_0000_0000,
            0x43F0_0000_0000_0000,
        ),
    ];
    for &(op, src32, to64, uns, lo, hi) in traps.iter() {
        let enc = sat_enc(src32, to64, uns);
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let va = src_fp(e, a_cls, src32, 0, 8, X10);
                mov_imm64(e, X13, lo);
                if src32 {
                    e.fmov_s_w(1, X13);
                } else {
                    e.fmov_d_x(1, X13);
                }
                e.fcmp(src32, va, 1);
                {
                    let delta = (slow_stub as i64 - e.here() as i64) / 4;
                    e.b_cond(LE, delta as i32); // x <= lo, or NaN
                }
                mov_imm64(e, X13, hi);
                if src32 {
                    e.fmov_s_w(1, X13);
                } else {
                    e.fmov_d_x(1, X13);
                }
                e.fcmp(src32, va, 1);
                {
                    let delta = (slow_stub as i64 - e.here() as i64) / 4;
                    e.b_cond(GE, delta as i32); // x >= hi
                }
                let rd = dst_target(d);
                e.fp1(enc, rd, va);
                finish(e, d, rd);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // popcnt via NEON (cnt per byte + uaddlv); i32 slots are
    // zero-extended, so the 64-bit byte-sum is exact for both widths
    for &op in [Op::I32_Popcnt, Op::I64_Popcnt].iter() {
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let ra = src_a(e, a_cls, X10);
                e.fmov_d_x(0, ra);
                e.i(0x0E20_5800); // cnt v0.8b, v0.8b
                e.i(0x2E30_3800); // uaddlv h0, v0.8b
                let rd = dst_target(d);
                e.fmov_w_s(rd, 0);
                finish(e, d, rd);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // reinterpret: raw-bit moves, but the accumulator edges are
    // domain-typed — float->int reads the float acc, int->float lands
    // in it
    for &(op, f32w) in [
        (Op::I32_ReinterpretF32, true),
        (Op::I64_ReinterpretF64, false),
    ]
    .iter()
    {
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let rd = dst_target(d);
                if matches!(a_cls, Cls::Acc | Cls::L0 | Cls::L1) {
                    let v = src_fp(e, a_cls, f32w, 0, 8, X10);
                    fp_result(e, f32w, rd, v);
                } else {
                    let ra = src_a(e, a_cls, X10);
                    if ra != rd {
                        e.mov_x(rd, ra);
                    }
                }
                finish(e, d, rd);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }
    for &(op, f32w) in [
        (Op::F32_ReinterpretI32, true),
        (Op::F64_ReinterpretI64, false),
    ]
    .iter()
    {
        for a_cls in CLASSES {
            for d in DSTS {
                let off = e.here();
                pre(e);
                let ra = src_a(e, a_cls, X10);
                if f32w {
                    e.fmov_s_w(fp_target(d), ra);
                } else {
                    e.fmov_d_x(fp_target(d), ra);
                }
                finish_fp(e, d);
                tail(e, counted(a_cls, Cls::Slot, d));
                def(handlers, op, variant(a_cls, Cls::Slot, d), off);
            }
        }
    }

    // ---- fused staging-mov pairs (strict order: src2 may be dst1) ----
    for a_cls in [Cls::Slot, Cls::L0, Cls::L1] {
        for b_cls in [Cls::Slot, Cls::L0, Cls::L1] {
            let off = e.here();
            pre(e);
            if a_cls == Cls::Slot && b_cls == Cls::Slot {
                e.ldp_x_base_imm(X10, X11, PC, 8);
            } else if a_cls == Cls::Slot {
                e.ldr_x_imm(X10, PC, 8);
            } else if b_cls == Cls::Slot {
                e.ldr_x_imm(X11, PC, 16);
            }
            e.ldr_x_imm(X12, PC, 24); // dst1*8 << 32 | dst2*8
            let v1 = match a_cls {
                Cls::L0 => L0R,
                Cls::L1 => L1R,
                _ => {
                    e.ldr_x_reg(X13, FRAME, X10);
                    X13
                }
            };
            e.lsr_x(X9, X12, 32);
            e.str_x_reg(v1, FRAME, X9);
            let v2 = match b_cls {
                Cls::L0 => L0R,
                Cls::L1 => L1R,
                _ => {
                    e.ldr_x_reg(X13, FRAME, X11);
                    X13
                }
            };
            e.ubfx_x(X12, X12, 0, 32);
            e.str_x_reg(v2, FRAME, X12);
            tail(e, counted(a_cls, b_cls, DstCls::Mem));
            def(
                handlers,
                Op::MovPair,
                variant(a_cls, b_cls, DstCls::Mem),
                off,
            );
        }
    }

    EmitOut {
        entry,
        slow_stub,
        call_handler,
        callindirect_handler,
        return_exit,
    }
}
