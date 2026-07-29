//! ARM32 handler backend: A32 and Thumb-2 from one source.
//!
//! The two encodings share unified assembly syntax almost completely, so
//! the only place the mode appears below is conditional execution — A32
//! puts the condition on the instruction, Thumb-2 needs a preceding `IT`.
//!
//! Register contract (chain-invariant across every handler):
//! ```text
//!   r4 = pc          r5 = frame base   r6 = &EnterState
//!   r7 = memory 0 base                 r8 = memory 0 length
//!   r9:r10 = accumulator (low:high)    r11:r12 = l0 (low:high)
//!   lr = prefetched next handler word
//!   r0-r3 = scratch
//! ```
//!
//! Fourteen usable registers is the tightest budget of any backend here,
//! and a wasm value is eight bytes, so the accumulator and the pinned local
//! are register PAIRS — which uses four of them. What gets pushed out to
//! `EnterState` as a result: the cell array base, the globals base, the
//! return-stack cursor and limit, and the value-stack limit. All five are
//! read by cold handlers only.
//!
//! Two scope decisions, both about the same budget:
//!
//! - **One pinned local, no `l1`.** A second would take a second pair.
//! - **Calls run through the shared executor.** The call protocol packs
//!   its operands into the high halves of 64-bit cell fields; threading six
//!   half-registers through the activation path costs more here than the
//!   two chain crossings a driver-run call pays. `Return` stays native —
//!   the driver plants a sentinel record per activation and the native
//!   return pops it, so the two compose.
//!
//! Float is not native either: the M-profile targets have no
//! double-precision unit at all, and routing f32/f64 to the shared
//! executor keeps one arm32 handler set instead of two.

use super::asm::Asm;
use super::instr::Op;
use super::isa::{Caps, Isa, PairDstSplit, Stubs, Variant, CLASSES_NO_L1, DSTS_NO_L1};
use super::layout::{Cls, DstCls, Fam};

const PC: &str = "r4";
const FRAME: &str = "r5";
const STATE: &str = "r6";
const MEM: &str = "r7";
const MEMLEN: &str = "r8";
const ACC: &str = "r9";
const ACC_HI: &str = "r10";
const L0R: &str = "r11";
const L0_HI: &str = "r12";
const NEXT: &str = "lr";
const T0: &str = "r0";
const T1: &str = "r1";
const T2: &str = "r2";
const T3: &str = "r3";

/// State-block offsets for the roles that do not get a register here.
const S_CODE: u32 = 40;
const S_GLOBALS: u32 = 48;
const S_RET_CURSOR: u32 = 56;
const S_DISPATCHES: u32 = 80;

/// `(condition, 32-bit operands)`. Every compare here is on i32 operands;
/// the 64-bit forms that stay native fold their pair to one word first.
fn int_cmp(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        I32_Eq | I32_BrEq => ("eq", true),
        I32_Ne | I32_BrNe => ("ne", true),
        I32_LtS | I32_BrLtS => ("lt", true),
        I32_LtU | I32_BrLtU => ("lo", true),
        I32_GtS | I32_BrGtS => ("gt", true),
        I32_GtU | I32_BrGtU => ("hi", true),
        I32_LeS | I32_BrLeS => ("le", true),
        I32_LeU | I32_BrLeU => ("ls", true),
        I32_GeS | I32_BrGeS => ("ge", true),
        I32_GeU | I32_BrGeU => ("hs", true),
        I64_Eq | I64_BrEq => ("eq", false),
        I64_Ne | I64_BrNe => ("ne", false),
        I32_BrAnd => ("ne", true),
        I32_BrAndNot => ("eq", true),
        _ => return None,
    })
}

fn inv(c: &str) -> &'static str {
    match c {
        "eq" => "ne",
        "ne" => "eq",
        "lt" => "ge",
        "ge" => "lt",
        "gt" => "le",
        "le" => "gt",
        "lo" => "hs",
        "hs" => "lo",
        "hi" => "ls",
        "ls" => "hi",
        other => panic!("no inverse for {other}"),
    }
}

/// `(mnemonic, 32-bit, shift-style)`. Shift-style ops mask their amount to
/// 5 bits first: ARM's register shifts take the low BYTE and produce zero
/// past 31, where wasm wants the amount modulo the width.
fn int_bin(op: Op) -> Option<(&'static str, bool, bool)> {
    use Op::*;
    Some(match op {
        I32_Add => ("add", true, false),
        I32_Sub => ("sub", true, false),
        I32_And => ("and", true, false),
        I32_Or => ("orr", true, false),
        I32_Xor => ("eor", true, false),
        I32_Mul => ("mul", true, false),
        I32_Shl => ("lsl", true, true),
        I32_ShrU => ("lsr", true, true),
        I32_ShrS => ("asr", true, true),
        I32_Rotr => ("ror", true, true),
        I32_Rotl => ("rotl", true, true),
        I64_Add => ("add", false, false),
        I64_Sub => ("sub", false, false),
        I64_And => ("and", false, false),
        I64_Or => ("orr", false, false),
        I64_Xor => ("eor", false, false),
        _ => return None,
    })
}

/// `(size, is a load, kind)`; kind 0 = u32, 1 = 64-bit, 2 = u8, 3 = u16,
/// 4 = i8, 5 = i16.
fn mem_kind(op: Op) -> Option<(u32, bool, u8)> {
    use Op::*;
    Some(match op {
        I32_Load => (4, true, 0),
        I64_Load => (8, true, 1),
        I32_Load8U => (1, true, 2),
        I32_Load16U => (2, true, 3),
        I32_Load8S => (1, true, 4),
        I32_Load16S => (2, true, 5),
        I32_Store => (4, false, 0),
        I64_Store => (8, false, 1),
        I64_Store32 => (4, false, 0),
        I32_Store8 | I64_Store8 => (1, false, 2),
        I32_Store16 | I64_Store16 => (2, false, 3),
        _ => return None,
    })
}

/// The ops this backend keeps native: the whole i32 domain, plus the
/// 64-bit ones a register pair handles in a couple of instructions.
fn native_op(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    if (d >= I32_Add as u16 && d <= I32_GeU as u16)
        || (d >= I32_BrEq as u16 && d <= I32_BrGeU as u16)
        || matches!(op, I32_BrAnd | I32_BrAndNot | I32_SubBrIf | I64_SubBrIf)
    {
        return true;
    }
    matches!(
        op,
        MovSlot
            | MovConst
            | MovPair
            | Select
            | Br
            | BrIf
            | BrIfNot
            | BrTable
            | Return
            | GlobalGet
            | GlobalSet
            | MemoryFill
            | MemoryCopy
            | I32_Load
            | I32_Load8S
            | I32_Load8U
            | I32_Load16S
            | I32_Load16U
            | I32_Store
            | I32_Store8
            | I32_Store16
            | I64_Load
            | I64_Store
            | I64_Store32
            | I64_Store8
            | I64_Store16
            | I64_Eqz
            | I64_Eq
            | I64_Ne
            | I64_Add
            | I64_Sub
            | I64_And
            | I64_Or
            | I64_Xor
            | I64_BrEq
            | I64_BrNe
            | I32_WrapI64
            | I64_ExtendI32S
            | I64_ExtendI32U
            | I32_Extend8S
            | I32_Extend16S
    )
}

pub struct Arm32 {
    /// Emit Thumb-2 rather than A32.
    pub thumb: bool,
    /// Whether the assembler has to be *told* about `sdiv`/`udiv`. The
    /// backend always emits them; the question is only whether the target's
    /// base architecture already includes them.
    pub idiv_directive: bool,
}

impl Arm32 {
    fn pre(&self, a: &mut Asm) {
        a.ins(&format!("ldr {NEXT}, [{PC}, #32]"));
    }

    fn bump(&self, a: &mut Asm, on: bool) {
        if on {
            a.ins(&format!("ldr {T0}, [{STATE}, #{S_DISPATCHES}]"));
            a.ins(&format!("adds {T0}, {T0}, #1"));
            a.ins(&format!("str {T0}, [{STATE}, #{S_DISPATCHES}]"));
        }
    }

    fn tail(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins(&format!("add {PC}, {PC}, #32"));
        a.ins(&format!("bx {NEXT}"));
    }

    /// A conditional `mov`. The instruction carries the condition suffix
    /// in both encodings — unified syntax requires it inside a Thumb `IT`
    /// block, and it is the only spelling A32 has.
    fn cond_mov_imm(&self, a: &mut Asm, cond: &str, rd: &str, imm: u32) {
        if self.thumb {
            a.ins(&format!("it {cond}"));
        }
        a.ins(&format!("mov{cond} {rd}, #{imm}"));
    }

    /// A conditional register move, same rule as above.
    fn cond_mov(&self, a: &mut Asm, cond: &str, rd: &str, rs: &str) {
        if self.thumb {
            a.ins(&format!("it {cond}"));
        }
        a.ins(&format!("mov{cond} {rd}, {rs}"));
    }

    /// Materialize an arbitrary 32-bit constant without a literal pool: a
    /// pool inside a 200 KB blob would need placement passes the generator
    /// deliberately does not have.
    fn mov_imm32(&self, a: &mut Asm, rd: &str, v: u32) {
        a.ins(&format!("movw {rd}, #{}", v & 0xffff));
        if v >> 16 != 0 {
            a.ins(&format!("movt {rd}, #{}", v >> 16));
        }
    }

    fn slot_addr(&self, a: &mut Asm, field: u32, tmp: &str) {
        a.ins(&format!("ldr {tmp}, [{PC}, #{field}]"));
        a.ins(&format!("add {tmp}, {FRAME}, {tmp}"));
    }

    /// A source operand's low word.
    fn src(&self, a: &mut Asm, cls: Cls, field: u32, tmp: &'static str) -> &'static str {
        match cls {
            Cls::Acc => ACC,
            Cls::L0 => L0R,
            Cls::L1 => unreachable!("arm32 has no l1"),
            Cls::Const => {
                a.ins(&format!("ldr {tmp}, [{PC}, #{field}]"));
                tmp
            }
            Cls::Slot => {
                self.slot_addr(a, field, tmp);
                a.ins(&format!("ldr {tmp}, [{tmp}]"));
                tmp
            }
        }
    }

    /// A source operand as a `(low, high)` pair.
    fn pair(
        &self,
        a: &mut Asm,
        cls: Cls,
        field: u32,
        lo_tmp: &'static str,
        hi_tmp: &'static str,
    ) -> (&'static str, &'static str) {
        match cls {
            Cls::Acc => (ACC, ACC_HI),
            Cls::L0 => (L0R, L0_HI),
            Cls::L1 => unreachable!("arm32 has no l1"),
            Cls::Const => {
                a.ins(&format!("ldr {lo_tmp}, [{PC}, #{field}]"));
                a.ins(&format!("ldr {hi_tmp}, [{PC}, #{}]", field + 4));
                (lo_tmp, hi_tmp)
            }
            Cls::Slot => {
                self.slot_addr(a, field, hi_tmp);
                a.ins(&format!("ldr {lo_tmp}, [{hi_tmp}]"));
                a.ins(&format!("ldr {hi_tmp}, [{hi_tmp}, #4]"));
                (lo_tmp, hi_tmp)
            }
        }
    }

    fn dst_target(&self, dc: DstCls) -> &'static str {
        match dc {
            DstCls::L0 => L0R,
            _ => ACC,
        }
    }

    fn dst_hi(&self, dc: DstCls) -> &'static str {
        match dc {
            DstCls::L0 => L0_HI,
            _ => ACC_HI,
        }
    }

    /// Store a value to the destination slot.
    ///
    /// `wide` says the high word of the result pair carries meaning. A
    /// 32-bit result zeroes it FIRST, before the `Acc` early return: the
    /// accumulator and the pinned local are register PAIRS here, and a
    /// later consumer reads both halves whether or not a store happened.
    /// Skipping that on the store-less path leaves an i32 result carrying
    /// a stale upper word into the next i64 operation.
    fn finish(&self, a: &mut Asm, dc: DstCls, src: &str, wide: bool, scratch: &'static str) {
        if !wide {
            a.ins(&format!("mov {}, #0", self.dst_hi(dc)));
        }
        if dc == DstCls::Acc {
            return;
        }
        self.slot_addr(a, 24, scratch);
        a.ins(&format!("str {src}, [{scratch}]"));
        a.ins(&format!("str {}, [{scratch}, #4]", self.dst_hi(dc)));
    }
}

impl Isa for Arm32 {
    fn caps(&self) -> Caps {
        Caps {
            classes: &CLASSES_NO_L1,
            dsts: &DSTS_NO_L1,
            ptr_bytes: 4,
            has_l1: false,
            has_float_regs: false,
            float_pin_f32: false,
            native_calls: false,
        }
    }

    fn emit_prelude(&mut self, a: &mut Asm, st: &Stubs) {
        if self.idiv_directive {
            // In the A-profile `sdiv`/`udiv` are an optional extension that
            // the bare `armv7` CPU model leaves off, so the directive is
            // required. In the M-profile they are baseline and the
            // assembler rejects the directive outright.
            a.raw("\t.arch_extension idiv");
        }

        a.label(&st.exit_common);
        a.ins(&format!("str {T0}, [{STATE}, #0]")); // state.reason
        a.ins(&format!("str {PC}, [{STATE}, #8]"));
        a.ins(&format!("str {FRAME}, [{STATE}, #16]"));
        // The accumulator crosses activation boundaries at full width.
        a.ins(&format!("str {ACC}, [{STATE}, #104]"));
        a.ins(&format!("str {ACC_HI}, [{STATE}, #108]"));
        a.ins("pop {r4-r12, lr}");
        a.ins("bx lr");

        for (label, reason) in [
            (&st.slow, 1u32),
            (&st.return_exit, 2),
            (&st.trap_oob, 16),
            (&st.trap_exhaust, 17),
        ] {
            a.label(label);
            a.ins(&format!("mov {T0}, #{reason}"));
            a.ins(&format!("b {}", st.exit_common));
        }

        // ---- entry trampoline: extern "C" fn(*mut EnterState) ----
        a.label(&st.entry);
        a.ins("push {r4-r12, lr}");
        a.ins(&format!("mov {STATE}, r0"));
        a.ins(&format!("ldr {PC}, [{STATE}, #8]"));
        a.ins(&format!("ldr {FRAME}, [{STATE}, #16]"));
        a.ins(&format!("ldr {MEM}, [{STATE}, #24]"));
        a.ins(&format!("ldr {MEMLEN}, [{STATE}, #32]"));
        a.ins(&format!("ldr {L0R}, [{STATE}, #88]"));
        a.ins(&format!("ldr {L0_HI}, [{STATE}, #92]"));
        a.ins(&format!("ldr {ACC}, [{STATE}, #104]"));
        a.ins(&format!("ldr {ACC_HI}, [{STATE}, #108]"));
        a.ins(&format!("ldr {T0}, [{PC}]"));
        a.ins(&format!("bx {T0}"));

        // Both call flavours link to the slow stub on this backend; these
        // labels exist only because the meta table names them.
        a.label(&st.call);
        a.label(&st.call_indirect);
        a.label(&st.call_core);
        a.ins(&format!("b {}", st.slow));
    }

    fn wants(&self, v: &Variant) -> bool {
        use Op::*;
        if !native_op(v.op) {
            return false;
        }
        match v.op {
            MemoryFillCopy => false,
            MovSlot => v.a != Cls::Const,
            MovPair => {
                !matches!(v.a, Cls::Const | Cls::Acc) && !matches!(v.b, Cls::Const | Cls::Acc)
            }
            BrTable => v.a != Cls::Const,
            I32_SubBrIf | I64_SubBrIf => !matches!(v.a, Cls::Const | Cls::Acc),
            _ => {
                if v.fused && v.a == Cls::Const {
                    return false;
                }
                true
            }
        }
    }

    fn emit_handler(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        use Op::*;
        match v.op {
            Return => return self.emit_return(a, v),
            Br => {
                self.bump(a, v.counted);
                a.ins(&format!("ldr {PC}, [{PC}, #24]")); // c is absolute
                a.ins(&format!("ldr {T0}, [{PC}]"));
                a.ins(&format!("bx {T0}"));
                return;
            }
            BrIf | BrIfNot => {
                self.pre(a);
                // Target and its handler word load at entry: the two are
                // dependent, so on the taken path they otherwise sit
                // nose-to-tail on the critical path.
                a.ins(&format!("ldr {T2}, [{PC}, #24]"));
                a.ins(&format!("ldr {T3}, [{T2}]"));
                let ra = self.src(a, v.a, 8, T0);
                let nt = a.fresh("nt");
                a.ins(&format!("cmp {ra}, #0"));
                a.ins(&format!("b{} {nt}", if v.op == BrIf { "eq" } else { "ne" }));
                self.bump(a, v.counted);
                a.ins(&format!("mov {PC}, {T2}"));
                a.ins(&format!("bx {T3}"));
                a.label(&nt);
                self.tail(a, v.counted);
                return;
            }
            I32_SubBrIf => {
                a.ins(&format!("ldr {T2}, [{PC}, #24]"));
                a.ins(&format!("ldr {T3}, [{T2}]"));
                let x = self.src(a, v.a, 8, T0);
                let y = self.src(a, v.b, 16, T1);
                let (rd, rdh) = match v.a {
                    Cls::L0 => (L0R, L0_HI),
                    Cls::Slot => (ACC, ACC_HI),
                    Cls::Const | Cls::Acc | Cls::L1 => unreachable!(),
                };
                a.ins(&format!("subs {rd}, {x}, {y}"));
                // Neither MOV nor the address/load/store sequence below
                // changes Z, so it remains the subtraction's branch test.
                a.ins(&format!("mov {rdh}, #0"));
                self.slot_addr(a, 8, T0);
                a.ins(&format!("str {rd}, [{T0}]"));
                a.ins(&format!("str {rdh}, [{T0}, #4]"));
                let nt = a.fresh("nt");
                a.ins(&format!("beq {nt}"));
                self.bump(a, v.counted);
                a.ins(&format!("mov {PC}, {T2}"));
                a.ins(&format!("bx {T3}"));
                a.label(&nt);
                self.pre(a);
                self.tail(a, v.counted);
                return;
            }
            I64_SubBrIf => {
                // A full-width subtraction occupies all four scratch
                // registers, so unlike the i32 form the target loads wait
                // until the taken path is known.
                // Preserve an accumulator RHS before loading a slot/L0 lhs
                // into that same register pair.
                if v.b == Cls::Acc {
                    a.ins(&format!("mov {T2}, {ACC}"));
                    a.ins(&format!("mov {T3}, {ACC_HI}"));
                }
                match v.a {
                    Cls::L0 => {
                        a.ins(&format!("mov {ACC}, {L0R}"));
                        a.ins(&format!("mov {ACC_HI}, {L0_HI}"));
                    }
                    Cls::Slot => {
                        self.slot_addr(a, 8, T0);
                        a.ins(&format!("ldr {ACC}, [{T0}]"));
                        a.ins(&format!("ldr {ACC_HI}, [{T0}, #4]"));
                    }
                    Cls::Const | Cls::Acc | Cls::L1 => unreachable!(),
                }
                match v.b {
                    Cls::Acc => {}
                    Cls::L0 => {
                        a.ins(&format!("mov {T2}, {L0R}"));
                        a.ins(&format!("mov {T3}, {L0_HI}"));
                    }
                    Cls::Const => {
                        a.ins(&format!("ldr {T2}, [{PC}, #16]"));
                        a.ins(&format!("ldr {T3}, [{PC}, #20]"));
                    }
                    Cls::Slot => {
                        self.slot_addr(a, 16, T1);
                        a.ins(&format!("ldr {T2}, [{T1}]"));
                        a.ins(&format!("ldr {T3}, [{T1}, #4]"));
                    }
                    Cls::L1 => unreachable!(),
                }
                a.ins(&format!("subs {ACC}, {ACC}, {T2}"));
                a.ins(&format!("sbc {ACC_HI}, {ACC_HI}, {T3}"));
                self.slot_addr(a, 8, T0);
                a.ins(&format!("str {ACC}, [{T0}]"));
                a.ins(&format!("str {ACC_HI}, [{T0}, #4]"));
                if v.a == Cls::L0 {
                    a.ins(&format!("mov {L0R}, {ACC}"));
                    a.ins(&format!("mov {L0_HI}, {ACC_HI}"));
                }
                a.ins(&format!("orrs {T1}, {ACC}, {ACC_HI}"));
                let nt = a.fresh("nt");
                a.ins(&format!("beq {nt}"));
                self.bump(a, v.counted);
                a.ins(&format!("ldr {T2}, [{PC}, #24]"));
                a.ins(&format!("ldr {T3}, [{T2}]"));
                a.ins(&format!("mov {PC}, {T2}"));
                a.ins(&format!("bx {T3}"));
                a.label(&nt);
                self.pre(a);
                self.tail(a, v.counted);
                return;
            }
            BrTable => {
                self.bump(a, v.counted);
                let ra = self.src(a, v.a, 8, T0);
                a.ins(&format!("ldr {T1}, [{PC}, #16]")); // flat table
                a.ins(&format!("ldr {T2}, [{PC}, #24]")); // len - 1
                a.ins(&format!("mov {T3}, {ra}"));
                // Clamp: any out-of-range index picks the default, which
                // the link pass parked in the last slot.
                a.ins(&format!("cmp {T3}, {T2}"));
                self.cond_mov(a, "hs", T3, T2);
                a.ins(&format!("ldr {T3}, [{T1}, {T3}, lsl #2]"));
                a.ins(&format!("ldr {T1}, [{STATE}, #{S_CODE}]"));
                a.ins(&format!("add {PC}, {T1}, {T3}, lsl #5"));
                a.ins(&format!("ldr {T0}, [{PC}]"));
                a.ins(&format!("bx {T0}"));
                return;
            }
            MemoryFill | MemoryCopy => return self.emit_bulk(a, st, v),
            _ => {}
        }
        if let Some((cond, w32)) = int_cmp(v.op) {
            if super::layout::family(v.op) == Fam::SrcAB {
                self.pre(a);
                // Loading the branch target and its handler word at entry
                // keeps the taken path's two dependent loads off the
                // critical path — but it needs two registers held across
                // the compare, and a 64-bit compare folds a register PAIR
                // and has none to spare. Four scratch registers is the
                // whole budget, so the wide forms load the target after
                // the compare instead.
                let early = w32;
                if early {
                    a.ins(&format!("ldr {T2}, [{PC}, #24]"));
                    a.ins(&format!("ldr {T3}, [{T2}]"));
                }
                self.emit_compare(a, v, w32);
                let nt = a.fresh("nt");
                a.ins(&format!("b{} {nt}", inv(cond)));
                self.bump(a, v.counted);
                if !early {
                    a.ins(&format!("ldr {T2}, [{PC}, #24]"));
                    a.ins(&format!("ldr {T3}, [{T2}]"));
                }
                a.ins(&format!("mov {PC}, {T2}"));
                a.ins(&format!("bx {T3}"));
                a.label(&nt);
                self.tail(a, v.counted);
                return;
            }
        }

        self.pre(a);
        match v.op {
            MovSlot | MovConst => {
                let rd = self.dst_target(v.d);
                let rdh = self.dst_hi(v.d);
                match v.a {
                    Cls::Acc | Cls::L0 => {
                        let (lo, hi) = self.pair(a, v.a, 8, T0, T1);
                        if lo != rd {
                            a.ins(&format!("mov {rd}, {lo}"));
                            a.ins(&format!("mov {rdh}, {hi}"));
                        }
                    }
                    Cls::Const => {
                        a.ins(&format!("ldr {rd}, [{PC}, #8]"));
                        a.ins(&format!("ldr {rdh}, [{PC}, #12]"));
                    }
                    Cls::Slot => {
                        self.slot_addr(a, 8, T0);
                        a.ins(&format!("ldr {rd}, [{T0}]"));
                        a.ins(&format!("ldr {rdh}, [{T0}, #4]"));
                    }
                    Cls::L1 => unreachable!(),
                }
                self.finish(a, v.d, rd, true, T0);
            }
            MovPair => {
                // Strictly ordered: commit dst1 (including L0, when
                // pinned) before reading src2.
                let (v1, v1h) = self.pair(a, v.a, 8, T0, T1);
                a.ins(&format!("ldr {T2}, [{PC}, #28]")); // dst1*8
                a.ins(&format!("add {T2}, {FRAME}, {T2}"));
                a.ins(&format!("str {v1}, [{T2}]"));
                a.ins(&format!("str {v1h}, [{T2}, #4]"));
                if let Some(d1) = v.pair_d.first() {
                    debug_assert_eq!(d1, DstCls::L0);
                    if v1 != L0R {
                        a.ins(&format!("mov {L0R}, {v1}"));
                        a.ins(&format!("mov {L0_HI}, {v1h}"));
                    }
                }
                let (v2, v2h) = self.pair(a, v.b, 16, ACC, ACC_HI);
                if v2 != ACC {
                    a.ins(&format!("mov {ACC}, {v2}"));
                    a.ins(&format!("mov {ACC_HI}, {v2h}"));
                }
                a.ins(&format!("ldr {T2}, [{PC}, #24]")); // dst2*8
                a.ins(&format!("add {T2}, {FRAME}, {T2}"));
                a.ins(&format!("str {ACC}, [{T2}]"));
                a.ins(&format!("str {ACC_HI}, [{T2}, #4]"));
                if let Some(d2) = v.pair_d.second() {
                    debug_assert_eq!(d2, DstCls::L0);
                    if ACC != L0R {
                        a.ins(&format!("mov {L0R}, {ACC}"));
                        a.ins(&format!("mov {L0_HI}, {ACC_HI}"));
                    }
                }
            }
            Select => {
                // The condition is always a materialized i32 slot.
                a.ins(&format!("ldr {T2}, [{PC}, #28]")); // cond slot
                a.ins(&format!("ldr {T2}, [{FRAME}, {T2}]"));
                a.ins(&format!("cmp {T2}, #0"));
                let rd = self.dst_target(v.d);
                let rdh = self.dst_hi(v.d);
                let takeb = a.fresh("selb");
                let done = a.fresh("seldone");
                a.ins(&format!("beq {takeb}"));
                let (x, xh) = self.pair(a, v.a, 8, T0, T1);
                a.ins(&format!("mov {rd}, {x}"));
                a.ins(&format!("mov {rdh}, {xh}"));
                a.ins(&format!("b {done}"));
                a.label(&takeb);
                let (y, yh) = self.pair(a, v.b, 16, T0, T1);
                a.ins(&format!("mov {rd}, {y}"));
                a.ins(&format!("mov {rdh}, {yh}"));
                a.label(&done);
                if v.d != DstCls::Acc {
                    a.ins(&format!("ldr {T2}, [{PC}, #24]"));
                    a.ins(&format!("add {T2}, {FRAME}, {T2}"));
                    a.ins(&format!("str {rd}, [{T2}]"));
                    a.ins(&format!("str {rdh}, [{T2}, #4]"));
                }
            }
            GlobalGet => {
                a.ins(&format!("ldr {T0}, [{PC}, #8]"));
                a.ins(&format!("ldr {T1}, [{STATE}, #{S_GLOBALS}]"));
                a.ins(&format!("add {T0}, {T1}, {T0}"));
                let rd = self.dst_target(v.d);
                a.ins(&format!("ldr {rd}, [{T0}]"));
                a.ins(&format!("ldr {}, [{T0}, #4]", self.dst_hi(v.d)));
                self.finish(a, v.d, rd, true, T0);
            }
            GlobalSet => {
                let (x, xh) = self.pair(a, v.a, 8, T0, T1);
                a.ins(&format!("ldr {T2}, [{PC}, #24]"));
                a.ins(&format!("ldr {T3}, [{STATE}, #{S_GLOBALS}]"));
                a.ins(&format!("add {T2}, {T3}, {T2}"));
                a.ins(&format!("str {x}, [{T2}]"));
                a.ins(&format!("str {xh}, [{T2}, #4]"));
            }
            I32_Eqz | I64_Eqz => {
                let rd = self.dst_target(v.d);
                if v.op == I32_Eqz {
                    let ra = self.src(a, v.a, 8, T0);
                    a.ins(&format!("cmp {ra}, #0"));
                } else {
                    let (lo, hi) = self.pair(a, v.a, 8, T0, T1);
                    a.ins(&format!("orrs {T2}, {lo}, {hi}"));
                    a.ins(&format!("cmp {T2}, #0"));
                }
                a.ins(&format!("mov {rd}, #0"));
                self.cond_mov_imm(a, "eq", rd, 1);
                self.finish(a, v.d, rd, false, T0);
            }
            I32_WrapI64 | I64_ExtendI32U => {
                let ra = self.src(a, v.a, 8, T0);
                let rd = self.dst_target(v.d);
                a.ins(&format!("mov {rd}, {ra}"));
                self.finish(a, v.d, rd, false, T1);
            }
            I64_ExtendI32S => {
                let ra = self.src(a, v.a, 8, T0);
                let rd = self.dst_target(v.d);
                let rdh = self.dst_hi(v.d);
                a.ins(&format!("mov {rd}, {ra}"));
                a.ins(&format!("asr {rdh}, {rd}, #31"));
                self.finish(a, v.d, rd, true, T1);
            }
            I32_Extend8S | I32_Extend16S => {
                let ra = self.src(a, v.a, 8, T0);
                let rd = self.dst_target(v.d);
                let m = if v.op == I32_Extend8S { "sxtb" } else { "sxth" };
                a.ins(&format!("{m} {rd}, {ra}"));
                self.finish(a, v.d, rd, false, T1);
            }
            I32_Clz | I32_Ctz => {
                let ra = self.src(a, v.a, 8, T0);
                let rd = self.dst_target(v.d);
                if v.op == I32_Ctz {
                    a.ins(&format!("rbit {rd}, {ra}"));
                    a.ins(&format!("clz {rd}, {rd}"));
                } else {
                    a.ins(&format!("clz {rd}, {ra}"));
                }
                self.finish(a, v.d, rd, false, T1);
            }
            I32_Popcnt => {
                // The SWAR count: no `popcnt` in either profile's base ISA.
                let ra = self.src(a, v.a, 8, T0);
                let rd = self.dst_target(v.d);
                a.ins(&format!("lsr {T1}, {ra}, #1"));
                self.mov_imm32(a, T2, 0x5555_5555);
                a.ins(&format!("and {T1}, {T1}, {T2}"));
                a.ins(&format!("sub {T1}, {ra}, {T1}"));
                self.mov_imm32(a, T2, 0x3333_3333);
                a.ins(&format!("and {T3}, {T1}, {T2}"));
                a.ins(&format!("lsr {T1}, {T1}, #2"));
                a.ins(&format!("and {T1}, {T1}, {T2}"));
                a.ins(&format!("add {T1}, {T3}, {T1}"));
                a.ins(&format!("lsr {T3}, {T1}, #4"));
                a.ins(&format!("add {T1}, {T1}, {T3}"));
                self.mov_imm32(a, T2, 0x0f0f_0f0f);
                a.ins(&format!("and {T1}, {T1}, {T2}"));
                self.mov_imm32(a, T2, 0x0101_0101);
                a.ins(&format!("mul {T1}, {T1}, {T2}"));
                a.ins(&format!("lsr {rd}, {T1}, #24"));
                self.finish(a, v.d, rd, false, T1);
            }
            I32_DivS | I32_DivU | I32_RemS | I32_RemU => self.emit_div(a, st, v),
            _ => {
                if let Some((m, w32, shifty)) = int_bin(v.op) {
                    self.emit_bin(a, m, w32, shifty, v);
                } else if let Some((cond, w32)) = int_cmp(v.op) {
                    self.emit_compare(a, v, w32);
                    let rd = self.dst_target(v.d);
                    a.ins(&format!("mov {rd}, #0"));
                    self.cond_mov_imm(a, cond, rd, 1);
                    self.finish(a, v.d, rd, false, T0);
                } else if mem_kind(v.op).is_some() {
                    self.emit_mem(a, st, v);
                } else {
                    panic!("arm32: no handler shape for {:?}", v.op);
                }
            }
        }
        self.tail(a, v.counted);
    }
}

impl Arm32 {
    /// Set the flags for a compare. The 64-bit equality forms fold their
    /// pair into one word first, which is why only `eq`/`ne` of them stay
    /// native.
    fn emit_compare(&mut self, a: &mut Asm, v: &Variant, w32: bool) {
        use Op::*;
        if matches!(v.op, I32_BrAnd | I32_BrAndNot) {
            let x = self.src(a, v.a, 8, T0);
            let y = self.src(a, v.b, 16, T1);
            a.ins(&format!("tst {x}, {y}"));
            return;
        }
        if w32 {
            let x = self.src(a, v.a, 8, T0);
            let y = self.src(a, v.b, 16, T1);
            a.ins(&format!("cmp {x}, {y}"));
            return;
        }
        let (xl, xh) = self.pair(a, v.a, 8, T0, T1);
        a.ins(&format!("mov {T2}, {xl}"));
        a.ins(&format!("mov {T3}, {xh}"));
        let (yl, yh) = self.pair(a, v.b, 16, T0, T1);
        a.ins(&format!("eor {T2}, {T2}, {yl}"));
        a.ins(&format!("eor {T3}, {T3}, {yh}"));
        a.ins(&format!("orrs {T2}, {T2}, {T3}"));
        a.ins(&format!("cmp {T2}, #0"));
    }

    fn emit_bin(&mut self, a: &mut Asm, m: &str, w32: bool, shifty: bool, v: &Variant) {
        let rd = self.dst_target(v.d);
        let rdh = self.dst_hi(v.d);
        if !w32 {
            let (xl, xh) = self.pair(a, v.a, 8, T0, T1);
            a.ins(&format!("mov {T2}, {xl}"));
            a.ins(&format!("mov {T3}, {xh}"));
            let (yl, yh) = self.pair(a, v.b, 16, T0, T1);
            match m {
                "add" => {
                    a.ins(&format!("adds {T2}, {T2}, {yl}"));
                    a.ins(&format!("adc {T3}, {T3}, {yh}"));
                }
                "sub" => {
                    a.ins(&format!("subs {T2}, {T2}, {yl}"));
                    a.ins(&format!("sbc {T3}, {T3}, {yh}"));
                }
                "and" | "orr" | "eor" => {
                    a.ins(&format!("{m} {T2}, {T2}, {yl}"));
                    a.ins(&format!("{m} {T3}, {T3}, {yh}"));
                }
                other => panic!("arm32: no 64-bit form for {other}"),
            }
            a.ins(&format!("mov {rd}, {T2}"));
            a.ins(&format!("mov {rdh}, {T3}"));
            self.finish(a, v.d, rd, true, T0);
            return;
        }
        let x = self.src(a, v.a, 8, T0);
        let y = self.src(a, v.b, 16, T1);
        if shifty {
            // ARM's register shifts read the low BYTE and flush to zero
            // past 31; wasm wants the amount modulo the width.
            a.ins(&format!("and {T2}, {y}, #31"));
            if m == "rotl" {
                a.ins(&format!("rsb {T2}, {T2}, #32"));
                a.ins(&format!("ror {rd}, {x}, {T2}"));
            } else {
                a.ins(&format!("{m} {rd}, {x}, {T2}"));
            }
        } else {
            a.ins(&format!("{m} {rd}, {x}, {y}"));
        }
        self.finish(a, v.d, rd, false, T0);
    }

    /// ARM's `sdiv`/`udiv` do not fault: divide-by-zero yields zero and
    /// MIN/-1 wraps. Both are wasm traps, so both are detected here and
    /// bail to the shared executor, which raises the right error.
    fn emit_div(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        use Op::*;
        let (signed, rem) = match v.op {
            I32_DivS => (true, false),
            I32_DivU => (false, false),
            I32_RemS => (true, true),
            _ => (false, true),
        };
        let x = self.src(a, v.a, 8, T0);
        let y = self.src(a, v.b, 16, T1);
        a.ins(&format!("mov {T2}, {x}"));
        a.ins(&format!("mov {T3}, {y}"));
        a.ins(&format!("cmp {T3}, #0"));
        a.ins(&format!("beq {}", st.slow));
        if signed {
            let go = a.fresh("div");
            a.ins(&format!("cmn {T3}, #1")); // y == -1 ?
            a.ins(&format!("bne {go}"));
            self.mov_imm32(a, T1, 0x8000_0000);
            a.ins(&format!("cmp {T2}, {T1}"));
            a.ins(&format!("beq {}", st.slow));
            a.label(&go);
        }
        let rd = self.dst_target(v.d);
        let m = if signed { "sdiv" } else { "udiv" };
        if rem {
            a.ins(&format!("{m} {T1}, {T2}, {T3}"));
            a.ins(&format!("mls {rd}, {T1}, {T3}, {T2}"));
        } else {
            a.ins(&format!("{m} {rd}, {T2}, {T3}"));
        }
        self.finish(a, v.d, rd, false, T0);
    }

    fn emit_mem(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let (size, load, kind) = mem_kind(v.op).unwrap();
        let addr = self.src(a, v.a, 8, T0);
        if v.fused {
            // ea = wrapping i32 sum of two slot operands, plus the static
            // offset — the corpus-universal base+index pattern.
            a.ins(&format!("ldr {T1}, [{PC}, #28]")); // addr2*8
            a.ins(&format!("ldr {T1}, [{FRAME}, {T1}]"));
            // The two address operands sum with i32 wrapping by
            // definition, so this add is allowed to carry out.
            a.ins(&format!("add {T1}, {addr}, {T1}"));
            if load {
                a.ins(&format!("ldr {T2}, [{PC}, #16]")); // static offset
            } else {
                a.ins(&format!("ldr {T2}, [{PC}, #24]"));
            }
            a.ins(&format!("adds {T2}, {T2}, {T1}"));
            a.ins(&format!("bcs {}", st.trap_oob));
        } else {
            a.ins(&format!(
                "ldr {T2}, [{PC}, #{}]",
                if load { 16 } else { 24 }
            ));
            a.ins(&format!("adds {T2}, {T2}, {addr}"));
            a.ins(&format!("bcs {}", st.trap_oob));
        }
        // Both the offset+address sum and the +size are 33-bit on a 32-bit
        // host, so each carry-out is part of the check. Without it a
        // wrapped address lands back inside the memory and the access
        // silently succeeds where wasm requires a trap.
        a.ins(&format!("adds {T3}, {T2}, #{size}"));
        a.ins(&format!("bcs {}", st.trap_oob));
        a.ins(&format!("cmp {T3}, {MEMLEN}"));
        a.ins(&format!("bhi {}", st.trap_oob));
        a.ins(&format!("add {T2}, {MEM}, {T2}"));
        if load {
            let rd = self.dst_target(v.d);
            let rdh = self.dst_hi(v.d);
            let wide = kind == 1;
            if wide {
                a.ins(&format!("ldr {rd}, [{T2}]"));
                a.ins(&format!("ldr {rdh}, [{T2}, #4]"));
            } else {
                let m = match kind {
                    0 => "ldr",
                    2 => "ldrb",
                    3 => "ldrh",
                    4 => "ldrsb",
                    5 => "ldrsh",
                    _ => unreachable!(),
                };
                a.ins(&format!("{m} {rd}, [{T2}]"));
            }
            if v.fused {
                if v.d != DstCls::Acc {
                    a.ins(&format!("ldr {T1}, [{PC}, #24]"));
                    a.ins(&format!("add {T1}, {FRAME}, {T1}"));
                    a.ins(&format!("str {rd}, [{T1}]"));
                    if wide {
                        a.ins(&format!("str {rdh}, [{T1}, #4]"));
                    } else {
                        a.ins(&format!("mov {T3}, #0"));
                        a.ins(&format!("str {T3}, [{T1}, #4]"));
                    }
                }
            } else {
                self.finish(a, v.d, rd, wide, T1);
            }
        } else if kind == 1 {
            let (lo, hi) = self.pair(a, v.b, 16, T0, T1);
            a.ins(&format!("str {lo}, [{T2}]"));
            a.ins(&format!("str {hi}, [{T2}, #4]"));
        } else {
            let rb = self.src(a, v.b, 16, T0);
            let m = match kind {
                0 => "str",
                2 => "strb",
                3 => "strh",
                _ => unreachable!(),
            };
            a.ins(&format!("{m} {rb}, [{T2}]"));
        }
    }

    fn emit_return(&mut self, a: &mut Asm, v: &Variant) {
        self.bump(a, v.counted);
        a.ins(&format!("ldr {T0}, [{PC}, #8]")); // first-result slot*8
        a.ins(&format!("ldr {T1}, [{PC}, #16]")); // result count
        a.ins(&format!("add {T0}, {FRAME}, {T0}"));
        a.ins(&format!("mov {T2}, {FRAME}"));
        let pop = a.fresh("pop");
        let cl = a.fresh("cl");
        a.ins(&format!("cmp {T1}, #0"));
        a.ins(&format!("beq {pop}"));
        // The accumulator doubles as the copy scratch: after a
        // single-result copy it holds result 0, which is the
        // call-result-in-acc convention at zero extra instructions.
        a.label(&cl);
        a.ins(&format!("ldr {ACC}, [{T0}]"));
        a.ins(&format!("ldr {ACC_HI}, [{T0}, #4]"));
        a.ins(&format!("str {ACC}, [{T2}]"));
        a.ins(&format!("str {ACC_HI}, [{T2}, #4]"));
        a.ins(&format!("add {T0}, {T0}, #8"));
        a.ins(&format!("add {T2}, {T2}, #8"));
        a.ins(&format!("subs {T1}, {T1}, #1"));
        a.ins(&format!("bne {cl}"));
        a.label(&pop);
        a.ins(&format!("ldr {T3}, [{STATE}, #{S_RET_CURSOR}]"));
        a.ins(&format!("sub {T3}, {T3}, #32"));
        a.ins(&format!("str {T3}, [{STATE}, #{S_RET_CURSOR}]"));
        a.ins(&format!("ldr {T0}, [{T3}]")); // ret pc
        a.ins(&format!("ldr {FRAME}, [{T3}, #8]")); // caller frame
                                                    // The packed word straddles two machine words here: the cell base
                                                    // is the low half and the caller's l0 offset the high half's low
                                                    // 16 bits. Sentinel records carry a readable dummy frame, so the
                                                    // reload below is always safe.
        a.ins(&format!("ldr {T1}, [{T3}, #16]")); // caller cell base
        a.ins(&format!("str {T1}, [{STATE}, #{S_CODE}]"));
        a.ins(&format!("ldr {T2}, [{T3}, #20]"));
        a.ins(&format!("uxth {T2}, {T2}")); // caller l0off
        a.ins(&format!("add {T2}, {FRAME}, {T2}"));
        a.ins(&format!("ldr {L0R}, [{T2}]"));
        a.ins(&format!("ldr {L0_HI}, [{T2}, #4]"));
        a.ins(&format!("mov {PC}, {T0}"));
        a.ins(&format!("ldr {T0}, [{PC}]"));
        a.ins(&format!("bx {T0}"));
    }

    /// `memory.fill` / `memory.copy` on memory 0, a word at a time with a
    /// byte tail.
    fn emit_bulk(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let fill = v.op == Op::MemoryFill;
        self.pre(a);
        a.ins(&format!("ldr {T0}, [{PC}, #8]"));
        a.ins(&format!("add {T0}, {FRAME}, {T0}"));
        a.ins(&format!("ldr {T1}, [{T0}]")); // d
        a.ins(&format!("ldr {T2}, [{T0}, #8]")); // fill: value, copy: s
        a.ins(&format!("ldr {T3}, [{T0}, #16]")); // n
        a.ins(&format!("adds {T0}, {T1}, {T3}"));
        a.ins(&format!("bcs {}", st.slow));
        a.ins(&format!("cmp {T0}, {MEMLEN}"));
        a.ins(&format!("bhi {}", st.slow)); // dst out of bounds
        let lw = a.fresh("lw");
        let bytes = a.fresh("bytes");
        let done = a.fresh("done");
        if fill {
            a.ins(&format!("and {T2}, {T2}, #255"));
            a.ins(&format!("orr {T2}, {T2}, {T2}, lsl #8"));
            a.ins(&format!("orr {T2}, {T2}, {T2}, lsl #16"));
            a.ins(&format!("add {T1}, {MEM}, {T1}")); // cursor
            a.ins(&format!("add {T0}, {MEM}, {T0}")); // end
            a.label(&lw);
            a.ins(&format!("add {T3}, {T1}, #4"));
            a.ins(&format!("cmp {T3}, {T0}"));
            a.ins(&format!("bhi {bytes}"));
            a.ins(&format!("str {T2}, [{T1}]"));
            a.ins(&format!("mov {T1}, {T3}"));
            a.ins(&format!("b {lw}"));
            a.label(&bytes);
            a.ins(&format!("cmp {T1}, {T0}"));
            a.ins(&format!("bhs {done}"));
            a.ins(&format!("strb {T2}, [{T1}]"));
            a.ins(&format!("add {T1}, {T1}, #1"));
            a.ins(&format!("b {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
        } else {
            a.ins(&format!("adds {T0}, {T2}, {T3}"));
            a.ins(&format!("bcs {}", st.slow));
            a.ins(&format!("cmp {T0}, {MEMLEN}"));
            a.ins(&format!("bhi {}", st.slow)); // src out of bounds
            let back = a.fresh("copyback");
            let fwd = a.fresh("copyfwd");
            // A forward copy is wrong only when s < d < s+n.
            a.ins(&format!("cmp {T1}, {T2}"));
            a.ins(&format!("bls {fwd}"));
            a.ins(&format!("cmp {T1}, {T0}"));
            a.ins(&format!("blo {back}"));
            a.label(&fwd);
            a.ins(&format!("add {T1}, {MEM}, {T1}")); // dst cursor
            a.ins(&format!("add {T2}, {MEM}, {T2}")); // src cursor
            a.ins(&format!("add {T0}, {T1}, {T3}")); // dst end
            a.label(&lw);
            a.ins(&format!("add {T3}, {T1}, #4"));
            a.ins(&format!("cmp {T3}, {T0}"));
            a.ins(&format!("bhi {bytes}"));
            a.ins(&format!("ldr {ACC}, [{T2}]"));
            a.ins(&format!("str {ACC}, [{T1}]"));
            a.ins(&format!("add {T2}, {T2}, #4"));
            a.ins(&format!("mov {T1}, {T3}"));
            a.ins(&format!("b {lw}"));
            a.label(&bytes);
            a.ins(&format!("cmp {T1}, {T0}"));
            a.ins(&format!("bhs {done}"));
            a.ins(&format!("ldrb {ACC}, [{T2}]"));
            a.ins(&format!("strb {ACC}, [{T1}]"));
            a.ins(&format!("add {T1}, {T1}, #1"));
            a.ins(&format!("add {T2}, {T2}, #1"));
            a.ins(&format!("b {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
            // Overlapping-downward block, out of line past the tail:
            // T1 = d, T2 = s, T3 = n, all raw offsets.
            let bw = a.fresh("bw");
            let bb = a.fresh("bb");
            let bdone = a.fresh("bdone");
            a.label(&back);
            a.ins(&format!("add {T1}, {MEM}, {T1}"));
            a.ins(&format!("add {T2}, {MEM}, {T2}"));
            a.label(&bw);
            a.ins(&format!("cmp {T3}, #4"));
            a.ins(&format!("blo {bb}"));
            a.ins(&format!("sub {T3}, {T3}, #4"));
            a.ins(&format!("ldr {ACC}, [{T2}, {T3}]"));
            a.ins(&format!("str {ACC}, [{T1}, {T3}]"));
            a.ins(&format!("b {bw}"));
            a.label(&bb);
            a.ins(&format!("cmp {T3}, #0"));
            a.ins(&format!("beq {bdone}"));
            a.ins(&format!("sub {T3}, {T3}, #1"));
            a.ins(&format!("ldrb {ACC}, [{T2}, {T3}]"));
            a.ins(&format!("strb {ACC}, [{T1}, {T3}]"));
            a.ins(&format!("b {bb}"));
            a.label(&bdone);
            self.tail(a, v.counted);
        }
    }
}
