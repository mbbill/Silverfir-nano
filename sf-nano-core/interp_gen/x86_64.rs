//! x86-64 handler backend (SSE2 baseline, System V and Win64).
//!
//! Register contract (chain-invariant across every handler):
//! ```text
//!   rbx = pc (current 32-byte dispatch cell)   rbp = frame base
//!   r15 = &EnterState                          r14 = memory 0 base
//!   r13 = memory 0 length                      r9  = cell array base
//!   r12 = accumulator   r11 = l0   r10 = l1
//!   rdi = prefetched next handler word
//!   rax, rcx, rdx, rsi, r8 = scratch
//!   xmm3 = float acc, xmm4/xmm5 = float l0/l1, xmm0-xmm2 = float scratch
//! ```
//!
//! x86-64 has four fewer usable GPRs than arm64, so three chain roles that
//! arm64 pins live in `EnterState` instead: the globals base and the
//! return-stack cursor/limit and value-stack limit. All three are read by
//! cold handlers only (globals, calls, returns), and moving them out is
//! what leaves five scratch registers — enough for `idiv`'s fixed
//! `rdx:rax` pair and `shl`'s fixed `cl` without spilling.
//!
//! The float registers are deliberately xmm0-xmm5: those are volatile
//! under BOTH ABIs, so the entry trampoline never has to save any of them
//! (Win64 makes xmm6-xmm15 callee-saved).
//!
//! Baseline coverage gaps, all routed to the shared executor rather than
//! approximated: `ceil`/`floor`/`trunc`/`nearest` need SSE4.1 `roundsd`;
//! the saturating float->int family and the unsigned-64 conversions need
//! multi-branch fixups whose value does not pay for their size here. Every
//! one of those is a full decline, never a runtime bail — a handler that
//! bails on a path which could still SUCCEED would leave an accumulator
//! consumer reading a stale register.

use super::asm::Asm;
use super::instr::Op;
use super::isa::{Caps, Isa, PairDstSplit, Stubs, Variant, CLASSES, DSTS};
use super::layout::{Cls, DstCls, Fam};

const RAX: u32 = 0;
const RCX: u32 = 1;
const RDX: u32 = 2;
const RBX: u32 = 3;
const RBP: u32 = 5;
const RSI: u32 = 6;
const RDI: u32 = 7;
const R8: u32 = 8;
const R9: u32 = 9;
const ACC: u32 = 12;
const L0R: u32 = 11;
const L1R: u32 = 10;
const MEM: u32 = 14;
const MEMLEN: u32 = 13;
const STATE: u32 = 15;

const FACC: u32 = 3;
const FL0R: u32 = 4;
const FL1R: u32 = 5;

/// State-block offsets for the roles that do not get a register here.
const S_GLOBALS: u32 = 48;
const S_RET_CURSOR: u32 = 56;
const S_RET_LIMIT: u32 = 64;
const S_STACK_LIMIT: u32 = 72;
const S_DISPATCHES: u32 = 80;

const NAMES64: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
const NAMES32: [&str; 16] = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d",
    "r13d", "r14d", "r15d",
];
const NAMES16: [&str; 16] = [
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w",
    "r14w", "r15w",
];
const NAMES8: [&str; 16] = [
    "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
    "r13b", "r14b", "r15b",
];

fn q(n: u32) -> &'static str {
    NAMES64[n as usize]
}
fn d(n: u32) -> &'static str {
    NAMES32[n as usize]
}
fn h16(n: u32) -> &'static str {
    NAMES16[n as usize]
}
fn b8(n: u32) -> &'static str {
    NAMES8[n as usize]
}
fn r(w32: bool, n: u32) -> &'static str {
    if w32 {
        d(n)
    } else {
        q(n)
    }
}
fn xm(n: u32) -> String {
    format!("xmm{n}")
}

fn inv(c: &str) -> &'static str {
    match c {
        "e" => "ne",
        "ne" => "e",
        "l" => "ge",
        "ge" => "l",
        "g" => "le",
        "le" => "g",
        "b" => "ae",
        "ae" => "b",
        "a" => "be",
        "be" => "a",
        other => panic!("no inverse for {other}"),
    }
}

/// `(condition suffix, 32-bit form)` for integer compares and their fused
/// compare-branch twins.
fn int_cmp(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        I32_Eq | I32_BrEq => ("e", true),
        I32_Ne | I32_BrNe => ("ne", true),
        I32_LtS | I32_BrLtS => ("l", true),
        I32_LtU | I32_BrLtU => ("b", true),
        I32_GtS | I32_BrGtS => ("g", true),
        I32_GtU | I32_BrGtU => ("a", true),
        I32_LeS | I32_BrLeS => ("le", true),
        I32_LeU | I32_BrLeU => ("be", true),
        I32_GeS | I32_BrGeS => ("ge", true),
        I32_GeU | I32_BrGeU => ("ae", true),
        I64_Eq | I64_BrEq => ("e", false),
        I64_Ne | I64_BrNe => ("ne", false),
        I64_LtS | I64_BrLtS => ("l", false),
        I64_LtU | I64_BrLtU => ("b", false),
        I64_GtS | I64_BrGtS => ("g", false),
        I64_GtU | I64_BrGtU => ("a", false),
        I64_LeS | I64_BrLeS => ("le", false),
        I64_LeU | I64_BrLeU => ("be", false),
        I64_GeS | I64_BrGeS => ("ge", false),
        I64_GeU | I64_BrGeU => ("ae", false),
        I32_BrAnd => ("ne", true),
        I32_BrAndNot => ("e", true),
        _ => return None,
    })
}

/// `(mnemonic, 32-bit form, commutative, count-in-cl)`.
fn int_bin(op: Op) -> Option<(&'static str, bool, bool, bool)> {
    use Op::*;
    Some(match op {
        I32_Add => ("add", true, true, false),
        I32_Sub => ("sub", true, false, false),
        I32_And => ("and", true, true, false),
        I32_Or => ("or", true, true, false),
        I32_Xor => ("xor", true, true, false),
        I32_Mul => ("imul", true, true, false),
        I32_Shl => ("shl", true, false, true),
        I32_ShrU => ("shr", true, false, true),
        I32_ShrS => ("sar", true, false, true),
        I32_Rotr => ("ror", true, false, true),
        I32_Rotl => ("rol", true, false, true),
        I64_Add => ("add", false, true, false),
        I64_Sub => ("sub", false, false, false),
        I64_And => ("and", false, true, false),
        I64_Or => ("or", false, true, false),
        I64_Xor => ("xor", false, true, false),
        I64_Mul => ("imul", false, true, false),
        I64_Shl => ("shl", false, false, true),
        I64_ShrU => ("shr", false, false, true),
        I64_ShrS => ("sar", false, false, true),
        I64_Rotr => ("ror", false, false, true),
        I64_Rotl => ("rol", false, false, true),
        _ => return None,
    })
}

/// `(size in bytes, is a load, access kind)`; kind 0 = 32-bit zero-extend,
/// 1 = 64-bit, 2 = u8, 3 = u16, 4 = i8->i32, 5 = i16->i32, 6 = i8->i64,
/// 7 = i16->i64, 8 = i32->i64.
fn mem_kind(op: Op) -> Option<(u32, bool, u8)> {
    use Op::*;
    Some(match op {
        I32_Load | F32_Load => (4, true, 0),
        I64_Load | F64_Load => (8, true, 1),
        I32_Load8U | I64_Load8U => (1, true, 2),
        I32_Load16U | I64_Load16U => (2, true, 3),
        I32_Load8S => (1, true, 4),
        I32_Load16S => (2, true, 5),
        I64_Load8S => (1, true, 6),
        I64_Load16S => (2, true, 7),
        I64_Load32S => (4, true, 8),
        I64_Load32U => (4, true, 0),
        I32_Store | F32_Store => (4, false, 0),
        I64_Store | F64_Store => (8, false, 1),
        I32_Store8 | I64_Store8 => (1, false, 2),
        I32_Store16 | I64_Store16 => (2, false, 3),
        I64_Store32 => (4, false, 0),
        _ => return None,
    })
}

fn is_fp_mem(op: Op) -> bool {
    matches!(
        op,
        Op::F32_Load | Op::F64_Load | Op::F32_Store | Op::F64_Store
    )
}

/// `(mnemonic stem, f32 form)` for the float binaries with a direct SSE2
/// instruction. `min`/`max` are not here: `minsd` returns its SECOND
/// operand on any NaN and on equality, which is neither wasm's NaN rule
/// nor its signed-zero rule, so they get an explicit sequence.
fn float_bin(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        F32_Add => ("add", true),
        F32_Sub => ("sub", true),
        F32_Mul => ("mul", true),
        F32_Div => ("div", true),
        F64_Add => ("add", false),
        F64_Sub => ("sub", false),
        F64_Mul => ("mul", false),
        F64_Div => ("div", false),
        _ => return None,
    })
}

/// `(condition suffix, f32 form, needs the parity guard)`. `ucomis*` sets
/// PF on unordered, and the unordered answer must be false for every wasm
/// compare except `ne`. `lt`/`le` get it for free by comparing with the
/// operands swapped, so only `eq`/`ne` need an explicit parity term.
fn float_cmp(op: Op) -> Option<(&'static str, bool, bool, bool)> {
    use Op::*;
    // (suffix, f32, swap operands, parity-guarded)
    Some(match op {
        F32_Eq => ("e", true, false, true),
        F32_Ne => ("ne", true, false, true),
        F32_Lt => ("a", true, true, false),
        F32_Gt => ("a", true, false, false),
        F32_Le => ("ae", true, true, false),
        F32_Ge => ("ae", true, false, false),
        F64_Eq => ("e", false, false, true),
        F64_Ne => ("ne", false, false, true),
        F64_Lt => ("a", false, true, false),
        F64_Gt => ("a", false, false, false),
        F64_Le => ("ae", false, true, false),
        F64_Ge => ("ae", false, false, false),
        _ => return None,
    })
}

/// `(f32 source, 64-bit destination, low bound, high bound)` — the same
/// exclusive trap boundaries the arm64 backend checks, on the UN-truncated
/// operand, so a bail always means a trap. Signedness does not appear: the
/// conversion always targets a 64-bit GPR, and for the unsigned 32-bit
/// forms the in-range value is below 2^32, so its low half is the answer.
fn cvt_f2i_trap(op: Op) -> Option<(bool, bool, u64, u64)> {
    use Op::*;
    Some(match op {
        I32_TruncF32S => (true, false, 0xCF00_0001, 0x4F00_0000),
        I32_TruncF32U => (true, false, 0xBF80_0000, 0x4F80_0000),
        I32_TruncF64S => (false, false, 0xC1E0_0000_0020_0000, 0x41E0_0000_0000_0000),
        I32_TruncF64U => (false, false, 0xBFF0_0000_0000_0000, 0x41F0_0000_0000_0000),
        I64_TruncF32S => (true, true, 0xDF00_0001, 0x5F00_0000),
        I64_TruncF64S => (false, true, 0xC3E0_0000_0000_0001, 0x43E0_0000_0000_0000),
        // The unsigned-64 forms would need the subtract-2^63 fixup after
        // the range check; they are rare enough to leave slow.
        _ => return None,
    })
}

pub struct X86_64 {
    /// Win64 passes the argument in rcx and makes rdi/rsi callee-saved.
    pub windows: bool,
}

impl X86_64 {
    fn pre(&self, a: &mut Asm) {
        a.ins("mov rdi, [rbx + 32]");
    }

    fn bump(&self, a: &mut Asm, on: bool) {
        if on {
            a.ins(&format!("inc qword ptr [r15 + {S_DISPATCHES}]"));
        }
    }

    fn tail(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins("add rbx, 32");
        a.ins("jmp rdi");
    }

    /// Materialize a source operand. Register classes cost nothing;
    /// constants load inline from the cell; slots load through the frame.
    fn src(&self, a: &mut Asm, cls: Cls, field: u32, tmp: u32) -> u32 {
        match cls {
            Cls::Acc => ACC,
            Cls::L0 => L0R,
            Cls::L1 => L1R,
            Cls::Const => {
                a.ins(&format!("mov {}, [rbx + {field}]", q(tmp)));
                tmp
            }
            Cls::Slot => {
                a.ins(&format!("mov {}, [rbx + {field}]", q(tmp)));
                a.ins(&format!("mov {0}, [rbp + {0}]", q(tmp)));
                tmp
            }
        }
    }

    fn src_ab(&self, a: &mut Asm, ac: Cls, bc: Cls) -> (u32, u32) {
        (self.src(a, ac, 8, RAX), self.src(a, bc, 16, RCX))
    }

    fn dst_target(&self, dc: DstCls) -> u32 {
        match dc {
            DstCls::L0 => L0R,
            DstCls::L1 => L1R,
            _ => ACC,
        }
    }

    fn finish(&self, a: &mut Asm, dc: DstCls, src: u32) {
        if dc != DstCls::Acc {
            a.ins("mov rdx, [rbx + 24]");
            a.ins(&format!("mov [rbp + rdx], {}", q(src)));
        }
    }

    /// Two-operand ALU with the aliasing cases spelled out: x86 destroys
    /// its destination, so `dst == rb` needs either commutativity or a
    /// scratch round trip.
    fn alu2(&self, a: &mut Asm, m: &str, w32: bool, dst: u32, ra: u32, rb: u32, commutative: bool) {
        if dst == rb && dst != ra {
            if commutative {
                a.ins(&format!("{m} {}, {}", r(w32, dst), r(w32, ra)));
            } else {
                a.ins(&format!("mov {}, {}", q(RSI), q(ra)));
                a.ins(&format!("{m} {}, {}", r(w32, RSI), r(w32, rb)));
                a.ins(&format!("mov {}, {}", q(dst), q(RSI)));
            }
            return;
        }
        if dst != ra {
            a.ins(&format!("mov {}, {}", q(dst), q(ra)));
        }
        a.ins(&format!("{m} {}, {}", r(w32, dst), r(w32, rb)));
    }

    /// Variable shifts and rotates take their count in `cl` only.
    fn shift(&self, a: &mut Asm, m: &str, w32: bool, dst: u32, ra: u32, rb: u32) {
        if rb != RCX {
            a.ins(&format!("mov rcx, {}", q(rb)));
        }
        if dst == RCX {
            a.ins(&format!("mov {}, {}", q(RSI), q(ra)));
            a.ins(&format!("{m} {}, cl", r(w32, RSI)));
            a.ins(&format!("mov {}, {}", q(dst), q(RSI)));
            return;
        }
        if dst != ra {
            a.ins(&format!("mov {}, {}", q(dst), q(ra)));
        }
        a.ins(&format!("{m} {}, cl", r(w32, dst)));
    }

    fn fp_target(&self, dc: DstCls) -> u32 {
        match dc {
            DstCls::L0 => FL0R,
            DstCls::L1 => FL1R,
            _ => FACC,
        }
    }

    /// Load a float operand into an xmm register. `movss` zeroes the upper
    /// bits, so an f32 read of a zero-extended slot is exact.
    fn src_fp(&self, a: &mut Asm, cls: Cls, w32: bool, v: u32, field: u32, tmp: u32) -> u32 {
        let m = if w32 { "movss" } else { "movsd" };
        match cls {
            Cls::Slot => {
                a.ins(&format!("mov {}, [rbx + {field}]", q(tmp)));
                a.ins(&format!("{m} {}, [rbp + {}]", xm(v), q(tmp)));
                v
            }
            Cls::Const => {
                a.ins(&format!("{m} {}, [rbx + {field}]", xm(v)));
                v
            }
            Cls::Acc => FACC,
            Cls::L0 => FL0R,
            Cls::L1 => FL1R,
        }
    }

    fn src_fp_ab(&self, a: &mut Asm, ac: Cls, bc: Cls, w32: bool) -> (u32, u32) {
        (
            self.src_fp(a, ac, w32, 0, 8, RAX),
            self.src_fp(a, bc, w32, 1, 16, RCX),
        )
    }

    /// Land a float result in its destination slot.
    ///
    /// Always through a GPR, never a direct 8-byte `movsd`: an `ss`
    /// arithmetic op writes only the low 32 bits of its xmm and leaves the
    /// upper half untouched, so storing eight bytes from it would write
    /// stale bits into the slot and break the zero-extension convention
    /// that every f32 reader depends on.
    fn finish_fp(&self, a: &mut Asm, dc: DstCls, w32: bool) {
        if dc == DstCls::Acc {
            return;
        }
        let v = self.fp_target(dc);
        a.ins("mov rdx, [rbx + 24]");
        if w32 {
            a.ins(&format!("movd eax, {}", xm(v)));
        } else {
            a.ins(&format!("movq rax, {}", xm(v)));
        }
        a.ins("mov [rbp + rdx], rax");
    }

    /// Two-operand float ALU with the same aliasing care as [`Self::alu2`].
    fn fp_bin(&self, a: &mut Asm, m: &str, w32: bool, dst: u32, va: u32, vb: u32) {
        let sfx = if w32 { "ss" } else { "sd" };
        let mov = if w32 { "movaps" } else { "movapd" };
        if dst == vb && dst != va {
            a.ins(&format!("{mov} xmm2, {}", xm(va)));
            a.ins(&format!("{m}{sfx} xmm2, {}", xm(vb)));
            a.ins(&format!("{mov} {}, xmm2", xm(dst)));
            return;
        }
        if dst != va {
            a.ins(&format!("{mov} {}, {}", xm(dst), xm(va)));
        }
        a.ins(&format!("{m}{sfx} {}, {}", xm(dst), xm(vb)));
    }

    /// wasm `min`/`max`: NaN in either operand yields a quiet NaN, and
    /// `-0` and `+0` compare equal but must resolve by sign. `minsd` does
    /// neither, so the three cases are explicit. The result is built in
    /// xmm2 and moved out once, which keeps every operand alias safe.
    fn fp_minmax(&self, a: &mut Asm, is_min: bool, w32: bool, dst: u32, va: u32, vb: u32) {
        let sfx = if w32 { "ss" } else { "sd" };
        let mov = if w32 { "movaps" } else { "movapd" };
        let cmp = if w32 { "ucomiss" } else { "ucomisd" };
        let bits = if w32 { "ps" } else { "pd" };
        let nan = a.fresh("mmnan");
        let ne = a.fresh("mmne");
        let takea = a.fresh("mmtakea");
        let done = a.fresh("mmdone");
        a.ins(&format!("{cmp} {}, {}", xm(va), xm(vb)));
        a.ins(&format!("jp {nan}"));
        a.ins(&format!("jne {ne}"));
        // Equal, including -0 vs +0: `or` keeps the negative zero (min),
        // `and` keeps the positive one (max). For equal non-zeros both
        // reproduce the operand.
        a.ins(&format!("{mov} xmm2, {}", xm(va)));
        a.ins(&format!(
            "{} xmm2, {}",
            if is_min {
                format!("or{bits}")
            } else {
                format!("and{bits}")
            },
            xm(vb)
        ));
        a.ins(&format!("jmp {done}"));
        a.label(&ne);
        a.ins(&format!("j{} {takea}", if is_min { "b" } else { "a" }));
        a.ins(&format!("{mov} xmm2, {}", xm(vb)));
        a.ins(&format!("jmp {done}"));
        a.label(&takea);
        a.ins(&format!("{mov} xmm2, {}", xm(va)));
        a.ins(&format!("jmp {done}"));
        a.label(&nan);
        // Adding quiets a signalling NaN, which is what wasm requires of
        // an arithmetic NaN result.
        a.ins(&format!("{mov} xmm2, {}", xm(va)));
        a.ins(&format!("add{sfx} xmm2, {}", xm(vb)));
        a.label(&done);
        a.ins(&format!("{mov} {}, xmm2", xm(dst)));
    }

    /// Move an xmm's bits into a GPR (`movd` zero-extends, upholding the
    /// f32 slot convention).
    fn fp_to_gpr(&self, a: &mut Asm, w32: bool, rd: u32, v: u32) {
        if w32 {
            a.ins(&format!("movd {}, {}", d(rd), xm(v)));
        } else {
            a.ins(&format!("movq {}, {}", q(rd), xm(v)));
        }
    }
}

impl Isa for X86_64 {
    fn caps(&self) -> Caps {
        Caps {
            classes: &CLASSES,
            dsts: &DSTS,
            ptr_bytes: 8,
            has_l1: true,
            has_float_regs: true,
            float_pin_f32: true,
            native_calls: true,
        }
    }

    fn emit_prelude(&mut self, a: &mut Asm, st: &Stubs) {
        let win = self.windows;
        // ---- common exit: rax = reason ----
        a.label(&st.exit_common);
        a.ins("mov [r15 + 0], rax");
        a.ins("mov [r15 + 8], rbx");
        a.ins("mov [r15 + 16], rbp");
        a.ins("mov [r15 + 104], r12");
        if win {
            a.ins("pop rsi");
            a.ins("pop rdi");
        }
        a.ins("pop r15");
        a.ins("pop r14");
        a.ins("pop r13");
        a.ins("pop r12");
        a.ins("pop rbx");
        a.ins("pop rbp");
        a.ins("ret");

        for (label, reason) in [
            (&st.slow, 1u32),
            (&st.return_exit, 2),
            (&st.trap_oob, 16),
            (&st.trap_exhaust, 17),
        ] {
            a.label(label);
            a.ins(&format!("mov eax, {reason}"));
            a.ins(&format!("jmp {}", st.exit_common));
        }

        // ---- entry trampoline: extern "C" fn(*mut EnterState) ----
        a.label(&st.entry);
        a.ins("push rbp");
        a.ins("push rbx");
        a.ins("push r12");
        a.ins("push r13");
        a.ins("push r14");
        a.ins("push r15");
        if win {
            a.ins("push rdi");
            a.ins("push rsi");
            a.ins("mov r15, rcx");
        } else {
            a.ins("mov r15, rdi");
        }
        a.ins("mov rbx, [r15 + 8]");
        a.ins("mov rbp, [r15 + 16]");
        a.ins("mov r14, [r15 + 24]");
        a.ins("mov r13, [r15 + 32]");
        a.ins("mov r9, [r15 + 40]");
        a.ins("mov r11, [r15 + 88]");
        a.ins("mov r10, [r15 + 96]");
        a.ins("mov r12, [r15 + 104]");
        a.ins("movq xmm4, [r15 + 88]");
        a.ins("movq xmm5, [r15 + 96]");
        a.ins("mov rax, [rbx]");
        a.ins("jmp rax");

        // ---- Call (wired by the cross-function fixup, not the table) ----
        a.label(&st.call);
        self.bump(a, true);
        a.ins("mov rdx, [rbx + 24]");
        a.ins("mov rcx, [rbx + 16]");
        a.ins("mov rax, [rbx + 8]");

        // Shared activation entry: rax/rcx/rdx hold the a/b/c-shaped callee
        // description. The indirect handler composes them from a runtime
        // lookup instead of from cell fields.
        a.label(&st.call_core);
        a.ins(&format!("mov rsi, [r15 + {S_RET_CURSOR}]"));
        a.ins(&format!("cmp rsi, [r15 + {S_RET_LIMIT}]"));
        a.ins(&format!("jae {}", st.trap_exhaust));
        a.ins("mov r8d, ecx");
        a.ins("and r8d, 0x7fffffff"); // arg_base*8 (bit 31 is the fp flag)
        a.ins("add r8, rbp"); // new frame base
        a.ins("mov rdi, rdx");
        a.ins("shr rdi, 32");
        a.ins("movzx edi, di"); // frame_slots
        a.ins("lea rdi, [r8 + rdi*8]");
        a.ins(&format!("cmp rdi, [r15 + {S_STACK_LIMIT}]"));
        a.ins(&format!("ja {}", st.trap_exhaust));
        // push (ret_pc, caller frame, code | caller_l0off<<48, caller_l1off)
        a.ins("lea rdi, [rbx + 32]");
        a.ins("mov [rsi], rdi");
        a.ins("mov [rsi + 8], rbp");
        a.ins("mov rdi, rdx");
        a.ins("shr rdi, 48");
        a.ins("shl rdi, 48");
        a.ins("or rdi, r9");
        a.ins("mov [rsi + 16], rdi");
        a.ins("mov rdi, rax");
        a.ins("shr rdi, 48");
        a.ins("mov [rsi + 24], rdi");
        a.ins("add rsi, 32");
        a.ins(&format!("mov [r15 + {S_RET_CURSOR}], rsi"));
        a.ins("mov rbp, r8");
        // zero the fresh locals: [n_params*8, n_locals*8)
        a.ins("movzx edi, dx"); // n_params
        a.ins("lea rdi, [rbp + rdi*8]");
        a.ins("mov r8, rdx");
        a.ins("shr r8, 16");
        a.ins("movzx r8d, r8w"); // n_locals
        a.ins("lea r8, [rbp + r8*8]");
        a.ins("xor esi, esi");
        let zl = a.fresh("zl");
        let zdone = a.fresh("zdone");
        a.label(&zl);
        a.ins("cmp rdi, r8");
        a.ins(&format!("jae {zdone}"));
        a.ins("mov [rdi], rsi");
        a.ins("add rdi, 8");
        a.ins(&format!("jmp {zl}"));
        a.label(&zdone);
        a.ins("mov rdi, rcx");
        a.ins("shr rdi, 32");
        a.ins("movzx edi, di"); // callee l0off
        a.ins("mov r11, [rbp + rdi]");
        a.ins("mov rdi, rcx");
        a.ins("shr rdi, 48"); // callee l1off
        a.ins("mov r10, [rbp + rdi]");
        // Float twins only when the callee has float-pinned slots (cell b
        // bit 31). Integer code falls through; the transfer block sits out
        // of line, because making integer code take a branch here measured
        // 11% on a zero-float call microbench.
        let fp = a.fresh("callfp");
        let cont = a.fresh("callcont");
        a.ins("test ecx, ecx");
        a.ins(&format!("js {fp}"));
        a.label(&cont);
        a.ins("shl rax, 16");
        a.ins("shr rax, 16"); // clean 48-bit cells address
        a.ins("mov r9, rax");
        a.ins("mov rbx, rax");
        a.ins("mov rax, [rbx]");
        a.ins("jmp rax");
        a.label(&fp);
        a.ins("movq xmm4, r11");
        a.ins("movq xmm5, r10");
        a.ins(&format!("jmp {cont}"));

        // ---- CallIndirect (wired by the fixup pass) ----
        a.label(&st.call_indirect);
        self.bump(a, true);
        a.ins("mov rsi, [rbx + 8]"); // cell a
        a.ins("mov eax, esi"); // index_slot*8
        a.ins("mov eax, [rbp + rax]"); // t, zero-extended by the 32-bit load
        a.ins("cmp rax, [r15 + 120]"); // table 0 length
        a.ins(&format!("jae {}", st.slow));
        a.ins("mov rcx, [r15 + 112]"); // table 0 entries
                                       // Entries are `RefHandle` slots (8 bytes); a plain handle's payload
                                       // is the function index. Null is all-ones and a tagged handle has
                                       // high bits set, so both fail the bound and take the slow path.
        a.ins("mov rcx, [rcx + rax*8]"); // fi
        a.ins("mov rdx, rcx");
        a.ins("shr rdx, 32");
        a.ins(&format!("jnz {}", st.slow)); // null or tagged entry
        a.ins("mov rdx, [r15 + 128]"); // info base
        a.ins("lea rax, [rcx + rcx*2]"); // fi*3
        a.ins("lea rdx, [rdx + rax*8]"); // entry = info + fi*24
        a.ins("mov rcx, [rbx + 16]"); // cell b
        a.ins("mov rax, [rdx + 8]"); // l1off<<48 | l0off<<32 | canon
        a.ins("mov edi, eax"); // canonical actual
        a.ins("mov r8, rcx");
        a.ins("shr r8, 32");
        a.ins("movzx r8d, r8w"); // canonical expected
        a.ins("cmp rdi, r8");
        a.ins(&format!("jne {}", st.slow)); // type mismatch
        a.ins("mov r8, [rdx]"); // callee cells | fp flag (0 = slow)
        a.ins("test r8, r8");
        a.ins(&format!("jz {}", st.slow));
        a.ins("mov rdx, [rdx + 16]"); // frame metadata
                                      // compose the call_core inputs
        a.ins("shr rax, 32");
        a.ins("shl rax, 32"); // callee l0/l1 offsets, canon cleared
        a.ins("mov edi, ecx"); // arg_base*8
        a.ins("or rax, rdi");
        a.ins("mov rdi, r8");
        a.ins("and edi, 1"); // callee fp flag
        a.ins("shl rdi, 31");
        a.ins("or rax, rdi"); // b-equiv
        a.ins("shr rcx, 48");
        a.ins("shl rcx, 48");
        a.ins("or rcx, rdx"); // c-equiv
        a.ins("shr r8, 5");
        a.ins("shl r8, 5"); // clean cells address
        a.ins("shr rsi, 48");
        a.ins("shl rsi, 48");
        a.ins("or rsi, r8"); // a-equiv
        a.ins("mov rdx, rcx");
        a.ins("mov rcx, rax");
        a.ins("mov rax, rsi");
        a.ins(&format!("jmp {}", st.call_core));
    }

    fn wants(&self, v: &Variant) -> bool {
        use Op::*;
        // Ops with no SSE2 form, or whose fixups cost more than they buy.
        // A decline is total: the linker then strips any accumulator hint
        // on the pair, which a runtime bail could not do.
        if matches!(
            v.op,
            F32_Ceil
                | F32_Floor
                | F32_Trunc
                | F32_Nearest
                | F64_Ceil
                | F64_Floor
                | F64_Trunc
                | F64_Nearest
                | I32_TruncSatF32S
                | I32_TruncSatF32U
                | I32_TruncSatF64S
                | I32_TruncSatF64U
                | I64_TruncSatF32S
                | I64_TruncSatF32U
                | I64_TruncSatF64S
                | I64_TruncSatF64U
                | I64_TruncF32U
                | I64_TruncF64U
                | F32_ConvertI64U
                | F64_ConvertI64U
        ) {
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
                if v.fused {
                    if v.a == Cls::Const {
                        return false;
                    }
                    if matches!(
                        v.op,
                        I64_Load8S
                            | I64_Load8U
                            | I64_Load16S
                            | I64_Load16U
                            | I64_Load32S
                            | I64_Load32U
                    ) {
                        return false;
                    }
                }
                true
            }
        }
    }

    fn emit_handler(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        use Op::*;
        let dc = v.d;
        match v.op {
            Return => return self.emit_return(a, v),
            Br => {
                self.bump(a, v.counted);
                a.ins("mov rbx, [rbx + 24]"); // c is an absolute cell address
                a.ins("mov rax, [rbx]");
                a.ins("jmp rax");
                return;
            }
            BrIf | BrIfNot => {
                self.pre(a);
                // Target and its handler word load at entry, not after the
                // condition resolves: the two are dependent, and on the
                // taken path they otherwise sit nose-to-tail on the
                // critical path.
                a.ins("mov rdx, [rbx + 24]");
                a.ins("mov rsi, [rdx]");
                let ra = self.src(a, v.a, 8, RAX);
                let nt = a.fresh("nt");
                a.ins(&format!("test {0}, {0}", d(ra)));
                a.ins(&format!("{} {nt}", if v.op == BrIf { "jz" } else { "jnz" }));
                self.bump(a, v.counted);
                a.ins("mov rbx, rdx");
                a.ins("jmp rsi");
                a.label(&nt);
                self.tail(a, v.counted);
                return;
            }
            I32_SubBrIf | I64_SubBrIf => {
                a.ins("mov rdx, [rbx + 24]");
                a.ins("mov rsi, [rdx]");
                a.ins("mov r8, [rbx + 8]"); // a slot byte offset
                let ra = match v.a {
                    Cls::Slot => {
                        a.ins("mov rax, [rbp + r8]");
                        RAX
                    }
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    Cls::Const | Cls::Acc => unreachable!(),
                };
                let rb = self.src(a, v.b, 16, RCX);
                debug_assert_eq!(
                    ra,
                    match v.a {
                        Cls::L0 => L0R,
                        Cls::L1 => L1R,
                        _ => RAX,
                    }
                );
                if v.op == I32_SubBrIf {
                    a.ins(&format!("sub {}, {}", d(ra), d(rb)));
                } else {
                    a.ins(&format!("sub {}, {}", q(ra), q(rb)));
                }
                a.ins(&format!("mov [rbp + r8], {}", q(ra)));
                let nt = a.fresh("nt");
                a.ins(&format!("jz {nt}"));
                self.bump(a, v.counted);
                a.ins("mov rbx, rdx");
                a.ins("jmp rsi");
                a.label(&nt);
                self.pre(a);
                self.tail(a, v.counted);
                return;
            }
            BrTable => {
                self.bump(a, v.counted);
                let ra = self.src(a, v.a, 8, RAX);
                a.ins("mov rcx, [rbx + 16]"); // flat table
                a.ins("mov rdx, [rbx + 24]"); // len - 1
                a.ins(&format!("mov esi, {}", d(ra)));
                a.ins("cmp esi, edx");
                a.ins("cmova esi, edx"); // clamp: out of range takes the default
                a.ins("mov esi, [rcx + rsi*4]"); // target instruction index
                a.ins("shl rsi, 5");
                a.ins("lea rbx, [r9 + rsi]");
                a.ins("mov rax, [rbx]");
                a.ins("jmp rax");
                return;
            }
            MemoryFill | MemoryCopy => return self.emit_bulk(a, st, v),
            _ => {}
        }
        if int_cmp(v.op).is_some() && super::layout::family(v.op) == Fam::SrcAB {
            let (cond, w32) = int_cmp(v.op).unwrap();
            self.pre(a);
            a.ins("mov rdx, [rbx + 24]");
            a.ins("mov rsi, [rdx]");
            let (ra, rb) = self.src_ab(a, v.a, v.b);
            if matches!(v.op, I32_BrAnd | I32_BrAndNot) {
                a.ins(&format!("test {}, {}", d(ra), d(rb)));
            } else {
                a.ins(&format!("cmp {}, {}", r(w32, ra), r(w32, rb)));
            }
            let nt = a.fresh("nt");
            a.ins(&format!("j{} {nt}", inv(cond)));
            self.bump(a, v.counted);
            a.ins("mov rbx, rdx");
            a.ins("jmp rsi");
            a.label(&nt);
            self.tail(a, v.counted);
            return;
        }

        self.pre(a);
        match v.op {
            MovSlot | MovConst => {
                let rd = self.dst_target(dc);
                match v.a {
                    Cls::Acc | Cls::L0 | Cls::L1 => {
                        let s = match v.a {
                            Cls::Acc => ACC,
                            Cls::L0 => L0R,
                            _ => L1R,
                        };
                        if s != rd {
                            a.ins(&format!("mov {}, {}", q(rd), q(s)));
                        }
                    }
                    Cls::Const => a.ins(&format!("mov {}, [rbx + 8]", q(rd))),
                    Cls::Slot => {
                        a.ins("mov rax, [rbx + 8]");
                        a.ins(&format!("mov {}, [rbp + rax]", q(rd)));
                    }
                }
                self.finish(a, dc, rd);
            }
            MovPair => {
                // Strictly ordered: commit dst1 (including its pinned
                // register, when present) before reading src2.
                a.ins("mov rdx, [rbx + 24]"); // dst1*8 << 32 | dst2*8
                let v1 = match v.a {
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    _ => {
                        a.ins("mov rax, [rbx + 8]");
                        a.ins("mov rax, [rbp + rax]");
                        RAX
                    }
                };
                a.ins("mov rsi, rdx");
                a.ins("shr rsi, 32");
                a.ins(&format!("mov [rbp + rsi], {}", q(v1)));
                let d1 = match v.pair_d.first() {
                    None => None,
                    Some(DstCls::L0) => Some(L0R),
                    Some(DstCls::L1) => Some(L1R),
                    Some(DstCls::Mem | DstCls::Acc) => unreachable!(),
                };
                if let Some(rd) = d1 {
                    if rd != v1 {
                        a.ins(&format!("mov {}, {}", q(rd), q(v1)));
                    }
                }
                let v2 = match v.b {
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    _ => {
                        a.ins("mov rcx, [rbx + 16]");
                        a.ins(&format!("mov {}, [rbp + rcx]", q(ACC)));
                        ACC
                    }
                };
                if v2 != ACC {
                    a.ins(&format!("mov {}, {}", q(ACC), q(v2)));
                }
                a.ins("mov edx, edx");
                a.ins(&format!("mov [rbp + rdx], {}", q(ACC)));
                let d2 = match v.pair_d.second() {
                    None => None,
                    Some(DstCls::L0) => Some(L0R),
                    Some(DstCls::L1) => Some(L1R),
                    Some(DstCls::Mem | DstCls::Acc) => unreachable!(),
                };
                if let Some(rd) = d2 {
                    if rd != ACC {
                        a.ins(&format!("mov {}, {}", q(rd), q(ACC)));
                    }
                }
            }
            Select => {
                let (ra, rb) = self.src_ab(a, v.a, v.b);
                a.ins("mov rdx, [rbx + 24]");
                a.ins("mov rsi, rdx");
                a.ins("shr rsi, 32"); // cond slot byte offset
                a.ins("mov rsi, [rbp + rsi]");
                let rd = self.dst_target(dc);
                // Build in rsi so an alias between rd and either operand
                // cannot destroy the other.
                a.ins(&format!("test {0}, {0}", q(RSI)));
                a.ins(&format!("mov {}, {}", q(R8), q(rb)));
                a.ins(&format!("cmovne {}, {}", q(R8), q(ra)));
                a.ins(&format!("mov {}, {}", q(rd), q(R8)));
                if dc != DstCls::Acc {
                    a.ins("mov edx, edx"); // dst slot byte offset
                    a.ins(&format!("mov [rbp + rdx], {}", q(rd)));
                }
            }
            GlobalGet => {
                a.ins("mov rax, [rbx + 8]"); // index*8
                a.ins(&format!("mov rcx, [r15 + {S_GLOBALS}]"));
                let rd = self.dst_target(dc);
                a.ins(&format!("mov {}, [rcx + rax]", q(rd)));
                self.finish(a, dc, rd);
            }
            GlobalSet => {
                let ra = self.src(a, v.a, 8, RAX);
                a.ins("mov rdx, [rbx + 24]"); // index*8
                a.ins(&format!("mov rcx, [r15 + {S_GLOBALS}]"));
                a.ins(&format!("mov [rcx + rdx], {}", q(ra)));
            }
            I32_Eqz | I64_Eqz => {
                let w32 = v.op == I32_Eqz;
                let ra = self.src(a, v.a, 8, RAX);
                let rd = self.dst_target(dc);
                a.ins(&format!("test {0}, {0}", r(w32, ra)));
                a.ins(&format!("sete {}", b8(rd)));
                a.ins(&format!("movzx {}, {}", d(rd), b8(rd)));
                self.finish(a, dc, rd);
            }
            I32_DivS | I32_DivU | I32_RemS | I32_RemU | I64_DivS | I64_DivU | I64_RemS
            | I64_RemU => self.emit_div(a, st, v),
            I32_Popcnt | I64_Popcnt => self.emit_popcnt(a, v),
            I32_Clz | I64_Clz | I32_Ctz | I64_Ctz => self.emit_bitscan(a, v),
            I32_ReinterpretF32 | I64_ReinterpretF64 => {
                let w32 = v.op == I32_ReinterpretF32;
                let rd = self.dst_target(dc);
                if v.a.is_reg() {
                    let vs = self.src_fp(a, v.a, w32, 0, 8, RAX);
                    self.fp_to_gpr(a, w32, rd, vs);
                } else {
                    let ra = self.src(a, v.a, 8, RAX);
                    if ra != rd {
                        a.ins(&format!("mov {}, {}", q(rd), q(ra)));
                    }
                }
                self.finish(a, dc, rd);
            }
            F32_ReinterpretI32 | F64_ReinterpretI64 => {
                let w32 = v.op == F32_ReinterpretI32;
                let ra = self.src(a, v.a, 8, RAX);
                let vt = self.fp_target(dc);
                if w32 {
                    a.ins(&format!("movd {}, {}", xm(vt), d(ra)));
                } else {
                    a.ins(&format!("movq {}, {}", xm(vt), q(ra)));
                }
                self.finish_fp(a, dc, w32);
            }
            F32_Abs | F64_Abs | F32_Neg | F64_Neg => {
                // Sign-bit masks, built with pcmpeqd/shift so no constant
                // pool is needed.
                let w32 = matches!(v.op, F32_Abs | F32_Neg);
                let abs = matches!(v.op, F32_Abs | F64_Abs);
                let vs = self.src_fp(a, v.a, w32, 0, 8, RAX);
                let vt = self.fp_target(dc);
                a.ins("pcmpeqd xmm2, xmm2");
                match (abs, w32) {
                    (true, true) => a.ins("psrld xmm2, 1"),
                    (true, false) => a.ins("psrlq xmm2, 1"),
                    (false, true) => a.ins("pslld xmm2, 31"),
                    (false, false) => a.ins("psllq xmm2, 63"),
                }
                let mov = if w32 { "movaps" } else { "movapd" };
                let bits = if w32 { "ps" } else { "pd" };
                if vt != vs {
                    a.ins(&format!("{mov} {}, {}", xm(vt), xm(vs)));
                }
                a.ins(&format!(
                    "{} {}, xmm2",
                    if abs {
                        format!("and{bits}")
                    } else {
                        format!("xor{bits}")
                    },
                    xm(vt)
                ));
                self.finish_fp(a, dc, w32);
            }
            F32_Sqrt | F64_Sqrt => {
                let w32 = v.op == F32_Sqrt;
                let vs = self.src_fp(a, v.a, w32, 0, 8, RAX);
                let vt = self.fp_target(dc);
                a.ins(&format!(
                    "sqrt{} {}, {}",
                    if w32 { "ss" } else { "sd" },
                    xm(vt),
                    xm(vs)
                ));
                self.finish_fp(a, dc, w32);
            }
            F32_Min | F32_Max | F64_Min | F64_Max => {
                let w32 = matches!(v.op, F32_Min | F32_Max);
                let is_min = matches!(v.op, F32_Min | F64_Min);
                let (va, vb) = self.src_fp_ab(a, v.a, v.b, w32);
                self.fp_minmax(a, is_min, w32, self.fp_target(dc), va, vb);
                self.finish_fp(a, dc, w32);
            }
            F32_Copysign | F64_Copysign => {
                let w32 = v.op == F32_Copysign;
                let ra = if v.a.is_reg() {
                    let vs = self.src_fp(a, v.a, w32, 0, 8, RAX);
                    self.fp_to_gpr(a, w32, RAX, vs);
                    RAX
                } else {
                    self.src(a, v.a, 8, RAX)
                };
                let rb = if v.b.is_reg() {
                    let vs = self.src_fp(a, v.b, w32, 1, 16, RCX);
                    self.fp_to_gpr(a, w32, RCX, vs);
                    RCX
                } else {
                    self.src(a, v.b, 16, RCX)
                };
                let vt = self.fp_target(dc);
                if w32 {
                    a.ins(&format!("mov esi, {}", d(ra)));
                    a.ins("and esi, 0x7fffffff");
                    a.ins(&format!("mov r8d, {}", d(rb)));
                    a.ins("and r8d, 0x80000000");
                    a.ins("or esi, r8d");
                    a.ins(&format!("movd {}, esi", xm(vt)));
                } else {
                    a.ins(&format!("mov rsi, {}", q(ra)));
                    a.ins("shl rsi, 1");
                    a.ins("shr rsi, 1");
                    a.ins(&format!("mov r8, {}", q(rb)));
                    a.ins("shr r8, 63");
                    a.ins("shl r8, 63");
                    a.ins("or rsi, r8");
                    a.ins(&format!("movq {}, rsi", xm(vt)));
                }
                self.finish_fp(a, dc, w32);
            }
            F32_DemoteF64 => {
                let vs = self.src_fp(a, v.a, false, 0, 8, RAX);
                a.ins(&format!("cvtsd2ss {}, {}", xm(self.fp_target(dc)), xm(vs)));
                self.finish_fp(a, dc, true);
            }
            F64_PromoteF32 => {
                let vs = self.src_fp(a, v.a, true, 0, 8, RAX);
                a.ins(&format!("cvtss2sd {}, {}", xm(self.fp_target(dc)), xm(vs)));
                self.finish_fp(a, dc, false);
            }
            F32_ConvertI32S | F32_ConvertI32U | F32_ConvertI64S | F64_ConvertI32S
            | F64_ConvertI32U | F64_ConvertI64S => {
                let to32 = matches!(v.op, F32_ConvertI32S | F32_ConvertI32U | F32_ConvertI64S);
                // The unsigned 32-bit forms convert through a 64-bit
                // source: the slot is already zero-extended, so the value
                // is exactly right as a signed 64-bit integer.
                let src32 = matches!(v.op, F32_ConvertI32S | F64_ConvertI32S);
                let ra = self.src(a, v.a, 8, RAX);
                let vt = self.fp_target(dc);
                // cvtsi2s* leaves the destination's other lanes alone, so
                // clear it first to avoid a false dependency.
                a.ins(&format!("pxor {}, {}", xm(vt), xm(vt)));
                a.ins(&format!(
                    "cvtsi2{} {}, {}",
                    if to32 { "ss" } else { "sd" },
                    xm(vt),
                    r(src32, ra)
                ));
                self.finish_fp(a, dc, to32);
            }
            _ => {
                if let Some((m, w32, commutative, cl)) = int_bin(v.op) {
                    let (ra, rb) = self.src_ab(a, v.a, v.b);
                    let rd = self.dst_target(dc);
                    if cl {
                        self.shift(a, m, w32, rd, ra, rb);
                    } else {
                        self.alu2(a, m, w32, rd, ra, rb, commutative);
                    }
                    self.finish(a, dc, rd);
                } else if let Some((cond, w32)) = int_cmp(v.op) {
                    let (ra, rb) = self.src_ab(a, v.a, v.b);
                    a.ins(&format!("cmp {}, {}", r(w32, ra), r(w32, rb)));
                    let rd = self.dst_target(dc);
                    a.ins(&format!("set{cond} {}", b8(rd)));
                    a.ins(&format!("movzx {}, {}", d(rd), b8(rd)));
                    self.finish(a, dc, rd);
                } else if mem_kind(v.op).is_some() {
                    self.emit_mem(a, st, v);
                } else if let Some((m, w32)) = float_bin(v.op) {
                    let (va, vb) = self.src_fp_ab(a, v.a, v.b, w32);
                    self.fp_bin(a, m, w32, self.fp_target(dc), va, vb);
                    self.finish_fp(a, dc, w32);
                } else if let Some((cond, w32, swap, parity)) = float_cmp(v.op) {
                    let (va, vb) = self.src_fp_ab(a, v.a, v.b, w32);
                    let cmp = if w32 { "ucomiss" } else { "ucomisd" };
                    let (l, rr) = if swap { (vb, va) } else { (va, vb) };
                    a.ins(&format!("{cmp} {}, {}", xm(l), xm(rr)));
                    let rd = self.dst_target(dc);
                    a.ins(&format!("set{cond} {}", b8(RSI)));
                    if parity {
                        // eq must be false when unordered, ne must be true.
                        a.ins(&format!(
                            "set{} r8b",
                            if v.op == F32_Eq || v.op == F64_Eq {
                                "np"
                            } else {
                                "p"
                            }
                        ));
                        a.ins(&format!(
                            "{} sil, r8b",
                            if v.op == F32_Eq || v.op == F64_Eq {
                                "and"
                            } else {
                                "or"
                            }
                        ));
                    }
                    a.ins(&format!("movzx {}, sil", d(rd)));
                    self.finish(a, dc, rd);
                } else if let Some((src32, to64, lo, hi)) = cvt_f2i_trap(v.op) {
                    let vs = self.src_fp(a, v.a, src32, 0, 8, RAX);
                    let cmp = if src32 { "ucomiss" } else { "ucomisd" };
                    // Exclusive bounds on the un-truncated operand, so a
                    // bail is always a trap. The low test uses `jbe` on
                    // the swapped compare, which is also true when
                    // unordered, catching NaN.
                    a.ins(&format!("mov rsi, {lo}"));
                    if src32 {
                        a.ins("movd xmm2, esi");
                    } else {
                        a.ins("movq xmm2, rsi");
                    }
                    a.ins(&format!("{cmp} {}, xmm2", xm(vs)));
                    a.ins(&format!("jbe {}", st.slow));
                    a.ins(&format!("mov rsi, {hi}"));
                    if src32 {
                        a.ins("movd xmm2, esi");
                    } else {
                        a.ins("movq xmm2, rsi");
                    }
                    a.ins(&format!("{cmp} {}, xmm2", xm(vs)));
                    a.ins(&format!("jae {}", st.slow));
                    let rd = self.dst_target(dc);
                    // Always convert to a 64-bit GPR: for the unsigned
                    // 32-bit forms the in-range value is below 2^32 and
                    // the low half is the answer, zero-extended.
                    a.ins(&format!(
                        "cvtt{}2si {}, {}",
                        if src32 { "ss" } else { "sd" },
                        q(rd),
                        xm(vs)
                    ));
                    if !to64 {
                        a.ins(&format!("mov {0}, {0}", d(rd)));
                    }
                    self.finish(a, dc, rd);
                } else {
                    self.emit_int_un(a, v);
                }
            }
        }
        self.tail(a, v.counted);
    }
}

impl X86_64 {
    fn emit_int_un(&mut self, a: &mut Asm, v: &Variant) {
        use Op::*;
        let ra = self.src(a, v.a, 8, RAX);
        let rd = self.dst_target(v.d);
        match v.op {
            I32_Extend8S => a.ins(&format!("movsx {}, {}", d(rd), b8(ra))),
            I32_Extend16S => a.ins(&format!("movsx {}, {}", d(rd), h16(ra))),
            I64_Extend8S => a.ins(&format!("movsx {}, {}", q(rd), b8(ra))),
            I64_Extend16S => a.ins(&format!("movsx {}, {}", q(rd), h16(ra))),
            I64_Extend32S | I64_ExtendI32S => a.ins(&format!("movsxd {}, {}", q(rd), d(ra))),
            I32_WrapI64 | I64_ExtendI32U => a.ins(&format!("mov {}, {}", d(rd), d(ra))),
            other => panic!("x86_64: no handler shape for {other:?}"),
        }
        self.finish(a, v.d, rd);
    }

    /// `bsr`/`bsf` leave the destination undefined for a zero input and
    /// set ZF, so the wasm answer (the operand width) comes from a `cmov`
    /// on that flag. `mov` does not disturb flags, which is what lets the
    /// sentinel load sit between the scan and the `cmov`.
    fn emit_bitscan(&mut self, a: &mut Asm, v: &Variant) {
        use Op::*;
        let w32 = matches!(v.op, I32_Clz | I32_Ctz);
        let ctz = matches!(v.op, I32_Ctz | I64_Ctz);
        let width = if w32 { 32 } else { 64 };
        let ra = self.src(a, v.a, 8, RAX);
        let rd = self.dst_target(v.d);
        if ctz {
            a.ins(&format!("mov esi, {width}"));
            a.ins(&format!("bsf {}, {}", r(w32, R8), r(w32, ra)));
            a.ins(&format!("cmovz {}, {}", r(w32, R8), r(w32, RSI)));
            a.ins(&format!("mov {}, {}", r(w32, rd), r(w32, R8)));
        } else {
            // The sentinel must be all-ones at the OPERAND width: a
            // 32-bit `mov esi, -1` zero-extends, and `63 - 0xffffffff`
            // is not 64.
            a.ins(&format!("mov {}, -1", r(w32, RSI)));
            a.ins(&format!("bsr {}, {}", r(w32, R8), r(w32, ra)));
            a.ins(&format!("cmovz {}, {}", r(w32, R8), r(w32, RSI)));
            a.ins(&format!("mov esi, {}", width - 1));
            a.ins(&format!("sub {}, {}", r(w32, RSI), r(w32, R8)));
            a.ins(&format!("mov {}, {}", r(w32, rd), r(w32, RSI)));
        }
        self.finish(a, v.d, rd);
    }

    /// The SWAR population count. `popcnt` itself is an SSE4.2 opcode, and
    /// the baseline this backend targets does not have it.
    fn emit_popcnt(&mut self, a: &mut Asm, v: &Variant) {
        let w32 = v.op == Op::I32_Popcnt;
        let ra = self.src(a, v.a, 8, RAX);
        let rd = self.dst_target(v.d);
        if w32 {
            a.ins(&format!("mov esi, {}", d(ra)));
            a.ins("mov r8d, esi");
            a.ins("shr r8d, 1");
            a.ins("and r8d, 0x55555555");
            a.ins("sub esi, r8d");
            a.ins("mov r8d, esi");
            a.ins("and esi, 0x33333333");
            a.ins("shr r8d, 2");
            a.ins("and r8d, 0x33333333");
            a.ins("add esi, r8d");
            a.ins("mov r8d, esi");
            a.ins("shr r8d, 4");
            a.ins("add esi, r8d");
            a.ins("and esi, 0x0f0f0f0f");
            a.ins("imul esi, esi, 0x01010101");
            a.ins("shr esi, 24");
            a.ins(&format!("mov {}, esi", d(rd)));
        } else {
            a.ins(&format!("mov rsi, {}", q(ra)));
            a.ins("mov r8, rsi");
            a.ins("shr r8, 1");
            a.ins("mov rdx, 0x5555555555555555");
            a.ins("and r8, rdx");
            a.ins("sub rsi, r8");
            a.ins("mov rdx, 0x3333333333333333");
            a.ins("mov r8, rsi");
            a.ins("and rsi, rdx");
            a.ins("shr r8, 2");
            a.ins("and r8, rdx");
            a.ins("add rsi, r8");
            a.ins("mov r8, rsi");
            a.ins("shr r8, 4");
            a.ins("add rsi, r8");
            a.ins("mov rdx, 0x0f0f0f0f0f0f0f0f");
            a.ins("and rsi, rdx");
            a.ins("mov rdx, 0x0101010101010101");
            a.ins("imul rsi, rdx");
            a.ins("shr rsi, 56");
            a.ins(&format!("mov {}, rsi", q(rd)));
        }
        self.finish(a, v.d, rd);
    }

    /// `idiv` traps in hardware on both wasm-defined edges, so both are
    /// detected first and bail to the slow path, where the shared executor
    /// raises the proper wasm trap. `rem_s` still needs the MIN/-1 edge
    /// here (unlike arm64's `sdiv`+`msub`, which wraps to the right
    /// answer) because the hardware would fault.
    fn emit_div(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        use Op::*;
        let (w32, signed, rem) = match v.op {
            I32_DivS => (true, true, false),
            I32_DivU => (true, false, false),
            I32_RemS => (true, true, true),
            I32_RemU => (true, false, true),
            I64_DivS => (false, true, false),
            I64_DivU => (false, false, false),
            I64_RemS => (false, true, true),
            I64_RemU => (false, false, true),
            _ => unreachable!(),
        };
        let (ra, rb) = self.src_ab(a, v.a, v.b);
        // `idiv`'s divisor may not be rax or rdx; `src_ab` never places an
        // operand there, but a register class could still collide with the
        // dividend move, so stage the divisor in rsi.
        a.ins(&format!("mov {}, {}", q(RSI), q(rb)));
        a.ins(&format!("test {0}, {0}", r(w32, RSI)));
        a.ins(&format!("jz {}", st.slow));
        if signed {
            let go = a.fresh("div");
            a.ins(&format!("cmp {}, -1", r(w32, RSI)));
            a.ins(&format!("jne {go}"));
            if w32 {
                a.ins("mov r8d, 0x80000000");
            } else {
                a.ins("mov r8, 0x8000000000000000");
            }
            a.ins(&format!("cmp {}, {}", r(w32, ra), r(w32, R8)));
            a.ins(&format!("je {}", st.slow));
            a.label(&go);
        }
        a.ins(&format!("mov {}, {}", q(RAX), q(ra)));
        if signed {
            a.ins(if w32 { "cdq" } else { "cqo" });
            a.ins(&format!("idiv {}", r(w32, RSI)));
        } else {
            a.ins("xor edx, edx");
            a.ins(&format!("div {}", r(w32, RSI)));
        }
        let rd = self.dst_target(v.d);
        let res = if rem { RDX } else { RAX };
        // A 32-bit divide writes eax/edx, which zero-extends into the full
        // register, so the move above already upholds the slot convention.
        if rd != res {
            a.ins(&format!("mov {}, {}", q(rd), q(res)));
        }
        self.finish(a, v.d, rd);
    }

    fn emit_mem(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let (size, load, kind) = mem_kind(v.op).unwrap();
        let fp = is_fp_mem(v.op);
        let dc = v.d;
        // The address is an i32. Reading it as a 32-bit value is what
        // makes the bounds check independent of the slot zero-extension
        // convention rather than dependent on it.
        let addr = match v.a {
            Cls::Slot => {
                a.ins("mov rax, [rbx + 8]");
                a.ins("mov eax, [rbp + rax]");
                RAX
            }
            Cls::Const => {
                a.ins("mov eax, [rbx + 8]");
                RAX
            }
            Cls::Acc | Cls::L0 | Cls::L1 => {
                let s = match v.a {
                    Cls::Acc => ACC,
                    Cls::L0 => L0R,
                    _ => L1R,
                };
                a.ins(&format!("mov eax, {}", d(s)));
                RAX
            }
        };
        if v.fused {
            // ea = zext32(addr1 + addr2) + static offset — the
            // corpus-universal base+index pattern folded into one
            // dispatch. addr2 always comes from its slot, packed in c's
            // high half.
            a.ins("mov rsi, [rbx + 24]");
            a.ins("mov r8, rsi");
            a.ins("shr r8, 32");
            a.ins("mov r8d, [rbp + r8]"); // addr2
            a.ins(&format!("add {}, r8d", d(addr))); // wrapping i32 sum
            if load {
                a.ins("mov esi, esi"); // dst byte offset
                a.ins("mov rcx, [rbx + 16]"); // static offset
            } else {
                a.ins("mov esi, esi"); // static offset
                a.ins("mov rcx, rsi");
            }
            a.ins(&format!("add rcx, {}", q(addr))); // ea
        } else {
            a.ins(&format!("mov rcx, [rbx + {}]", if load { 16 } else { 24 }));
            a.ins(&format!("add rcx, {}", q(addr))); // ea = offset + zext(addr)
        }
        a.ins(&format!("lea rdx, [rcx + {size}]"));
        a.ins("cmp rdx, r13");
        a.ins(&format!("ja {}", st.trap_oob));
        if load {
            if fp {
                let vt = self.fp_target(dc);
                a.ins(&format!(
                    "{} {}, [r14 + rcx]",
                    if kind == 0 { "movss" } else { "movsd" },
                    xm(vt)
                ));
                if dc != DstCls::Acc {
                    // Both widths are clean here: `movss` zeroes the upper
                    // half, so an 8-byte store upholds the convention.
                    if v.fused {
                        a.ins(&format!("movq rax, {}", xm(vt)));
                        a.ins("mov [rbp + rsi], rax");
                    } else {
                        a.ins("mov rdx, [rbx + 24]");
                        a.ins(&format!("movq rax, {}", xm(vt)));
                        a.ins("mov [rbp + rdx], rax");
                    }
                }
            } else {
                let rd = self.dst_target(dc);
                let m = match kind {
                    0 => format!("mov {}, [r14 + rcx]", d(rd)),
                    1 => format!("mov {}, [r14 + rcx]", q(rd)),
                    2 => format!("movzx {}, byte ptr [r14 + rcx]", d(rd)),
                    3 => format!("movzx {}, word ptr [r14 + rcx]", d(rd)),
                    4 => format!("movsx {}, byte ptr [r14 + rcx]", d(rd)),
                    5 => format!("movsx {}, word ptr [r14 + rcx]", d(rd)),
                    6 => format!("movsx {}, byte ptr [r14 + rcx]", q(rd)),
                    7 => format!("movsx {}, word ptr [r14 + rcx]", q(rd)),
                    8 => format!("movsxd {}, dword ptr [r14 + rcx]", q(rd)),
                    _ => unreachable!(),
                };
                a.ins(&m);
                if v.fused {
                    if dc != DstCls::Acc {
                        a.ins(&format!("mov [rbp + rsi], {}", q(rd)));
                    }
                } else {
                    self.finish(a, dc, rd);
                }
            }
        } else if fp && v.b.is_reg() {
            let vs = match v.b {
                Cls::L0 => FL0R,
                Cls::L1 => FL1R,
                _ => FACC,
            };
            a.ins(&format!(
                "{} [r14 + rcx], {}",
                if kind == 0 { "movss" } else { "movsd" },
                xm(vs)
            ));
        } else {
            let rb = self.src(a, v.b, 16, RSI);
            let m = match kind {
                0 => format!("mov [r14 + rcx], {}", d(rb)),
                1 => format!("mov [r14 + rcx], {}", q(rb)),
                2 => format!("mov [r14 + rcx], {}", b8(rb)),
                3 => format!("mov [r14 + rcx], {}", h16(rb)),
                _ => unreachable!(),
            };
            a.ins(&m);
        }
    }

    fn emit_return(&mut self, a: &mut Asm, v: &Variant) {
        self.bump(a, v.counted);
        a.ins("mov rax, [rbx + 8]"); // first-result slot*8
        a.ins("mov rcx, [rbx + 16]"); // result count
        a.ins("add rax, rbp");
        a.ins("mov rdx, rbp");
        let pop = a.fresh("pop");
        let cl = a.fresh("cl");
        a.ins("test rcx, rcx");
        a.ins(&format!("jz {pop}"));
        // The accumulator doubles as the copy scratch: after a
        // single-result copy it holds result 0, which is the
        // call-result-in-acc convention at zero extra instructions.
        a.label(&cl);
        a.ins("mov r12, [rax]");
        a.ins("mov [rdx], r12");
        a.ins("add rax, 8");
        a.ins("add rdx, 8");
        a.ins("dec rcx");
        a.ins(&format!("jnz {cl}"));
        a.label(&pop);
        a.ins(&format!("mov rsi, [r15 + {S_RET_CURSOR}]"));
        a.ins("sub rsi, 32");
        a.ins(&format!("mov [r15 + {S_RET_CURSOR}], rsi"));
        a.ins("mov rax, [rsi]"); // ret pc
        a.ins("mov rbp, [rsi + 8]"); // caller frame
        a.ins("mov rdx, [rsi + 16]"); // caller code | caller_l0off<<48
        a.ins("mov rcx, [rsi + 24]"); // caller l1off
        a.ins("mov r9, rdx");
        a.ins("shl r9, 16");
        a.ins("shr r9, 16"); // caller cell base
        a.ins("shr rdx, 48"); // caller l0off, bit 0 = float-pinned flag
                              // Sentinel records carry a readable dummy frame, so these loads
                              // are always safe.
        let fp = a.fresh("retfp");
        let join = a.fresh("retjoin");
        a.ins("test dl, 1");
        a.ins(&format!("jnz {fp}"));
        a.ins("mov r11, [rbp + rdx]");
        a.ins("mov r10, [rbp + rcx]");
        a.label(&join);
        a.ins("mov rbx, rax");
        a.ins("mov rax, [rbx]");
        a.ins("jmp rax");
        a.label(&fp);
        a.ins("dec rdx");
        a.ins("mov r11, [rbp + rdx]");
        a.ins("mov r10, [rbp + rcx]");
        a.ins("movq xmm4, r11");
        a.ins("movq xmm5, r10");
        a.ins(&format!("jmp {join}"));
    }

    /// `memory.fill` / `memory.copy` on memory 0, in 64-byte SSE2 blocks
    /// with word and byte tails. Block width matters more than anything
    /// else here: a word-at-a-time loop measured 28% BELOW wasm3 and wasmi
    /// on STREAM Copy, the one benchmark where this engine lost to another
    /// interpreter. xmm0-xmm2 are scratch; the pinned float registers are
    /// xmm3-xmm5.
    fn emit_bulk(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let fill = v.op == Op::MemoryFill;
        self.pre(a);
        a.ins("mov rax, [rbx + 8]");
        a.ins("add rax, rbp");
        a.ins("mov rcx, [rax]"); // d
        a.ins("mov rdx, [rax + 8]"); // fill: value, copy: s
        a.ins("mov rsi, [rax + 16]"); // n
        a.ins("lea r8, [rcx + rsi]");
        a.ins("cmp r8, r13");
        a.ins(&format!("ja {}", st.slow)); // dst out of bounds
        let l64 = a.fresh("l64");
        let l8 = a.fresh("l8");
        let bytes = a.fresh("bytes");
        let done = a.fresh("done");
        if fill {
            a.ins("movzx edx, dl"); // splat the fill byte
            a.ins("mov rax, 0x0101010101010101");
            a.ins("imul rdx, rax");
            a.ins("add rcx, r14"); // cursor
            a.ins("add r8, r14"); // end
            a.ins("movq xmm0, rdx");
            a.ins("punpcklqdq xmm0, xmm0");
            a.label(&l64);
            a.ins("lea rax, [rcx + 64]");
            a.ins("cmp rax, r8");
            a.ins(&format!("ja {l8}"));
            a.ins("movdqu [rcx], xmm0");
            a.ins("movdqu [rcx + 16], xmm0");
            a.ins("movdqu [rcx + 32], xmm0");
            a.ins("movdqu [rcx + 48], xmm0");
            a.ins("mov rcx, rax");
            a.ins(&format!("jmp {l64}"));
            a.label(&l8);
            a.ins("lea rax, [rcx + 8]");
            a.ins("cmp rax, r8");
            a.ins(&format!("ja {bytes}"));
            a.ins("mov [rcx], rdx");
            a.ins("mov rcx, rax");
            a.ins(&format!("jmp {l8}"));
            a.label(&bytes);
            a.ins("cmp rcx, r8");
            a.ins(&format!("jae {done}"));
            a.ins("mov [rcx], dl");
            a.ins("inc rcx");
            a.ins(&format!("jmp {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
        } else {
            a.ins("lea r8, [rdx + rsi]");
            a.ins("cmp r8, r13");
            a.ins(&format!("ja {}", st.slow)); // src out of bounds
            let back = a.fresh("copyback");
            let fwd = a.fresh("copyfwd");
            // A forward copy is wrong only when s < d < s+n.
            a.ins("cmp rcx, rdx");
            a.ins(&format!("jbe {fwd}"));
            a.ins("cmp rcx, r8");
            a.ins(&format!("jb {back}"));
            a.label(&fwd);
            a.ins("add rcx, r14"); // dst cursor
            a.ins("add rdx, r14"); // src cursor
            a.ins("lea r8, [rcx + rsi]"); // dst end
            a.label(&l64);
            a.ins("lea rax, [rcx + 64]");
            a.ins("cmp rax, r8");
            a.ins(&format!("ja {l8}"));
            a.ins("movdqu xmm0, [rdx]");
            a.ins("movdqu xmm1, [rdx + 16]");
            a.ins("movdqu [rcx], xmm0");
            a.ins("movdqu [rcx + 16], xmm1");
            a.ins("movdqu xmm0, [rdx + 32]");
            a.ins("movdqu xmm1, [rdx + 48]");
            a.ins("movdqu [rcx + 32], xmm0");
            a.ins("movdqu [rcx + 48], xmm1");
            a.ins("add rdx, 64");
            a.ins("mov rcx, rax");
            a.ins(&format!("jmp {l64}"));
            a.label(&l8);
            a.ins("lea rax, [rcx + 8]");
            a.ins("cmp rax, r8");
            a.ins(&format!("ja {bytes}"));
            a.ins("mov rsi, [rdx]");
            a.ins("mov [rcx], rsi");
            a.ins("add rdx, 8");
            a.ins("mov rcx, rax");
            a.ins(&format!("jmp {l8}"));
            a.label(&bytes);
            a.ins("cmp rcx, r8");
            a.ins(&format!("jae {done}"));
            a.ins("mov sil, [rdx]");
            a.ins("mov [rcx], sil");
            a.ins("inc rcx");
            a.ins("inc rdx");
            a.ins(&format!("jmp {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
            // Overlapping-downward block, out of line past the tail:
            // rcx = d, rdx = s, rsi = n, all raw offsets.
            let b8l = a.fresh("b8");
            let bb = a.fresh("bb");
            let bdone = a.fresh("bdone");
            a.label(&back);
            a.ins("add rcx, r14");
            a.ins("add rdx, r14");
            a.label(&b8l);
            a.ins("cmp rsi, 8");
            a.ins(&format!("jb {bb}"));
            a.ins("sub rsi, 8");
            a.ins("mov rax, [rdx + rsi]");
            a.ins("mov [rcx + rsi], rax");
            a.ins(&format!("jmp {b8l}"));
            a.label(&bb);
            a.ins("test rsi, rsi");
            a.ins(&format!("jz {bdone}"));
            a.ins("dec rsi");
            a.ins("mov al, [rdx + rsi]");
            a.ins("mov [rcx + rsi], al");
            a.ins(&format!("jmp {bb}"));
            a.label(&bdone);
            self.tail(a, v.counted);
        }
    }
}
