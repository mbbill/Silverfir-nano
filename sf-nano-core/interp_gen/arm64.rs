//! ARM64 handler backend.
//!
//! Register contract (chain-invariant across every handler):
//! ```text
//!   x19 = pc (current 32-byte dispatch cell)   x20 = frame base
//!   x21 = &EnterState                          x22 = memory 0 base
//!   x23 = memory 0 length                      x24 = cell array base
//!   x25 = globals base                         x26 = return-stack cursor
//!   x27 = return-stack limit                   x28 = value-stack limit
//!   x8  = accumulator     x17 = l0     x14 = l1     x15 = dispatch counter
//!   x7  = prefetched next handler word
//!   x6, x9-x13, x16 = scratch;  v16 = float acc, v17/v18 = float l0/l1
//!   v0, v1 = float scratch;  sp belongs to the entry trampoline's frame
//! ```
//!
//! Cell layout is `[ handler | a | b | c ]` at 0/8/16/24. At link time slot
//! operands are pre-scaled to byte offsets and branch targets to absolute
//! cell addresses.

use super::asm::Asm;
use super::instr::Op;
use super::isa::{
    Caps, Isa, PairDstSplit, Stubs, Variant, CLASSES, DSTS, SUPER_ADD4, SUPER_ADD_BRNE,
    SUPER_AND_BREQ, SUPER_HANDLER_SLOTS, SUPER_MOVPAIR_LOAD_STORE_BRIF, SUPER_SHRU_AND,
};
use super::layout::{Cls, DstCls, Fam};

const ACC: u32 = 8;
const L0R: u32 = 17;
const L1R: u32 = 14;
const FACC: u32 = 16;
const FL0R: u32 = 17;
const FL1R: u32 = 18;

fn r(w32: bool, n: u32) -> String {
    if w32 {
        format!("w{n}")
    } else {
        format!("x{n}")
    }
}
fn x(n: u32) -> String {
    format!("x{n}")
}
fn w(n: u32) -> String {
    format!("w{n}")
}
fn f(w32: bool, n: u32) -> String {
    if w32 {
        format!("s{n}")
    } else {
        format!("d{n}")
    }
}

/// Inverted condition, for the fall-through side of a fused branch.
fn inv(c: &str) -> &'static str {
    match c {
        "eq" => "ne",
        "ne" => "eq",
        "hs" => "lo",
        "lo" => "hs",
        "hi" => "ls",
        "ls" => "hi",
        "ge" => "lt",
        "lt" => "ge",
        "gt" => "le",
        "le" => "gt",
        "mi" => "pl",
        "pl" => "mi",
        other => panic!("no inverse for {other}"),
    }
}

pub struct Arm64 {
    /// Apple ARM64 gets the measured wide-core tuning. Other ARM64 targets
    /// retain the portable handler shapes and linker behavior.
    pub apple_tuning: bool,
    /// Multi-cell handlers are omitted from exact dispatch-counting builds.
    pub superhandlers: bool,
}

impl Arm64 {
    /// Prefetch the NEXT cell's handler word at handler entry: the load
    /// issues alongside the handler body, so the dispatch branch's operand
    /// is ready (and a misprediction resolves) sooner. Every handler that
    /// ends in [`Self::tail`] must open with this.
    fn pre(&self, a: &mut Asm) {
        a.ins("ldr x7, [x19, #32]");
    }

    fn bump(&self, a: &mut Asm, on: bool) {
        if on {
            a.ins("add x15, x15, #1");
        }
    }

    /// `add pc, #32; br prefetched`. A pre-index `ldr x7, [x19, #32]!`
    /// form measured no better on Apple silicon: the writeback uop
    /// serializes against the load while the separate add schedules
    /// freely.
    fn tail(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins("add x19, x19, #32");
        a.ins("br x7");
    }

    /// Materialize operand a into a register: the register classes ARE
    /// registers (no code); consts load inline from the cell; slots load
    /// through the frame.
    fn src_a(&self, a: &mut Asm, cls: Cls, tmp: u32) -> u32 {
        match cls {
            Cls::Acc => ACC,
            Cls::L0 => L0R,
            Cls::L1 => L1R,
            Cls::Const => {
                a.ins(&format!("ldr {}, [x19, #8]", x(tmp)));
                tmp
            }
            Cls::Slot => {
                a.ins(&format!("ldr {}, [x19, #8]", x(tmp)));
                a.ins(&format!("ldr {0}, [x20, {0}]", x(tmp)));
                tmp
            }
        }
    }

    fn src_b(&self, a: &mut Asm, cls: Cls, tmp: u32) -> u32 {
        match cls {
            Cls::Acc => ACC,
            Cls::L0 => L0R,
            Cls::L1 => L1R,
            Cls::Const => {
                a.ins(&format!("ldr {}, [x19, #16]", x(tmp)));
                tmp
            }
            Cls::Slot => {
                a.ins(&format!("ldr {}, [x19, #16]", x(tmp)));
                a.ins(&format!("ldr {0}, [x20, {0}]", x(tmp)));
                tmp
            }
        }
    }

    /// Fetch both operands. When both live in slots the two cell fields
    /// load with a single `ldp` — one instruction saved on the hottest
    /// binop variant. Pairing the cell PAYLOAD is safe; pairing two FRAME
    /// slot reads would not be, because store-to-load forwarding does not
    /// work through pair instructions on Apple silicon and frame-slot
    /// forwarding is exactly what the slot-edge cost model depends on.
    fn src_ab(&self, a: &mut Asm, ac: Cls, bc: Cls) -> (u32, u32) {
        if ac == Cls::Slot && bc == Cls::Slot {
            a.ins("ldp x10, x11, [x19, #8]");
            a.ins("ldr x10, [x20, x10]");
            a.ins("ldr x11, [x20, x11]");
            (10, 11)
        } else {
            (self.src_a(a, ac, 10), self.src_b(a, bc, 11))
        }
    }

    fn dst_target(&self, d: DstCls) -> u32 {
        match d {
            DstCls::L0 => L0R,
            DstCls::L1 => L1R,
            _ => ACC,
        }
    }

    fn store_dst(&self, a: &mut Asm, src: u32) {
        a.ins("ldr x12, [x19, #24]");
        a.ins(&format!("str {}, [x20, x12]", x(src)));
    }

    /// Mem and pinned destinations store to the slot (write-through keeps
    /// the slot authoritative for the slow path and the per-entry reload);
    /// Acc skips the store.
    fn finish(&self, a: &mut Asm, d: DstCls, src: u32) {
        if d != DstCls::Acc {
            self.store_dst(a, src);
        }
    }

    fn fp_target(&self, d: DstCls) -> u32 {
        match d {
            DstCls::L0 => FL0R,
            DstCls::L1 => FL1R,
            _ => FACC,
        }
    }

    /// Land a float result. Non-acc destinations store D-width even for
    /// f32 — S-form producers zero the upper bits, so the slot
    /// zero-extension convention holds for free.
    fn finish_fp(&self, a: &mut Asm, d: DstCls) {
        if d != DstCls::Acc {
            a.ins("ldr x12, [x19, #24]");
            a.ins(&format!("str {}, [x20, x12]", f(false, self.fp_target(d))));
        }
    }

    /// Load a floating operand into an FP register. Register classes move
    /// raw bits over; f32 values are zero-extended in slots and registers,
    /// so an S-view read is always exact.
    fn src_fp(&self, a: &mut Asm, cls: Cls, w32: bool, v: u32, pcoff: u32, tmp: u32) -> u32 {
        match cls {
            Cls::Slot => {
                a.ins(&format!("ldr {}, [x19, #{pcoff}]", x(tmp)));
                a.ins(&format!("ldr {}, [x20, {}]", f(w32, v), x(tmp)));
                v
            }
            Cls::Const => {
                a.ins(&format!("ldr {}, [x19, #{pcoff}]", f(w32, v)));
                v
            }
            Cls::Acc => FACC,
            // A float-domain access classes L0/L1 only inside a
            // float-pinned function, where the value is resident in the
            // NEON file — no cross-domain move.
            Cls::L0 => FL0R,
            Cls::L1 => FL1R,
        }
    }

    fn src_fp_ab(&self, a: &mut Asm, ac: Cls, bc: Cls, w32: bool) -> (u32, u32) {
        if ac == Cls::Slot && bc == Cls::Slot {
            a.ins("ldp x10, x11, [x19, #8]");
            a.ins(&format!("ldr {}, [x20, x10]", f(w32, 0)));
            a.ins(&format!("ldr {}, [x20, x11]", f(w32, 1)));
            (0, 1)
        } else {
            (
                self.src_fp(a, ac, w32, 0, 8, 10),
                self.src_fp(a, bc, w32, 1, 16, 11),
            )
        }
    }

    /// Move an FP register's bits into an integer register (the S-form
    /// zero-extends, upholding the f32 slot convention).
    fn fp_result(&self, a: &mut Asm, w32: bool, rd: u32, v: u32) {
        a.ins(&format!("fmov {}, {}", r(w32, rd), f(w32, v)));
    }

    fn mov_imm64(&self, a: &mut Asm, rd: u32, val: u64) {
        let mut set = false;
        for hw in 0..4 {
            let part = (val >> (hw * 16)) & 0xFFFF;
            if part == 0 {
                continue;
            }
            let sh = hw * 16;
            if set {
                a.ins(&format!("movk {}, #{part}, lsl #{sh}", x(rd)));
            } else {
                a.ins(&format!("movz {}, #{part}, lsl #{sh}", x(rd)));
                set = true;
            }
        }
        if !set {
            a.ins(&format!("movz {}, #0", x(rd)));
        }
    }

    /// `MovPair -> i32.load -> i32.store -> br_if` list walk. The linker
    /// admits only the shape where the load and store use the same address
    /// and width, so one bounds check preserves the original trap order.
    fn emit_super_list_step(&self, a: &mut Asm, st: &Stubs, counted: bool) {
        self.bump(a, counted);
        a.ins("mov x6, x17"); // ordered copy source 1: old L0
        a.ins("ldr x10, [x19, #16]"); // ordered copy source 2 slot
        a.ins("ldr x11, [x20, x10]");
        a.ins("ldr x13, [x19, #24]"); // dst1 << 32 | dst2
        a.ins("lsr x10, x13, #32");
        a.ins("str x6, [x20, x10]");
        a.ins("ubfx x10, x13, #0, #32");
        a.ins("str x11, [x20, x10]");
        a.ins("mov x17, x11");

        // Publish the load cell before its bounds check. A trap must resume
        // at the load, after the ordered copies that already committed.
        a.ins("add x19, x19, #32");
        a.ins("ldr x12, [x19, #88]"); // final branch target
        a.ins("ldr x9, [x12]");
        a.ins("mov w10, w17"); // zero-extended memory address
        a.ins("add x11, x10, #4");
        a.ins("cmp x11, x23");
        a.ins(&format!("b.hi {}", st.trap_oob));
        a.ins("ldr w8, [x22, x10]");
        a.ins("ldr x11, [x19, #24]"); // load destination slot
        a.ins("str x8, [x20, x11]");
        a.ins("str w6, [x22, x10]"); // store old L0 to same address

        let fall = a.fresh("super_list_fall");
        a.ins(&format!("cbz w8, {fall}"));
        a.ins("mov x19, x12");
        a.ins("br x9");
        a.label(&fall);
        a.ins("ldr x7, [x19, #96]");
        a.ins("add x19, x19, #96");
        a.ins("br x7");
    }

    /// Four exact accumulator/pinned/slot i32 adds used by the matrix inner
    /// loop. Payload offsets remain dynamic; only the proven residency and
    /// dependency shape is specialized.
    fn emit_super_add4(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins("add w8, w8, w14"); // Acc + L1 -> Acc

        a.ins("ldr x10, [x19, #40]"); // cell 1 source A
        a.ins("ldr x10, [x20, x10]");
        a.ins("add w14, w10, w8"); // Slot + Acc -> L1
        a.ins("ldr x11, [x19, #56]");
        a.ins("str x14, [x20, x11]");

        a.ins("ldr x10, [x19, #80]"); // cell 2 source B
        a.ins("ldr x10, [x20, x10]");
        a.ins("add w17, w17, w10"); // L0 + Slot -> L0
        a.ins("ldr x11, [x19, #72]");
        a.ins("str x17, [x20, x11]");

        a.ins("ldr x10, [x19, #104]"); // cell 3 in-place slot
        a.ins("ldr w8, [x20, x10]");
        a.ins("add w8, w8, #4");
        a.ins("str x8, [x20, x10]");
        a.ins("ldr x7, [x19, #128]");
        a.ins("add x19, x19, #128");
        a.ins("br x7");
    }

    /// In-place constant add followed by `i32.ne` branch consuming the new
    /// accumulator value.
    fn emit_super_add_brne(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins("ldr x10, [x19, #8]");
        a.ins("ldr w8, [x20, x10]");
        a.ins("ldr w11, [x19, #16]");
        a.ins("add w8, w8, w11");
        a.ins("str x8, [x20, x10]");
        a.ins("ldr x12, [x19, #56]");
        a.ins("ldr x9, [x12]");
        a.ins("ldr x11, [x19, #40]");
        a.ins("ldr w11, [x20, x11]");
        a.ins("cmp w11, w8");
        let fall = a.fresh("super_addne_fall");
        a.ins(&format!("b.eq {fall}"));
        a.ins("mov x19, x12");
        a.ins("br x9");
        a.label(&fall);
        a.ins("ldr x7, [x19, #64]");
        a.ins("add x19, x19, #64");
        a.ins("br x7");
    }

    /// Accumulator shift-right followed by a constant mask.
    fn emit_super_shru_and(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins("ldr w11, [x19, #16]");
        a.ins("lsr w8, w8, w11");
        a.ins("ldr w11, [x19, #48]");
        a.ins("and w8, w8, w11");
        a.ins("ldr x10, [x19, #56]");
        a.ins("str x8, [x20, x10]");
        a.ins("ldr x7, [x19, #64]");
        a.ins("add x19, x19, #64");
        a.ins("br x7");
    }

    /// `i32.and` whose accumulator result feeds an equality branch.
    fn emit_super_and_breq(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins("ldr x10, [x19, #8]");
        a.ins("ldr w10, [x20, x10]");
        a.ins("ldr w11, [x19, #16]");
        a.ins("and w8, w10, w11");
        a.ins("ldr x10, [x19, #24]");
        a.ins("str x8, [x20, x10]");
        a.ins("ldr x12, [x19, #56]");
        a.ins("ldr x9, [x12]");
        a.ins("ldr w11, [x19, #48]");
        a.ins("cmp w8, w11");
        let fall = a.fresh("super_andeq_fall");
        a.ins(&format!("b.ne {fall}"));
        a.ins("mov x19, x12");
        a.ins("br x9");
        a.label(&fall);
        a.ins("ldr x7, [x19, #64]");
        a.ins("add x19, x19, #64");
        a.ins("br x7");
    }
}

/// `(mnemonic, 32-bit form, needs the rotate-left fixup)`.
fn int_bin(op: Op) -> Option<(&'static str, bool, bool)> {
    use Op::*;
    Some(match op {
        I32_Add => ("add", true, false),
        I32_Sub => ("sub", true, false),
        I32_And => ("and", true, false),
        I32_Or => ("orr", true, false),
        I32_Xor => ("eor", true, false),
        I32_Mul => ("mul", true, false),
        I32_Shl => ("lsl", true, false),
        I32_ShrU => ("lsr", true, false),
        I32_ShrS => ("asr", true, false),
        I32_Rotr => ("ror", true, false),
        I32_Rotl => ("ror", true, true),
        I64_Add => ("add", false, false),
        I64_Sub => ("sub", false, false),
        I64_And => ("and", false, false),
        I64_Or => ("orr", false, false),
        I64_Xor => ("eor", false, false),
        I64_Mul => ("mul", false, false),
        I64_Shl => ("lsl", false, false),
        I64_ShrU => ("lsr", false, false),
        I64_ShrS => ("asr", false, false),
        I64_Rotr => ("ror", false, false),
        I64_Rotl => ("ror", false, true),
        _ => return None,
    })
}

/// `(condition, 32-bit form)` for integer compares and their fused
/// compare-branch twins.
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
        I64_LtS | I64_BrLtS => ("lt", false),
        I64_LtU | I64_BrLtU => ("lo", false),
        I64_GtS | I64_BrGtS => ("gt", false),
        I64_GtU | I64_BrGtU => ("hi", false),
        I64_LeS | I64_BrLeS => ("le", false),
        I64_LeU | I64_BrLeU => ("ls", false),
        I64_GeS | I64_BrGeS => ("ge", false),
        I64_GeU | I64_BrGeU => ("hs", false),
        I32_BrAnd => ("ne", true),
        I32_BrAndNot => ("eq", true),
        _ => return None,
    })
}

/// `(access size, is a load, access kind)`; kind 0 = w, 1 = x, 2 = b,
/// 3 = h, 4 = sb->w, 5 = sh->w, 6 = sb->x, 7 = sh->x, 8 = sw->x.
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

/// `(mnemonic, f32 form)` for the direct-instruction float binaries.
/// Hardware `fmin`/`fmax` match wasm's NaN and signed-zero semantics.
fn float_bin(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        F32_Add => ("fadd", true),
        F32_Sub => ("fsub", true),
        F32_Mul => ("fmul", true),
        F32_Div => ("fdiv", true),
        F32_Min => ("fmin", true),
        F32_Max => ("fmax", true),
        F64_Add => ("fadd", false),
        F64_Sub => ("fsub", false),
        F64_Mul => ("fmul", false),
        F64_Div => ("fdiv", false),
        F64_Min => ("fmin", false),
        F64_Max => ("fmax", false),
        _ => return None,
    })
}

/// `(mnemonic, f32 form)` for the unary float ops.
fn float_un(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        F32_Abs => ("fabs", true),
        F32_Neg => ("fneg", true),
        F32_Sqrt => ("fsqrt", true),
        F32_Ceil => ("frintp", true),
        F32_Floor => ("frintm", true),
        F32_Trunc => ("frintz", true),
        F32_Nearest => ("frintn", true),
        F64_Abs => ("fabs", false),
        F64_Neg => ("fneg", false),
        F64_Sqrt => ("fsqrt", false),
        F64_Ceil => ("frintp", false),
        F64_Floor => ("frintm", false),
        F64_Trunc => ("frintz", false),
        F64_Nearest => ("frintn", false),
        _ => return None,
    })
}

/// `(condition, f32 form)`; the mapping is unordered-false except `Ne`,
/// which is unordered-true via `ne`.
fn float_cmp(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        F32_Eq => ("eq", true),
        F32_Ne => ("ne", true),
        F32_Lt => ("mi", true),
        F32_Gt => ("gt", true),
        F32_Le => ("ls", true),
        F32_Ge => ("ge", true),
        F64_Eq => ("eq", false),
        F64_Ne => ("ne", false),
        F64_Lt => ("mi", false),
        F64_Gt => ("gt", false),
        F64_Le => ("ls", false),
        F64_Ge => ("ge", false),
        _ => return None,
    })
}

/// `(unsigned, 64-bit source, f32 destination)` for int -> float.
fn cvt_i2f(op: Op) -> Option<(bool, bool, bool)> {
    use Op::*;
    Some(match op {
        F32_ConvertI32S => (false, false, true),
        F32_ConvertI32U => (true, false, true),
        F32_ConvertI64S => (false, true, true),
        F32_ConvertI64U => (true, true, true),
        F64_ConvertI32S => (false, false, false),
        F64_ConvertI32U => (true, false, false),
        F64_ConvertI64S => (false, true, false),
        F64_ConvertI64U => (true, true, false),
        _ => return None,
    })
}

/// `(f32 source, 64-bit destination, unsigned)` for the saturating
/// float -> int family. `fcvtzs`/`fcvtzu` implement wasm `trunc_sat`
/// exactly: clamp at the bounds, NaN -> 0.
fn cvt_f2i_sat(op: Op) -> Option<(bool, bool, bool)> {
    use Op::*;
    Some(match op {
        I32_TruncSatF32S => (true, false, false),
        I32_TruncSatF32U => (true, false, true),
        I32_TruncSatF64S => (false, false, false),
        I32_TruncSatF64U => (false, false, true),
        I64_TruncSatF32S => (true, true, false),
        I64_TruncSatF32U => (true, true, true),
        I64_TruncSatF64S => (false, true, false),
        I64_TruncSatF64U => (false, true, true),
        _ => return None,
    })
}

/// `(f32 source, 64-bit destination, unsigned, low bound, high bound)`
/// for the trapping float -> int family. The bounds are the EXCLUSIVE
/// trap boundaries on the un-truncated operand, so a native bail always
/// means a trap: a bail that could still succeed on the slow path would
/// break the accumulator pairing, whose soundness rests on "producer
/// bails => execution never reaches the consumer". The low compare uses
/// `le`, which is unordered-true, so it catches NaN for free.
fn cvt_f2i_trap(op: Op) -> Option<(bool, bool, bool, u64, u64)> {
    use Op::*;
    Some(match op {
        I32_TruncF32S => (true, false, false, 0xCF00_0001, 0x4F00_0000),
        I32_TruncF32U => (true, false, true, 0xBF80_0000, 0x4F80_0000),
        I32_TruncF64S => (
            false,
            false,
            false,
            0xC1E0_0000_0020_0000,
            0x41E0_0000_0000_0000,
        ),
        I32_TruncF64U => (
            false,
            false,
            true,
            0xBFF0_0000_0000_0000,
            0x41F0_0000_0000_0000,
        ),
        I64_TruncF32S => (true, true, false, 0xDF00_0001, 0x5F00_0000),
        I64_TruncF32U => (true, true, true, 0xBF80_0000, 0x5F80_0000),
        I64_TruncF64S => (
            false,
            true,
            false,
            0xC3E0_0000_0000_0001,
            0x43E0_0000_0000_0000,
        ),
        I64_TruncF64U => (
            false,
            true,
            true,
            0xBFF0_0000_0000_0000,
            0x43F0_0000_0000_0000,
        ),
        _ => return None,
    })
}

impl Isa for Arm64 {
    fn caps(&self) -> Caps {
        Caps {
            classes: &CLASSES,
            dsts: &DSTS,
            ptr_bytes: 8,
            has_l1: true,
            has_float_regs: true,
            float_pin_f32: true,
            native_calls: true,
            page_aligned_blob: true,
        }
    }

    fn emit_prelude(&mut self, a: &mut Asm, st: &Stubs) {
        // ---- common exit path: x9 = reason ----
        a.label(&st.exit_common);
        a.ins("str x9, [x21, #0]"); // state.reason
        a.ins("str x19, [x21, #8]"); // state.pc
        a.ins("str x20, [x21, #16]"); // calls move the frame; write it back
        a.ins("str x26, [x21, #56]"); // ...and the return-stack cursor
        a.ins("str x15, [x21, #80]"); // ...and the dispatch counter
        a.ins("str x8, [x21, #104]"); // ...and the accumulator (call results)
        a.ins("ldp x19, x20, [sp, #16]");
        a.ins("ldp x21, x22, [sp, #32]");
        a.ins("ldp x23, x24, [sp, #48]");
        a.ins("ldp x25, x26, [sp, #64]");
        a.ins("ldp x27, x28, [sp, #80]");
        a.ins("ldp x29, x30, [sp], #96");
        a.ins("ret");

        for (label, reason) in [
            (&st.slow, 1u32),
            (&st.return_exit, 2),
            (&st.trap_oob, 16),
            (&st.trap_exhaust, 17),
        ] {
            a.label(label);
            a.ins(&format!("mov x9, #{reason}"));
            a.ins(&format!("b {}", st.exit_common));
        }

        // ---- entry trampoline: extern "C" fn(*mut EnterState) ----
        a.label(&st.entry);
        a.ins("stp x29, x30, [sp, #-96]!");
        a.ins("stp x19, x20, [sp, #16]");
        a.ins("stp x21, x22, [sp, #32]");
        a.ins("stp x23, x24, [sp, #48]");
        a.ins("stp x25, x26, [sp, #64]");
        a.ins("stp x27, x28, [sp, #80]");
        a.ins("mov x21, x0");
        a.ins("ldr x19, [x21, #8]");
        a.ins("ldr x20, [x21, #16]");
        a.ins("ldr x22, [x21, #24]");
        a.ins("ldr x23, [x21, #32]");
        a.ins("ldr x24, [x21, #40]");
        a.ins("ldr x25, [x21, #48]");
        a.ins("ldr x26, [x21, #56]");
        a.ins("ldr x27, [x21, #64]");
        a.ins("ldr x28, [x21, #72]");
        a.ins("ldr x15, [x21, #80]");
        a.ins("ldr x17, [x21, #88]");
        a.ins("ldr x14, [x21, #96]");
        a.ins("ldr x8, [x21, #104]");
        a.ins("ldr d17, [x21, #88]");
        a.ins("ldr d18, [x21, #96]");
        a.ins("ldr x9, [x19]");
        a.ins("br x9");

        // ---- Call (wired by the cross-function fixup, not the table) ----
        // cell a = caller_l1off<<48 | callee cells (48-bit VA)
        //      b = callee_l1off<<48 | callee_l0off<<32 | fp<<31 | arg_base*8
        //      c = caller_l0off<<48 | frame_slots<<32 | n_locals<<16 | n_params
        a.label(&st.call);
        // The counter is unconditional here: the call handler is not a
        // class variant, so there is no per-variant predicate to consult.
        a.ins("add x15, x15, #1");
        a.ins("ldr x12, [x19, #24]");
        a.ins("ldr x11, [x19, #16]");
        a.ins("ldr x9, [x19, #8]");

        // The shared activation-entry tail. x9/x11/x12 hold the a/b/c-shaped
        // callee description; the indirect handler composes them from its
        // runtime lookup instead of from cell fields.
        a.label(&st.call_core);
        a.ins("cmp x26, x27");
        a.ins(&format!("b.hs {}", st.trap_exhaust));
        a.ins("ubfx x10, x11, #0, #31"); // arg_base*8 (bit 31 is the fp flag)
        a.ins("add x10, x20, x10"); // new frame base
        a.ins("ubfx x13, x12, #32, #16"); // frame_slots
        a.ins("add x13, x10, x13, lsl #3");
        a.ins("cmp x13, x28");
        a.ins(&format!("b.hi {}", st.trap_exhaust));
        // push (ret_pc, caller frame, code | caller_l0off<<48, caller_l1off)
        a.ins("add x13, x19, #32");
        a.ins("stp x13, x20, [x26]");
        a.ins("lsr x13, x12, #48");
        a.ins("orr x13, x24, x13, lsl #48");
        a.ins("str x13, [x26, #16]");
        a.ins("lsr x13, x9, #48");
        a.ins("str x13, [x26, #24]");
        a.ins("add x26, x26, #32");
        // The callee's first handler is independent of frame initialization
        // and pinned-local reloads. Start that load here so the final
        // indirect branch does not wait on a fresh pointer chase.
        a.ins("ubfx x24, x9, #0, #48");
        a.ins("ldr x6, [x24]");
        a.ins("mov x20, x10");
        // zero the fresh locals: [n_params*8, n_locals*8)
        a.ins("ubfx x10, x12, #0, #16");
        a.ins("ubfx x13, x12, #16, #16");
        a.ins("add x10, x20, x10, lsl #3");
        a.ins("add x13, x20, x13, lsl #3");
        let zl = a.fresh("zl");
        let zdone = a.fresh("zdone");
        if self.apple_tuning {
            let zpair = a.fresh("zpair");
            a.ins("cmp x10, x13");
            a.ins(&format!("b.hs {zdone}"));
            // Clear an odd leading slot, then write the even remainder two
            // at a time. The zero-local fast path stays at two instructions.
            a.ins("sub x16, x13, x10");
            a.ins(&format!("tbz x16, #3, {zpair}"));
            a.ins("str xzr, [x10], #8");
            a.ins("cmp x10, x13");
            a.ins(&format!("b.hs {zdone}"));
            a.label(&zpair);
            a.label(&zl);
            a.ins("stp xzr, xzr, [x10], #16");
            a.ins("cmp x10, x13");
            a.ins(&format!("b.lo {zl}"));
        } else {
            a.label(&zl);
            a.ins("cmp x10, x13");
            a.ins(&format!("b.hs {zdone}"));
            a.ins("str xzr, [x10], #8");
            a.ins(&format!("b {zl}"));
        }
        a.label(&zdone);
        a.ins("ubfx x13, x11, #32, #16"); // callee l0off
        a.ins("ldr x17, [x20, x13]");
        a.ins("lsr x13, x11, #48"); // callee l1off
        a.ins("ldr x14, [x20, x13]");
        // Float twins only when the callee has float-pinned slots (cell b
        // bit 31, stamped by the fixup): int->FP transfers are expensive
        // on this core, and integer-only code predicts the skip perfectly.
        // Polarity matters: with the integer case as the branch TARGET the
        // gate still cost 11%.
        let fp = a.fresh("callfp");
        let cont = a.fresh("callcont");
        a.ins(&format!("tbnz x11, #31, {fp}"));
        a.label(&cont);
        a.ins("mov x19, x24");
        a.ins("br x6");
        a.label(&fp);
        a.ins("fmov d17, x17");
        a.ins("fmov d18, x14");
        a.ins(&format!("b {cont}"));

        // ---- CallIndirect (wired by the fixup pass) ----
        // cell a = caller_l1off<<48 | index_slot*8
        //      b = caller_l0off<<48 | canon_expected<<32 | arg_base*8
        // Table 0's base/length and the per-function info table come from
        // the entry state, refreshed on every chain entry, so table.grow
        // and table.set need no invalidation protocol. Every guard failure
        // (bounds, null, type mismatch, import callee) bails to the slow
        // stub, which re-executes the cell and raises the proper trap or
        // routes the host call.
        a.label(&st.call_indirect);
        a.ins("add x15, x15, #1");
        a.ins("ldr x9, [x19, #8]");
        a.ins("ubfx x10, x9, #0, #32"); // index_slot*8
        a.ins("ldr x10, [x20, x10]"); // t (i32, zero-extended)
        a.ins("ldr x11, [x21, #112]"); // table 0 entries
        a.ins("ldr x12, [x21, #120]"); // table 0 len
        a.ins("cmp w10, w12");
        a.ins(&format!("b.hs {}", st.slow)); // out of bounds (or no table)
                                             // Entries are `RefValue` slots (8 bytes). A plain handle's payload
                                             // is the function index, so the fi*3/fi*24 arithmetic below is
                                             // unchanged. The fast path requires the upper 32 bits to be zero;
                                             // normalized null and current 64-bit tagged encodings fail that test.
        a.ins("ldr x12, [x11, w10, uxtw #3]"); // fi = entries[t]
        a.ins("lsr x13, x12, #32");
        a.ins(&format!("cbnz x13, {}", st.slow)); // upper 32 bits are nonzero
        a.ins("ldr x13, [x21, #136]"); // info length
        a.ins("cmp x12, x13");
        a.ins(&format!("b.hs {}", st.slow)); // function index out of bounds
        a.ins("ldr x13, [x21, #128]"); // info base
        a.ins("add x11, x12, x12, lsl #1"); // fi*3
        a.ins("add x13, x13, x11, lsl #3"); // entry = info + fi*24
        a.ins("ldr x11, [x13, #8]"); // l1off<<48 | l0off<<32 | canon type
        a.ins("ldr x10, [x19, #16]"); // cell b
        a.ins("ubfx x12, x11, #0, #32"); // canonical actual
        a.ins("ubfx x16, x10, #32, #16"); // canonical expected
        a.ins("cmp x12, x16");
        a.ins(&format!("b.ne {}", st.slow)); // type mismatch
        a.ins("ldr x12, [x13, #0]"); // callee cells | fp flag (0 = slow)
        a.ins(&format!("cbz x12, {}", st.slow));
        a.ins("ldr x16, [x13, #16]"); // frame metadata
                                      // compose the call_core inputs; the callee fp flag moves from
                                      // entry bit 0 (cells are 32-byte aligned, low bits free) to
                                      // the b-equivalent's bit 31
        a.ins("ubfx x13, x12, #0, #1");
        a.ins("lsr x12, x12, #5");
        a.ins("lsl x12, x12, #5"); // clean cells address
        a.ins("lsr x9, x9, #48");
        a.ins("orr x9, x12, x9, lsl #48"); // a-equiv: caller_l1off | cells
        a.ins("lsr x12, x10, #48");
        a.ins("orr x12, x16, x12, lsl #48"); // c-equiv: caller_l0off | meta
        a.ins("lsr x11, x11, #32");
        a.ins("lsl x11, x11, #32"); // callee l0/l1 offsets, canon cleared
        a.ins("ubfx x16, x10, #0, #32");
        a.ins("orr x11, x11, x16"); // b-equiv: | arg_base*8
        a.ins("orr x11, x11, x13, lsl #31"); // | callee fp flag
        a.ins(&format!("b {}", st.call_core));
    }

    fn wants(&self, v: &Variant) -> bool {
        use Op::*;
        match v.op {
            // MovSlot's source is never an inline constant: the predecoder
            // emits MovConst for those.
            MovSlot => v.a != Cls::Const,
            // MovPair merges two plain ordered copies; neither source is
            // ever a constant or the accumulator.
            MovPair => {
                !matches!(v.a, Cls::Const | Cls::Acc) && !matches!(v.b, Cls::Const | Cls::Acc)
            }
            BrTable => v.a != Cls::Const,
            I32_SubBrIf | I64_SubBrIf => !matches!(v.a, Cls::Const | Cls::Acc),
            _ => {
                if v.fused {
                    // The fused bank folds a second slot operand into the
                    // address, so the first one is never a constant. The
                    // narrow i64 loads are left out: they are rare and the
                    // bank costs a handler each.
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
        let d = v.d;
        // ---- control flow: no operand plumbing ----
        match v.op {
            Return => return self.emit_return(a, v),
            Br => {
                self.bump(a, v.counted);
                a.ins("ldr x12, [x19, #24]");
                a.ins("mov x19, x12"); // c is an absolute cell address
                a.ins("ldr x9, [x19]");
                a.ins("br x9");
                return;
            }
            BrIf | BrIfNot => {
                self.pre(a);
                // Load the branch target and ITS handler word at entry, not
                // after the condition resolves: the two loads are dependent,
                // so on the taken path they otherwise sit nose-to-tail on the
                // critical path (+4.25 cycles versus falling through).
                // Issued here they overlap the condition load, and x9/x12
                // survive it because the operand helpers only touch x10/x11.
                a.ins("ldr x12, [x19, #24]");
                a.ins("ldr x9, [x12]");
                let ra = self.src_a(a, v.a, 10);
                let nt = a.fresh("nt");
                let br = if v.op == BrIf { "cbz" } else { "cbnz" };
                a.ins(&format!("{br} {}, {nt}", w(ra)));
                self.bump(a, v.counted);
                a.ins("mov x19, x12");
                a.ins("br x9");
                a.label(&nt);
                self.tail(a, v.counted);
                return;
            }
            I32_SubBrIf | I64_SubBrIf => {
                // Keep both dependent target loads off the taken critical
                // path while the arithmetic sources are being prepared.
                a.ins("ldr x12, [x19, #24]");
                a.ins("ldr x9, [x12]");
                a.ins("ldr x13, [x19, #8]"); // a slot byte offset
                let ra = match v.a {
                    Cls::Slot => {
                        a.ins("ldr x10, [x20, x13]");
                        10
                    }
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    Cls::Const | Cls::Acc => unreachable!(),
                };
                let rb = self.src_b(a, v.b, 11);
                let rd = match v.a {
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    _ => 10,
                };
                if v.op == I32_SubBrIf {
                    // i32 arithmetic through a W register zero-extends the
                    // authoritative 64-bit frame slot.
                    a.ins(&format!("sub {}, {}, {}", w(rd), w(ra), w(rb)));
                } else {
                    a.ins(&format!("sub {}, {}, {}", x(rd), x(ra), x(rb)));
                }
                a.ins(&format!("str {}, [x20, x13]", x(rd)));
                let nt = a.fresh("nt");
                a.ins(&format!(
                    "cbz {}, {nt}",
                    if v.op == I32_SubBrIf { w(rd) } else { x(rd) }
                ));
                self.bump(a, v.counted);
                a.ins("mov x19, x12");
                a.ins("br x9");
                a.label(&nt);
                // Falling through happens once in the canonical countdown
                // loop. Fetch its next handler here instead of spending
                // that load on every taken back edge.
                self.pre(a);
                self.tail(a, v.counted);
                return;
            }
            BrTable => {
                // The flat table is `[targets..., default]`; clamping the
                // index to len-1 selects the default for any out-of-range
                // value.
                self.bump(a, v.counted);
                let ra = self.src_a(a, v.a, 10);
                a.ins("ldr x11, [x19, #16]");
                a.ins("ldr x12, [x19, #24]");
                a.ins(&format!("cmp {}, w12", w(ra)));
                a.ins(&format!("csel w10, {}, w12, lo", w(ra)));
                a.ins("ldr w10, [x11, w10, uxtw #2]");
                a.ins("lsl x10, x10, #5");
                a.ins("add x19, x24, x10");
                a.ins("ldr x9, [x19]");
                a.ins("br x9");
                return;
            }
            MemoryFill | MemoryCopy => return self.emit_bulk(a, st, v),
            MemoryFillCopy => return self.emit_fill_copy(a, st, v),
            _ => {}
        }
        if let Some((cond, w32)) = int_cmp(v.op) {
            if matches!(super::layout::family(v.op), Fam::SrcAB) {
                self.pre(a);
                a.ins("ldr x12, [x19, #24]");
                a.ins("ldr x9, [x12]");
                let (ra, rb) = self.src_ab(a, v.a, v.b);
                if matches!(v.op, I32_BrAnd | I32_BrAndNot) {
                    a.ins(&format!("tst {}, {}", w(ra), w(rb)));
                } else {
                    a.ins(&format!("cmp {}, {}", r(w32, ra), r(w32, rb)));
                }
                let nt = a.fresh("nt");
                a.ins(&format!("b.{} {nt}", inv(cond)));
                self.bump(a, v.counted);
                a.ins("mov x19, x12");
                a.ins("br x9");
                a.label(&nt);
                self.tail(a, v.counted);
                return;
            }
        }

        self.pre(a);
        match v.op {
            MovSlot | MovConst => {
                let rd = self.dst_target(d);
                match v.a {
                    Cls::Acc => {
                        if rd != ACC {
                            a.ins(&format!("mov {}, {}", x(rd), x(ACC)));
                        }
                    }
                    Cls::L0 => {
                        if rd != L0R {
                            a.ins(&format!("mov {}, {}", x(rd), x(L0R)));
                        }
                    }
                    Cls::L1 => {
                        if rd != L1R {
                            a.ins(&format!("mov {}, {}", x(rd), x(L1R)));
                        }
                    }
                    Cls::Const => a.ins(&format!("ldr {}, [x19, #8]", x(rd))),
                    Cls::Slot => {
                        a.ins("ldr x10, [x19, #8]");
                        a.ins(&format!("ldr {}, [x20, x10]", x(rd)));
                    }
                }
                self.finish(a, d, rd);
            }
            MovPair => {
                // Strictly ordered: commit dst1 (including its pinned
                // register, when present) before reading src2.
                if v.a == Cls::Slot && v.b == Cls::Slot {
                    a.ins("ldp x10, x11, [x19, #8]");
                } else if v.a == Cls::Slot {
                    a.ins("ldr x10, [x19, #8]");
                } else if v.b == Cls::Slot {
                    a.ins("ldr x11, [x19, #16]");
                }
                a.ins("ldr x12, [x19, #24]"); // dst1*8 << 32 | dst2*8
                let v1 = match v.a {
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    _ => {
                        a.ins("ldr x13, [x20, x10]");
                        13
                    }
                };
                a.ins("lsr x9, x12, #32");
                a.ins(&format!("str {}, [x20, x9]", x(v1)));
                let d1 = match v.pair_d.first() {
                    None => None,
                    Some(DstCls::L0) => Some(L0R),
                    Some(DstCls::L1) => Some(L1R),
                    Some(DstCls::Mem | DstCls::Acc) => unreachable!(),
                };
                if let Some(rd) = d1 {
                    if rd != v1 {
                        a.ins(&format!("mov {}, {}", x(rd), x(v1)));
                    }
                }
                let v2 = match v.b {
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    _ => {
                        a.ins(&format!("ldr {}, [x20, x11]", x(ACC)));
                        ACC
                    }
                };
                if v2 != ACC {
                    a.ins(&format!("mov {}, {}", x(ACC), x(v2)));
                }
                a.ins("ubfx x12, x12, #0, #32");
                a.ins(&format!("str {}, [x20, x12]", x(ACC)));
                let d2 = match v.pair_d.second() {
                    None => None,
                    Some(DstCls::L0) => Some(L0R),
                    Some(DstCls::L1) => Some(L1R),
                    Some(DstCls::Mem | DstCls::Acc) => unreachable!(),
                };
                if let Some(rd) = d2 {
                    if rd != ACC {
                        a.ins(&format!("mov {}, {}", x(rd), x(ACC)));
                    }
                }
            }
            Select => {
                let (ra, rb) = self.src_ab(a, v.a, v.b);
                a.ins("ldr x12, [x19, #24]");
                a.ins("lsr x13, x12, #32"); // cond slot byte offset
                a.ins("ldr x13, [x20, x13]");
                a.ins("cmp x13, xzr");
                let rd = self.dst_target(d);
                a.ins(&format!("csel {}, {}, {}, ne", x(rd), x(ra), x(rb)));
                if d != DstCls::Acc {
                    a.ins("mov w12, w12"); // dst slot byte offset
                    a.ins(&format!("str {}, [x20, x12]", x(rd)));
                }
            }
            GlobalGet => {
                a.ins("ldr x10, [x19, #8]"); // a = index*8
                let rd = self.dst_target(d);
                a.ins(&format!("ldr {}, [x25, x10]", x(rd)));
                self.finish(a, d, rd);
            }
            GlobalSet => {
                let ra = self.src_a(a, v.a, 10);
                a.ins("ldr x12, [x19, #24]"); // c = index*8
                a.ins(&format!("str {}, [x25, x12]", x(ra)));
            }
            I32_Eqz | I64_Eqz => {
                let w32 = v.op == I32_Eqz;
                let ra = self.src_a(a, v.a, 10);
                a.ins(&format!(
                    "cmp {}, {}",
                    r(w32, ra),
                    if w32 { "wzr" } else { "xzr" }
                ));
                let rd = self.dst_target(d);
                a.ins(&format!("cset {}, eq", w(rd)));
                self.finish(a, d, rd);
            }
            I32_DivS | I32_DivU | I32_RemS | I32_RemU | I64_DivS | I64_DivU | I64_RemS
            | I64_RemU => self.emit_div(a, st, v),
            I32_Popcnt | I64_Popcnt => {
                // NEON cnt + uaddlv; i32 slots are zero-extended, so the
                // 64-bit byte sum is exact for both widths.
                let ra = self.src_a(a, v.a, 10);
                a.ins(&format!("fmov d0, {}", x(ra)));
                a.ins("cnt v0.8b, v0.8b");
                a.ins("uaddlv h0, v0.8b");
                let rd = self.dst_target(d);
                a.ins(&format!("fmov {}, s0", w(rd)));
                self.finish(a, d, rd);
            }
            I32_ReinterpretF32 | I64_ReinterpretF64 => {
                // Raw-bit moves, but the accumulator edges are
                // domain-typed: float->int reads the float acc.
                let w32 = v.op == I32_ReinterpretF32;
                let rd = self.dst_target(d);
                if v.a.is_reg() {
                    let vs = self.src_fp(a, v.a, w32, 0, 8, 10);
                    self.fp_result(a, w32, rd, vs);
                } else {
                    let ra = self.src_a(a, v.a, 10);
                    if ra != rd {
                        a.ins(&format!("mov {}, {}", x(rd), x(ra)));
                    }
                }
                self.finish(a, d, rd);
            }
            F32_ReinterpretI32 | F64_ReinterpretI64 => {
                let w32 = v.op == F32_ReinterpretI32;
                let ra = self.src_a(a, v.a, 10);
                a.ins(&format!(
                    "fmov {}, {}",
                    f(w32, self.fp_target(d)),
                    r(w32, ra)
                ));
                self.finish_fp(a, d);
            }
            F32_Copysign | F64_Copysign => {
                // Pure bit splice: no FP unit involved. Float-domain
                // operands arrive in NEON registers and move over.
                let w32 = v.op == F32_Copysign;
                let ra = if v.a.is_reg() {
                    let vs = self.src_fp(a, v.a, w32, 0, 8, 10);
                    self.fp_result(a, w32, 10, vs);
                    10
                } else {
                    self.src_a(a, v.a, 10)
                };
                let rb = if v.b.is_reg() {
                    let vs = self.src_fp(a, v.b, w32, 1, 16, 11);
                    self.fp_result(a, w32, 11, vs);
                    11
                } else {
                    self.src_b(a, v.b, 11)
                };
                if w32 {
                    a.ins(&format!("lsl x12, {}, #33", x(ra)));
                    a.ins("lsr x12, x12, #33");
                    a.ins(&format!("ubfx x13, {}, #31, #1", x(rb)));
                    a.ins("orr x12, x12, x13, lsl #31");
                    a.ins(&format!("fmov {}, w12", f(true, self.fp_target(d))));
                } else {
                    a.ins(&format!("lsl x12, {}, #1", x(ra)));
                    a.ins("lsr x12, x12, #1");
                    a.ins(&format!("lsr x13, {}, #63", x(rb)));
                    a.ins("orr x12, x12, x13, lsl #63");
                    a.ins(&format!("fmov {}, x12", f(false, self.fp_target(d))));
                }
                self.finish_fp(a, d);
            }
            F32_DemoteF64 | F64_PromoteF32 => {
                let to32 = v.op == F32_DemoteF64;
                let vs = self.src_fp(a, v.a, !to32, 0, 8, 10);
                a.ins(&format!(
                    "fcvt {}, {}",
                    f(to32, self.fp_target(d)),
                    f(!to32, vs)
                ));
                self.finish_fp(a, d);
            }
            _ => {
                if let Some((m, w32, rotl)) = int_bin(v.op) {
                    let (ra, mut rb) = self.src_ab(a, v.a, v.b);
                    if rotl {
                        // rotl x, n == rotr x, -n (the rotate masks it)
                        a.ins(&format!("neg {}, {}", r(w32, 11), r(w32, rb)));
                        rb = 11;
                    }
                    let rd = self.dst_target(d);
                    a.ins(&format!(
                        "{m} {}, {}, {}",
                        r(w32, rd),
                        r(w32, ra),
                        r(w32, rb)
                    ));
                    self.finish(a, d, rd);
                } else if let Some((cond, w32)) = int_cmp(v.op) {
                    let (ra, rb) = self.src_ab(a, v.a, v.b);
                    a.ins(&format!("cmp {}, {}", r(w32, ra), r(w32, rb)));
                    let rd = self.dst_target(d);
                    a.ins(&format!("cset {}, {cond}", w(rd)));
                    self.finish(a, d, rd);
                } else if mem_kind(v.op).is_some() {
                    self.emit_mem(a, st, v);
                } else if let Some((m, w32)) = float_bin(v.op) {
                    let (va, vb) = self.src_fp_ab(a, v.a, v.b, w32);
                    a.ins(&format!(
                        "{m} {}, {}, {}",
                        f(w32, self.fp_target(d)),
                        f(w32, va),
                        f(w32, vb)
                    ));
                    self.finish_fp(a, d);
                } else if let Some((cond, w32)) = float_cmp(v.op) {
                    let (va, vb) = self.src_fp_ab(a, v.a, v.b, w32);
                    a.ins(&format!("fcmp {}, {}", f(w32, va), f(w32, vb)));
                    let rd = self.dst_target(d);
                    a.ins(&format!("cset {}, {cond}", w(rd)));
                    self.finish(a, d, rd);
                } else if let Some((m, w32)) = float_un(v.op) {
                    let vs = self.src_fp(a, v.a, w32, 0, 8, 10);
                    a.ins(&format!(
                        "{m} {}, {}",
                        f(w32, self.fp_target(d)),
                        f(w32, vs)
                    ));
                    self.finish_fp(a, d);
                } else if let Some((uns, src64, dst32)) = cvt_i2f(v.op) {
                    let ra = self.src_a(a, v.a, 10);
                    let m = if uns { "ucvtf" } else { "scvtf" };
                    a.ins(&format!(
                        "{m} {}, {}",
                        f(dst32, self.fp_target(d)),
                        r(!src64, ra)
                    ));
                    self.finish_fp(a, d);
                } else if let Some((src32, to64, uns)) = cvt_f2i_sat(v.op) {
                    let vs = self.src_fp(a, v.a, src32, 0, 8, 10);
                    let rd = self.dst_target(d);
                    let m = if uns { "fcvtzu" } else { "fcvtzs" };
                    a.ins(&format!("{m} {}, {}", r(!to64, rd), f(src32, vs)));
                    self.finish(a, d, rd);
                } else if let Some((src32, to64, uns, lo, hi)) = cvt_f2i_trap(v.op) {
                    let vs = self.src_fp(a, v.a, src32, 0, 8, 10);
                    self.mov_imm64(a, 13, lo);
                    a.ins(&format!("fmov {}, {}", f(src32, 1), r(src32, 13)));
                    a.ins(&format!("fcmp {}, {}", f(src32, vs), f(src32, 1)));
                    a.ins(&format!("b.le {}", st.slow)); // x <= lo, or NaN
                    self.mov_imm64(a, 13, hi);
                    a.ins(&format!("fmov {}, {}", f(src32, 1), r(src32, 13)));
                    a.ins(&format!("fcmp {}, {}", f(src32, vs), f(src32, 1)));
                    a.ins(&format!("b.ge {}", st.slow)); // x >= hi
                    let rd = self.dst_target(d);
                    let m = if uns { "fcvtzu" } else { "fcvtzs" };
                    a.ins(&format!("{m} {}, {}", r(!to64, rd), f(src32, vs)));
                    self.finish(a, d, rd);
                } else {
                    self.emit_int_un(a, v);
                }
            }
        }
        self.tail(a, v.counted);
    }

    fn emit_superhandlers(
        &mut self,
        a: &mut Asm,
        st: &Stubs,
        counting: bool,
    ) -> Vec<Option<String>> {
        let mut labels = vec![None; SUPER_HANDLER_SLOTS];
        if !self.superhandlers {
            return labels;
        }
        assert!(
            !counting,
            "interp-count must retain one ordinary handler per measured dispatch"
        );

        let label = a.fresh("super_list");
        a.label(&label);
        self.emit_super_list_step(a, st, counting);
        labels[SUPER_MOVPAIR_LOAD_STORE_BRIF] = Some(label);

        let label = a.fresh("super_add4");
        a.label(&label);
        self.emit_super_add4(a, counting);
        labels[SUPER_ADD4] = Some(label);

        let label = a.fresh("super_add_brne");
        a.label(&label);
        self.emit_super_add_brne(a, counting);
        labels[SUPER_ADD_BRNE] = Some(label);

        let label = a.fresh("super_shru_and");
        a.label(&label);
        self.emit_super_shru_and(a, counting);
        labels[SUPER_SHRU_AND] = Some(label);

        let label = a.fresh("super_and_breq");
        a.label(&label);
        self.emit_super_and_breq(a, counting);
        labels[SUPER_AND_BREQ] = Some(label);
        labels
    }
}

impl Arm64 {
    fn emit_int_un(&mut self, a: &mut Asm, v: &Variant) {
        use Op::*;
        let ra = self.src_a(a, v.a, 10);
        let rd = self.dst_target(v.d);
        match v.op {
            I32_Clz => a.ins(&format!("clz {}, {}", w(rd), w(ra))),
            I64_Clz => a.ins(&format!("clz {}, {}", x(rd), x(ra))),
            I32_Ctz => {
                a.ins(&format!("rbit {}, {}", w(rd), w(ra)));
                a.ins(&format!("clz {0}, {0}", w(rd)));
            }
            I64_Ctz => {
                a.ins(&format!("rbit {}, {}", x(rd), x(ra)));
                a.ins(&format!("clz {0}, {0}", x(rd)));
            }
            I32_Extend8S => a.ins(&format!("sxtb {}, {}", w(rd), w(ra))),
            I32_Extend16S => a.ins(&format!("sxth {}, {}", w(rd), w(ra))),
            I64_Extend8S => a.ins(&format!("sxtb {}, {}", x(rd), w(ra))),
            I64_Extend16S => a.ins(&format!("sxth {}, {}", x(rd), w(ra))),
            I64_Extend32S | I64_ExtendI32S => a.ins(&format!("sxtw {}, {}", x(rd), w(ra))),
            I32_WrapI64 | I64_ExtendI32U => a.ins(&format!("mov {}, {}", w(rd), w(ra))),
            other => panic!("arm64: no handler shape for {other:?}"),
        }
        self.finish(a, v.d, rd);
    }

    fn emit_return(&mut self, a: &mut Asm, v: &Variant) {
        // a = first-result slot*8, b = result count. Results copy to the
        // frame base — the caller's staged-argument area, because frames
        // overlap — then a return record pops. Sentinel records route
        // through the driver's exit cell into `return_exit`.
        self.bump(a, v.counted);
        a.ins("ldr x10, [x19, #8]");
        a.ins("ldr x11, [x19, #16]");
        a.ins("add x10, x20, x10");
        a.ins("mov x12, x20");
        let pop = a.fresh("pop");
        let cl = a.fresh("cl");
        a.ins(&format!("cbz x11, {pop}"));
        // The accumulator doubles as the copy scratch: after a
        // single-result copy it holds result 0, which is the
        // call-result-in-acc convention at zero extra instructions. A
        // dedicated load here instead cost a call-recursive microbench
        // 6-8%: the native return path tolerates no extra
        // store-forwarded load.
        a.label(&cl);
        a.ins("ldr x8, [x10], #8");
        a.ins("str x8, [x12], #8");
        a.ins("sub x11, x11, #1");
        a.ins(&format!("cbnz x11, {cl}"));
        a.label(&pop);
        a.ins("ldp x13, x20, [x26, #-32]!");
        // Overlap the caller continuation's handler load with the remaining
        // return-record and pinned-local reloads.
        a.ins("ldr x9, [x13]");
        a.ins("ldr x12, [x26, #16]"); // caller code | caller_l0off<<48
        a.ins("ldr x10, [x26, #24]"); // caller l1off
        a.ins("lsr x11, x12, #48");
        a.ins("ubfx x24, x12, #0, #48");
        // Reload the caller's pinned locals. Sentinel records carry a
        // readable dummy frame, so these loads are always safe. The
        // caller's float-pinned flag rides bit 0 of its recorded l0 offset
        // (offsets are byte-scaled, so bit 0 is structurally free).
        let fp = a.fresh("retfp");
        let join = a.fresh("retjoin");
        a.ins(&format!("tbnz x11, #0, {fp}"));
        a.ins("ldr x17, [x20, x11]");
        a.ins("ldr x14, [x20, x10]");
        a.label(&join);
        a.ins("mov x19, x13");
        a.ins("br x9");
        a.label(&fp);
        a.ins("sub x11, x11, #1");
        a.ins("ldr x17, [x20, x11]");
        a.ins("ldr x14, [x20, x10]");
        a.ins("fmov d17, x17");
        a.ins("fmov d18, x14");
        a.ins(&format!("b {join}"));
    }

    fn emit_div(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        use Op::*;
        // The trap edges (div0, and MIN/-1 for div_s) branch to the slow
        // stub, where the cell re-executes and raises the proper trap.
        // rem_s needs no overflow edge: sdiv wraps MIN/-1 to MIN and msub
        // then yields the correct 0.
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
        a.ins(&format!("cbz {}, {}", r(w32, rb), st.slow));
        if signed && !rem {
            let go = a.fresh("div");
            a.ins(&format!("mov {}, #-1", r(w32, 12)));
            a.ins(&format!("cmp {}, {}", r(w32, rb), r(w32, 12)));
            a.ins(&format!("b.ne {go}"));
            a.ins("movz x13, #0x8000");
            a.ins(&format!("lsl x13, x13, #{}", if w32 { 16 } else { 48 }));
            a.ins(&format!("cmp {}, {}", r(w32, ra), r(w32, 13)));
            a.ins(&format!("b.eq {}", st.slow));
            a.label(&go);
        }
        let rd = self.dst_target(v.d);
        let div = if signed { "sdiv" } else { "udiv" };
        if rem {
            a.ins(&format!(
                "{div} {}, {}, {}",
                r(w32, 12),
                r(w32, ra),
                r(w32, rb)
            ));
            a.ins(&format!(
                "msub {}, {}, {}, {}",
                r(w32, rd),
                r(w32, 12),
                r(w32, rb),
                r(w32, ra)
            ));
        } else {
            a.ins(&format!(
                "{div} {}, {}, {}",
                r(w32, rd),
                r(w32, ra),
                r(w32, rb)
            ));
        }
        self.finish(a, v.d, rd);
    }

    fn emit_mem(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let (size, load, kind) = mem_kind(v.op).unwrap();
        let fp = is_fp_mem(v.op);
        let d = v.d;
        let ra1 = self.src_a(a, v.a, 10); // address (u32 low bits)
        if v.fused {
            // ea = zext32(addr1 + addr2) + static offset, the
            // corpus-universal base+index pattern folded into one
            // dispatch. addr2 always loads from its slot (packed in c's
            // high half, like Select's cond); addr1 takes the classes.
            a.ins("ldr x11, [x19, #24]");
            a.ins("lsr x13, x11, #32");
            if load {
                a.ins("ubfx x11, x11, #0, #32"); // dst byte offset
                a.ins("ldr x13, [x20, x13]"); // addr2
                a.ins(&format!("add w13, {}, w13", w(ra1))); // wrapping i32 sum
                a.ins("ldr x12, [x19, #16]"); // static offset
            } else {
                a.ins("ldr x13, [x20, x13]"); // addr2
                a.ins(&format!("add w13, {}, w13", w(ra1)));
                a.ins("ubfx x12, x11, #0, #32"); // static offset
            }
            a.ins("add x12, x12, w13, uxtw"); // ea
        } else {
            if load {
                a.ins("ldr x11, [x19, #16]"); // static offset
            } else {
                a.ins("ldr x11, [x19, #24]"); // stores: offset in c
            }
            a.ins(&format!("add x12, x11, {}, uxtw", w(ra1))); // ea = offset + zext(addr)
        }
        a.ins(&format!("add x13, x12, #{size}"));
        a.ins("cmp x13, x23");
        a.ins(&format!("b.hi {}", st.trap_oob));
        if load {
            if fp {
                // Float loads land in their destination's NEON register
                // (acc or pinned) and write through from it — no integer
                // round trip.
                let vt = self.fp_target(d);
                a.ins(&format!("ldr {}, [x22, x12]", f(kind == 0, vt)));
                if v.fused {
                    if d != DstCls::Acc {
                        a.ins(&format!("str {}, [x20, x11]", f(false, vt)));
                    }
                } else {
                    self.finish_fp(a, d);
                }
            } else {
                let rd = self.dst_target(d);
                let m = match kind {
                    0 => format!("ldr {}", w(rd)),
                    1 => format!("ldr {}", x(rd)),
                    2 => format!("ldrb {}", w(rd)),
                    3 => format!("ldrh {}", w(rd)),
                    4 => format!("ldrsb {}", w(rd)),
                    5 => format!("ldrsh {}", w(rd)),
                    6 => format!("ldrsb {}", x(rd)),
                    7 => format!("ldrsh {}", x(rd)),
                    8 => format!("ldrsw {}", x(rd)),
                    _ => unreachable!(),
                };
                a.ins(&format!("{m}, [x22, x12]"));
                if v.fused {
                    if d != DstCls::Acc {
                        a.ins(&format!("str {}, [x20, x11]", x(rd)));
                    }
                } else {
                    self.finish(a, d, rd);
                }
            }
        } else if fp && v.b.is_reg() {
            // The float store value is already register-resident.
            let vs = match v.b {
                Cls::L0 => FL0R,
                Cls::L1 => FL1R,
                _ => FACC,
            };
            a.ins(&format!("str {}, [x22, x12]", f(kind == 0, vs)));
        } else {
            // Value operand; the offset in x11 is consumed by now.
            let rb = self.src_b(a, v.b, 11);
            let m = match kind {
                0 => format!("str {}", w(rb)),
                1 => format!("str {}", x(rb)),
                2 => format!("strb {}", w(rb)),
                3 => format!("strh {}", w(rb)),
                _ => unreachable!(),
            };
            a.ins(&format!("{m}, [x22, x12]"));
        }
    }

    /// `memory.fill` / `memory.copy` on memory 0. `a` is the base slot of
    /// the three contiguous operands. Bulk-memory handlers must move wide
    /// blocks, not machine words: an 8-byte-per-iteration loop put copy
    /// 28% BELOW wasm3 and wasmi on STREAM Copy. 64-byte NEON blocks took
    /// copy past both and fill to 2.2x. 128-byte blocks measured a wash,
    /// so the loop is bandwidth-bound past 64. q0-q3 are free: the pinned
    /// float registers are v16-v18.
    fn emit_bulk(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let fill = v.op == Op::MemoryFill;
        self.pre(a);
        // No bump at entry: every exit runs through `tail`, which counts.
        // Bumping here as well counted each fill and copy dispatch twice.
        a.ins("ldr x9, [x19, #8]");
        a.ins("add x9, x20, x9");
        a.ins("ldr x10, [x9, #0]"); // d
        a.ins("ldr x11, [x9, #8]"); // fill: value, copy: s
        a.ins("ldr x12, [x9, #16]"); // n
        a.ins("add x13, x10, x12");
        a.ins("cmp x13, x23");
        a.ins(&format!("b.hi {}", st.slow)); // dst out of bounds
        let l64 = a.fresh("l64");
        let l8 = a.fresh("l8");
        let bytes = a.fresh("bytes");
        let done = a.fresh("done");
        if fill {
            a.ins("ubfx x11, x11, #0, #8"); // splat the fill byte
            a.ins("orr x11, x11, x11, lsl #8");
            a.ins("orr x11, x11, x11, lsl #16");
            a.ins("orr x11, x11, x11, lsl #32");
            a.ins("add x10, x22, x10"); // cursor
            a.ins("add x13, x22, x13"); // end
            a.ins("dup v0.2d, x11");
            a.label(&l64);
            a.ins("add x9, x10, #64");
            a.ins("cmp x9, x13");
            a.ins(&format!("b.hi {l8}"));
            a.ins("stp q0, q0, [x10], #32");
            a.ins("stp q0, q0, [x10], #32");
            a.ins(&format!("b {l64}"));
            a.label(&l8);
            a.ins("add x9, x10, #8");
            a.ins("cmp x9, x13");
            a.ins(&format!("b.hi {bytes}"));
            a.ins("str x11, [x10], #8");
            a.ins(&format!("b {l8}"));
            a.label(&bytes);
            a.ins("cmp x10, x13");
            a.ins(&format!("b.hs {done}"));
            a.ins("strb w11, [x10]");
            a.ins("add x10, x10, #1");
            a.ins(&format!("b {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
        } else {
            a.ins("add x13, x11, x12");
            a.ins("cmp x13, x23");
            a.ins(&format!("b.hi {}", st.slow)); // src out of bounds
            let back = a.fresh("copyback");
            let fwd = a.fresh("copyfwd");
            // A forward copy is wrong only when s < d < s+n; that
            // overlapping-downward move takes the backward block.
            a.ins("cmp x10, x11");
            a.ins(&format!("b.ls {fwd}"));
            a.ins("cmp x10, x13");
            a.ins(&format!("b.lo {back}"));
            a.label(&fwd);
            a.ins("add x10, x22, x10"); // dst cursor
            a.ins("add x11, x22, x11"); // src cursor
            a.ins("add x13, x10, x12"); // dst end
            a.label(&l64);
            a.ins("add x9, x10, #64");
            a.ins("cmp x9, x13");
            a.ins(&format!("b.hi {l8}"));
            a.ins("ldp q0, q1, [x11], #32");
            a.ins("ldp q2, q3, [x11], #32");
            a.ins("stp q0, q1, [x10], #32");
            a.ins("stp q2, q3, [x10], #32");
            a.ins(&format!("b {l64}"));
            a.label(&l8);
            a.ins("add x9, x10, #8");
            a.ins("cmp x9, x13");
            a.ins(&format!("b.hi {bytes}"));
            a.ins("ldr x12, [x11], #8");
            a.ins("str x12, [x10], #8");
            a.ins(&format!("b {l8}"));
            a.label(&bytes);
            a.ins("cmp x10, x13");
            a.ins(&format!("b.hs {done}"));
            a.ins("ldrb w12, [x11]");
            a.ins("strb w12, [x10]");
            a.ins("add x10, x10, #1");
            a.ins("add x11, x11, #1");
            a.ins(&format!("b {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
            // Overlapping-downward block, out of line past the tail:
            // x10 = d, x11 = s, x12 = n, all raw offsets.
            let b32 = a.fresh("b32");
            let bfinish = a.fresh("bfinish");
            let b8 = a.fresh("b8");
            let bb = a.fresh("bb");
            let bdone = a.fresh("bdone");
            a.label(&back);
            a.ins("add x9, x22, x10"); // original dst
            a.ins("add x10, x22, x11"); // original src
            a.ins("mov x11, x12"); // remaining length
            a.ins("add x12, x9, x11"); // dst end
            a.ins("add x10, x10, x11"); // src end
            a.ins("cmp x11, #64");
            a.ins(&format!("b.lo {b8}"));

            // Darwin's memmove strategy for medium backward copies: align
            // the destination end to 32 bytes and pipeline one 32-byte GPR
            // block ahead of the stores. On Apple silicon this schedules
            // better than descending NEON pairs, while preserving x7/x8,
            // pinned locals, the dispatch counter, and every vector register.
            a.ins("ldp x0, x1, [x10, #-16]");
            a.ins("ldp x2, x3, [x10, #-32]");
            a.ins("sub x13, x12, #1");
            a.ins("and x13, x13, #0xffffffffffffffe0"); // aligned dst cursor
            a.ins("sub x16, x12, x13"); // 1..32-byte end cap
            a.ins("sub x10, x10, x16");
            a.ins("sub x11, x11, x16");
            a.ins("ldp x4, x5, [x10, #-16]");
            a.ins("ldp x6, x16, [x10, #-32]");
            a.ins("stp x0, x1, [x12, #-16]");
            a.ins("stp x2, x3, [x12, #-32]");
            a.ins("sub x10, x10, #32");
            a.ins("subs x11, x11, #64");
            a.ins(&format!("b.ls {bfinish}"));
            a.label(&b32);
            a.ins("stp x4, x5, [x13, #-16]");
            a.ins("stp x6, x16, [x13, #-32]");
            a.ins("sub x13, x13, #32");
            a.ins("ldp x4, x5, [x10, #-16]");
            a.ins("ldp x6, x16, [x10, #-32]");
            a.ins("sub x10, x10, #32");
            a.ins("subs x11, x11, #32");
            a.ins(&format!("b.hi {b32}"));
            a.label(&bfinish);
            a.ins("sub x10, x10, x11");
            a.ins("ldp x0, x1, [x10, #-16]");
            a.ins("ldp x2, x3, [x10, #-32]");
            a.ins("stp x4, x5, [x13, #-16]");
            a.ins("stp x6, x16, [x13, #-32]");
            a.ins("stp x0, x1, [x9, #16]");
            a.ins("stp x2, x3, [x9]");
            a.ins(&format!("b {bdone}"));

            // Tiny backward copies do not enter the overlapping wide-store
            // scheme above.
            a.label(&b8);
            a.ins("cmp x11, #8");
            a.ins(&format!("b.lo {bb}"));
            a.ins("sub x11, x11, #8");
            a.ins("ldr x0, [x10, #-8]!");
            a.ins("str x0, [x12, #-8]!");
            a.ins(&format!("b {b8}"));
            a.label(&bb);
            a.ins(&format!("cbz x11, {bdone}"));
            a.ins("sub x11, x11, #1");
            a.ins("ldrb w0, [x10, #-1]!");
            a.ins("strb w0, [x12, #-1]!");
            a.ins(&format!("b {bb}"));
            a.label(&bdone);
            self.tail(a, v.counted);
        }
    }

    /// Adjacent fill+copy on memory 0.
    ///
    /// The predecoder proves only adjacency and same-memory identity. This
    /// handler dynamically proves the useful semantic fact: the copy source
    /// lies wholly inside the range made uniform by the fill. When the
    /// destination overlaps that range, the pair is one fill over their
    /// union. Any failed guard replays the exact pair in the shared executor,
    /// which also preserves the required "fill commits before copy traps"
    /// ordering.
    fn emit_fill_copy(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        self.pre(a);
        a.ins("ldp x9, x10, [x19, #8]"); // fill and copy operand-pack offsets
        a.ins("add x9, x20, x9");
        a.ins("add x10, x20, x10");
        a.ins("ldr x11, [x9]"); // fill dst
        a.ins("ldr x12, [x9, #16]"); // fill len
        a.ins("ldr x9, [x9, #8]"); // fill value
        a.ins("ubfx x9, x9, #0, #8");
        a.ins("orr x9, x9, x9, lsl #8");
        a.ins("orr x9, x9, x9, lsl #16");
        a.ins("orr x9, x9, x9, lsl #32");
        a.ins("dup v0.2d, x9");
        a.ins("add x13, x11, x12"); // fill end
        a.ins("cmp x13, x23");
        a.ins(&format!("b.hi {}", st.slow));

        a.ins("ldr x16, [x10]"); // copy dst
        a.ins("ldr x9, [x10, #16]"); // copy len
        a.ins("add x12, x16, x9"); // copy dst end
        a.ins("cmp x12, x23");
        a.ins(&format!("b.hi {}", st.slow));
        a.ins("ldr x10, [x10, #8]"); // copy src
        a.ins("cmp x10, x11");
        a.ins(&format!("b.lo {}", st.slow));
        a.ins("add x9, x10, x9"); // copy src end
        a.ins("cmp x9, x23");
        a.ins(&format!("b.hi {}", st.slow));
        a.ins("cmp x9, x13");
        a.ins(&format!("b.hi {}", st.slow));

        // Only a single contiguous union belongs in this fast path.
        a.ins("cmp x16, x13");
        a.ins(&format!("b.hi {}", st.slow));
        a.ins("cmp x12, x11");
        a.ins(&format!("b.lo {}", st.slow));
        a.ins("cmp x11, x16");
        a.ins("csel x10, x11, x16, lo"); // union start
        a.ins("cmp x13, x12");
        a.ins("csel x13, x13, x12, hs"); // union end

        // Recover a scalar copy of the already-splatted byte for the tails.
        a.ins("fmov x11, d0");
        a.ins("add x10, x22, x10");
        a.ins("add x13, x22, x13");

        let l128 = a.fresh("fc128");
        let l64 = a.fresh("fc64");
        let l8 = a.fresh("fc8");
        let bytes = a.fresh("fcbytes");
        let done = a.fresh("fcdone");
        a.label(&l128);
        a.ins("add x9, x10, #128");
        a.ins("cmp x9, x13");
        a.ins(&format!("b.hi {l64}"));
        a.ins("stp q0, q0, [x10], #32");
        a.ins("stp q0, q0, [x10], #32");
        a.ins("stp q0, q0, [x10], #32");
        a.ins("stp q0, q0, [x10], #32");
        a.ins(&format!("b {l128}"));
        a.label(&l64);
        a.ins("add x9, x10, #64");
        a.ins("cmp x9, x13");
        a.ins(&format!("b.hi {l8}"));
        a.ins("stp q0, q0, [x10], #32");
        a.ins("stp q0, q0, [x10], #32");
        a.ins(&format!("b {l64}"));
        a.label(&l8);
        a.ins("add x9, x10, #8");
        a.ins("cmp x9, x13");
        a.ins(&format!("b.hi {bytes}"));
        a.ins("str x11, [x10], #8");
        a.ins(&format!("b {l8}"));
        a.label(&bytes);
        a.ins("cmp x10, x13");
        a.ins(&format!("b.hs {done}"));
        a.ins("strb w11, [x10]");
        a.ins("add x10, x10, #1");
        a.ins(&format!("b {bytes}"));
        a.label(&done);
        self.tail(a, v.counted);
    }
}
