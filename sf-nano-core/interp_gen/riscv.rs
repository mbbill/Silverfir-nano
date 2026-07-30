//! RISC-V handler backend, RV64 and RV32 from one source.
//!
//! Register contract (chain-invariant across every handler):
//! ```text
//!   s2 = pc          s3 = frame base   s4 = &EnterState
//!   s5 = memory 0 base                 s6 = memory 0 length
//!   s7 = cell array base               s8 = globals base
//!   s9 = return-stack cursor           s10 = return-stack limit
//!   s11 = value-stack limit            a1  = dispatch counter
//!   s1 = accumulator   s0 = l0   a0 = l1 (RV64 only)
//!   t0 = prefetched next handler word
//!   t1-t6, a2-a5 = scratch;  fs0 = float acc, fs1/fs2 = float l0/l1
//! ```
//!
//! Three things shape this backend.
//!
//! **No base+index addressing.** Every frame-slot access costs an extra
//! `add` that arm64 and x86-64 get inside the load. That is the RISC-V tax
//! on a threaded interpreter and nothing removes it.
//!
//! **RV64's 32-bit ops sign-extend.** The engine's convention is that an
//! i32 value is ZERO-extended in its 8-byte slot, and that convention is
//! shared with the single-instruction executor, the host boundary, globals
//! and invocation results. Rather than fork it per target, RV64 pays two
//! instructions to re-establish it after the arithmetic that sign-extends.
//! Adopting RISC-V's native sign-extended convention instead would move the
//! cost onto unsigned compares AND change a cross-boundary contract, which
//! is the per-target divergence this design exists to avoid.
//!
//! **On RV32 a wasm value is a register PAIR.** A wasm value still occupies
//! eight bytes, so the accumulator and the pinned local are two registers
//! each. The 64-bit ops that stay native are the ones a pair handles in a
//! couple of instructions: moves, loads and stores, add and sub with carry,
//! the bitwise family, and equality. Multiply, divide, the variable shifts,
//! the rotates and the ordered 64-bit compares go to the shared executor.
//! So do calls: the call protocol packs its operands into the high halves
//! of 64-bit cell fields, and threading six half-registers through the
//! shared activation path buys less than it costs on a target whose profile
//! is an MCU. RV32 calls therefore cross the chain boundary twice, which is
//! correct — the driver plants a sentinel per activation and the native
//! `Return` still pops it.

use super::asm::Asm;
use super::instr::Op;
use super::isa::{Caps, Isa, Stubs, Variant, CLASSES, CLASSES_NO_L1, DSTS, DSTS_NO_L1};
use super::layout::{Cls, DstCls, Fam};

const PC: &str = "s2";
const FRAME: &str = "s3";
const STATE: &str = "s4";
const MEM: &str = "s5";
const MEMLEN: &str = "s6";
const CODE: &str = "s7";
const GLOB: &str = "s8";
const RETSP: &str = "s9";
const RETLIM: &str = "s10";
const STKLIM: &str = "s11";
const DCNT: &str = "a1";
const NEXT: &str = "t0";
const ACC: &str = "s1";
const L0R: &str = "s0";
const L1R: &str = "a0";
/// High halves of the value registers; RV32 only.
const ACC_HI: &str = "a6";
const L0_HI: &str = "a7";

const FACC: &str = "fs0";
const FL0R: &str = "fs1";
const FL1R: &str = "fs2";

const T1: &str = "t1";
const T2: &str = "t2";
const T3: &str = "t3";
const T4: &str = "t4";
const T5: &str = "t5";
const T6: &str = "t6";
const A2: &str = "a2";
const A3: &str = "a3";
const A4: &str = "a4";
const A5: &str = "a5";

/// `(kind, 32-bit operands, needs sign-extended operands on RV64)`.
fn int_cmp(op: Op) -> Option<(&'static str, bool, bool)> {
    use Op::*;
    Some(match op {
        I32_Eq | I32_BrEq => ("eq", true, false),
        I32_Ne | I32_BrNe => ("ne", true, false),
        I32_LtS | I32_BrLtS => ("lt", true, true),
        I32_LtU | I32_BrLtU => ("ltu", true, false),
        I32_GtS | I32_BrGtS => ("gt", true, true),
        I32_GtU | I32_BrGtU => ("gtu", true, false),
        I32_LeS | I32_BrLeS => ("le", true, true),
        I32_LeU | I32_BrLeU => ("leu", true, false),
        I32_GeS | I32_BrGeS => ("ge", true, true),
        I32_GeU | I32_BrGeU => ("geu", true, false),
        I64_Eq | I64_BrEq => ("eq", false, false),
        I64_Ne | I64_BrNe => ("ne", false, false),
        I64_LtS | I64_BrLtS => ("lt", false, true),
        I64_LtU | I64_BrLtU => ("ltu", false, false),
        I64_GtS | I64_BrGtS => ("gt", false, true),
        I64_GtU | I64_BrGtU => ("gtu", false, false),
        I64_LeS | I64_BrLeS => ("le", false, true),
        I64_LeU | I64_BrLeU => ("leu", false, false),
        I64_GeS | I64_BrGeS => ("ge", false, true),
        I64_GeU | I64_BrGeU => ("geu", false, false),
        I32_BrAnd => ("and_ne", true, false),
        I32_BrAndNot => ("and_eq", true, false),
        _ => return None,
    })
}

fn int_bin(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        I32_Add => ("add", true),
        I32_Sub => ("sub", true),
        I32_And => ("and", true),
        I32_Or => ("or", true),
        I32_Xor => ("xor", true),
        I32_Mul => ("mul", true),
        I32_Shl => ("sll", true),
        I32_ShrU => ("srl", true),
        I32_ShrS => ("sra", true),
        I32_Rotr => ("rotr", true),
        I32_Rotl => ("rotl", true),
        I64_Add => ("add", false),
        I64_Sub => ("sub", false),
        I64_And => ("and", false),
        I64_Or => ("or", false),
        I64_Xor => ("xor", false),
        I64_Mul => ("mul", false),
        I64_Shl => ("sll", false),
        I64_ShrU => ("srl", false),
        I64_ShrS => ("sra", false),
        I64_Rotr => ("rotr", false),
        I64_Rotl => ("rotl", false),
        _ => return None,
    })
}

/// `(size, is a load, kind)`; kind 0 = u32, 1 = 64-bit, 2 = u8, 3 = u16,
/// 4 = i8->i32, 5 = i16->i32, 6 = i8->i64, 7 = i16->i64, 8 = i32->i64.
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

fn float_bin(op: Op) -> Option<(&'static str, bool)> {
    use Op::*;
    Some(match op {
        F32_Add => ("fadd", true),
        F32_Sub => ("fsub", true),
        F32_Mul => ("fmul", true),
        F32_Div => ("fdiv", true),
        F64_Add => ("fadd", false),
        F64_Sub => ("fsub", false),
        F64_Mul => ("fmul", false),
        F64_Div => ("fdiv", false),
        _ => return None,
    })
}

/// `(mnemonic, f32, swap operands, invert)`. RISC-V's `feq`/`flt`/`fle`
/// answer false for an unordered pair — exactly wasm's rule for every
/// compare except `ne`, which is the inversion.
fn float_cmp(op: Op) -> Option<(&'static str, bool, bool, bool)> {
    use Op::*;
    Some(match op {
        F32_Eq => ("feq", true, false, false),
        F32_Ne => ("feq", true, false, true),
        F32_Lt => ("flt", true, false, false),
        F32_Gt => ("flt", true, true, false),
        F32_Le => ("fle", true, false, false),
        F32_Ge => ("fle", true, true, false),
        F64_Eq => ("feq", false, false, false),
        F64_Ne => ("feq", false, false, true),
        F64_Lt => ("flt", false, false, false),
        F64_Gt => ("flt", false, true, false),
        F64_Le => ("fle", false, false, false),
        F64_Ge => ("fle", false, true, false),
        _ => return None,
    })
}

/// `(f32 source, 64-bit destination, unsigned)`.
fn cvt_f2i(op: Op) -> Option<(bool, bool, bool)> {
    use Op::*;
    Some(match op {
        I32_TruncSatF32S | I32_TruncF32S => (true, false, false),
        I32_TruncSatF32U | I32_TruncF32U => (true, false, true),
        I32_TruncSatF64S | I32_TruncF64S => (false, false, false),
        I32_TruncSatF64U | I32_TruncF64U => (false, false, true),
        I64_TruncSatF32S | I64_TruncF32S => (true, true, false),
        I64_TruncSatF32U | I64_TruncF32U => (true, true, true),
        I64_TruncSatF64S | I64_TruncF64S => (false, true, false),
        I64_TruncSatF64U | I64_TruncF64U => (false, true, true),
        _ => return None,
    })
}

/// Exclusive trap boundaries for the trapping float->int family, as raw
/// bits of the source format — the same values every backend checks.
fn trap_bounds(op: Op) -> Option<(u64, u64)> {
    use Op::*;
    Some(match op {
        I32_TruncF32S => (0xCF00_0001, 0x4F00_0000),
        I32_TruncF32U => (0xBF80_0000, 0x4F80_0000),
        I32_TruncF64S => (0xC1E0_0000_0020_0000, 0x41E0_0000_0000_0000),
        I32_TruncF64U => (0xBFF0_0000_0000_0000, 0x41F0_0000_0000_0000),
        I64_TruncF32S => (0xDF00_0001, 0x5F00_0000),
        I64_TruncF32U => (0xBF80_0000, 0x5F80_0000),
        I64_TruncF64S => (0xC3E0_0000_0000_0001, 0x43E0_0000_0000_0000),
        I64_TruncF64U => (0xBFF0_0000_0000_0000, 0x43F0_0000_0000_0000),
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

fn is_float_op(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    (d >= F32_Abs as u16 && d <= F32_Ge as u16)
        || (d >= F64_Abs as u16 && d <= F64_Ge as u16)
        || (d >= I32_TruncF32S as u16 && d <= F64_ReinterpretI64 as u16)
        || matches!(op, F32_Load | F64_Load | F32_Store | F64_Store)
}

/// The ops RV32 keeps native: every i32-domain op, plus the 64-bit ones a
/// register pair handles in a couple of instructions.
fn rv32_native(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    if (d >= I32_Add as u16 && d <= I32_GeU as u16)
        || matches!(
            op,
            I32_Load | I32_Load8S | I32_Load8U | I32_Load16S | I32_Load16U
        )
        || matches!(op, I32_Store | I32_Store8 | I32_Store16)
        || (d >= I32_BrEq as u16 && d <= I32_BrGeU as u16)
        || matches!(op, I32_BrAnd | I32_BrAndNot | I32_SubBrIf | I64_SubBrIf)
    {
        // The 32-bit shifts, rotates, multiply and divide are native here;
        // RV32 has them as base instructions.
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
            | I64_Load
            | I64_Store
            | I64_Store32
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
    )
}

/// Which `b*` realises a compare kind, and in which operand order. RISC-V
/// has no `gt`/`le` branch, so those swap.
fn branch_form<'s>(kind: &str, x: &'s str, y: &'s str) -> (&'static str, &'s str, &'s str) {
    match kind {
        "eq" | "and_eq" => ("beq", x, y),
        "ne" | "and_ne" => ("bne", x, y),
        "lt" => ("blt", x, y),
        "ltu" => ("bltu", x, y),
        "gt" => ("blt", y, x),
        "gtu" => ("bltu", y, x),
        "le" => ("bge", y, x),
        "leu" => ("bgeu", y, x),
        "ge" => ("bge", x, y),
        "geu" => ("bgeu", x, y),
        other => panic!("no branch form for {other}"),
    }
}

fn invert_branch(br: &str) -> &'static str {
    match br {
        "beq" => "bne",
        "bne" => "beq",
        "blt" => "bge",
        "bge" => "blt",
        "bltu" => "bgeu",
        "bgeu" => "bltu",
        other => panic!("no inverse for {other}"),
    }
}

/// Materialize a compare into a 0/1 register.
fn emit_setcc(a: &mut Asm, kind: &str, rd: &str, x: &str, y: &str, tmp: &str) {
    match kind {
        "eq" => {
            a.ins(&format!("xor {tmp}, {x}, {y}"));
            a.ins(&format!("seqz {rd}, {tmp}"));
        }
        "ne" => {
            a.ins(&format!("xor {tmp}, {x}, {y}"));
            a.ins(&format!("snez {rd}, {tmp}"));
        }
        "lt" => a.ins(&format!("slt {rd}, {x}, {y}")),
        "ltu" => a.ins(&format!("sltu {rd}, {x}, {y}")),
        "gt" => a.ins(&format!("slt {rd}, {y}, {x}")),
        "gtu" => a.ins(&format!("sltu {rd}, {y}, {x}")),
        "le" => {
            a.ins(&format!("slt {rd}, {y}, {x}"));
            a.ins(&format!("xori {rd}, {rd}, 1"));
        }
        "leu" => {
            a.ins(&format!("sltu {rd}, {y}, {x}"));
            a.ins(&format!("xori {rd}, {rd}, 1"));
        }
        "ge" => {
            a.ins(&format!("slt {rd}, {x}, {y}"));
            a.ins(&format!("xori {rd}, {rd}, 1"));
        }
        "geu" => {
            a.ins(&format!("sltu {rd}, {x}, {y}"));
            a.ins(&format!("xori {rd}, {rd}, 1"));
        }
        other => panic!("no setcc for {other}"),
    }
}

pub struct RiscV {
    pub xlen: u32,
    /// Whether the target has the F and D extensions.
    pub fp: bool,
}

impl RiscV {
    fn rv64(&self) -> bool {
        self.xlen == 64
    }
    /// Pointer-width load / store mnemonics. On RV32 these also read the
    /// LOW half of a 64-bit cell field, which is what every field except an
    /// i64 constant needs.
    fn lp(&self) -> &'static str {
        if self.rv64() {
            "ld"
        } else {
            "lw"
        }
    }
    fn sp(&self) -> &'static str {
        if self.rv64() {
            "sd"
        } else {
            "sw"
        }
    }
    /// Zero-extending 32-bit load.
    fn lw32(&self) -> &'static str {
        if self.rv64() {
            "lwu"
        } else {
            "lw"
        }
    }
    fn wsz(&self) -> u32 {
        self.xlen / 8
    }

    fn pre(&self, a: &mut Asm) {
        a.ins(&format!("{} {NEXT}, 32({PC})", self.lp()));
    }

    fn bump(&self, a: &mut Asm, on: bool) {
        if on {
            a.ins(&format!("addi {DCNT}, {DCNT}, 1"));
        }
    }

    fn tail(&self, a: &mut Asm, counted: bool) {
        self.bump(a, counted);
        a.ins(&format!("addi {PC}, {PC}, 32"));
        a.ins(&format!("jr {NEXT}"));
    }

    /// A conditional branch to a label hundreds of kilobytes away.
    /// RISC-V's conditional branches reach only +-4 KB, so a far target is
    /// written as an inverted short branch over an unconditional jump
    /// rather than left to assembler relaxation.
    /// Guard a 32-bit address addition against wrapping. RISC-V has no
    /// carry flag, so the carry-out is recovered by comparing the sum with
    /// one addend. A wrapped effective address would land back inside the
    /// memory and let an access wasm requires to trap succeed silently.
    /// Not needed on RV64, where a 48-bit offset plus a 32-bit address
    /// cannot overflow the register.
    fn carry_guard(&self, a: &mut Asm, sum: &str, addend: &str, target: &str) {
        if self.rv64() {
            return;
        }
        a.ins(&format!("sltu {A5}, {sum}, {addend}"));
        self.br_far(a, "bne", A5, "zero", target);
    }

    fn br_far(&self, a: &mut Asm, br: &str, x: &str, y: &str, target: &str) {
        let skip = a.fresh("nf");
        a.ins(&format!("{} {x}, {y}, {skip}", invert_branch(br)));
        a.ins(&format!("j {target}"));
        a.label(&skip);
    }

    /// The frame address of a slot whose byte offset sits in a cell field.
    fn slot_addr(&self, a: &mut Asm, field: u32, tmp: &str) {
        a.ins(&format!("{} {tmp}, {field}({PC})", self.lp()));
        a.ins(&format!("add {tmp}, {FRAME}, {tmp}"));
    }

    /// A source operand's low word (the whole value on RV64).
    fn src(&self, a: &mut Asm, cls: Cls, field: u32, tmp: &'static str) -> &'static str {
        match cls {
            Cls::Acc => ACC,
            Cls::L0 => L0R,
            Cls::L1 => L1R,
            Cls::Const => {
                a.ins(&format!("{} {tmp}, {field}({PC})", self.lp()));
                tmp
            }
            Cls::Slot => {
                self.slot_addr(a, field, tmp);
                a.ins(&format!("{} {tmp}, 0({tmp})", self.lp()));
                tmp
            }
        }
    }

    /// A source operand as a `(low, high)` pair. On RV64 the high name is
    /// never consulted.
    fn pair(
        &self,
        a: &mut Asm,
        cls: Cls,
        field: u32,
        lo_tmp: &'static str,
        hi_tmp: &'static str,
    ) -> (&'static str, &'static str) {
        if self.rv64() {
            return (self.src(a, cls, field, lo_tmp), "zero");
        }
        match cls {
            Cls::Acc => (ACC, ACC_HI),
            Cls::L0 => (L0R, L0_HI),
            Cls::L1 => unreachable!("rv32 has no l1"),
            Cls::Const => {
                a.ins(&format!("lw {lo_tmp}, {field}({PC})"));
                a.ins(&format!("lw {hi_tmp}, {}({PC})", field + 4));
                (lo_tmp, hi_tmp)
            }
            Cls::Slot => {
                self.slot_addr(a, field, hi_tmp);
                a.ins(&format!("lw {lo_tmp}, 0({hi_tmp})"));
                a.ins(&format!("lw {hi_tmp}, 4({hi_tmp})"));
                (lo_tmp, hi_tmp)
            }
        }
    }

    fn dst_target(&self, dc: DstCls) -> &'static str {
        match dc {
            DstCls::L0 => L0R,
            DstCls::L1 => L1R,
            _ => ACC,
        }
    }

    fn dst_hi(&self, dc: DstCls) -> &'static str {
        match dc {
            DstCls::L0 => L0_HI,
            _ => ACC_HI,
        }
    }

    /// Re-establish the zero-extended slot convention for a 32-bit result.
    fn zext32(&self, a: &mut Asm, rd: &str, dc: DstCls) {
        if self.rv64() {
            a.ins(&format!("slli {rd}, {rd}, 32"));
            a.ins(&format!("srli {rd}, {rd}, 32"));
        } else {
            a.ins(&format!("li {}, 0", self.dst_hi(dc)));
        }
    }

    /// Store a value to the destination slot. `wide` says the high word
    /// carries meaning on RV32; a 32-bit result writes a zero there.
    fn finish(&self, a: &mut Asm, dc: DstCls, src: &str, wide: bool) {
        if dc == DstCls::Acc {
            return;
        }
        self.slot_addr(a, 24, T5);
        a.ins(&format!("{} {src}, 0({T5})", self.sp()));
        if !self.rv64() {
            let hi = if wide { self.dst_hi(dc) } else { "zero" };
            a.ins(&format!("sw {hi}, 4({T5})"));
        }
    }

    fn fp_target(&self, dc: DstCls) -> &'static str {
        match dc {
            DstCls::L0 => FL0R,
            DstCls::L1 => FL1R,
            _ => FACC,
        }
    }

    fn src_fp(
        &self,
        a: &mut Asm,
        cls: Cls,
        w32: bool,
        v: &'static str,
        field: u32,
        tmp: &'static str,
    ) -> &'static str {
        let l = if w32 { "flw" } else { "fld" };
        match cls {
            Cls::Slot => {
                self.slot_addr(a, field, tmp);
                a.ins(&format!("{l} {v}, 0({tmp})"));
                v
            }
            Cls::Const => {
                a.ins(&format!("{l} {v}, {field}({PC})"));
                v
            }
            Cls::Acc => FACC,
            Cls::L0 => FL0R,
            Cls::L1 => FL1R,
        }
    }

    /// Land a float result. An f32 producer leaves NaN-boxed upper bits in
    /// the register, so the f32 case goes out through a GPR to keep the
    /// slot's zero-extension convention.
    fn finish_fp(&self, a: &mut Asm, dc: DstCls, w32: bool) {
        if dc == DstCls::Acc {
            return;
        }
        let v = self.fp_target(dc);
        self.slot_addr(a, 24, T5);
        if w32 {
            a.ins(&format!("fmv.x.w {T6}, {v}"));
            a.ins(&format!("slli {T6}, {T6}, 32"));
            a.ins(&format!("srli {T6}, {T6}, 32"));
            a.ins(&format!("sd {T6}, 0({T5})"));
        } else {
            a.ins(&format!("fsd {v}, 0({T5})"));
        }
    }

    /// Both compare operands, width- and sign-adjusted. On RV64 an i32
    /// SIGNED compare needs sign-extended inputs; the unsigned forms work
    /// directly on the zero-extended slot value.
    fn cmp_operands(
        &self,
        a: &mut Asm,
        v: &Variant,
        w32: bool,
        signed: bool,
        kind: &str,
    ) -> (&'static str, &'static str) {
        if matches!(kind, "and_ne" | "and_eq") {
            let x = self.src(a, v.a, 8, T1);
            let y = self.src(a, v.b, 16, T2);
            a.ins(&format!("and {T3}, {x}, {y}"));
            if self.rv64() {
                a.ins(&format!("sext.w {T3}, {T3}"));
            }
            return (T3, "zero");
        }
        if !self.rv64() && !w32 {
            // RV32 keeps only 64-bit equality, which folds the pair into
            // one difference word.
            let (xl, xh) = self.pair(a, v.a, 8, T1, T2);
            a.ins(&format!("mv {A4}, {xl}"));
            a.ins(&format!("mv {A5}, {xh}"));
            let (yl, yh) = self.pair(a, v.b, 16, T1, T2);
            a.ins(&format!("xor {A4}, {A4}, {yl}"));
            a.ins(&format!("xor {A5}, {A5}, {yh}"));
            a.ins(&format!("or {A4}, {A4}, {A5}"));
            return (A4, "zero");
        }
        let x = self.src(a, v.a, 8, T1);
        let y = self.src(a, v.b, 16, T2);
        if self.rv64() && w32 && signed {
            a.ins(&format!("sext.w {T3}, {x}"));
            a.ins(&format!("sext.w {T4}, {y}"));
            return (T3, T4);
        }
        (x, y)
    }
}

impl Isa for RiscV {
    fn caps(&self) -> Caps {
        if self.rv64() {
            Caps {
                classes: &CLASSES,
                dsts: &DSTS,
                ptr_bytes: 8,
                has_l1: true,
                has_float_regs: self.fp,
                float_pin_f32: false,
                native_calls: true,
            }
        } else {
            // One pinned local on RV32: a second would take a second
            // register PAIR and roughly double the emitted blob, which is
            // flash on the profiles that select this width.
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
    }

    fn emit_prelude(&mut self, a: &mut Asm, st: &Stubs) {
        // `global_asm!` assembles against the target's BASE ISA, not the
        // triple's full feature string, so the extensions this backend
        // emits have to be requested explicitly. Without this, `mul` and
        // every `f`/`d` instruction are rejected on rv64gc.
        if self.fp {
            a.raw("\t.option arch, +m, +f, +d");
        } else {
            a.raw("\t.option arch, +m");
        }
        let w = self.wsz();
        let (lp, sp) = (self.lp(), self.sp());
        let frame = if self.rv64() { 128 } else { 64 };
        let saved = [
            "ra", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        ];

        a.label(&st.exit_common);
        a.ins(&format!("{sp} {T1}, 0({STATE})")); // state.reason
        a.ins(&format!("{sp} {PC}, 8({STATE})"));
        a.ins(&format!("{sp} {FRAME}, 16({STATE})"));
        a.ins(&format!("{sp} {RETSP}, 56({STATE})"));
        a.ins(&format!("{sp} {DCNT}, 80({STATE})"));
        // The accumulator crosses activation boundaries, so it goes out at
        // full value width even on RV32.
        a.ins(&format!("{sp} {ACC}, 104({STATE})"));
        if !self.rv64() {
            a.ins(&format!("sw {ACC_HI}, 108({STATE})"));
        }
        for (i, reg) in saved.iter().enumerate() {
            a.ins(&format!("{lp} {reg}, {}(sp)", i as u32 * w));
        }
        if self.fp {
            a.ins("fld fs0, 104(sp)");
            a.ins("fld fs1, 112(sp)");
            a.ins("fld fs2, 120(sp)");
        }
        a.ins(&format!("addi sp, sp, {frame}"));
        a.ins("ret");

        for (label, reason) in [
            (&st.slow, 1u32),
            (&st.return_exit, 2),
            (&st.trap_oob, 16),
            (&st.trap_exhaust, 17),
        ] {
            a.label(label);
            a.ins(&format!("li {T1}, {reason}"));
            a.ins(&format!("j {}", st.exit_common));
        }

        // ---- entry trampoline: extern "C" fn(*mut EnterState) ----
        a.label(&st.entry);
        a.ins(&format!("addi sp, sp, -{frame}"));
        for (i, reg) in saved.iter().enumerate() {
            a.ins(&format!("{sp} {reg}, {}(sp)", i as u32 * w));
        }
        if self.fp {
            a.ins("fsd fs0, 104(sp)");
            a.ins("fsd fs1, 112(sp)");
            a.ins("fsd fs2, 120(sp)");
        }
        a.ins(&format!("mv {STATE}, a0"));
        a.ins(&format!("{lp} {PC}, 8({STATE})"));
        a.ins(&format!("{lp} {FRAME}, 16({STATE})"));
        a.ins(&format!("{lp} {MEM}, 24({STATE})"));
        a.ins(&format!("{lp} {MEMLEN}, 32({STATE})"));
        a.ins(&format!("{lp} {CODE}, 40({STATE})"));
        a.ins(&format!("{lp} {GLOB}, 48({STATE})"));
        a.ins(&format!("{lp} {RETSP}, 56({STATE})"));
        a.ins(&format!("{lp} {RETLIM}, 64({STATE})"));
        a.ins(&format!("{lp} {STKLIM}, 72({STATE})"));
        a.ins(&format!("{lp} {DCNT}, 80({STATE})"));
        a.ins(&format!("{lp} {L0R}, 88({STATE})"));
        a.ins(&format!("{lp} {ACC}, 104({STATE})"));
        if self.rv64() {
            a.ins(&format!("{lp} {L1R}, 96({STATE})"));
        } else {
            a.ins(&format!("lw {L0_HI}, 92({STATE})"));
            a.ins(&format!("lw {ACC_HI}, 108({STATE})"));
        }
        if self.fp {
            a.ins(&format!("fld {FL0R}, 88({STATE})"));
            a.ins(&format!("fld {FL1R}, 96({STATE})"));
        }
        a.ins(&format!("{lp} {T1}, 0({PC})"));
        a.ins(&format!("jr {T1}"));

        if !self.rv64() {
            // RV32 links both call flavours to the slow stub; these labels
            // exist only because the meta table names them.
            a.label(&st.call);
            a.label(&st.call_indirect);
            a.label(&st.call_core);
            a.ins(&format!("j {}", st.slow));
            return;
        }

        // ---- Call (wired by the cross-function fixup, not the table) ----
        a.label(&st.call);
        a.ins(&format!("addi {DCNT}, {DCNT}, 1"));
        a.ins(&format!("{lp} {A2}, 8({PC})")); // a-equiv
        a.ins(&format!("{lp} {A3}, 16({PC})")); // b-equiv
        a.ins(&format!("{lp} {T6}, 24({PC})")); // c-equiv
        a.label(&st.call_core);
        self.emit_call_core(a, st);

        a.label(&st.call_indirect);
        a.ins(&format!("addi {DCNT}, {DCNT}, 1"));
        self.emit_call_indirect(a, st);
    }

    fn wants(&self, v: &Variant) -> bool {
        use Op::*;
        // The bit-manipulation ops need Zbb, in neither baseline.
        if matches!(
            v.op,
            I32_Clz | I64_Clz | I32_Ctz | I64_Ctz | I32_Popcnt | I64_Popcnt
        ) {
            return false;
        }
        // RISC-V has no float-to-float rounding instruction at all.
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
        ) {
            return false;
        }
        if !self.fp && is_float_op(v.op) {
            return false;
        }
        if !self.rv64() && !rv32_native(v.op) {
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
        let lp = self.lp();
        match v.op {
            Return => return self.emit_return(a, v),
            Br => {
                self.bump(a, v.counted);
                a.ins(&format!("{lp} {PC}, 24({PC})")); // c is absolute
                a.ins(&format!("{lp} {T1}, 0({PC})"));
                a.ins(&format!("jr {T1}"));
                return;
            }
            BrIf | BrIfNot => {
                self.pre(a);
                // Target and its handler word load at entry: the two are
                // dependent, so on the taken path they otherwise sit
                // nose-to-tail on the critical path.
                a.ins(&format!("{lp} {T5}, 24({PC})"));
                a.ins(&format!("{lp} {T6}, 0({T5})"));
                let ra = self.src(a, v.a, 8, T1);
                let nt = a.fresh("nt");
                if self.rv64() {
                    a.ins(&format!("sext.w {T2}, {ra}"));
                } else {
                    a.ins(&format!("mv {T2}, {ra}"));
                }
                a.ins(&format!(
                    "{} {T2}, zero, {nt}",
                    if v.op == BrIf { "beq" } else { "bne" }
                ));
                self.bump(a, v.counted);
                a.ins(&format!("mv {PC}, {T5}"));
                a.ins(&format!("jr {T6}"));
                a.label(&nt);
                self.tail(a, v.counted);
                return;
            }
            I32_SubBrIf => {
                a.ins(&format!("{lp} {T5}, 24({PC})"));
                a.ins(&format!("{lp} {T6}, 0({T5})"));
                let x = self.src(a, v.a, 8, T1);
                let y = self.src(a, v.b, 16, T2);
                let rd = match v.a {
                    Cls::Slot => ACC,
                    Cls::L0 => L0R,
                    Cls::L1 => L1R,
                    Cls::Const | Cls::Acc => unreachable!(),
                };
                a.ins(&format!("sub {rd}, {x}, {y}"));
                if self.rv64() {
                    // RV64 arithmetic sign-extends through bit 31; frame
                    // slots use the engine-wide zero-extended i32 form.
                    a.ins(&format!("slli {rd}, {rd}, 32"));
                    a.ins(&format!("srli {rd}, {rd}, 32"));
                } else {
                    let hi = if v.a == Cls::L0 { L0_HI } else { ACC_HI };
                    a.ins(&format!("li {hi}, 0"));
                }
                self.slot_addr(a, 8, T3);
                a.ins(&format!("{} {rd}, 0({T3})", self.sp()));
                if !self.rv64() {
                    a.ins(&format!("sw zero, 4({T3})"));
                }
                let nt = a.fresh("nt");
                a.ins(&format!("beq {rd}, zero, {nt}"));
                self.bump(a, v.counted);
                a.ins(&format!("mv {PC}, {T5}"));
                a.ins(&format!("jr {T6}"));
                a.label(&nt);
                self.pre(a);
                self.tail(a, v.counted);
                return;
            }
            I64_SubBrIf => {
                a.ins(&format!("{lp} {T5}, 24({PC})"));
                a.ins(&format!("{lp} {T6}, 0({T5})"));
                let nz = if self.rv64() {
                    let x = self.src(a, v.a, 8, T1);
                    let y = self.src(a, v.b, 16, T2);
                    a.ins(&format!("sub {A2}, {x}, {y}"));
                    self.slot_addr(a, 8, T3);
                    a.ins(&format!("sd {A2}, 0({T3})"));
                    match v.a {
                        Cls::L0 => a.ins(&format!("mv {L0R}, {A2}")),
                        Cls::L1 => a.ins(&format!("mv {L1R}, {A2}")),
                        Cls::Slot => {}
                        Cls::Const | Cls::Acc => unreachable!(),
                    }
                    A2
                } else {
                    let (xl, xh) = self.pair(a, v.a, 8, T1, T2);
                    let (yl, yh) = self.pair(a, v.b, 16, T3, T4);
                    a.ins(&format!("sub {A2}, {xl}, {yl}"));
                    a.ins(&format!("sltu {A4}, {xl}, {yl}")); // borrow
                    a.ins(&format!("sub {A3}, {xh}, {yh}"));
                    a.ins(&format!("sub {A3}, {A3}, {A4}"));
                    self.slot_addr(a, 8, T3);
                    a.ins(&format!("sw {A2}, 0({T3})"));
                    a.ins(&format!("sw {A3}, 4({T3})"));
                    match v.a {
                        Cls::L0 => {
                            a.ins(&format!("mv {L0R}, {A2}"));
                            a.ins(&format!("mv {L0_HI}, {A3}"));
                        }
                        Cls::Slot => {}
                        Cls::Const | Cls::Acc | Cls::L1 => unreachable!(),
                    }
                    a.ins(&format!("or {A4}, {A2}, {A3}"));
                    A4
                };
                let nt = a.fresh("nt");
                a.ins(&format!("beq {nz}, zero, {nt}"));
                self.bump(a, v.counted);
                a.ins(&format!("mv {PC}, {T5}"));
                a.ins(&format!("jr {T6}"));
                a.label(&nt);
                self.pre(a);
                self.tail(a, v.counted);
                return;
            }
            BrTable => {
                self.bump(a, v.counted);
                let ra = self.src(a, v.a, 8, T1);
                a.ins(&format!("{lp} {T2}, 16({PC})")); // flat table
                a.ins(&format!("{lp} {T3}, 24({PC})")); // len - 1
                if self.rv64() {
                    a.ins(&format!("slli {T4}, {ra}, 32"));
                    a.ins(&format!("srli {T4}, {T4}, 32"));
                } else {
                    a.ins(&format!("mv {T4}, {ra}"));
                }
                // Clamp: any out-of-range index picks the default, which
                // the link pass parked in the last slot.
                let keep = a.fresh("bt");
                a.ins(&format!("bltu {T4}, {T3}, {keep}"));
                a.ins(&format!("mv {T4}, {T3}"));
                a.label(&keep);
                a.ins(&format!("slli {T4}, {T4}, 2"));
                a.ins(&format!("add {T4}, {T2}, {T4}"));
                a.ins(&format!("lw {T4}, 0({T4})")); // target cell byte offset
                a.ins(&format!("add {PC}, {CODE}, {T4}"));
                a.ins(&format!("{lp} {T1}, 0({PC})"));
                a.ins(&format!("jr {T1}"));
                return;
            }
            MemoryFill | MemoryCopy => return self.emit_bulk(a, st, v),
            _ => {}
        }
        if let Some((kind, w32, signed)) = int_cmp(v.op) {
            if super::layout::family(v.op) == Fam::SrcAB {
                self.pre(a);
                a.ins(&format!("{lp} {T5}, 24({PC})"));
                a.ins(&format!("{lp} {T6}, 0({T5})"));
                let (x, y) = self.cmp_operands(a, v, w32, signed, kind);
                let nt = a.fresh("nt");
                let (br, bx, by) = branch_form(kind, x, y);
                a.ins(&format!("{} {bx}, {by}, {nt}", invert_branch(br)));
                self.bump(a, v.counted);
                a.ins(&format!("mv {PC}, {T5}"));
                a.ins(&format!("jr {T6}"));
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
                    Cls::Acc | Cls::L0 | Cls::L1 => {
                        let (lo, hi) = self.pair(a, v.a, 8, T1, T2);
                        if lo != rd {
                            a.ins(&format!("mv {rd}, {lo}"));
                            if !self.rv64() {
                                a.ins(&format!("mv {rdh}, {hi}"));
                            }
                        }
                    }
                    Cls::Const => {
                        a.ins(&format!("{lp} {rd}, 8({PC})"));
                        if !self.rv64() {
                            a.ins(&format!("lw {rdh}, 12({PC})"));
                        }
                    }
                    Cls::Slot => {
                        self.slot_addr(a, 8, T1);
                        a.ins(&format!("{lp} {rd}, 0({T1})"));
                        if !self.rv64() {
                            a.ins(&format!("lw {rdh}, 4({T1})"));
                        }
                    }
                }
                self.finish(a, v.d, rd, true);
            }
            MovPair => {
                // Strictly ordered: commit dst1 (including its pinned
                // register, when present) before reading src2.
                let (v1, v1h) = self.pair(a, v.a, 8, T1, T2);
                if self.rv64() {
                    a.ins(&format!("{lp} {T3}, 24({PC})"));
                    a.ins(&format!("srli {T3}, {T3}, 32"));
                } else {
                    a.ins(&format!("lw {T3}, 28({PC})"));
                }
                a.ins(&format!("add {T3}, {FRAME}, {T3}"));
                a.ins(&format!("{} {v1}, 0({T3})", self.sp()));
                if !self.rv64() {
                    a.ins(&format!("sw {v1h}, 4({T3})"));
                }
                if let Some(d1) = v.pair_d.first() {
                    let rd = self.dst_target(d1);
                    if rd != v1 {
                        a.ins(&format!("mv {rd}, {v1}"));
                        if !self.rv64() {
                            a.ins(&format!("mv {}, {v1h}", self.dst_hi(d1)));
                        }
                    }
                }
                let (v2, v2h) = self.pair(a, v.b, 16, ACC, ACC_HI);
                if v2 != ACC {
                    a.ins(&format!("mv {ACC}, {v2}"));
                    if !self.rv64() {
                        a.ins(&format!("mv {ACC_HI}, {v2h}"));
                    }
                }
                a.ins(&format!("{lp} {T5}, 24({PC})"));
                if self.rv64() {
                    a.ins(&format!("slli {T5}, {T5}, 32"));
                    a.ins(&format!("srli {T5}, {T5}, 32"));
                }
                a.ins(&format!("add {T5}, {FRAME}, {T5}"));
                a.ins(&format!("{} {ACC}, 0({T5})", self.sp()));
                if !self.rv64() {
                    a.ins(&format!("sw {ACC_HI}, 4({T5})"));
                }
                if let Some(d2) = v.pair_d.second() {
                    let rd = self.dst_target(d2);
                    if rd != ACC {
                        a.ins(&format!("mv {rd}, {ACC}"));
                        if !self.rv64() {
                            a.ins(&format!("mv {}, {ACC_HI}", self.dst_hi(d2)));
                        }
                    }
                }
            }
            Select => {
                // The condition is always a materialized i32 slot.
                if self.rv64() {
                    a.ins(&format!("{lp} {T6}, 24({PC})"));
                    a.ins(&format!("srli {T6}, {T6}, 32"));
                } else {
                    a.ins(&format!("lw {T6}, 28({PC})"));
                }
                a.ins(&format!("add {T6}, {FRAME}, {T6}"));
                a.ins(&format!("lw {T6}, 0({T6})"));
                let rd = self.dst_target(v.d);
                let rdh = self.dst_hi(v.d);
                let takeb = a.fresh("selb");
                let done = a.fresh("seldone");
                a.ins(&format!("beq {T6}, zero, {takeb}"));
                let (x, xh) = self.pair(a, v.a, 8, T1, T2);
                a.ins(&format!("mv {rd}, {x}"));
                if !self.rv64() {
                    a.ins(&format!("mv {rdh}, {xh}"));
                }
                a.ins(&format!("j {done}"));
                a.label(&takeb);
                let (y, yh) = self.pair(a, v.b, 16, T1, T2);
                a.ins(&format!("mv {rd}, {y}"));
                if !self.rv64() {
                    a.ins(&format!("mv {rdh}, {yh}"));
                }
                a.label(&done);
                if v.d != DstCls::Acc {
                    a.ins(&format!("{lp} {T5}, 24({PC})"));
                    if self.rv64() {
                        a.ins(&format!("slli {T5}, {T5}, 32"));
                        a.ins(&format!("srli {T5}, {T5}, 32"));
                    }
                    a.ins(&format!("add {T5}, {FRAME}, {T5}"));
                    a.ins(&format!("{} {rd}, 0({T5})", self.sp()));
                    if !self.rv64() {
                        a.ins(&format!("sw {rdh}, 4({T5})"));
                    }
                }
            }
            GlobalGet => {
                a.ins(&format!("{lp} {T1}, 8({PC})"));
                a.ins(&format!("add {T1}, {GLOB}, {T1}"));
                let rd = self.dst_target(v.d);
                a.ins(&format!("{lp} {rd}, 0({T1})"));
                if !self.rv64() {
                    a.ins(&format!("lw {}, 4({T1})", self.dst_hi(v.d)));
                }
                self.finish(a, v.d, rd, true);
            }
            GlobalSet => {
                let (x, xh) = self.pair(a, v.a, 8, T1, T2);
                a.ins(&format!("{lp} {T5}, 24({PC})"));
                a.ins(&format!("add {T5}, {GLOB}, {T5}"));
                a.ins(&format!("{} {x}, 0({T5})", self.sp()));
                if !self.rv64() {
                    a.ins(&format!("sw {xh}, 4({T5})"));
                }
            }
            I32_Eqz | I64_Eqz => {
                let rd = self.dst_target(v.d);
                if v.op == I32_Eqz {
                    let ra = self.src(a, v.a, 8, T1);
                    if self.rv64() {
                        a.ins(&format!("sext.w {T2}, {ra}"));
                        a.ins(&format!("seqz {rd}, {T2}"));
                    } else {
                        a.ins(&format!("seqz {rd}, {ra}"));
                    }
                } else if self.rv64() {
                    let ra = self.src(a, v.a, 8, T1);
                    a.ins(&format!("seqz {rd}, {ra}"));
                } else {
                    let (lo, hi) = self.pair(a, v.a, 8, T1, T2);
                    a.ins(&format!("or {T3}, {lo}, {hi}"));
                    a.ins(&format!("seqz {rd}, {T3}"));
                }
                if !self.rv64() {
                    a.ins(&format!("li {}, 0", self.dst_hi(v.d)));
                }
                self.finish(a, v.d, rd, false);
            }
            I32_WrapI64 => {
                let ra = self.src(a, v.a, 8, T1);
                let rd = self.dst_target(v.d);
                a.ins(&format!("mv {rd}, {ra}"));
                self.zext32(a, rd, v.d);
                self.finish(a, v.d, rd, false);
            }
            I64_ExtendI32U => {
                let ra = self.src(a, v.a, 8, T1);
                let rd = self.dst_target(v.d);
                a.ins(&format!("mv {rd}, {ra}"));
                if self.rv64() {
                    a.ins(&format!("slli {rd}, {rd}, 32"));
                    a.ins(&format!("srli {rd}, {rd}, 32"));
                } else {
                    a.ins(&format!("li {}, 0", self.dst_hi(v.d)));
                }
                self.finish(a, v.d, rd, true);
            }
            I64_ExtendI32S | I64_Extend32S => {
                let ra = self.src(a, v.a, 8, T1);
                let rd = self.dst_target(v.d);
                if self.rv64() {
                    a.ins(&format!("sext.w {rd}, {ra}"));
                } else {
                    a.ins(&format!("mv {rd}, {ra}"));
                    a.ins(&format!("srai {}, {rd}, 31", self.dst_hi(v.d)));
                }
                self.finish(a, v.d, rd, true);
            }
            I32_Extend8S | I32_Extend16S | I64_Extend8S | I64_Extend16S => {
                let ra = self.src(a, v.a, 8, T1);
                let rd = self.dst_target(v.d);
                let sh = if matches!(v.op, I32_Extend8S | I64_Extend8S) {
                    self.xlen - 8
                } else {
                    self.xlen - 16
                };
                a.ins(&format!("slli {rd}, {ra}, {sh}"));
                a.ins(&format!("srai {rd}, {rd}, {sh}"));
                let wide = matches!(v.op, I64_Extend8S | I64_Extend16S);
                if wide {
                    if !self.rv64() {
                        a.ins(&format!("srai {}, {rd}, 31", self.dst_hi(v.d)));
                    }
                } else {
                    self.zext32(a, rd, v.d);
                }
                self.finish(a, v.d, rd, wide);
            }
            I32_DivS | I32_DivU | I32_RemS | I32_RemU | I64_DivS | I64_DivU | I64_RemS
            | I64_RemU => self.emit_div(a, st, v),
            _ => {
                if let Some((kind, w32)) = int_bin(v.op) {
                    self.emit_bin(a, kind, w32, v);
                } else if let Some((kind, w32, signed)) = int_cmp(v.op) {
                    let (x, y) = self.cmp_operands(a, v, w32, signed, kind);
                    let rd = self.dst_target(v.d);
                    emit_setcc(a, kind, rd, x, y, T5);
                    if !self.rv64() {
                        a.ins(&format!("li {}, 0", self.dst_hi(v.d)));
                    }
                    self.finish(a, v.d, rd, false);
                } else if mem_kind(v.op).is_some() {
                    self.emit_mem(a, st, v);
                } else {
                    self.emit_float(a, st, v);
                }
            }
        }
        self.tail(a, v.counted);
    }
}

impl RiscV {
    fn emit_bin(&mut self, a: &mut Asm, kind: &str, w32: bool, v: &Variant) {
        let rd = self.dst_target(v.d);
        if !self.rv64() && !w32 {
            // The RV32 64-bit set: add and sub with carry, plus bitwise.
            let (xl, xh) = self.pair(a, v.a, 8, T1, T2);
            a.ins(&format!("mv {A4}, {xl}"));
            a.ins(&format!("mv {A5}, {xh}"));
            let (yl, yh) = self.pair(a, v.b, 16, T1, T2);
            let rdh = self.dst_hi(v.d);
            match kind {
                "add" => {
                    a.ins(&format!("add {T3}, {A4}, {yl}"));
                    a.ins(&format!("sltu {T4}, {T3}, {A4}")); // carry out
                    a.ins(&format!("add {A5}, {A5}, {yh}"));
                    a.ins(&format!("add {rdh}, {A5}, {T4}"));
                    a.ins(&format!("mv {rd}, {T3}"));
                }
                "sub" => {
                    a.ins(&format!("sltu {T4}, {A4}, {yl}")); // borrow
                    a.ins(&format!("sub {T3}, {A4}, {yl}"));
                    a.ins(&format!("sub {A5}, {A5}, {yh}"));
                    a.ins(&format!("sub {rdh}, {A5}, {T4}"));
                    a.ins(&format!("mv {rd}, {T3}"));
                }
                "and" | "or" | "xor" => {
                    a.ins(&format!("{kind} {T3}, {A4}, {yl}"));
                    a.ins(&format!("{kind} {rdh}, {A5}, {yh}"));
                    a.ins(&format!("mv {rd}, {T3}"));
                }
                other => panic!("rv32: no 64-bit form for {other}"),
            }
            self.finish(a, v.d, rd, true);
            return;
        }
        let x = self.src(a, v.a, 8, T1);
        let y = self.src(a, v.b, 16, T2);
        let mut needs_zext = w32;
        match kind {
            "and" | "or" | "xor" => {
                // Zero-extended operands stay zero-extended, so the 32-bit
                // case needs no fixup at all.
                a.ins(&format!("{kind} {rd}, {x}, {y}"));
                needs_zext = false;
                if !self.rv64() && w32 {
                    a.ins(&format!("li {}, 0", self.dst_hi(v.d)));
                }
            }
            "sll" | "srl" | "sra" => {
                let bits = if w32 { 31 } else { self.xlen - 1 };
                a.ins(&format!("andi {T3}, {y}, {bits}"));
                let src = if kind == "sra" && w32 && self.rv64() {
                    a.ins(&format!("sext.w {T4}, {x}"));
                    T4
                } else {
                    x
                };
                a.ins(&format!("{kind} {rd}, {src}, {T3}"));
                if !self.rv64() {
                    needs_zext = false;
                    a.ins(&format!("li {}, 0", self.dst_hi(v.d)));
                }
            }
            "rotr" | "rotl" => {
                // No Zbb: two shifts and an or. The amounts are masked, so
                // a zero rotate takes the `x >> 0 | x << 0` path and
                // reproduces the operand.
                let bits = if w32 { 31 } else { self.xlen - 1 };
                a.ins(&format!("andi {T3}, {y}, {bits}"));
                let src = if w32 && self.rv64() {
                    a.ins(&format!("slli {T4}, {x}, 32"));
                    a.ins(&format!("srli {T4}, {T4}, 32"));
                    T4
                } else {
                    x
                };
                a.ins(&format!("sub {T5}, zero, {T3}"));
                a.ins(&format!("andi {T5}, {T5}, {bits}"));
                let (first, second) = if kind == "rotr" {
                    ("srl", "sll")
                } else {
                    ("sll", "srl")
                };
                a.ins(&format!("{first} {T6}, {src}, {T3}"));
                a.ins(&format!("{second} {A4}, {src}, {T5}"));
                a.ins(&format!("or {rd}, {T6}, {A4}"));
                if !self.rv64() {
                    needs_zext = false;
                    a.ins(&format!("li {}, 0", self.dst_hi(v.d)));
                }
            }
            _ => {
                a.ins(&format!("{kind} {rd}, {x}, {y}"));
                if !self.rv64() {
                    needs_zext = false;
                    a.ins(&format!("li {}, 0", self.dst_hi(v.d)));
                }
            }
        }
        if needs_zext {
            self.zext32(a, rd, v.d);
        }
        self.finish(a, v.d, rd, !w32);
    }

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
        let x = self.src(a, v.a, 8, T1);
        let y = self.src(a, v.b, 16, T2);
        // RISC-V division never faults: divide-by-zero yields all-ones and
        // MIN/-1 wraps. Both are wasm traps, so both are detected here and
        // bail to the shared executor, which raises the right error.
        let (xs, ys) = if w32 && self.rv64() {
            a.ins(&format!("sext.w {T3}, {x}"));
            a.ins(&format!("sext.w {T4}, {y}"));
            (T3, T4)
        } else {
            (x, y)
        };
        self.br_far(a, "beq", ys, "zero", &st.slow);
        if signed {
            let go = a.fresh("div");
            a.ins(&format!("li {T5}, -1"));
            a.ins(&format!("bne {ys}, {T5}, {go}"));
            let sh = if w32 { 31 } else { self.xlen - 1 };
            a.ins(&format!("li {T6}, 1"));
            a.ins(&format!("slli {T6}, {T6}, {sh}"));
            if w32 && self.rv64() {
                a.ins(&format!("sext.w {T6}, {T6}"));
            }
            self.br_far(a, "beq", xs, T6, &st.slow);
            a.label(&go);
        }
        let rd = self.dst_target(v.d);
        let m = match (signed, rem, w32 && self.rv64()) {
            (true, false, true) => "divw",
            (false, false, true) => "divuw",
            (true, true, true) => "remw",
            (false, true, true) => "remuw",
            (true, false, false) => "div",
            (false, false, false) => "divu",
            (true, true, false) => "rem",
            (false, true, false) => "remu",
        };
        a.ins(&format!("{m} {rd}, {xs}, {ys}"));
        if w32 {
            self.zext32(a, rd, v.d);
        }
        self.finish(a, v.d, rd, !w32);
    }

    fn emit_mem(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let (size, load, kind) = mem_kind(v.op).unwrap();
        let fp = is_fp_mem(v.op);
        let lp = self.lp();
        let lw32 = self.lw32();
        // Read the address as 32 bits so the bounds check does not depend
        // on the slot's zero-extension holding.
        let addr = match v.a {
            Cls::Slot => {
                self.slot_addr(a, 8, T1);
                a.ins(&format!("{lw32} {T1}, 0({T1})"));
                T1
            }
            Cls::Const => {
                a.ins(&format!("{lw32} {T1}, 8({PC})"));
                T1
            }
            _ => {
                let s = self.src(a, v.a, 8, T1);
                if self.rv64() {
                    a.ins(&format!("slli {T1}, {s}, 32"));
                    a.ins(&format!("srli {T1}, {T1}, 32"));
                    T1
                } else {
                    s
                }
            }
        };
        if v.fused {
            // ea = zext32(addr1 + addr2) + static offset.
            if self.rv64() {
                a.ins(&format!("{lp} {T5}, 24({PC})"));
                a.ins(&format!("srli {T3}, {T5}, 32"));
                a.ins(&format!("slli {T5}, {T5}, 32"));
                a.ins(&format!("srli {T5}, {T5}, 32"));
            } else {
                a.ins(&format!("lw {T3}, 28({PC})"));
                a.ins(&format!("lw {T5}, 24({PC})"));
            }
            a.ins(&format!("add {T3}, {FRAME}, {T3}"));
            a.ins(&format!("{lw32} {T3}, 0({T3})")); // addr2
            a.ins(&format!("add {T3}, {addr}, {T3}"));
            if self.rv64() {
                a.ins(&format!("slli {T3}, {T3}, 32"));
                a.ins(&format!("srli {T3}, {T3}, 32")); // wrapping i32 sum
            }
            if load {
                a.ins(&format!("{lp} {T2}, 16({PC})")); // static offset
                a.ins(&format!("add {T2}, {T2}, {T3}")); // ea
            } else {
                a.ins(&format!("add {T2}, {T5}, {T3}")); // ea; offset is in T5
            }
            self.carry_guard(a, T2, T3, &st.trap_oob);
        } else {
            a.ins(&format!("{lp} {T2}, {}({PC})", if load { 16 } else { 24 }));
            a.ins(&format!("add {T2}, {T2}, {addr}"));
            self.carry_guard(a, T2, addr, &st.trap_oob);
        }
        a.ins(&format!("addi {T4}, {T2}, {size}"));
        self.carry_guard(a, T4, T2, &st.trap_oob);
        self.br_far(a, "bltu", MEMLEN, T4, &st.trap_oob);
        a.ins(&format!("add {T2}, {MEM}, {T2}"));
        if load {
            if fp {
                let vt = self.fp_target(v.d);
                a.ins(&format!(
                    "{} {vt}, 0({T2})",
                    if kind == 0 { "flw" } else { "fld" }
                ));
                if v.d != DstCls::Acc {
                    if v.fused {
                        a.ins(&format!("add {T5}, {FRAME}, {T5}"));
                    } else {
                        self.slot_addr(a, 24, T5);
                    }
                    if kind == 0 {
                        a.ins(&format!("fmv.x.w {T6}, {vt}"));
                        a.ins(&format!("slli {T6}, {T6}, 32"));
                        a.ins(&format!("srli {T6}, {T6}, 32"));
                        a.ins(&format!("sd {T6}, 0({T5})"));
                    } else {
                        a.ins(&format!("fsd {vt}, 0({T5})"));
                    }
                }
                return;
            }
            let rd = self.dst_target(v.d);
            let rdh = self.dst_hi(v.d);
            let wide = matches!(kind, 1 | 6 | 7 | 8) || v.op == Op::I64_Load32U;
            if !self.rv64() && kind == 1 {
                a.ins(&format!("lw {rd}, 0({T2})"));
                a.ins(&format!("lw {rdh}, 4({T2})"));
            } else {
                let m = match kind {
                    0 => lw32,
                    1 => "ld",
                    2 => "lbu",
                    3 => "lhu",
                    4 | 6 => "lb",
                    5 | 7 => "lh",
                    8 => "lw",
                    _ => unreachable!(),
                };
                a.ins(&format!("{m} {rd}, 0({T2})"));
                // The sign-extending narrow loads must land zero-extended
                // when the wasm result type is i32.
                if matches!(kind, 4 | 5) {
                    self.zext32(a, rd, v.d);
                } else if !self.rv64() {
                    let hi = if matches!(kind, 6 | 7) { "srai" } else { "li" };
                    if hi == "srai" {
                        a.ins(&format!("srai {rdh}, {rd}, 31"));
                    } else {
                        a.ins(&format!("li {rdh}, 0"));
                    }
                }
            }
            if v.fused {
                if v.d != DstCls::Acc {
                    a.ins(&format!("add {T5}, {FRAME}, {T5}"));
                    a.ins(&format!("{} {rd}, 0({T5})", self.sp()));
                    if !self.rv64() {
                        let hi = if wide { rdh } else { "zero" };
                        a.ins(&format!("sw {hi}, 4({T5})"));
                    }
                }
            } else {
                self.finish(a, v.d, rd, wide);
            }
        } else if fp && v.b.is_reg() {
            let vs = match v.b {
                Cls::L0 => FL0R,
                Cls::L1 => FL1R,
                _ => FACC,
            };
            a.ins(&format!(
                "{} {vs}, 0({T2})",
                if kind == 0 { "fsw" } else { "fsd" }
            ));
        } else if !self.rv64() && kind == 1 {
            let (lo, hi) = self.pair(a, v.b, 16, T5, T6);
            a.ins(&format!("sw {lo}, 0({T2})"));
            a.ins(&format!("sw {hi}, 4({T2})"));
        } else {
            let rb = self.src(a, v.b, 16, T5);
            let m = match kind {
                0 => "sw",
                1 => "sd",
                2 => "sb",
                3 => "sh",
                _ => unreachable!(),
            };
            a.ins(&format!("{m} {rb}, 0({T2})"));
        }
    }

    fn emit_float(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        use Op::*;
        let dc = v.d;
        if let Some((m, w32)) = float_bin(v.op) {
            let sfx = if w32 { "s" } else { "d" };
            let x = self.src_fp(a, v.a, w32, "ft0", 8, T1);
            let y = self.src_fp(a, v.b, w32, "ft1", 16, T2);
            a.ins(&format!("{m}.{sfx} {}, {x}, {y}", self.fp_target(dc)));
            self.finish_fp(a, dc, w32);
            return;
        }
        if let Some((m, w32, swap, invert)) = float_cmp(v.op) {
            let sfx = if w32 { "s" } else { "d" };
            let x = self.src_fp(a, v.a, w32, "ft0", 8, T1);
            let y = self.src_fp(a, v.b, w32, "ft1", 16, T2);
            let (l, r) = if swap { (y, x) } else { (x, y) };
            let rd = self.dst_target(dc);
            a.ins(&format!("{m}.{sfx} {rd}, {l}, {r}"));
            if invert {
                a.ins(&format!("xori {rd}, {rd}, 1"));
            }
            self.finish(a, dc, rd, false);
            return;
        }
        match v.op {
            F32_Abs | F64_Abs | F32_Neg | F64_Neg | F32_Sqrt | F64_Sqrt => {
                let w32 = matches!(v.op, F32_Abs | F32_Neg | F32_Sqrt);
                let sfx = if w32 { "s" } else { "d" };
                let m = match v.op {
                    F32_Abs | F64_Abs => "fabs",
                    F32_Neg | F64_Neg => "fneg",
                    _ => "fsqrt",
                };
                let x = self.src_fp(a, v.a, w32, "ft0", 8, T1);
                a.ins(&format!("{m}.{sfx} {}, {x}", self.fp_target(dc)));
                self.finish_fp(a, dc, w32);
            }
            F32_Copysign | F64_Copysign => {
                // `fsgnj` IS wasm copysign, bit for bit.
                let w32 = v.op == F32_Copysign;
                let sfx = if w32 { "s" } else { "d" };
                let x = self.src_fp(a, v.a, w32, "ft0", 8, T1);
                let y = self.src_fp(a, v.b, w32, "ft1", 16, T2);
                a.ins(&format!("fsgnj.{sfx} {}, {x}, {y}", self.fp_target(dc)));
                self.finish_fp(a, dc, w32);
            }
            F32_Min | F32_Max | F64_Min | F64_Max => {
                // RISC-V `fmin`/`fmax` return the NON-NaN operand when only
                // one is a NaN; wasm wants a NaN whenever either is. The
                // signed-zero rule already matches, so only the NaN case
                // needs its own path.
                let w32 = matches!(v.op, F32_Min | F32_Max);
                let sfx = if w32 { "s" } else { "d" };
                let m = if matches!(v.op, F32_Min | F64_Min) {
                    "fmin"
                } else {
                    "fmax"
                };
                let x = self.src_fp(a, v.a, w32, "ft0", 8, T1);
                let y = self.src_fp(a, v.b, w32, "ft1", 16, T2);
                let vt = self.fp_target(dc);
                let nan = a.fresh("fmnan");
                let done = a.fresh("fmdone");
                a.ins(&format!("feq.{sfx} {T3}, {x}, {x}"));
                a.ins(&format!("feq.{sfx} {T4}, {y}, {y}"));
                a.ins(&format!("and {T3}, {T3}, {T4}"));
                a.ins(&format!("beq {T3}, zero, {nan}"));
                a.ins(&format!("{m}.{sfx} {vt}, {x}, {y}"));
                a.ins(&format!("j {done}"));
                a.label(&nan);
                // Adding quiets a signalling NaN and propagates a quiet one.
                a.ins(&format!("fadd.{sfx} {vt}, {x}, {y}"));
                a.label(&done);
                self.finish_fp(a, dc, w32);
            }
            F32_DemoteF64 => {
                let x = self.src_fp(a, v.a, false, "ft0", 8, T1);
                a.ins(&format!("fcvt.s.d {}, {x}", self.fp_target(dc)));
                self.finish_fp(a, dc, true);
            }
            F64_PromoteF32 => {
                let x = self.src_fp(a, v.a, true, "ft0", 8, T1);
                a.ins(&format!("fcvt.d.s {}, {x}", self.fp_target(dc)));
                self.finish_fp(a, dc, false);
            }
            I32_ReinterpretF32 | I64_ReinterpretF64 => {
                let w32 = v.op == I32_ReinterpretF32;
                let rd = self.dst_target(dc);
                if v.a.is_reg() {
                    let x = self.src_fp(a, v.a, w32, "ft0", 8, T1);
                    a.ins(&format!(
                        "{} {rd}, {x}",
                        if w32 { "fmv.x.w" } else { "fmv.x.d" }
                    ));
                    if w32 {
                        a.ins(&format!("slli {rd}, {rd}, 32"));
                        a.ins(&format!("srli {rd}, {rd}, 32"));
                    }
                } else {
                    let x = self.src(a, v.a, 8, T1);
                    a.ins(&format!("mv {rd}, {x}"));
                }
                self.finish(a, dc, rd, !w32);
            }
            F32_ReinterpretI32 | F64_ReinterpretI64 => {
                let w32 = v.op == F32_ReinterpretI32;
                let x = self.src(a, v.a, 8, T1);
                a.ins(&format!(
                    "{} {}, {x}",
                    if w32 { "fmv.w.x" } else { "fmv.d.x" },
                    self.fp_target(dc)
                ));
                self.finish_fp(a, dc, w32);
            }
            _ => {
                if let Some((uns, src64, dst32)) = cvt_i2f(v.op) {
                    let x = self.src(a, v.a, 8, T1);
                    let s = match (src64, uns) {
                        (false, false) => "w",
                        (false, true) => "wu",
                        (true, false) => "l",
                        (true, true) => "lu",
                    };
                    a.ins(&format!(
                        "fcvt.{}.{s} {}, {x}",
                        if dst32 { "s" } else { "d" },
                        self.fp_target(dc)
                    ));
                    self.finish_fp(a, dc, dst32);
                } else if let Some((src32, to64, uns)) = cvt_f2i(v.op) {
                    let sfx = if src32 { "s" } else { "d" };
                    let x = self.src_fp(a, v.a, src32, "ft0", 8, T1);
                    let rd = self.dst_target(dc);
                    let w = match (to64, uns) {
                        (false, false) => "w",
                        (false, true) => "wu",
                        (true, false) => "l",
                        (true, true) => "lu",
                    };
                    if let Some((lo, hi)) = trap_bounds(v.op) {
                        // Exclusive bounds on the un-truncated operand, so
                        // a bail is always a trap.
                        a.ins(&format!("feq.{sfx} {T3}, {x}, {x}"));
                        self.br_far(a, "beq", T3, "zero", &st.slow); // NaN
                        a.ins(&format!("li {T4}, {}", lo as i64));
                        a.ins(&format!(
                            "{} ft1, {T4}",
                            if src32 { "fmv.w.x" } else { "fmv.d.x" }
                        ));
                        a.ins(&format!("fle.{sfx} {T3}, {x}, ft1"));
                        self.br_far(a, "bne", T3, "zero", &st.slow);
                        a.ins(&format!("li {T4}, {}", hi as i64));
                        a.ins(&format!(
                            "{} ft1, {T4}",
                            if src32 { "fmv.w.x" } else { "fmv.d.x" }
                        ));
                        a.ins(&format!("fle.{sfx} {T3}, ft1, {x}"));
                        self.br_far(a, "bne", T3, "zero", &st.slow);
                        a.ins(&format!("fcvt.{w}.{sfx} {rd}, {x}, rtz"));
                    } else {
                        // Saturating: RISC-V already clamps at the bounds;
                        // only NaN differs (it yields the maximum where
                        // wasm wants zero), so mask by "operand is ordered".
                        a.ins(&format!("fcvt.{w}.{sfx} {rd}, {x}, rtz"));
                        a.ins(&format!("feq.{sfx} {T3}, {x}, {x}"));
                        a.ins(&format!("sub {T3}, zero, {T3}"));
                        a.ins(&format!("and {rd}, {rd}, {T3}"));
                    }
                    if !to64 {
                        a.ins(&format!("slli {rd}, {rd}, 32"));
                        a.ins(&format!("srli {rd}, {rd}, 32"));
                    }
                    self.finish(a, dc, rd, to64);
                } else {
                    panic!("riscv: no handler shape for {:?}", v.op);
                }
            }
        }
    }

    /// The shared activation entry. RV64 only: `a2` = a-equiv, `a3` =
    /// b-equiv, `t6` = c-equiv, each the same packed word every backend's
    /// call cells carry.
    fn emit_call_core(&mut self, a: &mut Asm, st: &Stubs) {
        self.br_far(a, "bgeu", RETSP, RETLIM, &st.trap_exhaust);
        a.ins(&format!("slli {T1}, {A3}, 33"));
        a.ins(&format!("srli {T1}, {T1}, 33")); // arg_base*8 (bit 31 is fp)
        a.ins(&format!("add {T1}, {FRAME}, {T1}")); // new frame base
        a.ins(&format!("srli {T2}, {T6}, 32"));
        a.ins(&format!("slli {T2}, {T2}, 48"));
        a.ins(&format!("srli {T2}, {T2}, 48")); // frame_slots
        a.ins(&format!("slli {T2}, {T2}, 3"));
        a.ins(&format!("add {T2}, {T1}, {T2}"));
        self.br_far(a, "bltu", STKLIM, T2, &st.trap_exhaust);
        // push (ret_pc, caller frame, code | caller_l0off<<48, caller_l1off)
        a.ins(&format!("addi {T3}, {PC}, 32"));
        a.ins(&format!("sd {T3}, 0({RETSP})"));
        a.ins(&format!("sd {FRAME}, 8({RETSP})"));
        a.ins(&format!("srli {T3}, {T6}, 48"));
        a.ins(&format!("slli {T3}, {T3}, 48"));
        a.ins(&format!("or {T3}, {T3}, {CODE}"));
        a.ins(&format!("sd {T3}, 16({RETSP})"));
        a.ins(&format!("srli {T3}, {A2}, 48"));
        a.ins(&format!("sd {T3}, 24({RETSP})"));
        a.ins(&format!("addi {RETSP}, {RETSP}, 32"));
        a.ins(&format!("mv {FRAME}, {T1}"));
        // zero the fresh locals: [n_params*8, n_locals*8)
        a.ins(&format!("slli {T1}, {T6}, 48"));
        a.ins(&format!("srli {T1}, {T1}, 48")); // n_params
        a.ins(&format!("slli {T1}, {T1}, 3"));
        a.ins(&format!("add {T1}, {FRAME}, {T1}"));
        a.ins(&format!("srli {T2}, {T6}, 16"));
        a.ins(&format!("slli {T2}, {T2}, 48"));
        a.ins(&format!("srli {T2}, {T2}, 48")); // n_locals
        a.ins(&format!("slli {T2}, {T2}, 3"));
        a.ins(&format!("add {T2}, {FRAME}, {T2}"));
        let zl = a.fresh("zl");
        let zdone = a.fresh("zdone");
        a.label(&zl);
        a.ins(&format!("bgeu {T1}, {T2}, {zdone}"));
        a.ins(&format!("sd zero, 0({T1})"));
        a.ins(&format!("addi {T1}, {T1}, 8"));
        a.ins(&format!("j {zl}"));
        a.label(&zdone);
        a.ins(&format!("srli {T3}, {A3}, 32"));
        a.ins(&format!("slli {T3}, {T3}, 48"));
        a.ins(&format!("srli {T3}, {T3}, 48")); // callee l0off
        a.ins(&format!("add {T3}, {FRAME}, {T3}"));
        a.ins(&format!("ld {L0R}, 0({T3})"));
        a.ins(&format!("srli {T4}, {A3}, 48")); // callee l1off
        a.ins(&format!("add {T4}, {FRAME}, {T4}"));
        a.ins(&format!("ld {L1R}, 0({T4})"));
        let cont = a.fresh("callcont");
        if self.fp {
            // Float twins only when the callee has float-pinned slots
            // (cell b bit 31). Integer code falls through; the transfer
            // block sits out of line.
            let fp = a.fresh("callfp");
            a.ins(&format!("slli {T5}, {A3}, 32"));
            a.ins(&format!("srli {T5}, {T5}, 63"));
            a.ins(&format!("bne {T5}, zero, {fp}"));
            a.label(&cont);
            a.ins(&format!("slli {CODE}, {A2}, 16"));
            a.ins(&format!("srli {CODE}, {CODE}, 16"));
            a.ins(&format!("mv {PC}, {CODE}"));
            a.ins(&format!("ld {T1}, 0({PC})"));
            a.ins(&format!("jr {T1}"));
            a.label(&fp);
            a.ins(&format!("fmv.d.x {FL0R}, {L0R}"));
            a.ins(&format!("fmv.d.x {FL1R}, {L1R}"));
            a.ins(&format!("j {cont}"));
        } else {
            a.ins(&format!("slli {CODE}, {A2}, 16"));
            a.ins(&format!("srli {CODE}, {CODE}, 16"));
            a.ins(&format!("mv {PC}, {CODE}"));
            a.ins(&format!("ld {T1}, 0({PC})"));
            a.ins(&format!("jr {T1}"));
        }
    }

    /// RV64 only. Table 0's base and length and the per-function info
    /// table come from the entry state, refreshed on every chain entry, so
    /// `table.grow` and `table.set` need no invalidation protocol. Every
    /// guard failure bails to the slow stub, which re-executes the cell and
    /// raises the proper trap or routes the host call.
    fn emit_call_indirect(&mut self, a: &mut Asm, st: &Stubs) {
        a.ins(&format!("ld {T1}, 8({PC})")); // cell a
        a.ins(&format!("slli {T2}, {T1}, 32"));
        a.ins(&format!("srli {T2}, {T2}, 32")); // index_slot*8
        a.ins(&format!("add {T2}, {FRAME}, {T2}"));
        a.ins(&format!("lwu {T2}, 0({T2})")); // t
        a.ins(&format!("ld {T3}, 120({STATE})")); // table 0 length
        self.br_far(a, "bgeu", T2, T3, &st.slow);
        a.ins(&format!("ld {T3}, 112({STATE})")); // table 0 entries
                                                  // Entries are `RefHandle` slots (8 bytes); a plain handle's payload
                                                  // is the function index. Null is all-ones and a tagged handle has
                                                  // high bits set, so both exceed u32::MAX and take the slow path.
        a.ins(&format!("slli {T4}, {T2}, 3"));
        a.ins(&format!("add {T3}, {T3}, {T4}"));
        a.ins(&format!("ld {T4}, 0({T3})")); // fi (all-ones = null)
        a.ins(&format!("li {T5}, -1"));
        self.br_far(a, "beq", T4, T5, &st.slow);
        a.ins(&format!("ld {T5}, 128({STATE})")); // info base
        a.ins(&format!("slli {T6}, {T4}, 1"));
        a.ins(&format!("add {T6}, {T6}, {T4}")); // fi*3
        a.ins(&format!("slli {T6}, {T6}, 3"));
        a.ins(&format!("add {T5}, {T5}, {T6}")); // entry
        a.ins(&format!("ld {A3}, 8({T5})")); // l1off<<48|l0off<<32|canon
        a.ins(&format!("ld {A2}, 16({PC})")); // cell b
        a.ins(&format!("slli {T3}, {A3}, 32"));
        a.ins(&format!("srli {T3}, {T3}, 32")); // canonical actual
        a.ins(&format!("srli {T4}, {A2}, 32"));
        a.ins(&format!("slli {T4}, {T4}, 48"));
        a.ins(&format!("srli {T4}, {T4}, 48")); // canonical expected
        self.br_far(a, "bne", T3, T4, &st.slow);
        a.ins(&format!("ld {T6}, 0({T5})")); // callee cells | fp flag
        self.br_far(a, "beq", T6, "zero", &st.slow);
        a.ins(&format!("ld {T5}, 16({T5})")); // frame metadata
                                              // compose the call_core inputs
        a.ins(&format!("andi {T4}, {T6}, 1")); // callee fp flag
        a.ins(&format!("srli {T6}, {T6}, 5"));
        a.ins(&format!("slli {T6}, {T6}, 5")); // clean cells address
        a.ins(&format!("srli {T3}, {T1}, 48")); // caller l1off
        a.ins(&format!("slli {T3}, {T3}, 48"));
        a.ins(&format!("or {T1}, {T6}, {T3}")); // a-equiv (staged in t1)
        a.ins(&format!("srli {T3}, {A3}, 32"));
        a.ins(&format!("slli {A3}, {T3}, 32")); // callee l0/l1, canon cleared
        a.ins(&format!("slli {T3}, {A2}, 32"));
        a.ins(&format!("srli {T3}, {T3}, 32")); // arg_base*8
        a.ins(&format!("or {A3}, {A3}, {T3}"));
        a.ins(&format!("slli {T4}, {T4}, 31"));
        a.ins(&format!("or {A3}, {A3}, {T4}")); // b-equiv
        a.ins(&format!("srli {T3}, {A2}, 48"));
        a.ins(&format!("slli {T3}, {T3}, 48")); // caller l0off
        a.ins(&format!("or {T6}, {T3}, {T5}")); // c-equiv
        a.ins(&format!("mv {A2}, {T1}")); // a-equiv into place
        a.ins(&format!("j {}", st.call_core));
    }

    fn emit_return(&mut self, a: &mut Asm, v: &Variant) {
        let lp = self.lp();
        self.bump(a, v.counted);
        a.ins(&format!("{lp} {T1}, 8({PC})")); // first-result slot*8
        a.ins(&format!("{lp} {T2}, 16({PC})")); // result count
        a.ins(&format!("add {T1}, {FRAME}, {T1}"));
        a.ins(&format!("mv {T3}, {FRAME}"));
        let pop = a.fresh("pop");
        let cl = a.fresh("cl");
        a.ins(&format!("beq {T2}, zero, {pop}"));
        // The accumulator doubles as the copy scratch: after a
        // single-result copy it holds result 0, which is the
        // call-result-in-acc convention at zero extra instructions.
        a.label(&cl);
        if self.rv64() {
            a.ins(&format!("ld {ACC}, 0({T1})"));
            a.ins(&format!("sd {ACC}, 0({T3})"));
        } else {
            a.ins(&format!("lw {ACC}, 0({T1})"));
            a.ins(&format!("lw {ACC_HI}, 4({T1})"));
            a.ins(&format!("sw {ACC}, 0({T3})"));
            a.ins(&format!("sw {ACC_HI}, 4({T3})"));
        }
        a.ins(&format!("addi {T1}, {T1}, 8"));
        a.ins(&format!("addi {T3}, {T3}, 8"));
        a.ins(&format!("addi {T2}, {T2}, -1"));
        a.ins(&format!("bne {T2}, zero, {cl}"));
        a.label(&pop);
        a.ins(&format!("addi {RETSP}, {RETSP}, -32"));
        a.ins(&format!("{lp} {T1}, 0({RETSP})")); // ret pc
        a.ins(&format!("{lp} {FRAME}, 8({RETSP})"));
        a.ins(&format!("{lp} {T3}, 24({RETSP})")); // caller l1off
        if self.rv64() {
            a.ins(&format!("ld {T2}, 16({RETSP})")); // code | caller_l0off<<48
            a.ins(&format!("slli {CODE}, {T2}, 16"));
            a.ins(&format!("srli {CODE}, {CODE}, 16"));
            a.ins(&format!("srli {T2}, {T2}, 48")); // caller l0off, bit 0 = fp
        } else {
            // On RV32 the packed word straddles two machine words: the
            // cell base is the low half and the caller's l0 offset the
            // high half's low 16 bits.
            a.ins(&format!("lw {CODE}, 16({RETSP})"));
            a.ins(&format!("lw {T2}, 20({RETSP})"));
            a.ins(&format!("slli {T2}, {T2}, 16"));
            a.ins(&format!("srli {T2}, {T2}, 16"));
        }
        let join = a.fresh("retjoin");
        // Sentinel records carry a readable dummy frame, so these loads are
        // always safe. The caller's float-pinned flag rides bit 0 of its
        // recorded l0 offset (offsets are byte-scaled, so bit 0 is free).
        if self.fp {
            let fp = a.fresh("retfp");
            a.ins(&format!("andi {T4}, {T2}, 1"));
            a.ins(&format!("bne {T4}, zero, {fp}"));
            a.ins(&format!("add {T4}, {FRAME}, {T2}"));
            a.ins(&format!("{lp} {L0R}, 0({T4})"));
            a.ins(&format!("add {T4}, {FRAME}, {T3}"));
            a.ins(&format!("{lp} {L1R}, 0({T4})"));
            a.label(&join);
            a.ins(&format!("mv {PC}, {T1}"));
            a.ins(&format!("{lp} {T5}, 0({PC})"));
            a.ins(&format!("jr {T5}"));
            a.label(&fp);
            a.ins(&format!("addi {T2}, {T2}, -1"));
            a.ins(&format!("add {T4}, {FRAME}, {T2}"));
            a.ins(&format!("{lp} {L0R}, 0({T4})"));
            a.ins(&format!("add {T4}, {FRAME}, {T3}"));
            a.ins(&format!("{lp} {L1R}, 0({T4})"));
            a.ins(&format!("fmv.d.x {FL0R}, {L0R}"));
            a.ins(&format!("fmv.d.x {FL1R}, {L1R}"));
            a.ins(&format!("j {join}"));
        } else {
            a.ins(&format!("add {T4}, {FRAME}, {T2}"));
            a.ins(&format!("{lp} {L0R}, 0({T4})"));
            if self.rv64() {
                a.ins(&format!("add {T4}, {FRAME}, {T3}"));
                a.ins(&format!("{lp} {L1R}, 0({T4})"));
            } else {
                a.ins(&format!("lw {L0_HI}, 4({T4})"));
            }
            a.ins(&format!("mv {PC}, {T1}"));
            a.ins(&format!("{lp} {T5}, 0({PC})"));
            a.ins(&format!("jr {T5}"));
        }
    }

    /// `memory.fill` / `memory.copy` on memory 0, a machine word at a time
    /// with a byte tail. Neither baseline has a wider move: RISC-V's vector
    /// extension is not in RV64GC or RV32IMAC.
    fn emit_bulk(&mut self, a: &mut Asm, st: &Stubs, v: &Variant) {
        let fill = v.op == Op::MemoryFill;
        let lp = self.lp();
        let w = self.wsz();
        self.pre(a);
        a.ins(&format!("{lp} {T1}, 8({PC})"));
        a.ins(&format!("add {T1}, {FRAME}, {T1}"));
        a.ins(&format!("{lp} {T2}, 0({T1})")); // d
        a.ins(&format!("{lp} {T3}, 8({T1})")); // fill: value, copy: s
        a.ins(&format!("{lp} {T4}, 16({T1})")); // n
        a.ins(&format!("add {T5}, {T2}, {T4}"));
        self.carry_guard(a, T5, T4, &st.slow);
        self.br_far(a, "bltu", MEMLEN, T5, &st.slow);
        let lwl = a.fresh("lw");
        let bytes = a.fresh("bytes");
        let done = a.fresh("done");
        if fill {
            a.ins(&format!("andi {T3}, {T3}, 255"));
            a.ins(&format!("slli {T6}, {T3}, 8"));
            a.ins(&format!("or {T3}, {T3}, {T6}"));
            a.ins(&format!("slli {T6}, {T3}, 16"));
            a.ins(&format!("or {T3}, {T3}, {T6}"));
            if self.rv64() {
                a.ins(&format!("slli {T6}, {T3}, 32"));
                a.ins(&format!("or {T3}, {T3}, {T6}"));
            }
            a.ins(&format!("add {T2}, {MEM}, {T2}"));
            a.ins(&format!("add {T5}, {MEM}, {T5}"));
            a.label(&lwl);
            a.ins(&format!("addi {T6}, {T2}, {w}"));
            a.ins(&format!("bltu {T5}, {T6}, {bytes}"));
            a.ins(&format!("{} {T3}, 0({T2})", self.sp()));
            a.ins(&format!("mv {T2}, {T6}"));
            a.ins(&format!("j {lwl}"));
            a.label(&bytes);
            a.ins(&format!("bgeu {T2}, {T5}, {done}"));
            a.ins(&format!("sb {T3}, 0({T2})"));
            a.ins(&format!("addi {T2}, {T2}, 1"));
            a.ins(&format!("j {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
        } else {
            a.ins(&format!("add {T5}, {T3}, {T4}"));
            self.carry_guard(a, T5, T4, &st.slow);
            self.br_far(a, "bltu", MEMLEN, T5, &st.slow);
            let back = a.fresh("copyback");
            let fwd = a.fresh("copyfwd");
            // A forward copy is wrong only when s < d < s+n.
            a.ins(&format!("bgeu {T3}, {T2}, {fwd}"));
            a.ins(&format!("bltu {T2}, {T5}, {back}"));
            a.label(&fwd);
            a.ins(&format!("add {T2}, {MEM}, {T2}"));
            a.ins(&format!("add {T3}, {MEM}, {T3}"));
            a.ins(&format!("add {T5}, {T2}, {T4}"));
            a.label(&lwl);
            a.ins(&format!("addi {T6}, {T2}, {w}"));
            a.ins(&format!("bltu {T5}, {T6}, {bytes}"));
            a.ins(&format!("{lp} {A4}, 0({T3})"));
            a.ins(&format!("{} {A4}, 0({T2})", self.sp()));
            a.ins(&format!("addi {T3}, {T3}, {w}"));
            a.ins(&format!("mv {T2}, {T6}"));
            a.ins(&format!("j {lwl}"));
            a.label(&bytes);
            a.ins(&format!("bgeu {T2}, {T5}, {done}"));
            a.ins(&format!("lbu {A4}, 0({T3})"));
            a.ins(&format!("sb {A4}, 0({T2})"));
            a.ins(&format!("addi {T2}, {T2}, 1"));
            a.ins(&format!("addi {T3}, {T3}, 1"));
            a.ins(&format!("j {bytes}"));
            a.label(&done);
            self.tail(a, v.counted);
            // Overlapping-downward block, out of line past the tail.
            let bw = a.fresh("bw");
            let bb = a.fresh("bb");
            let bdone = a.fresh("bdone");
            a.label(&back);
            a.ins(&format!("add {T2}, {MEM}, {T2}"));
            a.ins(&format!("add {T3}, {MEM}, {T3}"));
            a.label(&bw);
            a.ins(&format!("li {T6}, {w}"));
            a.ins(&format!("bltu {T4}, {T6}, {bb}"));
            a.ins(&format!("addi {T4}, {T4}, -{w}"));
            a.ins(&format!("add {T6}, {T3}, {T4}"));
            a.ins(&format!("{lp} {A4}, 0({T6})"));
            a.ins(&format!("add {T6}, {T2}, {T4}"));
            a.ins(&format!("{} {A4}, 0({T6})", self.sp()));
            a.ins(&format!("j {bw}"));
            a.label(&bb);
            a.ins(&format!("beq {T4}, zero, {bdone}"));
            a.ins(&format!("addi {T4}, {T4}, -1"));
            a.ins(&format!("add {T6}, {T3}, {T4}"));
            a.ins(&format!("lbu {A4}, 0({T6})"));
            a.ins(&format!("add {T6}, {T2}, {T4}"));
            a.ins(&format!("sb {A4}, 0({T6})"));
            a.ins(&format!("j {bb}"));
            a.label(&bdone);
            self.tail(a, v.counted);
        }
    }
}
