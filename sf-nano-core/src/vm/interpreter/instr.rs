//! Instruction format of the folded stack machine.
//!
//! Fixed-width 32-byte cells, two per cache line (design doc §6):
//!
//! ```text
//! stage A: [ op:u16 | flags:u16 | pad:u32 | a:u64 | b:u64 | c:u64 ]
//! ```
//!
//! `a`/`b` are source operands: a frame-slot index, or an inline 64-bit
//! constant when the matching `FLAG_*_CONST` bit is set. Locals and
//! materialized temps share one flat frame (`[params|locals|temps]`), so a
//! folded local operand and a temp operand are both plain slot indices —
//! the distinction exists only at predecode time.
//!
//! `c` is the destination slot for value-producing ops (dst-folding
//! retro-patches it to a local slot), the branch target (instruction
//! index) for control ops, or op-specific payload as documented on [`Op`].

/// Source operand `a` is an inline constant, not a frame-slot index.
pub const FLAG_A_CONST: u16 = 1 << 0;
/// Source operand `b` is an inline constant, not a frame-slot index.
pub const FLAG_B_CONST: u16 = 1 << 1;
/// Acc residency hints (design doc §8): the producer immediately before
/// this instruction keeps its result in the accumulator register, and this
/// instruction reads it from there. Slot fields stay valid — a backend is
/// free to ignore the hints (and strips them when either side of a pair
/// lacks an acc-capable handler), and the slow path always uses the slots.
pub(crate) const FLAG_A_ACC: u16 = 1 << 2;
pub(crate) const FLAG_B_ACC: u16 = 1 << 3;
/// This instruction's result lives in the accumulator; the frame-slot
/// write is skipped by acc-honoring backends.
pub(crate) const FLAG_DST_ACC: u16 = 1 << 4;
/// Address-add fusion: this memory op's effective address is the
/// wrapping i32 sum of TWO slot operands (the census's universal
/// base+index pattern). Loads pack `addr2 << 32 | dst` in c; stores
/// pack `addr2 << 32 | static_offset` (offset < 2^32, memory 0 only).
pub(crate) const FLAG_FUSED: u16 = 1 << 5;

/// Value-domain classification for accumulator pairing: the accumulator
/// is a PER-DOMAIN register (x8 for integer bits, a NEON register for
/// float values), so a producer/consumer pair is only valid when the
/// produced value's domain matches the register the consumer reads.
/// Domain-agnostic bit movers (MovSlot, Select, globals) classify as
/// integer: float values flowing through them simply fall back to slots.
pub(crate) fn result_is_float(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    (d >= F32_Abs as u16 && d <= F32_Copysign as u16)
        || (d >= F64_Abs as u16 && d <= F64_Copysign as u16)
        || (d >= F32_ConvertI32S as u16 && d <= F64_PromoteF32 as u16)
        || matches!(
            op,
            F32_ReinterpretI32 | F64_ReinterpretI64 | F32_Load | F64_Load
        )
}

/// Whether the a (is_b = false) or b (is_b = true) operand of `op` is
/// read in the float domain.
pub(crate) fn operand_is_float(op: Op, is_b: bool) -> bool {
    use Op::*;
    let d = op as u16;
    // float arithmetic, unary, and compares read float operands
    if (d >= F32_Abs as u16 && d <= F32_Ge as u16) || (d >= F64_Abs as u16 && d <= F64_Ge as u16) {
        return true;
    }
    if is_b {
        // a store's b operand is the stored value
        return matches!(op, F32_Store | F64_Store);
    }
    (d >= I32_TruncF32S as u16 && d <= I64_TruncSatF64U as u16)
        || matches!(
            op,
            F32_DemoteF64 | F64_PromoteF32 | I32_ReinterpretF32 | I64_ReinterpretF64
        )
}

/// Semantic operations. One dispatch each; routing opcodes never appear.
///
/// Field conventions unless noted: `a`/`b` sources per flags, `c` = dst slot.
/// Control ops: `c` = target instruction index.
/// `Call`: `a` = function index, `b` = first-argument slot.
/// `CallIndirect`: `a` = table-target operand, `b` = first-argument slot,
/// `c` = type index.
/// `Return`: `a` = first-result slot, `b` = result count.
/// `Select`: `a`/`b` = the two values, `c` = `cond_slot << 32 | dst_slot`
/// (the condition is always materialized).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Op {
    // data movement (materialization / unfolded local.set)
    MovSlot,
    MovConst,

    // i32 ALU
    I32_Add,
    I32_Sub,
    I32_Mul,
    I32_DivS,
    I32_DivU,
    I32_RemS,
    I32_RemU,
    I32_And,
    I32_Or,
    I32_Xor,
    I32_Shl,
    I32_ShrS,
    I32_ShrU,
    I32_Rotl,
    I32_Rotr,
    I32_Clz,
    I32_Ctz,
    I32_Popcnt,
    I32_Extend8S,
    I32_Extend16S,

    // i32 compares (result: 0/1 in dst)
    I32_Eqz,
    I32_Eq,
    I32_Ne,
    I32_LtS,
    I32_LtU,
    I32_GtS,
    I32_GtU,
    I32_LeS,
    I32_LeU,
    I32_GeS,
    I32_GeU,

    // i64 ALU
    I64_Add,
    I64_Sub,
    I64_Mul,
    I64_DivS,
    I64_DivU,
    I64_RemS,
    I64_RemU,
    I64_And,
    I64_Or,
    I64_Xor,
    I64_Shl,
    I64_ShrS,
    I64_ShrU,
    I64_Rotl,
    I64_Rotr,
    I64_Clz,
    I64_Ctz,
    I64_Popcnt,
    I64_Extend8S,
    I64_Extend16S,
    I64_Extend32S,

    // i64 compares
    I64_Eqz,
    I64_Eq,
    I64_Ne,
    I64_LtS,
    I64_LtU,
    I64_GtS,
    I64_GtU,
    I64_LeS,
    I64_LeU,
    I64_GeS,
    I64_GeU,

    // int width conversions
    I32_WrapI64,
    I64_ExtendI32S,
    I64_ExtendI32U,

    // f32 ALU (values are IEEE bits in the low 32 of the slot)
    F32_Abs,
    F32_Neg,
    F32_Ceil,
    F32_Floor,
    F32_Trunc,
    F32_Nearest,
    F32_Sqrt,
    F32_Add,
    F32_Sub,
    F32_Mul,
    F32_Div,
    F32_Min,
    F32_Max,
    F32_Copysign,
    F32_Eq,
    F32_Ne,
    F32_Lt,
    F32_Gt,
    F32_Le,
    F32_Ge,

    // f64 ALU
    F64_Abs,
    F64_Neg,
    F64_Ceil,
    F64_Floor,
    F64_Trunc,
    F64_Nearest,
    F64_Sqrt,
    F64_Add,
    F64_Sub,
    F64_Mul,
    F64_Div,
    F64_Min,
    F64_Max,
    F64_Copysign,
    F64_Eq,
    F64_Ne,
    F64_Lt,
    F64_Gt,
    F64_Le,
    F64_Ge,

    // int <-> float conversions (trapping and saturating forms)
    I32_TruncF32S,
    I32_TruncF32U,
    I32_TruncF64S,
    I32_TruncF64U,
    I64_TruncF32S,
    I64_TruncF32U,
    I64_TruncF64S,
    I64_TruncF64U,
    I32_TruncSatF32S,
    I32_TruncSatF32U,
    I32_TruncSatF64S,
    I32_TruncSatF64U,
    I64_TruncSatF32S,
    I64_TruncSatF32U,
    I64_TruncSatF64S,
    I64_TruncSatF64U,
    F32_ConvertI32S,
    F32_ConvertI32U,
    F32_ConvertI64S,
    F32_ConvertI64U,
    F32_DemoteF64,
    F64_ConvertI32S,
    F64_ConvertI32U,
    F64_ConvertI64S,
    F64_ConvertI64U,
    F64_PromoteF32,
    I32_ReinterpretF32,
    I64_ReinterpretF64,
    F32_ReinterpretI32,
    F64_ReinterpretI64,

    // memory: loads (`a` = address operand, `b` = static offset, `c` = dst)
    I32_Load,
    I64_Load,
    F32_Load,
    F64_Load,
    I32_Load8S,
    I32_Load8U,
    I32_Load16S,
    I32_Load16U,
    I64_Load8S,
    I64_Load8U,
    I64_Load16S,
    I64_Load16U,
    I64_Load32S,
    I64_Load32U,
    // memory: stores (`a` = address operand, `b` = value operand, `c` = offset)
    I32_Store,
    I64_Store,
    F32_Store,
    F64_Store,
    I32_Store8,
    I32_Store16,
    I64_Store8,
    I64_Store16,
    I64_Store32,
    /// `c` = dst.
    MemorySize,
    /// `a` = delta operand, `c` = dst.
    MemoryGrow,
    /// Three materialized operands at consecutive slots starting at `a`
    /// (dst-addr, value, count).
    MemoryFill,
    /// Three materialized operands at consecutive slots starting at `a`
    /// (dst-addr, src-addr, count).
    MemoryCopy,
    /// Like MemoryFill layout (dst-addr, src-offset, count); `b` = data
    /// segment index.
    MemoryInit,
    /// `a` = data segment index.
    DataDrop,

    // globals: `a` = global index (get) / source operand (set), `c` = dst
    // slot (get) / global index (set)
    GlobalGet,
    GlobalSet,

    // reference / table ops (funcref values are function indices, null =
    // u32::MAX, in the low bits of a slot)
    /// `a` = ref operand, `c` = dst.
    RefIsNull,
    /// `a` = index operand, `b` = table index, `c` = dst.
    TableGet,
    /// `a` = index operand, `b` = value operand, `c` = table index.
    TableSet,
    /// `b` = table index, `c` = dst.
    TableSize,
    /// `a` = init-value operand, `b` = delta operand,
    /// `c` = `table_index << 32 | dst`.
    TableGrow,
    /// Three materialized operands at consecutive slots starting at `a`
    /// (start, value, count); `b` = table index.
    TableFill,
    /// Three materialized operands starting at `a` (dst, src, count);
    /// `b` = `dst_table << 32 | src_table`.
    TableCopy,
    /// Three materialized operands starting at `a` (dst, src, count);
    /// `b` = `table_index << 32 | elem_segment`.
    TableInit,
    /// `a` = element segment index.
    ElemDrop,

    // parametric
    Select,

    // control
    Br,
    BrIf,
    /// Branch when the condition is zero (lowered `if`, and the guard of the
    /// general `br_if` form that must move values on the taken path).
    BrIfNot,
    // Fused compare-and-branch (design doc §8 fixed combos): a compare
    // whose sole, same-region consumer is a conditional branch becomes the
    // branch itself. `a`/`b` = compare operands per flags, `c` = target.
    // Inverted senses are closed within the set, so `br_if_not` guards use
    // the opposite compare.
    I32_BrEq,
    I32_BrNe,
    I32_BrLtS,
    I32_BrLtU,
    I32_BrGtS,
    I32_BrGtU,
    I32_BrLeS,
    I32_BrLeU,
    I32_BrGeS,
    I32_BrGeU,
    I64_BrEq,
    I64_BrNe,
    I64_BrLtS,
    I64_BrLtU,
    I64_BrGtS,
    I64_BrGtU,
    I64_BrLeS,
    I64_BrLeU,
    I64_BrGeS,
    I64_BrGeU,
    /// Fused `i32.and` + branch: taken when (a & b) != 0.
    I32_BrAnd,
    /// Fused `i32.and` + `br_if_not`: taken when (a & b) == 0.
    I32_BrAndNot,
    /// `a` = index operand, `c` = side-table id in
    /// `PredecodedFunction::br_tables` (last entry is the default).
    BrTable,
    Return,
    Call,
    CallIndirect,
    /// Two fused materialization movs: `frame[c>>32] = frame[a]` then
    /// `frame[c & 0xffffffff] = frame[b]`, in that order (src2 may be
    /// dst1). Emitted by the predecoder for adjacent staging movs.
    MovPair,
    Unreachable,
}

/// One 32-byte instruction cell. Stage B replaces the leading
/// `op`/`flags`/pad word with the dispatch word (handler pointer or
/// index×stride); the operand layout is shared by both stages.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct Instr {
    pub op: Op,
    pub flags: u16,
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

impl Instr {
    pub fn new(op: Op, flags: u16, a: u64, b: u64, c: u64) -> Self {
        Instr { op, flags, a, b, c }
    }
}
