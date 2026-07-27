//! The handler variant space: operand residency classes, the per-op
//! variant family, and the cell-field transforms the linker applies.
//!
//! This module is compiled TWICE. The crate compiles it so the linker can
//! classify a cell into the variant whose handler it should point at;
//! `build.rs` compiles it (via `#[path]`) so the handler generator can
//! enumerate exactly the variants it must emit. One source of truth is
//! what makes those two agree — a divergence would not fail to build, it
//! would silently demote cells to the slow path or, worse, point a cell at
//! a handler that reads its operands from the wrong place.
//!
//! Handler slots are packed per family rather than allocated as a dense
//! `op x 200` matrix: most ops vary only one or two of the three positions,
//! and the dense form costs ~160 KB of table for ~10.5 k live handlers.

use super::instr::{
    op_from_index, operand_is_float, result_is_float, Instr, Op, FLAG_A_ACC, FLAG_A_CONST,
    FLAG_B_ACC, FLAG_B_CONST, FLAG_DST_ACC, FLAG_FUSED,
};

/// Whether a float-domain RESULT is 32 bits wide. Only meaningful where
/// [`result_is_float`] holds.
pub(crate) fn result_is_f32(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    (d >= F32_Abs as u16 && d <= F32_Copysign as u16)
        || (d >= F32_ConvertI32S as u16 && d <= F32_DemoteF64 as u16)
        || matches!(op, F32_ReinterpretI32 | F32_Load)
}

/// Whether a float-domain OPERAND is 32 bits wide. Only meaningful where
/// [`operand_is_float`] holds.
pub(crate) fn operand_is_f32(op: Op, is_b: bool) -> bool {
    use Op::*;
    let d = op as u16;
    if d >= F32_Abs as u16 && d <= F32_Ge as u16 {
        return true;
    }
    if is_b {
        return op == F32_Store;
    }
    matches!(
        op,
        I32_TruncF32S
            | I32_TruncF32U
            | I64_TruncF32S
            | I64_TruncF32U
            | I32_TruncSatF32S
            | I32_TruncSatF32U
            | I64_TruncSatF32S
            | I64_TruncSatF32U
            | F64_PromoteF32
            | I32_ReinterpretF32
    )
}

/// Bytes per dispatch cell. Two per 64-byte cache line; branch targets and
/// the pc advance are both denominated in it.
pub(crate) const CELL: u32 = 32;

/// Number of distinct `Op` discriminants.
pub(crate) const N_OPS: usize = Op::Unreachable as usize + 1;

/// Operand residency class: where a handler finds one of its inputs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cls {
    /// A frame slot, addressed by the pre-scaled byte offset in the cell.
    Slot,
    /// An inline constant in the cell field.
    Const,
    /// The accumulator register (span-1 producer/consumer edge).
    Acc,
    /// The function's first pinned local.
    L0,
    /// The function's second pinned local.
    L1,
}

/// Destination residency class. `Mem` computes into the accumulator and
/// stores; `Acc` skips the store; `L0`/`L1` compute into the pinned
/// register AND store (write-through — the slot stays authoritative for
/// the slow path and for the reload at every chain entry).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DstCls {
    Mem,
    Acc,
    L0,
    L1,
}

impl Cls {
    pub(crate) fn index(self) -> usize {
        match self {
            Cls::Slot => 0,
            Cls::Const => 1,
            Cls::Acc => 2,
            Cls::L0 => 3,
            Cls::L1 => 4,
        }
    }
}

impl DstCls {
    pub(crate) fn index(self) -> usize {
        match self {
            DstCls::Mem => 0,
            DstCls::Acc => 1,
            DstCls::L0 => 2,
            DstCls::L1 => 3,
        }
    }
}

/// Which operand positions an op's handler set varies over. The family
/// fixes both the number of handler slots the op owns and the index of a
/// given class combination inside them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fam {
    /// No native handler set: the op is permanently slow, or is wired
    /// outside the table (`Call`/`CallIndirect` take callee addresses from
    /// the cross-function fixup pass).
    None,
    /// One handler, no operand classes.
    Fixed,
    /// Destination only.
    Dst,
    /// Destination only; the single source is always an inline constant.
    ConstDst,
    /// Source a only.
    SrcA,
    /// Source a x destination.
    SrcADst,
    /// Sources a x b, no value destination.
    SrcAB,
    /// Sources a x b x destination.
    SrcABDst,
    /// Like `SrcADst`, plus a second bank for address-fused variants
    /// (b is the static offset, not an operand).
    Load,
    /// Like `SrcAB`, plus a second bank for address-fused variants.
    Store,
}

impl Fam {
    /// Handler slots this family owns.
    pub(crate) fn slots(self) -> usize {
        match self {
            Fam::None => 0,
            Fam::Fixed => 1,
            Fam::Dst | Fam::ConstDst => 4,
            Fam::SrcA => 5,
            Fam::SrcADst => 20,
            Fam::SrcAB => 25,
            Fam::SrcABDst => 100,
            Fam::Load => 40,
            Fam::Store => 50,
        }
    }

    /// Whether this family has an address-fused second bank.
    pub(crate) fn has_fused_bank(self) -> bool {
        matches!(self, Fam::Load | Fam::Store)
    }

    /// Index of one class combination inside the family, or `None` when
    /// the combination does not exist in it.
    pub(crate) fn index_of(self, a: Cls, b: Cls, d: DstCls, fused: bool) -> Option<usize> {
        if fused && !self.has_fused_bank() {
            return None;
        }
        let (ai, bi, di) = (a.index(), b.index(), d.index());
        Some(match self {
            Fam::None => return None,
            Fam::Fixed => 0,
            Fam::Dst => di,
            Fam::ConstDst => {
                if a != Cls::Const {
                    return None;
                }
                di
            }
            Fam::SrcA => ai,
            Fam::SrcADst => ai + 5 * di,
            Fam::SrcAB => ai + 5 * bi,
            Fam::SrcABDst => ai + 5 * bi + 25 * di,
            Fam::Load => ai + 5 * di + if fused { 20 } else { 0 },
            Fam::Store => ai + 5 * bi + if fused { 25 } else { 0 },
        })
    }
}

fn between(op: Op, lo: Op, hi: Op) -> bool {
    (op as u16) >= (lo as u16) && (op as u16) <= (hi as u16)
}

/// The variant family of an op. Every backend emits exactly the
/// combinations this enumerates (minus the ones its ISA declines), and the
/// linker looks them up the same way.
pub(crate) fn family(op: Op) -> Fam {
    use Op::*;
    // Integer and float binary value ops: both sources and a destination.
    if between(op, I32_Add, I32_Rotr)
        || between(op, I32_Eq, I32_GeU)
        || between(op, I64_Add, I64_Rotr)
        || between(op, I64_Eq, I64_GeU)
        || between(op, F32_Add, F32_Copysign)
        || between(op, F32_Eq, F32_Ge)
        || between(op, F64_Add, F64_Copysign)
        || between(op, F64_Eq, F64_Ge)
    {
        return Fam::SrcABDst;
    }
    // Unary value ops, width changes, and every conversion.
    if between(op, I32_Clz, I32_Eqz)
        || between(op, I64_Clz, I64_Eqz)
        || between(op, I32_WrapI64, I64_ExtendI32U)
        || between(op, F32_Abs, F32_Sqrt)
        || between(op, F64_Abs, F64_Sqrt)
        || between(op, I32_TruncF32S, F64_ReinterpretI64)
    {
        return Fam::SrcADst;
    }
    if between(op, I32_Load, I64_Load32U) {
        return Fam::Load;
    }
    if between(op, I32_Store, I64_Store32) {
        return Fam::Store;
    }
    if between(op, I32_BrEq, I32_SubBrIf) {
        return Fam::SrcAB;
    }
    match op {
        MovSlot => Fam::SrcADst,
        MovConst => Fam::ConstDst,
        MovPair => Fam::SrcAB,
        Select => Fam::SrcABDst,
        GlobalGet => Fam::Dst,
        GlobalSet | BrIf | BrIfNot | BrTable => Fam::SrcA,
        Br | Return | MemoryFill | MemoryCopy | MemoryFillCopy => Fam::Fixed,
        // Permanently slow (design doc SS12: host calls, memory/table
        // grow and size, segment and reference ops), plus the two call
        // flavours, which the cross-function fixup wires directly.
        _ => Fam::None,
    }
}

/// Base index of an op's handler slots in the packed table. Both the
/// generator and the linker build this the same way, from `family`.
pub(crate) fn build_op_base() -> [u32; N_OPS] {
    let mut base = [0u32; N_OPS];
    let mut next = 0u32;
    for (i, slot) in base.iter_mut().enumerate() {
        *slot = next;
        next += family(op_from_index(i)).slots() as u32;
    }
    base
}

/// Total packed handler slots across every op.
pub(crate) fn total_slots() -> usize {
    let mut n = 0usize;
    for i in 0..N_OPS {
        n += family(op_from_index(i)).slots();
    }
    n
}

/// Which cell fields are frame-slot references for this op:
/// `(a_is_slot_operand, b_is_slot_operand, c_is_value_dst)`. Exactness
/// matters — a wrong `true` mis-classes an operand into a variant nobody
/// emitted and silently demotes the cell to the slow path.
///
/// This is deliberately independent of [`family`] in the two places they
/// disagree: `Select` and the fused memory ops pack their destination into
/// `c` alongside another field, so `c` is not a plain destination slot.
pub(crate) fn slot_fields(op: Op) -> (bool, bool, bool) {
    use Op::*;
    match family(op) {
        Fam::SrcABDst => {
            if op == Select {
                // dst and cond are packed in c; the link guard reads them.
                (true, true, false)
            } else {
                (true, true, true)
            }
        }
        Fam::SrcADst => (true, false, true),
        Fam::ConstDst => (false, false, true),
        Fam::Dst => (false, false, true),
        Fam::SrcA => (true, false, false),
        // Loads: a = address, b = static offset, c = dst.
        Fam::Load => (true, false, true),
        // Stores: a = address, b = value, c = static offset.
        Fam::Store => (true, true, false),
        // Fused compare-branches and MovPair: a, b operands, c = target
        // / packed destinations.
        Fam::SrcAB => (true, true, false),
        Fam::Fixed | Fam::None => (false, false, false),
    }
}

/// Whether an op's native handler leaves its result in the accumulator
/// (every value producer computes into it; see the backend's `finish`).
/// Ops that are never native are harmlessly included — the linker's
/// handler-table lookup is the real gate.
pub(crate) fn writes_acc(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    (d >= MovSlot as u16 && d <= F64_ReinterpretI64 as u16)
        || (d >= I32_Load as u16 && d <= I64_Load32U as u16)
        || matches!(op, GlobalGet | Select)
}

/// Ops whose static offset packs a memory index in the high bits can only
/// run natively against memory 0.
pub(crate) fn native_guard(ins: &Instr) -> bool {
    match family(ins.op) {
        Fam::Load => ins.b >> 48 == 0,
        Fam::Store => ins.c >> 48 == 0,
        // b packs the memory index (copy: dst << 32 | src)
        _ => match ins.op {
            Op::MemoryFill | Op::MemoryCopy => ins.b == 0,
            Op::MemoryFillCopy => ins.c == 0,
            _ => true,
        },
    }
}

/// Whether this op's `c` field is a branch target cell index, which the
/// link pass turns into an absolute cell address. Must list exactly the
/// ops [`transform_bc`] multiplies by [`CELL`] — a missed one would leave
/// a relative offset and the handler would branch into hyperspace.
pub(crate) fn c_is_branch_target(op: Op) -> bool {
    use Op::*;
    matches!(op, Br | BrIf | BrIfNot) || between(op, I32_BrEq, I32_SubBrIf)
}

/// Per-op `b`/`c` pre-scaling for native handlers (`a` is handled
/// uniformly at the call site: slot index x8 unless const). `flags` are
/// the link-resolved flags, with acc hints possibly already stripped.
pub(crate) fn transform_bc(ins: &Instr, flags: u16) -> (u64, u64) {
    use Op::*;
    let scaled_b = |ins: &Instr| {
        if flags & FLAG_B_CONST != 0 {
            ins.b
        } else {
            ins.b * 8
        }
    };
    if flags & FLAG_FUSED != 0 {
        return if family(ins.op) == Fam::Load {
            // loads: b = static offset (raw), c = addr2*8 << 32 | dst*8
            (ins.b, ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8)
        } else {
            // stores: b = value, c = addr2*8 << 32 | static offset (raw)
            (
                scaled_b(ins),
                ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff),
            )
        };
    }
    if c_is_branch_target(ins.op) {
        // control: c = target cell byte offset
        let b = match ins.op {
            Br | BrIf | BrIfNot => ins.b,
            _ => scaled_b(ins),
        };
        return (b, ins.c * CELL as u64);
    }
    match family(ins.op) {
        // loads: b = static offset (stays raw), c = dst slot
        Fam::Load => (ins.b, ins.c * 8),
        // stores: b = value operand, c = static offset (stays raw)
        Fam::Store => (scaled_b(ins), ins.c),
        _ => match ins.op {
            // GlobalSet: c = global index
            GlobalSet => (ins.b, ins.c * 8),
            // MovPair: b = second source slot, c = dst1<<32 | dst2
            MovPair => (
                ins.b * 8,
                ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8,
            ),
            // MemoryFillCopy: a and b are two operand-pack base slots.
            // The linker scales a uniformly; scale the second pack here.
            MemoryFillCopy => (ins.b * 8, ins.c),
            // Return: b = result count, c unused — both stay raw
            Return => (ins.b, ins.c),
            // Select: c = cond_slot << 32 | dst_slot, both scaled
            Select => (
                scaled_b(ins),
                ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8,
            ),
            // plain value ops (incl. GlobalGet): b = operand, c = dst slot
            _ => (scaled_b(ins), ins.c * 8),
        },
    }
}

/// The pinned-local selection a function was linked with. `u64::MAX` means
/// the class is unused; the `_float` flags say which register file is
/// authoritative for that slot.
#[derive(Clone, Copy)]
pub(crate) struct Pinned {
    pub(crate) l0: u64,
    pub(crate) l1: u64,
    pub(crate) l0_float: bool,
    pub(crate) l1_float: bool,
}

impl Pinned {
    pub(crate) const NONE: Pinned = Pinned {
        l0: u64::MAX,
        l1: u64::MAX,
        l0_float: false,
        l1_float: false,
    };
}

/// Classify one cell into its packed handler slot, given link-resolved
/// flags and the function's pinned slots. Pinned classes take precedence
/// over acc hints on the same operand; a `const` flag wins over both.
///
/// Returns `None` when the op has no handler family at all.
pub(crate) fn op_slot(
    ins: &Instr,
    flags: u16,
    pin: &Pinned,
    op_base: &[u32; N_OPS],
) -> Option<usize> {
    let fam = family(ins.op);
    if fam == Fam::None {
        return None;
    }
    let (a_s, b_s, c_d) = slot_fields(ins.op);
    // Domain demotion: a pinned class is taken only when the access's
    // value domain matches the slot's pinned register file; otherwise the
    // access falls back to the (write-through, hence current) slot.
    let af = operand_is_float(ins.op, false);
    let bf = operand_is_float(ins.op, true);
    let rf = result_is_float(ins.op);
    let a = if flags & FLAG_A_CONST != 0 {
        Cls::Const
    } else if a_s && ins.a == pin.l0 && af == pin.l0_float {
        Cls::L0
    } else if a_s && ins.a == pin.l1 && af == pin.l1_float {
        Cls::L1
    } else if flags & FLAG_A_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    };
    let b = if flags & FLAG_B_CONST != 0 {
        Cls::Const
    } else if b_s && ins.b == pin.l0 && bf == pin.l0_float {
        Cls::L0
    } else if b_s && ins.b == pin.l1 && bf == pin.l1_float {
        Cls::L1
    } else if flags & FLAG_B_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    };
    // Select and the fused loads pack their destination in c's low half.
    let packed_dst = ins.op == Op::Select || (flags & FLAG_FUSED != 0 && c_d);
    let dslot = if packed_dst {
        ins.c & 0xffff_ffff
    } else {
        ins.c
    };
    // Select's destination is an integer-domain write: a float-pinned dst
    // slot cannot occur, because a mixed-domain writer makes a slot
    // unpinnable in the first place.
    let dfloat = if ins.op == Op::Select { false } else { rf };
    let d = if (c_d || packed_dst) && dslot == pin.l0 && dfloat == pin.l0_float {
        DstCls::L0
    } else if (c_d || packed_dst) && dslot == pin.l1 && dfloat == pin.l1_float {
        DstCls::L1
    } else if flags & FLAG_DST_ACC != 0 {
        DstCls::Acc
    } else {
        DstCls::Mem
    };
    let idx = fam.index_of(a, b, d, flags & FLAG_FUSED != 0)?;
    Some(op_base[ins.op as usize] as usize + idx)
}
