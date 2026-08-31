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
// The op count belongs to the instruction set, but every table in this
// module is that wide, and the generator reaches it through `layout`.
pub(crate) use super::instr::N_OPS;

/// Bytes per dispatch cell. Two per 64-byte cache line; branch targets and
/// the pc advance are both denominated in it.
pub(crate) const CELL: u32 = 32;

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

/// Residency of `MovPair`'s two ordered destinations. Equal destinations
/// are never paired, so seven states cover every unpinned/L0/L1 mapping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PairDstCls {
    None,
    FirstL0,
    FirstL1,
    SecondL0,
    SecondL1,
    L0L1,
    L1L0,
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

impl PairDstCls {
    pub(crate) fn index(self) -> usize {
        match self {
            PairDstCls::None => 0,
            PairDstCls::FirstL0 => 1,
            PairDstCls::FirstL1 => 2,
            PairDstCls::SecondL0 => 3,
            PairDstCls::SecondL1 => 4,
            PairDstCls::L0L1 => 5,
            PairDstCls::L1L0 => 6,
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
    /// Sources a x b and the seven ordered `MovPair` destination states.
    SrcABPairDst,
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
    pub(crate) const fn slots(self) -> usize {
        match self {
            Fam::None => 0,
            Fam::Fixed => 1,
            Fam::Dst | Fam::ConstDst => 4,
            Fam::SrcA => 5,
            Fam::SrcADst => 20,
            Fam::SrcAB => 25,
            Fam::SrcABPairDst => 175,
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
    pub(crate) fn index_of(
        self,
        a: Cls,
        b: Cls,
        d: DstCls,
        pair_d: PairDstCls,
        fused: bool,
    ) -> Option<usize> {
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
            Fam::SrcABPairDst => ai + 5 * bi + 25 * pair_d.index(),
            Fam::SrcABDst => ai + 5 * bi + 25 * di,
            Fam::Load => ai + 5 * di + if fused { 20 } else { 0 },
            Fam::Store => ai + 5 * bi + if fused { 25 } else { 0 },
        })
    }
}

// `build.rs` includes this file and consumes the generic indexing helper.
// Naming the function item here keeps the runtime crate's dead-code audit
// aware of that second consumer without emitting a call or suppressing lint.
const _: fn(Fam, Cls, Cls, DstCls, PairDstCls, bool) -> Option<usize> = Fam::index_of;

// ---------------------------------------------------------------------------
// The per-op fact table
//
// `family`, `slot_fields`, `writes_acc` and `c_is_branch_target` all answer
// a question about the opcode alone, and the linker asks them repeatedly
// for every cell it builds: `op_slot`, `transform_bc` and `select_pinned`
// each start by classifying the op again. Walking the
// discriminant ranges every time cost 13.7% of interpreter instantiation
// (`family` plus its `between` scans) on a 6 k-function module. The answers
// fit in six bytes per op, so they are computed once at compile time — in
// this crate AND in the build script, which compiles this same file — and
// read by index at run time.
//
// The `compute_*` functions below stay the single definition of each fact;
// nothing outside this table calls them.
// ---------------------------------------------------------------------------

/// `slot_fields().0` — cell field `a` holds a frame-slot operand.
const P_A_SLOT: u8 = 1 << 0;
/// `slot_fields().1` — cell field `b` holds a frame-slot operand.
const P_B_SLOT: u8 = 1 << 1;
/// `slot_fields().2` — cell field `c` holds a plain value destination.
const P_C_DST: u8 = 1 << 2;
const P_WRITES_ACC: u8 = 1 << 3;
const P_C_BRANCH_TARGET: u8 = 1 << 4;

/// How a cell's `b` and `c` fields are pre-scaled for its native handler.
///
/// One value per op, so [`transform_bc`] reads the shape it must apply
/// instead of re-deriving it from `family`, `c_is_branch_target` and the
/// opcode on every cell. The variants name the shape, not the op: several
/// unrelated ops share one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BcShape {
    /// b = operand (scaled unless const), c = destination slot. Also the
    /// shape of every op with no native handler, where it is inert.
    Plain,
    /// `MovSlot`: b mirrors the destination byte offset in c. Its source is
    /// carried by a, so the otherwise-unused b word can pair both offsets
    /// for backends that benefit from an early payload load.
    MirroredDst,
    /// b = static offset (stays raw), c = destination slot.
    LoadOffset,
    /// b = value operand, c = static offset (stays raw).
    StoreOffset,
    /// `br` / `br_if` / `br_if_not`: b stays raw, c = target cell.
    BranchRawB,
    /// Fused compare-and-branch: b = operand, c = target cell.
    BranchScaledB,
    /// b stays raw, c = global index scaled to a byte offset.
    GlobalIndex,
    /// b = second source slot, c = two packed destination slots.
    PackedPairDst,
    /// b = second operand-pack base slot, c stays raw.
    OperandPackB,
    /// b = result count, c unused; both stay raw.
    RawBoth,
    /// b = operand, c = condition and destination slots, both packed.
    PackedCondDst,
}

/// Which cell field carries a memory index that only memory 0 can satisfy
/// natively. Retained only as the independent test oracle for predecode's
/// cached `FLAG_NO_NATIVE` decision.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum MemGuard {
    /// Nothing to check.
    None,
    /// A load's static offset in `b` packs the memory index above bit 47.
    OffsetInB,
    /// A store's static offset in `c` packs it the same way.
    OffsetInC,
    /// `b` is the memory index outright (copy: `dst << 32 | src`).
    IndexInB,
    /// `c` is the memory index outright.
    IndexInC,
}

#[derive(Clone, Copy)]
struct OpProps {
    fam: Fam,
    /// Base index of this op's handler slots in the packed table.
    base: u32,
    bits: u8,
    bc: BcShape,
    #[cfg(test)]
    guard: MemGuard,
}

const OP_PROPS: [OpProps; N_OPS] = build_op_props();

const fn build_op_props() -> [OpProps; N_OPS] {
    let mut props = [OpProps {
        fam: Fam::None,
        base: 0,
        bits: 0,
        bc: BcShape::Plain,
        #[cfg(test)]
        guard: MemGuard::None,
    }; N_OPS];
    let mut i = 0;
    let mut next = 0u32;
    while i < N_OPS {
        let op = op_from_index(i);
        let fam = compute_family(op);
        let (a_s, b_s, c_d) = compute_slot_fields(op);
        let mut bits = 0u8;
        if a_s {
            bits |= P_A_SLOT;
        }
        if b_s {
            bits |= P_B_SLOT;
        }
        if c_d {
            bits |= P_C_DST;
        }
        if compute_writes_acc(op) {
            bits |= P_WRITES_ACC;
        }
        if compute_c_is_branch_target(op) {
            bits |= P_C_BRANCH_TARGET;
        }
        props[i] = OpProps {
            fam,
            base: next,
            bits,
            bc: compute_bc_shape(op),
            #[cfg(test)]
            guard: compute_mem_guard(op),
        };
        next += fam.slots() as u32;
        i += 1;
    }
    props
}

const fn between(op: Op, lo: Op, hi: Op) -> bool {
    (op as u16) >= (lo as u16) && (op as u16) <= (hi as u16)
}

/// The variant family of an op. Every backend emits exactly the
/// combinations this enumerates (minus the ones its ISA declines), and the
/// linker looks them up the same way.
#[inline]
pub(crate) fn family(op: Op) -> Fam {
    OP_PROPS[op as usize].fam
}

const fn compute_family(op: Op) -> Fam {
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
    if between(op, I32_BrEq, I64_SubBrIf) {
        return Fam::SrcAB;
    }
    match op {
        MovSlot => Fam::SrcADst,
        MovConst => Fam::ConstDst,
        MovPair => Fam::SrcABPairDst,
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

/// Base index of an op's handler slots in the packed table. The generator
/// lays the table out in this order and the linker indexes it the same way.
#[inline]
pub(crate) fn op_base(op: Op) -> u32 {
    OP_PROPS[op as usize].base
}

// The build-time handler generator consumes this helper from the same source.
const _: fn(Op) -> u32 = op_base;

/// Total packed handler slots across every op.
pub(crate) const fn total_slots() -> usize {
    let last = OP_PROPS[N_OPS - 1];
    last.base as usize + last.fam.slots()
}

/// Which cell fields are frame-slot references for this op:
/// `(a_is_slot_operand, b_is_slot_operand, c_is_value_dst)`. Exactness
/// matters — a wrong `true` mis-classes an operand into a variant nobody
/// emitted and silently demotes the cell to the slow path.
///
/// This is deliberately independent of [`family`] where packed fields
/// disagree with a family's generic shape: `Select`, `MovPair`, and fused
/// memory ops do not carry one plain destination in `c`.
#[inline]
pub(crate) fn slot_fields(op: Op) -> (bool, bool, bool) {
    let bits = OP_PROPS[op as usize].bits;
    (
        bits & P_A_SLOT != 0,
        bits & P_B_SLOT != 0,
        bits & P_C_DST != 0,
    )
}

const fn compute_slot_fields(op: Op) -> (bool, bool, bool) {
    use Op::*;
    match compute_family(op) {
        Fam::SrcABDst => {
            if matches!(op, Select) {
                // Select packs the destination and condition in c.
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
        // Fused compare-branches: a, b operands, c = target.
        Fam::SrcAB => (true, true, false),
        // MovPair: a, b operands, c = two packed destinations.
        Fam::SrcABPairDst => (true, true, false),
        Fam::Fixed | Fam::None => (false, false, false),
    }
}

/// Whether an op's native handler leaves its result in the accumulator
/// (every value producer computes into it; see the backend's `finish`).
/// Ops that are never native are harmlessly included — the linker's
/// handler-table lookup is the real gate.
#[inline]
pub(crate) fn writes_acc(op: Op) -> bool {
    OP_PROPS[op as usize].bits & P_WRITES_ACC != 0
}

const fn compute_writes_acc(op: Op) -> bool {
    use Op::*;
    let d = op as u16;
    (d >= MovSlot as u16 && d <= F64_ReinterpretI64 as u16)
        || (d >= I32_Load as u16 && d <= I64_Load32U as u16)
        // MovPair's accumulator result is its second ordered copy. This
        // preserves the residency that its second constituent MovSlot
        // would have provided without adding another handler-table axis.
        || matches!(op, GlobalGet | Select | MovPair)
}

/// Ops whose static offset packs a memory index in the high bits can only
/// run natively against memory 0.
#[cfg(test)]
#[inline]
pub(crate) fn native_guard(ins: &Instr) -> bool {
    match OP_PROPS[ins.op as usize].guard {
        MemGuard::None => true,
        MemGuard::OffsetInB => ins.b >> 48 == 0,
        MemGuard::OffsetInC => ins.c >> 48 == 0,
        MemGuard::IndexInB => ins.b == 0,
        MemGuard::IndexInC => ins.c == 0,
    }
}

#[cfg(test)]
const fn compute_mem_guard(op: Op) -> MemGuard {
    match compute_family(op) {
        Fam::Load => MemGuard::OffsetInB,
        Fam::Store => MemGuard::OffsetInC,
        _ => match op {
            Op::MemoryFill | Op::MemoryCopy => MemGuard::IndexInB,
            Op::MemoryFillCopy => MemGuard::IndexInC,
            _ => MemGuard::None,
        },
    }
}

/// Whether this op's `c` field is a branch target cell index, which the
/// link pass turns into an absolute cell address. Must list exactly the
/// ops [`transform_bc`] multiplies by [`CELL`] — a missed one would leave
/// a relative offset and the handler would branch into hyperspace.
#[inline]
pub(crate) fn c_is_branch_target(op: Op) -> bool {
    OP_PROPS[op as usize].bits & P_C_BRANCH_TARGET != 0
}

const fn compute_c_is_branch_target(op: Op) -> bool {
    use Op::*;
    matches!(op, Br | BrIf | BrIfNot) || between(op, I32_BrEq, I64_SubBrIf)
}

/// Per-op `b`/`c` pre-scaling for native handlers (`a` is handled
/// uniformly at the call site: slot index x8 unless const). `flags` are
/// the link-resolved flags, with acc hints possibly already stripped.
#[inline]
pub(crate) fn transform_bc(ins: &Instr, flags: u16) -> (u64, u64) {
    // `b` as an operand: a slot index scaled to a byte offset, unless the
    // field holds an inline constant.
    let scaled_b = if flags & FLAG_B_CONST != 0 {
        ins.b
    } else {
        ins.b * 8
    };
    // The packed-slot-pair forms: both halves of `c` are slot indices, so
    // both are scaled to byte offsets in place.
    let packed_slots = ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8;
    let shape = OP_PROPS[ins.op as usize].bc;
    if flags & FLAG_FUSED != 0 {
        // Only the two memory families have a fused bank, so the shape
        // already says which one this is.
        return if shape == BcShape::LoadOffset {
            // loads: b = static offset (raw), c = addr2*8 << 32 | dst*8
            (ins.b, packed_slots)
        } else {
            // stores: b = value, c = addr2*8 << 32 | static offset (raw)
            (scaled_b, ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff))
        };
    }
    match shape {
        BcShape::LoadOffset => (ins.b, ins.c * 8),
        BcShape::StoreOffset => (scaled_b, ins.c),
        BcShape::BranchRawB => (ins.b, ins.c * CELL as u64),
        BcShape::BranchScaledB => (scaled_b, ins.c * CELL as u64),
        BcShape::GlobalIndex => (ins.b, ins.c * 8),
        BcShape::PackedPairDst => (ins.b * 8, packed_slots),
        // MemoryFillCopy: a and b are two operand-pack base slots. The
        // linker scales a uniformly; scale the second pack here.
        BcShape::OperandPackB => (ins.b * 8, ins.c),
        BcShape::RawBoth => (ins.b, ins.c),
        BcShape::PackedCondDst => (scaled_b, packed_slots),
        BcShape::MirroredDst => (ins.c * 8, ins.c * 8),
        BcShape::Plain => (scaled_b, ins.c * 8),
    }
}

const fn compute_bc_shape(op: Op) -> BcShape {
    use Op::*;
    // Control first: a branch target's `c` is scaled by the cell size, not
    // by the slot size, whatever family the op belongs to.
    if compute_c_is_branch_target(op) {
        return match op {
            Br | BrIf | BrIfNot => BcShape::BranchRawB,
            _ => BcShape::BranchScaledB,
        };
    }
    match compute_family(op) {
        Fam::Load => BcShape::LoadOffset,
        Fam::Store => BcShape::StoreOffset,
        _ => match op {
            GlobalSet => BcShape::GlobalIndex,
            MovPair => BcShape::PackedPairDst,
            MemoryFillCopy => BcShape::OperandPackB,
            Return => BcShape::RawBoth,
            Select => BcShape::PackedCondDst,
            MovSlot => BcShape::MirroredDst,
            // Plain value ops, GlobalGet included, and every op with no
            // native form at all.
            _ => BcShape::Plain,
        },
    }
}

/// The pinned-local selection a function was linked with. `u64::MAX` means
/// the class is unused; the `_float` flags say which register file is
/// authoritative for that slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[inline]
pub(crate) fn op_slot(ins: &Instr, flags: u16, pin: &Pinned) -> Option<usize> {
    let props = OP_PROPS[ins.op as usize];
    let fam = props.fam;
    if fam == Fam::None {
        return None;
    }
    let fused = flags & FLAG_FUSED != 0;
    if fused && !fam.has_fused_bank() {
        return None;
    }

    // Match the family before classifying operands. The old generic path
    // computed A, B, and destination classes (and loaded all three domain
    // facts) for every native cell, even when the handler bank varied over
    // none or only one of them. These formulas are the same layout encoded
    // by `Fam::index_of`, but each arm asks only for its live dimensions.
    let idx = match fam {
        Fam::None => unreachable!(),
        Fam::Fixed => 0,
        Fam::Dst => dst_class(ins, flags, pin, false).index(),
        Fam::ConstDst => {
            // `Fam::index_of` admits only the constant-source bank.
            if flags & FLAG_A_CONST == 0 {
                return None;
            }
            dst_class(ins, flags, pin, false).index()
        }
        Fam::SrcA => source_a_class(ins, flags, pin).index(),
        Fam::SrcADst => {
            source_a_class(ins, flags, pin).index() + 5 * dst_class(ins, flags, pin, false).index()
        }
        Fam::SrcAB => {
            source_a_class(ins, flags, pin).index() + 5 * source_b_class(ins, flags, pin).index()
        }
        Fam::SrcABPairDst => {
            source_a_class(ins, flags, pin).index()
                + 5 * source_b_class(ins, flags, pin).index()
                + 25 * pair_dst_class(ins, pin)?.index()
        }
        Fam::SrcABDst => {
            source_a_class(ins, flags, pin).index()
                + 5 * source_b_class(ins, flags, pin).index()
                + 25 * dst_class(ins, flags, pin, ins.op == Op::Select).index()
        }
        Fam::Load => {
            source_a_class(ins, flags, pin).index()
                + 5 * dst_class(ins, flags, pin, fused).index()
                + if fused { 20 } else { 0 }
        }
        Fam::Store => {
            source_a_class(ins, flags, pin).index()
                + 5 * source_b_class(ins, flags, pin).index()
                + if fused { 25 } else { 0 }
        }
    };
    Some(props.base as usize + idx)
}

/// Classify source A for families whose handler bank varies over it.
///
/// Domain facts are read only when the operand actually aliases a pinned
/// slot. Most operands therefore avoid that table lookup as well as families
/// which do not have an A dimension avoiding this function altogether.
#[inline]
fn source_a_class(ins: &Instr, flags: u16, pin: &Pinned) -> Cls {
    if flags & FLAG_A_CONST != 0 {
        Cls::Const
    } else if ins.a == pin.l0 && operand_is_float(ins.op, false) == pin.l0_float {
        Cls::L0
    } else if ins.a == pin.l1 && operand_is_float(ins.op, false) == pin.l1_float {
        Cls::L1
    } else if flags & FLAG_A_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    }
}

/// Classify source B for families whose handler bank varies over it.
#[inline]
fn source_b_class(ins: &Instr, flags: u16, pin: &Pinned) -> Cls {
    if flags & FLAG_B_CONST != 0 {
        Cls::Const
    } else if ins.b == pin.l0 && operand_is_float(ins.op, true) == pin.l0_float {
        Cls::L0
    } else if ins.b == pin.l1 && operand_is_float(ins.op, true) == pin.l1_float {
        Cls::L1
    } else if flags & FLAG_B_ACC != 0 {
        Cls::Acc
    } else {
        Cls::Slot
    }
}

/// Classify a plain or low-half-packed destination for families whose
/// handler bank varies over it.
#[inline]
fn dst_class(ins: &Instr, flags: u16, pin: &Pinned, packed: bool) -> DstCls {
    let dslot = if packed { ins.c & 0xffff_ffff } else { ins.c };
    // Select is a domain-agnostic bit mover and is pinned only in the integer
    // register file. For every other destination family, defer the result
    // domain lookup until a pinned-slot identity actually matches.
    if dslot == pin.l0 && dst_domain_matches(ins.op, pin.l0_float) {
        DstCls::L0
    } else if dslot == pin.l1 && dst_domain_matches(ins.op, pin.l1_float) {
        DstCls::L1
    } else if flags & FLAG_DST_ACC != 0 {
        DstCls::Acc
    } else {
        DstCls::Mem
    }
}

#[inline]
fn dst_domain_matches(op: Op, pin_float: bool) -> bool {
    if op == Op::Select {
        !pin_float
    } else {
        result_is_float(op) == pin_float
    }
}

/// Classify `MovPair`'s two ordered destinations. Equal pinned destinations
/// are a forged layout and deliberately have no native handler.
#[inline]
fn pair_dst_class(ins: &Instr, pin: &Pinned) -> Option<PairDstCls> {
    // MovPair has two ordered destinations. Classify both so the handler
    // updates every authoritative pinned register before a later source
    // in the same pair can observe it.
    let dst1 = ins.c >> 32;
    let dst2 = ins.c & 0xffff_ffff;
    Some(
        match (
            dst1 == pin.l0,
            dst1 == pin.l1,
            dst2 == pin.l0,
            dst2 == pin.l1,
        ) {
            (false, false, false, false) => PairDstCls::None,
            (true, false, false, false) => PairDstCls::FirstL0,
            (false, true, false, false) => PairDstCls::FirstL1,
            (false, false, true, false) => PairDstCls::SecondL0,
            (false, false, false, true) => PairDstCls::SecondL1,
            (true, false, false, true) => PairDstCls::L0L1,
            (false, true, true, false) => PairDstCls::L1L0,
            // The predecoder does not pair equal destinations. Keep this
            // defensive fallback slow if a forged function violates it.
            _ => return None,
        },
    )
}

/// The former generic implementation, retained as an independent test
/// oracle for the family-specialized production classifier above.
#[cfg(test)]
fn op_slot_generic_reference(ins: &Instr, flags: u16, pin: &Pinned) -> Option<usize> {
    let props = OP_PROPS[ins.op as usize];
    let fam = props.fam;
    if fam == Fam::None {
        return None;
    }
    let (a_s, b_s, c_d) = (
        props.bits & P_A_SLOT != 0,
        props.bits & P_B_SLOT != 0,
        props.bits & P_C_DST != 0,
    );
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
    let pair_d = if ins.op == Op::MovPair {
        let dst1 = ins.c >> 32;
        let dst2 = ins.c & 0xffff_ffff;
        match (
            dst1 == pin.l0,
            dst1 == pin.l1,
            dst2 == pin.l0,
            dst2 == pin.l1,
        ) {
            (false, false, false, false) => PairDstCls::None,
            (true, false, false, false) => PairDstCls::FirstL0,
            (false, true, false, false) => PairDstCls::FirstL1,
            (false, false, true, false) => PairDstCls::SecondL0,
            (false, false, false, true) => PairDstCls::SecondL1,
            (true, false, false, true) => PairDstCls::L0L1,
            (false, true, true, false) => PairDstCls::L1L0,
            _ => return None,
        }
    } else {
        PairDstCls::None
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
    let idx = fam.index_of(a, b, d, pair_d, flags & FLAG_FUSED != 0)?;
    Some(op_base(ins.op) as usize + idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the specialized classifier against the exact generic code it
    /// replaced. The low six bits cover every class/fusion combination;
    /// representative overlays cover each slow-only handler bit separately
    /// and together. Every opcode is crossed with integer, float, mixed, and
    /// partially populated pin sets, plus all source and destination
    /// relationships relevant to plain, packed, and paired destinations.
    #[test]
    fn specialized_op_slot_matches_generic_reference() {
        use super::super::instr::{FLAG_ADDR64, FLAG_SHARED_GLOBAL, FLAG_SHARED_TABLE};
        use crate::collections::Vec;

        const L0: u64 = 11;
        const L1: u64 = 22;
        const OTHER: u64 = 33;
        const OTHER_2: u64 = 44;
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

        // Exhaust every classification bit combination. The three slow bits
        // are checked independently and combined over representative class
        // states; op_slot deliberately leaves their rejection to handler_for.
        let mut flag_cases: Vec<u16> = (0..(FLAG_FUSED << 1)).collect();
        let representative_class_flags = [
            0,
            FLAG_A_CONST | FLAG_B_CONST,
            FLAG_A_ACC | FLAG_B_ACC | FLAG_DST_ACC,
            FLAG_FUSED,
            (FLAG_FUSED << 1) - 1,
        ];
        for slow in [
            FLAG_ADDR64,
            FLAG_SHARED_TABLE,
            FLAG_SHARED_GLOBAL,
            FLAG_ADDR64 | FLAG_SHARED_TABLE | FLAG_SHARED_GLOBAL,
        ] {
            for class_flags in representative_class_flags {
                flag_cases.push(class_flags | slow);
            }
        }

        let source_relations = [OTHER, L0, L1];
        let destination_relations = [
            // Plain destinations.
            OTHER,
            L0,
            L1,
            // Packed/paired destinations: every valid pin relation, plus
            // both equal-pinned defensive failures and distinct unpinned dsts.
            (OTHER << 32) | OTHER_2,
            (L0 << 32) | OTHER,
            (L1 << 32) | OTHER,
            (OTHER << 32) | L0,
            (OTHER << 32) | L1,
            (L0 << 32) | L1,
            (L1 << 32) | L0,
            (L0 << 32) | L0,
            (L1 << 32) | L1,
        ];

        for op_index in 0..N_OPS {
            let op = op_from_index(op_index);
            for (pin_index, pin) in pins.iter().enumerate() {
                for &flags in &flag_cases {
                    for &a in &source_relations {
                        for &b in &source_relations {
                            for &c in &destination_relations {
                                let ins = Instr::new(op, flags, a, b, c);
                                assert_eq!(
                                    op_slot(&ins, flags, pin),
                                    op_slot_generic_reference(&ins, flags, pin),
                                    "op={op:?} flags={flags:#x} pin={pin_index} a={a} b={b} c={c:#x}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mov_pair_classifies_every_ordered_pinned_destination_state() {
        let pin = Pinned {
            l0: 1,
            l1: 2,
            l0_float: false,
            l1_float: false,
        };
        let cases = [
            ((3, 4), PairDstCls::None),
            ((1, 3), PairDstCls::FirstL0),
            ((2, 3), PairDstCls::FirstL1),
            ((3, 1), PairDstCls::SecondL0),
            ((3, 2), PairDstCls::SecondL1),
            ((1, 2), PairDstCls::L0L1),
            ((2, 1), PairDstCls::L1L0),
        ];

        for ((dst1, dst2), pair_d) in cases {
            let ins = Instr::new(Op::MovPair, 0, 3, 4, (dst1 << 32) | dst2);
            let expected = op_base(Op::MovPair) as usize
                + family(Op::MovPair)
                    .index_of(Cls::Slot, Cls::Slot, DstCls::Mem, pair_d, false)
                    .unwrap();
            assert_eq!(
                op_slot(&ins, 0, &pin),
                Some(expected),
                "dst1={dst1}, dst2={dst2}"
            );
        }

        let equal_destinations = Instr::new(Op::MovPair, 0, 3, 4, (1u64 << 32) | 1);
        assert_eq!(op_slot(&equal_destinations, 0, &pin), None);
    }

    /// `BcShape` is a packed restatement of a derivation that used to run
    /// per cell, from `family`, `c_is_branch_target` and the opcode. A
    /// mis-assigned shape is silent: the cell still links to a real
    /// handler, which then reads a field the linker scaled the wrong way.
    /// So check the packed answer against the structural one for every op.
    #[test]
    fn packed_bc_shape_matches_the_structural_derivation() {
        fn reference(ins: &Instr, flags: u16) -> (u64, u64) {
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
                    (ins.b, ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8)
                } else {
                    (
                        scaled_b(ins),
                        ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff),
                    )
                };
            }
            if c_is_branch_target(ins.op) {
                let b = match ins.op {
                    Br | BrIf | BrIfNot => ins.b,
                    _ => scaled_b(ins),
                };
                return (b, ins.c * CELL as u64);
            }
            match family(ins.op) {
                Fam::Load => (ins.b, ins.c * 8),
                Fam::Store => (scaled_b(ins), ins.c),
                _ => match ins.op {
                    GlobalSet => (ins.b, ins.c * 8),
                    MovPair => (
                        ins.b * 8,
                        ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8,
                    ),
                    MemoryFillCopy => (ins.b * 8, ins.c),
                    Return => (ins.b, ins.c),
                    Select => (
                        scaled_b(ins),
                        ((ins.c >> 32) * 8) << 32 | (ins.c & 0xffff_ffff) * 8,
                    ),
                    MovSlot => (ins.c * 8, ins.c * 8),
                    _ => (scaled_b(ins), ins.c * 8),
                },
            }
        }

        for i in 0..N_OPS {
            let op = op_from_index(i);
            // Distinct values in every field, so a swapped or unscaled one
            // cannot coincide with the right answer.
            let ins = Instr::new(op, 0, 3, 5, (7u64 << 32) | 9);
            for flags in [0, FLAG_B_CONST, FLAG_FUSED, FLAG_FUSED | FLAG_B_CONST] {
                assert_eq!(
                    transform_bc(&ins, flags),
                    reference(&ins, flags),
                    "{op:?} flags={flags:#x}"
                );
            }
        }
    }

    /// Same check for the memory-index guard: which field a cell packs its
    /// memory index into, and hence whether it may run natively at all.
    #[test]
    fn packed_memory_guard_matches_the_structural_derivation() {
        fn reference(ins: &Instr) -> bool {
            match family(ins.op) {
                Fam::Load => ins.b >> 48 == 0,
                Fam::Store => ins.c >> 48 == 0,
                _ => match ins.op {
                    Op::MemoryFill | Op::MemoryCopy => ins.b == 0,
                    Op::MemoryFillCopy => ins.c == 0,
                    _ => true,
                },
            }
        }

        for i in 0..N_OPS {
            let op = op_from_index(i);
            for (b, c) in [(0, 0), (1 << 48, 0), (0, 1 << 48), (1, 0), (0, 1), (1, 1)] {
                let ins = Instr::new(op, 0, 0, b, c);
                assert_eq!(
                    native_guard(&ins),
                    reference(&ins),
                    "{op:?} b={b:#x} c={c:#x}"
                );
            }
        }
    }
}
