use super::cfg::MachineBranchCond;
use super::types::{
    MachineTrapKind,
    MachineAddr, MachineCompareKind, MachineConstId, MachineConvertOp, MachineExternId,
    MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth, MachineIntBinaryOp,
    MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineSign, MachineStorageType, MachineValue,
};

/// Helper call that falls through in the same function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MachineHelperCall {
    /// Opaque external target id. Sidecar binding data resolves this to the
    /// real Rust helper wrapper address during backend finalization.
    pub target: MachineExternId,
    /// Read-only sidecar metadata for this call site.
    ///
    /// The backend treats this as an opaque constant reference. Helper-specific
    /// interpretation stays out of the ISA layer. Helpers operate on canonical
    /// frame regions named by this metadata, so unrelated live machine values
    /// remain live across the call in machine semantics.
    pub metadata: MachineConstId,
}

/// One machine instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MachineInst {
    pub kind: MachineInstKind,
}

/// Straight-line machine instruction vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MachineInstKind {
    /// Copy-like transfer.
    ///
    /// For GP storage classes, move-like operations define width adaptation in
    /// the obvious low-word way:
    /// - `GpWord -> GpI64` zero-extends
    /// - `GpI64 -> GpWord` truncates to the low word
    ///
    /// This is about register occupancy, not a separate `i32` register bank.
    /// On 64-bit targets, semantic `i32` values still typically use `GpWord`
    /// storage and rely on the consuming instruction's `MachineIntWidth::I32`
    /// to select a 32-bit ALU form.
    ///
    /// Signed widening remains an explicit `Convert`.
    Move {
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    },
    FloatConst {
        width: MachineFloatWidth,
        dst: MachineReg,
        bits: u64,
    },
    Load {
        ty: MachineStorageType,
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    },
    Store {
        ty: MachineStorageType,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    },
    IntUnary {
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    },
    IntBinary {
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    /// Full-width product of two native-word integer operands.
    ///
    /// 64-bit integer binary op over legalized 32-bit GP register pairs.
    ///
    /// This keeps the shared 32-bit MachineIR compact for the operations that
    /// would otherwise explode into temp-heavy carry/borrow sequences.
    Int64PairBinary {
        op: MachineIntBinaryOp,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        lhs_lo: MachineValue,
        lhs_hi: MachineValue,
        rhs_lo: MachineValue,
        rhs_hi: MachineValue,
    },
    /// 64-bit div/rem over legalized 32-bit GP register pairs.
    ///
    /// This keeps 32-bit native MachineIR pair-aware without forcing the
    /// lowerer or backend to explode div/rem into long scalar helper-shaped
    /// sequences up front.
    Int64PairDivRem {
        sign: MachineSign,
        rem: bool,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        lhs_lo: MachineValue,
        lhs_hi: MachineValue,
        rhs_lo: MachineValue,
        rhs_hi: MachineValue,
    },
    /// 64-bit unary integer op over legalized 32-bit GP register pairs.
    ///
    /// This keeps the shared 32-bit native IR compact for i64 operations whose
    /// result is still a pair and whose backend lowering is simpler when kept
    /// explicit.
    Int64PairUnary {
        op: MachineIntUnaryOp,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        src_lo: MachineValue,
        src_hi: MachineValue,
    },
    /// 64-bit shift/rotate over legalized 32-bit GP register pairs.
    ///
    /// The value being shifted is split into low/high GP-word halves, while
    /// the shift count is already reduced to the low native word because Wasm
    /// shift counts only observe the low 6 bits for i64 operations.
    Int64PairShift {
        op: MachineIntBinaryOp,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        lhs_lo: MachineValue,
        lhs_hi: MachineValue,
        rhs: MachineValue,
    },
    IntCompare {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    /// Compare two legalized 64-bit GP-word pairs and materialize a GP-word
    /// boolean result.
    ///
    /// This keeps one compare result in MachineIR rather than forcing the
    /// producer to manufacture multiple temporary boolean registers first.
    Int64PairCompare {
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs_lo: MachineValue,
        lhs_hi: MachineValue,
        rhs_lo: MachineValue,
        rhs_hi: MachineValue,
    },
    /// 64-bit integer to float conversion from legalized GP register pairs.
    ///
    /// This is the pair-aware 32-bit native form for i64-to-float conversion.
    ConvertI64PairToFloat {
        width: MachineFloatWidth,
        sign: MachineSign,
        dst: MachineReg,
        src_lo: MachineValue,
        src_hi: MachineValue,
    },
    /// Float-to-i64 conversion into a legalized GP-word pair.
    ///
    /// The `op` is one of the i64 trunc/trunc_sat families. Keeping the exact
    /// conversion opcode here preserves its trapping vs saturating semantics in
    /// the shared 32-bit emulator.
    ConvertFloatToI64Pair {
        op: MachineConvertOp,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        src: MachineValue,
    },
    /// Raw bit reinterpret from one fp64 value into a legalized i64 pair.
    ReinterpretF64ToI64Pair {
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        src: MachineValue,
    },
    /// Raw bit reinterpret from a legalized i64 pair into one fp64 value.
    ReinterpretI64PairToF64 {
        dst: MachineReg,
        src_lo: MachineValue,
        src_hi: MachineValue,
    },
    FloatUnary {
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    },
    FloatBinary {
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    FloatCompare {
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    Convert {
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    },
    Select {
        ty: MachineStorageType,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    },
    TrapIf {
        kind: MachineTrapKind,
        cond: MachineBranchCond,
    },
    CallHelper(MachineHelperCall),
}
