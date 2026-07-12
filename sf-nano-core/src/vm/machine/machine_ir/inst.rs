use super::cfg::MachineBranchCond;
use super::types::{
    MachineAddr, MachineCompareKind, MachineConstId, MachineConvertOp, MachineFloatBinaryOp,
    MachineFloatUnaryOp, MachineFloatWidth, MachineIndexExtend, MachineIntBinaryOp,
    MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg,
    MachineRegOwner, MachineShiftOp, MachineSign, MachineStorageType, MachineTrapKind,
    MachineValue,
};
use crate::collections;
use crate::value_type::RefType;
use crate::vm::middle::frame::{FrameSlot, FrameSpan};

/// Inline runtime-dispatch call that falls through in the same function.
///
/// This is the MachineIR representation for operations that cross into the
/// runtime-call ABI but do not transfer control to another compiled MachineIR
/// function. In particular, Wasm calls that must round-trip through runtime
/// dispatch lower to this form: they are semantically calls, but they return
/// inline to the same block, so they are modeled as instructions rather than
/// CFG terminators.
///
/// By contrast, compiled Wasm calls become [`MachineTerminator::Call`] because
/// they leave the current MachineIR block and resume at an explicit
/// continuation block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MachineCallRuntime {
    /// Read-only constant-pool metadata for this call site.
    ///
    /// The backend treats this as an opaque constant-pool reference and always
    /// lowers this instruction to the shared `call_runtime_entry` runtime ABI.
    /// The metadata record describes how the entrypoint finds the runtime-call
    /// target and the frame regions used for arguments and results.
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
        /// Semantic owner for the value defined in `dst`.
        ///
        /// The destination can become either one linear machine value or a
        /// cached-local binding. Late machine passes must use this explicit
        /// owner rather than infer meaning from the register number.
        owner: MachineRegOwner,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    },
    FloatConst {
        width: MachineFloatWidth,
        dst: MachineReg,
        bits: u64,
    },
    /// Materialize one SIMD literal into a native vector register.
    #[cfg(sf_has_simd)]
    V128Const {
        dst: MachineReg,
        bytes: [u8; 16],
    },
    /// Convert one canonical raw v128 handle into a native vector register.
    ///
    /// The raw handle is the engine's canonical 8-byte frame/global/ABI form.
    #[cfg(sf_has_simd)]
    V128FromRaw {
        owner: MachineRegOwner,
        dst: MachineReg,
        raw: MachineValue,
    },
    /// Convert one native vector register into the canonical raw v128 handle.
    ///
    /// The result is a GP-word handle suitable for frame/global/ABI storage.
    #[cfg(sf_has_simd)]
    V128ToRaw {
        dst: MachineReg,
        src: MachineValue,
    },
    Load {
        /// Semantic owner for the value defined in `dst`.
        ///
        /// A load can materialize either one linear machine value or a cached
        /// local. The owner tag keeps that distinction alive after lowering.
        owner: MachineRegOwner,
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
    /// Indexed load: `dst <- [base + extend(index) + offset]`.
    ///
    /// Produced by the shared peephole from `i64.Add(base, index) + Load` or
    /// `cvt.I64ExtendI32U + i64.Add(base, ext) + Load` sequences. Each backend
    /// maps this to its best addressing mode (ARM64 UXTW register-indexed,
    /// x86_64 `[base + index + disp]`, etc.).
    IndexedLoad {
        dst: MachineReg,
        base: MachineReg,
        index: MachineReg,
        index_extend: MachineIndexExtend,
        offset: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    },
    /// Indexed store: `[base + extend(index) + offset] <- src`.
    IndexedStore {
        base: MachineReg,
        index: MachineReg,
        index_extend: MachineIndexExtend,
        offset: i32,
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
    /// Signed 32×32 → 64 multiply, producing an i64 pair from two narrow i32
    /// values. Created by the `fuse_smull_sign_ext` peephole when both
    /// operands of an `Int64PairBinary{Mul}` originated from `i64.extend_i32_s`
    /// (i.e. their hi halves are `lo >> 31`). Backends that have a single-
    /// instruction signed widening multiply (arm32 SMULL, arm64 SMULL X,W,W,
    /// x86_64 IMUL after MOVSXD) emit it directly; otherwise this variant is
    /// not produced (only 32-bit GP targets see the source pattern).
    Int64MulFromSignExt32 {
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        lhs: MachineValue,
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
    /// Unsigned bitfield extract: `dst = (src >> lsb) & ((1 << bits) - 1)`.
    ///
    /// Fused from `ShrU(imm) + And(mask)` where mask is a contiguous low-bit
    /// pattern. On ARM64 this maps to `UBFX`.
    BitfieldExtractU {
        width: MachineIntWidth,
        dst: MachineReg,
        src: MachineReg,
        lsb: u8,
        bits: u8,
    },
    /// Binary op with shifted second operand:
    /// `dst = lhs OP (rhs SHIFT amount)`.
    ///
    /// Fused from `shift(rhs, imm) + binary_op(lhs, shifted)`. Covers
    /// Add, Sub, And, Or, Xor with LSL/LSR/ASR shifted register.
    IntBinaryShifted {
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        shift: MachineShiftOp,
        amount: u8,
    },
    /// Test bits: `dst = ((src & mask) CMP 0)`.
    ///
    /// Fused from `And(src, mask) + IntCompare(result, 0, Eq/Ne)`. On ARM64
    /// this maps to `TST + CSET`, or `TST + B.cond` when further fused into a
    /// branch condition.
    TestBits {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        src: MachineReg,
        mask: MachineValue,
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
    /// Generic SIMD unary op.
    #[cfg(sf_has_simd)]
    SimdUnary {
        opcode: u32,
        dst_ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    },
    /// Generic SIMD binary op.
    #[cfg(sf_has_simd)]
    SimdBinary {
        opcode: u32,
        dst_ty: MachineStorageType,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    /// Generic SIMD ternary op.
    #[cfg(sf_has_simd)]
    SimdTernary {
        opcode: u32,
        dst_ty: MachineStorageType,
        dst: MachineReg,
        a: MachineValue,
        b: MachineValue,
        c: MachineValue,
    },
    /// Generic SIMD shift op (`vector op scalar-count`).
    #[cfg(sf_has_simd)]
    SimdShift {
        opcode: u32,
        dst: MachineReg,
        vector: MachineValue,
        shift: MachineValue,
    },
    /// SIMD lane extract with one immediate lane index.
    #[cfg(sf_has_simd)]
    SimdExtractLane {
        opcode: u32,
        lane: u8,
        dst_ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    },
    /// SIMD lane replace with one immediate lane index.
    #[cfg(sf_has_simd)]
    SimdReplaceLane {
        opcode: u32,
        lane: u8,
        dst: MachineReg,
        vector: MachineValue,
        scalar: MachineValue,
    },
    /// i8x16.shuffle with its fixed lane mask.
    #[cfg(sf_has_simd)]
    SimdShuffle {
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
        lanes: [u8; 16],
    },
    /// SIMD memory load into a native vector register.
    #[cfg(sf_has_simd)]
    SimdLoad {
        opcode: u32,
        dst: MachineReg,
        addr: MachineAddr,
    },
    /// SIMD memory store from a native vector register.
    #[cfg(sf_has_simd)]
    SimdStore {
        opcode: u32,
        addr: MachineAddr,
        src: MachineValue,
    },
    /// SIMD lane load: merge one loaded scalar lane into an input vector.
    #[cfg(sf_has_simd)]
    SimdLoadLane {
        opcode: u32,
        lane: u8,
        dst: MachineReg,
        addr: MachineAddr,
        vector: MachineValue,
    },
    /// SIMD lane store: write one scalar lane from the input vector.
    #[cfg(sf_has_simd)]
    SimdStoreLane {
        opcode: u32,
        lane: u8,
        addr: MachineAddr,
        vector: MachineValue,
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
    /// Runtime/host helper call that returns inline in the same block.
    ///
    /// This is intentionally not a terminator: control flow remains within the
    /// current MachineIR block. The surrounding lowering is responsible for any
    /// cached-local save/reload, mem0 cache refresh, and other state repair
    /// needed around the call. The ISA/backend only performs the actual foreign
    /// call ABI sequence for the helper target.
    ///
    /// Wasm calls that stay on the runtime-dispatch path use this form because
    /// they do not create a new compiled-frame activation or a MachineIR
    /// continuation edge; they simply invoke the runtime entry and continue in
    /// the same function.
    CallRuntime(MachineCallRuntime),
    EhThrow {
        tag_idx: u32,
        args: FrameSpan,
    },
    EhThrowRef {
        exnref_slot: FrameSlot,
    },
    EhAllocExnRef {
        tag_idx: u32,
        dst: MachineReg,
    },
    MemoryGrow {
        mem_idx: u32,
        dst: MachineReg,
        delta: MachineValue,
    },
    MemoryFill {
        mem_idx: u32,
        dest: MachineValue,
        val: MachineValue,
        len: MachineValue,
    },
    MemoryCopy {
        dst_mem: u32,
        src_mem: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    },
    MemoryInit {
        mem_idx: u32,
        data_idx: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    },
    DataDrop {
        data_idx: u32,
    },
    TableGrow {
        table_idx: u32,
        dst: MachineReg,
        init_val: MachineValue,
        delta: MachineValue,
    },
    TableFill {
        table_idx: u32,
        start: MachineValue,
        val: MachineValue,
        len: MachineValue,
    },
    TableCopy {
        dst_tbl: u32,
        src_tbl: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    },
    TableInit {
        table_idx: u32,
        elem_idx: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    },
    ElemDrop {
        elem_idx: u32,
    },
    RefFunc {
        func_idx: u32,
        dst: MachineReg,
    },
    RefAsNonNull {
        src: MachineValue,
        dst: MachineReg,
    },
    RefEq {
        lhs: MachineValue,
        rhs: MachineValue,
        dst: MachineReg,
    },
    RefI31 {
        src: MachineValue,
        dst: MachineReg,
    },
    I31GetS {
        src: MachineValue,
        dst: MachineReg,
    },
    I31GetU {
        src: MachineValue,
        dst: MachineReg,
    },
    AnyConvertExtern {
        src: MachineValue,
        dst: MachineReg,
    },
    ExternConvertAny {
        src: MachineValue,
        dst: MachineReg,
    },
    RefTest {
        ref_type: RefType,
        src: MachineValue,
        dst: MachineReg,
    },
    RefCast {
        ref_type: RefType,
        src: MachineValue,
        dst: MachineReg,
    },
    StructNew {
        type_idx: u32,
        fields: collections::Vec<(MachineValue, Option<MachineValue>)>,
        dst: MachineReg,
    },
    StructNewDefault {
        type_idx: u32,
        dst: MachineReg,
    },
    StructGet {
        type_idx: u32,
        field_idx: u32,
        signed: Option<bool>,
        ty: MachineStorageType,
        src: MachineValue,
        dst: MachineReg,
        dst_hi: Option<MachineReg>,
    },
    StructSet {
        type_idx: u32,
        field_idx: u32,
        ref_src: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
    },
    ArrayNew {
        type_idx: u32,
        init_lo: MachineValue,
        init_hi: Option<MachineValue>,
        length: MachineValue,
        dst: MachineReg,
    },
    ArrayNewDefault {
        type_idx: u32,
        length: MachineValue,
        dst: MachineReg,
    },
    ArrayNewFixed {
        type_idx: u32,
        elements: collections::Vec<(MachineValue, Option<MachineValue>)>,
        dst: MachineReg,
    },
    ArrayNewData {
        type_idx: u32,
        data_idx: u32,
        src: MachineValue,
        len: MachineValue,
        dst: MachineReg,
    },
    ArrayNewElem {
        type_idx: u32,
        elem_idx: u32,
        src: MachineValue,
        len: MachineValue,
        dst: MachineReg,
    },
    ArrayGet {
        type_idx: u32,
        signed: Option<bool>,
        ty: MachineStorageType,
        ref_src: MachineValue,
        index: MachineValue,
        dst: MachineReg,
        dst_hi: Option<MachineReg>,
    },
    ArraySet {
        type_idx: u32,
        ref_src: MachineValue,
        index: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
    },
    ArrayFill {
        type_idx: u32,
        ref_src: MachineValue,
        index: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
        len: MachineValue,
    },
    ArrayCopy {
        dst_type_idx: u32,
        src_type_idx: u32,
        dst_ref: MachineValue,
        dst_index: MachineValue,
        src_ref: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    },
    ArrayInitData {
        type_idx: u32,
        data_idx: u32,
        ref_src: MachineValue,
        dst_index: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    },
    ArrayInitElem {
        type_idx: u32,
        elem_idx: u32,
        ref_src: MachineValue,
        dst_index: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    },
    ArrayLen {
        src: MachineValue,
        dst: MachineReg,
    },
}

impl MachineInstKind {
    /// Semantic owner for the register definitions of this instruction.
    ///
    /// Most defining instructions always produce linear machine values.
    /// `Move` and `Load` are the ambiguous cases, so they carry explicit owner
    /// metadata in the IR itself.
    pub(crate) const fn def_owner(&self) -> Option<MachineRegOwner> {
        match self {
            Self::Move { owner, .. } | Self::Load { owner, .. } => Some(*owner),
            #[cfg(sf_has_simd)]
            Self::V128FromRaw { owner, .. } => Some(*owner),
            Self::FloatConst { .. }
            | Self::IndexedLoad { .. }
            | Self::IntUnary { .. }
            | Self::IntBinary { .. }
            | Self::Int64PairBinary { .. }
            | Self::Int64PairDivRem { .. }
            | Self::Int64PairUnary { .. }
            | Self::Int64PairShift { .. }
            | Self::Int64MulFromSignExt32 { .. }
            | Self::IntCompare { .. }
            | Self::BitfieldExtractU { .. }
            | Self::IntBinaryShifted { .. }
            | Self::TestBits { .. }
            | Self::Int64PairCompare { .. }
            | Self::ConvertI64PairToFloat { .. }
            | Self::ConvertFloatToI64Pair { .. }
            | Self::ReinterpretF64ToI64Pair { .. }
            | Self::ReinterpretI64PairToF64 { .. }
            | Self::FloatUnary { .. }
            | Self::FloatBinary { .. }
            | Self::FloatCompare { .. }
            | Self::Convert { .. }
            | Self::Select { .. }
            | Self::EhAllocExnRef { .. }
            | Self::MemoryGrow { .. }
            | Self::TableGrow { .. }
            | Self::RefFunc { .. }
            | Self::RefAsNonNull { .. }
            | Self::RefEq { .. }
            | Self::RefI31 { .. }
            | Self::I31GetS { .. }
            | Self::I31GetU { .. }
            | Self::AnyConvertExtern { .. }
            | Self::ExternConvertAny { .. }
            | Self::RefTest { .. }
            | Self::RefCast { .. }
            | Self::StructNew { .. }
            | Self::StructNewDefault { .. }
            | Self::ArrayNew { .. }
            | Self::ArrayNewDefault { .. }
            | Self::ArrayNewFixed { .. }
            | Self::ArrayNewData { .. }
            | Self::ArrayNewElem { .. }
            | Self::StructGet { .. }
            | Self::ArrayGet { .. } => Some(MachineRegOwner::LinearValue),
            #[cfg(sf_has_simd)]
            Self::V128Const { .. }
            | Self::V128ToRaw { .. }
            | Self::SimdUnary { .. }
            | Self::SimdBinary { .. }
            | Self::SimdTernary { .. }
            | Self::SimdShift { .. }
            | Self::SimdExtractLane { .. }
            | Self::SimdReplaceLane { .. }
            | Self::SimdShuffle { .. }
            | Self::SimdLoad { .. }
            | Self::SimdLoadLane { .. } => Some(MachineRegOwner::LinearValue),
            Self::ArrayLen { .. } => Some(MachineRegOwner::LinearValue),
            Self::Store { .. }
            | Self::IndexedStore { .. }
            | Self::TrapIf { .. }
            | Self::CallRuntime(_)
            | Self::EhThrow { .. }
            | Self::EhThrowRef { .. }
            | Self::MemoryFill { .. }
            | Self::MemoryCopy { .. }
            | Self::MemoryInit { .. }
            | Self::DataDrop { .. }
            | Self::TableFill { .. }
            | Self::TableCopy { .. }
            | Self::TableInit { .. }
            | Self::ElemDrop { .. }
            | Self::StructSet { .. }
            | Self::ArraySet { .. }
            | Self::ArrayFill { .. }
            | Self::ArrayCopy { .. }
            | Self::ArrayInitData { .. }
            | Self::ArrayInitElem { .. } => None,
            #[cfg(sf_has_simd)]
            Self::SimdStore { .. } | Self::SimdStoreLane { .. } => None,
        }
    }

    /// Visit every register this instruction defines.
    ///
    /// Most instructions define a single destination register, but legalized
    /// i64 pair operations on 32-bit backends define two GP lanes (`dst_lo`
    /// then `dst_hi`), and `StructGet` / `ArrayGet` define a second lane only
    /// when their `dst_hi` is present. This is the single source of truth for
    /// the machine backend's def set: ownership tracking, register allocation,
    /// and the late peepholes all route through it so they cannot drift.
    pub(crate) fn for_each_defined_reg(&self, mut f: impl FnMut(MachineReg)) {
        match self {
            Self::Move { dst, .. }
            | Self::FloatConst { dst, .. }
            | Self::Load { dst, .. }
            | Self::IntUnary { dst, .. }
            | Self::IntBinary { dst, .. }
            | Self::IntCompare { dst, .. }
            | Self::FloatUnary { dst, .. }
            | Self::FloatBinary { dst, .. }
            | Self::FloatCompare { dst, .. }
            | Self::Convert { dst, .. }
            | Self::Select { dst, .. }
            | Self::IndexedLoad { dst, .. }
            | Self::BitfieldExtractU { dst, .. }
            | Self::IntBinaryShifted { dst, .. }
            | Self::TestBits { dst, .. }
            | Self::EhAllocExnRef { dst, .. }
            | Self::RefFunc { dst, .. }
            | Self::RefAsNonNull { dst, .. }
            | Self::RefEq { dst, .. }
            | Self::RefI31 { dst, .. }
            | Self::I31GetS { dst, .. }
            | Self::I31GetU { dst, .. }
            | Self::AnyConvertExtern { dst, .. }
            | Self::ExternConvertAny { dst, .. }
            | Self::RefTest { dst, .. }
            | Self::RefCast { dst, .. }
            | Self::StructNew { dst, .. }
            | Self::StructNewDefault { dst, .. }
            | Self::ArrayNew { dst, .. }
            | Self::ArrayNewDefault { dst, .. }
            | Self::ArrayNewFixed { dst, .. }
            | Self::ArrayNewData { dst, .. }
            | Self::ArrayNewElem { dst, .. }
            | Self::ArrayLen { dst, .. }
            | Self::MemoryGrow { dst, .. }
            | Self::TableGrow { dst, .. }
            | Self::Int64PairCompare { dst, .. }
            | Self::ConvertI64PairToFloat { dst, .. }
            | Self::ReinterpretI64PairToF64 { dst, .. } => f(*dst),
            #[cfg(sf_has_simd)]
            Self::V128Const { dst, .. }
            | Self::V128FromRaw { dst, .. }
            | Self::V128ToRaw { dst, .. }
            | Self::SimdUnary { dst, .. }
            | Self::SimdBinary { dst, .. }
            | Self::SimdTernary { dst, .. }
            | Self::SimdShift { dst, .. }
            | Self::SimdExtractLane { dst, .. }
            | Self::SimdReplaceLane { dst, .. }
            | Self::SimdShuffle { dst, .. }
            | Self::SimdLoad { dst, .. }
            | Self::SimdLoadLane { dst, .. } => f(*dst),
            Self::StructGet { dst, dst_hi, .. } | Self::ArrayGet { dst, dst_hi, .. } => {
                f(*dst);
                if let Some(dst_hi) = dst_hi {
                    f(*dst_hi);
                }
            }
            Self::Int64PairBinary { dst_lo, dst_hi, .. }
            | Self::Int64PairUnary { dst_lo, dst_hi, .. }
            | Self::Int64PairDivRem { dst_lo, dst_hi, .. }
            | Self::Int64PairShift { dst_lo, dst_hi, .. }
            | Self::Int64MulFromSignExt32 { dst_lo, dst_hi, .. }
            | Self::ConvertFloatToI64Pair { dst_lo, dst_hi, .. }
            | Self::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
                f(*dst_lo);
                f(*dst_hi);
            }
            Self::Store { .. }
            | Self::IndexedStore { .. }
            | Self::TrapIf { .. }
            | Self::CallRuntime(_)
            | Self::EhThrow { .. }
            | Self::EhThrowRef { .. }
            | Self::MemoryFill { .. }
            | Self::MemoryCopy { .. }
            | Self::MemoryInit { .. }
            | Self::DataDrop { .. }
            | Self::TableFill { .. }
            | Self::TableCopy { .. }
            | Self::TableInit { .. }
            | Self::ElemDrop { .. }
            | Self::StructSet { .. }
            | Self::ArraySet { .. }
            | Self::ArrayFill { .. }
            | Self::ArrayCopy { .. }
            | Self::ArrayInitData { .. }
            | Self::ArrayInitElem { .. } => {}
            #[cfg(sf_has_simd)]
            Self::SimdStore { .. } | Self::SimdStoreLane { .. } => {}
        }
    }
}
