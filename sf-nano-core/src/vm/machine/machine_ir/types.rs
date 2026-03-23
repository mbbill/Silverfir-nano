/// One generic machine register.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MachineReg(pub(crate) u16);

/// One machine CFG block id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MachineBlockId(pub(crate) u32);

impl MachineBlockId {
    #[inline]
    pub(crate) const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// One local-function identifier in the machine-level call graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MachineFuncId(pub(crate) u32);

/// One opaque external target id referenced from machine IR.
///
/// The machine IR does not know whether this resolves to a helper wrapper or
/// some other external native target. Sidecar binding data owns that meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MachineExternId(pub(crate) u32);

/// One read-only sidecar constant record referenced from machine IR.
///
/// This is used for immutable helper metadata or other finalized constant data
/// that should live beside code, not inside writable frame scratch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MachineConstId(pub(crate) u32);

/// One readable machine operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineValue {
    Reg(MachineReg),
    Imm64(u64),
}

/// One explicit machine address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MachineAddr {
    pub base: MachineReg,
    pub offset: i32,
}

/// Scalar integer width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineIntWidth {
    I32,
    I64,
}

/// Scalar float width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineFloatWidth {
    F32,
    F64,
}

/// Storage class carried by a machine register or generic move/select.
///
/// This is intentionally about register occupancy, not the full semantic Wasm
/// type:
/// - `GpWord` means "one native GP register wide on this target"
/// - `GpI64` means "a true 64-bit GP integer value"
///
/// On 64-bit targets, many semantic `i32` values still live in `GpWord`
/// storage and use `MachineIntWidth::I32` on the consuming instruction. The
/// backend may encode those ops using a 32-bit subregister view such as
/// `eax`/`rax` or `w0`/`x0`, but that is still one physical GP register slot,
/// not a separate storage class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineStorageType {
    GpWord,
    GpI64,
    Fp32,
    Fp64,
}

impl MachineStorageType {
    #[inline]
    pub(crate) const fn is_fp(self) -> bool {
        matches!(self, Self::Fp32 | Self::Fp64)
    }

    #[inline]
    pub(crate) const fn float_width(self) -> Option<MachineFloatWidth> {
        match self {
            Self::Fp32 => Some(MachineFloatWidth::F32),
            Self::Fp64 => Some(MachineFloatWidth::F64),
            Self::GpWord | Self::GpI64 => None,
        }
    }
}

/// Width of one memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineMemWidth {
    U8,
    U16,
    U32,
    U64,
}

/// Returns the `MachineMemWidth` matching the selected backend GP register
/// width.
///
/// Shared lowering must key off the backend ABI it is targeting, not the host
/// compiler's `usize`, because armv7a lowering runs on 64-bit hosts.
#[inline]
pub(crate) const fn machine_ptr_width(gp_reg_width: u8) -> MachineMemWidth {
    match gp_reg_width {
        4 => MachineMemWidth::U32,
        8 => MachineMemWidth::U64,
        _ => panic!("unsupported GP register width"),
    }
}

/// Returns the scalar integer width matching one native GP register.
#[inline]
pub(crate) const fn machine_word_int_width(gp_reg_width: u8) -> MachineIntWidth {
    match gp_reg_width {
        4 => MachineIntWidth::I32,
        8 => MachineIntWidth::I64,
        _ => panic!("unsupported GP register width"),
    }
}

/// Integer sign mode where it matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineSign {
    Signed,
    Unsigned,
}

/// Integer unary ALU op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineIntUnaryOp {
    Eqz,
    Clz,
    Ctz,
    Popcnt,
    Extend8S,
    Extend16S,
    Extend32S,
}

/// Integer binary ALU op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineIntBinaryOp {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
}

/// Float unary ALU op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineFloatUnaryOp {
    Abs,
    Neg,
    Ceil,
    Floor,
    Trunc,
    Nearest,
    Sqrt,
}

/// Float binary ALU op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineFloatBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Copysign,
}

/// Compare relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineCompareKind {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

/// One conversion / reinterpret op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineConvertOp {
    I32WrapI64,
    I64ExtendI32S,
    I64ExtendI32U,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    I32TruncSatF32S,
    I32TruncSatF32U,
    I32TruncSatF64S,
    I32TruncSatF64U,
    I64TruncSatF32S,
    I64TruncSatF32U,
    I64TruncSatF64S,
    I64TruncSatF64U,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    F32DemoteF64,
    F64PromoteF32,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,
}

/// Memory load extension mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineLoadExtension {
    None,
    SignExtend,
    ZeroExtend,
}

/// Machine-level traps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineTrapKind {
    Unreachable,
    MemoryOutOfBounds,
    TableOutOfBounds,
    InvalidFunctionReference,
    IndirectCallTypeMismatch,
    IntegerDivideByZero,
    IntegerOverflow,
    StackOverflow,
    #[allow(dead_code)] // constructed by emulator backend (debug-only)
    HelperFailure,
}
