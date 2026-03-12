/// One generic machine register.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineReg(pub u16);

/// One machine CFG block id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineBlockId(pub u32);

impl MachineBlockId {
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// One local-function identifier in the machine-level call graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineFuncId(pub u32);

/// One opaque external target id referenced from machine IR.
///
/// The machine IR does not know whether this resolves to a helper wrapper or
/// some other external native target. Sidecar binding data owns that meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineExternId(pub u32);

/// One read-only sidecar constant record referenced from machine IR.
///
/// This is used for immutable helper metadata or other finalized constant data
/// that should live beside code, not inside writable frame scratch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineConstId(pub u32);

/// One readable machine operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineValue {
    Reg(MachineReg),
    Imm64(u64),
}

/// One explicit machine address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MachineAddr {
    pub base: MachineReg,
    pub offset: i32,
}

/// Scalar integer width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineIntWidth {
    I32,
    I64,
}

/// Scalar float width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineFloatWidth {
    F32,
    F64,
}

/// Width of one memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineMemWidth {
    U8,
    U16,
    U32,
    U64,
}

/// Integer sign mode where it matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineSign {
    Signed,
    Unsigned,
}

/// Integer unary ALU op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineIntUnaryOp {
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
pub enum MachineIntBinaryOp {
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
pub enum MachineFloatUnaryOp {
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
pub enum MachineFloatBinaryOp {
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
pub enum MachineCompareKind {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

/// One conversion / reinterpret op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineConvertOp {
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
pub enum MachineLoadExtension {
    None,
    SignExtend,
    ZeroExtend,
}

/// Machine-level traps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineTrapKind {
    Unreachable,
    MemoryOutOfBounds,
    IntegerDivideByZero,
    IntegerOverflow,
    CallStackExhausted,
    StackOverflow,
    HelperFailure,
}
