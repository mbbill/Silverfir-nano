//! New machine-facing IR.
//!
//! This module defines the target shape that `NativeIR` should converge to.
//! It intentionally avoids VM-flavored storage concepts such as hot locals,
//! TOS lanes, frame slots, or spills. The IR models only:
//! - generic registers
//! - explicit addresses
//! - explicit loads and stores
//! - arithmetic and conversion ops
//! - explicit control-flow and call boundaries
//!
//! It is added in parallel with the current `native::ir` so the refactor can
//! proceed incrementally.

use alloc::vec::Vec;

use crate::error::WasmError;

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

/// One helper signature identifier from the machine runtime contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineHelperId(pub u32);

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

/// One explicit edge into another block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineEdge {
    pub target: MachineBlockId,
    pub args: Vec<MachineValue>,
}

/// One explicit branch condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineBranchCond {
    Value(MachineValue),
    IntCompare {
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        lhs: MachineValue,
        rhs: MachineValue,
    },
    FloatCompare {
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        lhs: MachineValue,
        rhs: MachineValue,
    },
}

/// Helper call that falls through in the same function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineHelperCall {
    pub helper: MachineHelperId,
    pub args: Vec<MachineValue>,
    pub results: Vec<MachineReg>,
    /// Clobbered registers are explicit so the backend does not have to invent
    /// hidden helper-spill policy.
    pub clobbers: Vec<MachineReg>,
}

/// One machine instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineInst {
    pub kind: MachineInstKind,
}

/// Machine instruction vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MachineInstKind {
    Move {
        dst: MachineReg,
        src: MachineValue,
    },
    Lea {
        dst: MachineReg,
        addr: MachineAddr,
    },
    Load {
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    },
    Store {
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
    Convert {
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    },
    CallHelper(MachineHelperCall),
}

/// One machine IR block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineBlock {
    pub id: MachineBlockId,
    /// Block parameters are generic registers. Incoming values are supplied by
    /// the predecessor edge.
    pub params: Vec<MachineReg>,
    pub ops: Vec<MachineInst>,
    pub terminator: MachineTerminator,
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

/// One machine terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MachineTerminator {
    Jump(MachineEdge),
    Branch {
        cond: MachineBranchCond,
        then_edge: MachineEdge,
        else_edge: MachineEdge,
    },
    JumpTable {
        index: MachineValue,
        entries: Vec<MachineEdge>,
    },
    /// Direct local call. Call setup stores and call-link writes are explicit
    /// instructions before this terminator; the terminator performs the actual
    /// transfer to the callee body entry.
    CallDirect {
        callee: MachineFuncId,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    },
    /// Indirect local call after the target entry has already been resolved by
    /// earlier machine-level code.
    CallIndirect {
        callee_entry: MachineValue,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    },
    /// Helper-controlled transfer that resumes native execution using helper-
    /// returned native state.
    CallHelperTransfer {
        helper: MachineHelperId,
        args: Vec<MachineValue>,
    },
    Return {
        values: Vec<MachineValue>,
    },
    Trap {
        kind: MachineTrapKind,
    },
}

/// Full machine program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineProgram {
    pub entry: MachineBlockId,
    pub reg_count: u16,
    pub blocks: Vec<MachineBlock>,
}

impl MachineProgram {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self) -> Result<(), WasmError> {
        if self.blocks.is_empty() {
            if self.entry.as_usize() != 0 {
                return Err(WasmError::internal(
                    "empty machine program must use entry block 0".into(),
                ));
            }
            return Ok(());
        }

        if self.entry.as_usize() >= self.blocks.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine entry block {} is out of range for {} blocks",
                self.entry.as_usize(),
                self.blocks.len(),
            )));
        }

        for (index, block) in self.blocks.iter().enumerate() {
            if block.id.as_usize() != index {
                return Err(WasmError::internal(alloc::format!(
                    "machine block {} has mismatched id {}",
                    index,
                    block.id.as_usize(),
                )));
            }
            for param in &block.params {
                self.validate_reg(*param)?;
            }
            for inst in &block.ops {
                self.validate_inst(inst)?;
            }
            self.validate_term(&block.terminator, index)?;
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn validate(&self) -> Result<(), WasmError> {
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_inst(&self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src } => {
                self.validate_reg(*dst)?;
                self.validate_value(*src)?;
            }
            MachineInstKind::Lea { dst, addr } => {
                self.validate_reg(*dst)?;
                self.validate_addr(*addr)?;
            }
            MachineInstKind::Load { dst, addr, .. } => {
                self.validate_reg(*dst)?;
                self.validate_addr(*addr)?;
            }
            MachineInstKind::Store { addr, src, .. } => {
                self.validate_addr(*addr)?;
                self.validate_value(*src)?;
            }
            MachineInstKind::IntUnary { dst, src, .. }
            | MachineInstKind::FloatUnary { dst, src, .. }
            | MachineInstKind::Convert { dst, src, .. } => {
                self.validate_reg(*dst)?;
                self.validate_value(*src)?;
            }
            MachineInstKind::IntBinary { dst, lhs, rhs, .. }
            | MachineInstKind::FloatBinary { dst, lhs, rhs, .. } => {
                self.validate_reg(*dst)?;
                self.validate_value(*lhs)?;
                self.validate_value(*rhs)?;
            }
            MachineInstKind::CallHelper(call) => {
                for arg in &call.args {
                    self.validate_value(*arg)?;
                }
                for reg in &call.results {
                    self.validate_reg(*reg)?;
                }
                for reg in &call.clobbers {
                    self.validate_reg(*reg)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_term(
        &self,
        term: &MachineTerminator,
        source_block: usize,
    ) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => self.validate_edge(edge, source_block),
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                self.validate_branch_cond(*cond)?;
                self.validate_edge(then_edge, source_block)?;
                self.validate_edge(else_edge, source_block)
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.validate_value(*index)?;
                for edge in entries {
                    self.validate_edge(edge, source_block)?;
                }
                Ok(())
            }
            MachineTerminator::CallDirect {
                callee_frame_base,
                continuation,
                ..
            } => {
                self.validate_reg(*callee_frame_base)?;
                self.validate_block_id(*continuation, source_block, "continuation")
            }
            MachineTerminator::CallIndirect {
                callee_entry,
                callee_frame_base,
                continuation,
            } => {
                self.validate_value(*callee_entry)?;
                self.validate_reg(*callee_frame_base)?;
                self.validate_block_id(*continuation, source_block, "continuation")
            }
            MachineTerminator::CallHelperTransfer { args, .. } => {
                for arg in args {
                    self.validate_value(*arg)?;
                }
                Ok(())
            }
            MachineTerminator::Return { values } => {
                for value in values {
                    self.validate_value(*value)?;
                }
                Ok(())
            }
            MachineTerminator::Trap { .. } => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_branch_cond(&self, cond: MachineBranchCond) -> Result<(), WasmError> {
        match cond {
            MachineBranchCond::Value(value) => self.validate_value(value),
            MachineBranchCond::IntCompare { lhs, rhs, .. }
            | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
                self.validate_value(lhs)?;
                self.validate_value(rhs)
            }
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_edge(&self, edge: &MachineEdge, source_block: usize) -> Result<(), WasmError> {
        self.validate_block_id(edge.target, source_block, "edge target")?;
        let target = &self.blocks[edge.target.as_usize()];
        if edge.args.len() != target.params.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine block {} -> {} supplies {} args, but target expects {}",
                source_block,
                edge.target.as_usize(),
                edge.args.len(),
                target.params.len(),
            )));
        }
        for value in &edge.args {
            self.validate_value(*value)?;
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_block_id(
        &self,
        block: MachineBlockId,
        source_block: usize,
        role: &str,
    ) -> Result<(), WasmError> {
        if block.as_usize() >= self.blocks.len() {
            return Err(WasmError::internal(alloc::format!(
                "machine block {} has out-of-range {} {}",
                source_block,
                role,
                block.as_usize(),
            )));
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_value(&self, value: MachineValue) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(reg) => self.validate_reg(reg),
            MachineValue::Imm64(_) => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_addr(&self, addr: MachineAddr) -> Result<(), WasmError> {
        self.validate_reg(addr.base)
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_reg(&self, reg: MachineReg) -> Result<(), WasmError> {
        if reg.0 >= self.reg_count {
            return Err(WasmError::internal(alloc::format!(
                "machine register {} exceeds declared register count {}",
                reg.0,
                self.reg_count,
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_edge_arity_mismatch() {
        let program = MachineProgram {
            entry: MachineBlockId(0),
            reg_count: 2,
            blocks: alloc::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(1),
                        args: Vec::new(),
                    }),
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: alloc::vec![MachineReg(0)],
                    ops: Vec::new(),
                    terminator: MachineTerminator::Return { values: Vec::new() },
                },
            ],
        };

        let err = program.validate().unwrap_err();
        assert!(alloc::format!("{err}").contains("supplies 0 args"));
    }

    #[test]
    fn rejects_out_of_range_register() {
        let program = MachineProgram {
            entry: MachineBlockId(0),
            reg_count: 1,
            blocks: alloc::vec![MachineBlock {
                id: MachineBlockId(0),
                params: Vec::new(),
                ops: alloc::vec![MachineInst {
                    kind: MachineInstKind::Move {
                        dst: MachineReg(1),
                        src: MachineValue::Imm64(0),
                    },
                }],
                terminator: MachineTerminator::Return { values: Vec::new() },
            }],
        };

        let err = program.validate().unwrap_err();
        assert!(alloc::format!("{err}").contains("exceeds declared register count"));
    }
}
