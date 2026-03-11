//! Lower target-independent native IR into ARM64 code.
//!
//! This file keeps the ISA layer mechanical:
//! - map placed VM locations onto fixed ARM64 registers
//! - materialize frame/spill accesses
//! - emit block control flow
//!
//! In debug/test builds, unsupported native programs still use the shared
//! reference entry while direct ARM64 coverage is being grown.
//! Release builds reject unsupported programs instead of compiling that path.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        lir::{leaf::LirLeafOp, slot::FrameSlot},
        native::{
            abi::{NativeLocation, NativePlace, NativeValue},
            ir::{
                NativeBlockId, NativeEdge, NativeInst, NativeInstKind, NativeProgram,
                NativeTerminator,
            },
        },
    },
};

use super::{
    emit::Arm64TextEmitter,
    enc::{self, Cond},
    entry::Arm64EntryPatch,
    reg::Arm64Reg,
};
#[cfg(any(debug_assertions, test))]
use super::entry::shared_native_entry;

const REG_CTX: Arm64Reg = Arm64Reg::X19;
const REG_FP: Arm64Reg = Arm64Reg::X20;
const REG_HOT0: Arm64Reg = Arm64Reg::X21;
const REG_HOT1: Arm64Reg = Arm64Reg::X22;
const REG_HOT2: Arm64Reg = Arm64Reg::X23;
const REG_TOS0: Arm64Reg = Arm64Reg::X24;
const REG_TOS1: Arm64Reg = Arm64Reg::X25;
const REG_TOS2: Arm64Reg = Arm64Reg::X26;
const REG_TOS3: Arm64Reg = Arm64Reg::X27;
const REG_TMP0: Arm64Reg = Arm64Reg::X9;
const REG_TMP1: Arm64Reg = Arm64Reg::X10;
const REG_TMP2: Arm64Reg = Arm64Reg::X11;
const REG_TMP3: Arm64Reg = Arm64Reg::X12;
const REG_SCRATCH0: Arm64Reg = Arm64Reg::X13;
const REG_SCRATCH1: Arm64Reg = Arm64Reg::X14;
const REG_COPY_TEMP: Arm64Reg = Arm64Reg::X15;
const REG_SP: Arm64Reg = Arm64Reg::Xzr;
const SAVED_REGS: [Arm64Reg; 9] = [
    REG_CTX, REG_FP, REG_HOT0, REG_HOT1, REG_HOT2, REG_TOS0, REG_TOS1, REG_TOS2, REG_TOS3,
];
const SAVE_BYTES: u32 = 80;

/// Result of lowering one function to ARM64.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Arm64LoweredFunction {
    pub text: Vec<u8>,
    pub entry_patches: Vec<Arm64EntryPatch>,
}

pub fn lower_arm64(program: &NativeProgram) -> Result<Arm64LoweredFunction, WasmError> {
    if program.blocks.is_empty() {
        return Err(WasmError::internal(
            "arm64 lowering requires at least one native block".into(),
        ));
    }

    if !supports_direct_lowering(program) {
        #[cfg(any(debug_assertions, test))]
        {
            return Ok(lower_shared_entry());
        }
        #[cfg(not(any(debug_assertions, test)))]
        {
            return Err(WasmError::internal(
                "arm64 direct lowering does not support this native program yet; reference fallback is debug-only".into(),
            ));
        }
    }

    Arm64Lowerer::new(program)?.lower()
}

#[cfg(any(debug_assertions, test))]
fn lower_shared_entry() -> Arm64LoweredFunction {
    let mut emitter = Arm64TextEmitter::new();
    emitter.emit_tail_branch_literal(Arm64Reg::X16, shared_native_entry as usize as u64);
    Arm64LoweredFunction {
        text: emitter.finish(),
        entry_patches: Vec::new(),
    }
}

fn supports_direct_lowering(program: &NativeProgram) -> bool {
    for block in &program.blocks {
        if block.entry_abi.tos_width > 4 {
            return false;
        }
        for inst in &block.ops {
            match &inst.kind {
                NativeInstKind::Move(_) => {}
                NativeInstKind::Leaf { op, results, .. } => {
                    if results.len() > 1 || !supported_leaf(op) {
                        return false;
                    }
                }
                NativeInstKind::CallExternal { .. }
                | NativeInstKind::CallLocal { .. }
                | NativeInstKind::CallIndirect { .. } => return false,
            }
        }
        match &block.terminator {
            NativeTerminator::Goto(_)
            | NativeTerminator::Branch { .. }
            | NativeTerminator::Return { .. } => {}
            NativeTerminator::BrTable { .. } | NativeTerminator::TrapUnreachable => return false,
        }
    }
    true
}

fn supported_leaf(op: &LirLeafOp) -> bool {
    matches!(
        op,
        LirLeafOp::I32Add
            | LirLeafOp::I32Mul
            | LirLeafOp::I32Sub
            | LirLeafOp::I32And
            | LirLeafOp::I32Or
            | LirLeafOp::I32Xor
            | LirLeafOp::I32Shl
            | LirLeafOp::I32ShrS
            | LirLeafOp::I32ShrU
            | LirLeafOp::I32Rotr
            | LirLeafOp::I64Add
            | LirLeafOp::I64Mul
            | LirLeafOp::I64Sub
            | LirLeafOp::I64And
            | LirLeafOp::I64Or
            | LirLeafOp::I64Xor
            | LirLeafOp::I64Shl
            | LirLeafOp::I64ShrS
            | LirLeafOp::I64ShrU
            | LirLeafOp::I64Rotr
            | LirLeafOp::I32Eq
            | LirLeafOp::I32Ne
            | LirLeafOp::I32LtS
            | LirLeafOp::I32LtU
            | LirLeafOp::I32GtS
            | LirLeafOp::I32GtU
            | LirLeafOp::I32LeS
            | LirLeafOp::I32LeU
            | LirLeafOp::I32GeS
            | LirLeafOp::I32GeU
            | LirLeafOp::I64Eq
            | LirLeafOp::I64Ne
            | LirLeafOp::I64LtS
            | LirLeafOp::I64LtU
            | LirLeafOp::I64GtS
            | LirLeafOp::I64GtU
            | LirLeafOp::I64LeS
            | LirLeafOp::I64LeU
            | LirLeafOp::I64GeS
            | LirLeafOp::I64GeU
            | LirLeafOp::I32Eqz
            | LirLeafOp::I32Clz
            | LirLeafOp::I32Ctz
            | LirLeafOp::I64Eqz
            | LirLeafOp::I64Clz
            | LirLeafOp::I64Ctz
            | LirLeafOp::I32WrapI64
            | LirLeafOp::I64ExtendI32S
            | LirLeafOp::I64ExtendI32U
            | LirLeafOp::I32Extend8S
            | LirLeafOp::I32Extend16S
            | LirLeafOp::I64Extend8S
            | LirLeafOp::I64Extend16S
            | LirLeafOp::I64Extend32S
            | LirLeafOp::I32Const { .. }
            | LirLeafOp::I64Const { .. }
            | LirLeafOp::F32Const { .. }
            | LirLeafOp::F64Const { .. }
            | LirLeafOp::Nop
            | LirLeafOp::Drop
    )
}

#[derive(Clone, Copy, Debug)]
struct BlockBranchPatch {
    offset: usize,
    target: NativeBlockId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CopySource {
    Value(NativeValue),
    Saved,
}

struct Arm64Lowerer<'a> {
    program: &'a NativeProgram,
    emitter: Arm64TextEmitter,
    block_offsets: Vec<Option<usize>>,
    branch_patches: Vec<BlockBranchPatch>,
}

impl<'a> Arm64Lowerer<'a> {
    fn new(program: &'a NativeProgram) -> Result<Self, WasmError> {
        let max_slot = program.frame.operands.end().0 as u32 + program.spill_slots as u32;
        if max_slot >= 0x1000 {
            return Err(WasmError::internal(
                "arm64 direct lowering does not support >=4096 frame slots yet".into(),
            ));
        }
        if program.abi.ctx_register_count != 1
            || program.abi.fp_register_count != 1
            || program.abi.hot_local_count > 3
            || program.abi.tos_register_count > 4
            || program.abi.tmp_register_count > 4
        {
            return Err(WasmError::internal(
                "arm64 lowering does not support the configured native ABI budget".into(),
            ));
        }
        if program
            .blocks
            .iter()
            .any(|block| block.entry_abi.tos_width > 4)
        {
            return Err(WasmError::internal(
                "arm64 lowering does not support block entries wider than 4 TOS lanes".into(),
            ));
        }
        Ok(Self {
            program,
            emitter: Arm64TextEmitter::new(),
            block_offsets: alloc::vec![None; program.blocks.len()],
            branch_patches: Vec::new(),
        })
    }

    fn lower(mut self) -> Result<Arm64LoweredFunction, WasmError> {
        self.emit_prologue();
        self.emit_branch_to_block(self.program.entry);

        for block in &self.program.blocks {
            self.block_offsets[block.id.as_usize()] = Some(self.emitter.len());
            for inst in &block.ops {
                self.emit_inst(inst)?;
            }
            self.emit_terminator(&block.terminator)?;
        }

        self.patch_block_branches()?;

        Ok(Arm64LoweredFunction {
            text: self.emitter.finish(),
            entry_patches: Vec::new(),
        })
    }

    fn emit_prologue(&mut self) {
        self.emitter
            .emit_u32(enc::sub_imm_64(REG_SP, REG_SP, SAVE_BYTES));
        for (slot, reg) in SAVED_REGS.iter().copied().enumerate() {
            self.emitter.emit_u32(enc::str_64(reg, REG_SP, slot as u32));
        }

        self.move_if_needed(REG_CTX, Arm64Reg::X0);
        self.move_if_needed(REG_FP, Arm64Reg::X1);
        self.move_if_needed(REG_HOT0, Arm64Reg::X2);
        self.move_if_needed(REG_HOT1, Arm64Reg::X3);
        self.move_if_needed(REG_HOT2, Arm64Reg::X4);
        self.move_if_needed(REG_TOS0, Arm64Reg::X5);
        self.move_if_needed(REG_TOS1, Arm64Reg::X6);
        self.move_if_needed(REG_TOS2, Arm64Reg::X7);
        self.emitter
            .emit_u32(enc::ldr_64(REG_TOS3, REG_SP, SAVE_BYTES / 8));
    }

    fn emit_epilogue(&mut self) {
        for (slot, reg) in SAVED_REGS.iter().copied().enumerate().rev() {
            self.emitter.emit_u32(enc::ldr_64(reg, REG_SP, slot as u32));
        }
        self.emitter
            .emit_u32(enc::add_imm_64(REG_SP, REG_SP, SAVE_BYTES));
        self.emitter.emit_u32(enc::ret());
    }

    fn emit_inst(&mut self, inst: &NativeInst) -> Result<(), WasmError> {
        match &inst.kind {
            NativeInstKind::Move(mov) => self.emit_move(mov.dst, mov.src),
            NativeInstKind::Leaf { op, args, results } => self.emit_leaf(op, args, results),
            NativeInstKind::CallExternal { .. }
            | NativeInstKind::CallLocal { .. }
            | NativeInstKind::CallIndirect { .. } => Err(WasmError::internal(
                "call ops are not supported by direct arm64 lowering yet".into(),
            )),
        }
    }

    fn emit_leaf(
        &mut self,
        op: &LirLeafOp,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        match op {
            LirLeafOp::Nop | LirLeafOp::Drop => Ok(()),
            LirLeafOp::I32Const { value } => self.emit_constant(*value as u64, results),
            LirLeafOp::I64Const { value } => self.emit_constant(*value, results),
            LirLeafOp::F32Const { value } => self.emit_constant(*value as u64, results),
            LirLeafOp::F64Const { value } => self.emit_constant(*value, results),

            LirLeafOp::I32Add => self.emit_binary_op32(args, results, enc::add_reg_32),
            LirLeafOp::I32Sub => self.emit_binary_op32(args, results, enc::sub_reg_32),
            LirLeafOp::I32Mul => self.emit_binary_op32(args, results, enc::mul_32),
            LirLeafOp::I32And => self.emit_binary_op32(args, results, enc::and_reg_32),
            LirLeafOp::I32Or => self.emit_binary_op32(args, results, enc::orr_reg_32),
            LirLeafOp::I32Xor => self.emit_binary_op32(args, results, enc::eor_reg_32),
            LirLeafOp::I32Shl => self.emit_binary_op32(args, results, enc::lslv_32),
            LirLeafOp::I32ShrS => self.emit_binary_op32(args, results, enc::asrv_32),
            LirLeafOp::I32ShrU => self.emit_binary_op32(args, results, enc::lsrv_32),
            LirLeafOp::I32Rotr => self.emit_binary_op32(args, results, enc::rorv_32),

            LirLeafOp::I64Add => self.emit_binary_op64(args, results, enc::add_reg_64),
            LirLeafOp::I64Sub => self.emit_binary_op64(args, results, enc::sub_reg_64),
            LirLeafOp::I64Mul => self.emit_binary_op64(args, results, enc::mul_64),
            LirLeafOp::I64And => self.emit_binary_op64(args, results, enc::and_reg_64),
            LirLeafOp::I64Or => self.emit_binary_op64(args, results, enc::orr_reg_64),
            LirLeafOp::I64Xor => self.emit_binary_op64(args, results, enc::eor_reg_64),
            LirLeafOp::I64Shl => self.emit_binary_op64(args, results, enc::lslv_64),
            LirLeafOp::I64ShrS => self.emit_binary_op64(args, results, enc::asrv_64),
            LirLeafOp::I64ShrU => self.emit_binary_op64(args, results, enc::lsrv_64),
            LirLeafOp::I64Rotr => self.emit_binary_op64(args, results, enc::rorv_64),

            LirLeafOp::I32Eq => self.emit_compare32(args, results, Cond::Eq),
            LirLeafOp::I32Ne => self.emit_compare32(args, results, Cond::Ne),
            LirLeafOp::I32LtS => self.emit_compare32(args, results, Cond::Lt),
            LirLeafOp::I32LtU => self.emit_compare32(args, results, Cond::Lo),
            LirLeafOp::I32GtS => self.emit_compare32(args, results, Cond::Gt),
            LirLeafOp::I32GtU => self.emit_compare32(args, results, Cond::Hi),
            LirLeafOp::I32LeS => self.emit_compare32(args, results, Cond::Le),
            LirLeafOp::I32LeU => self.emit_compare32(args, results, Cond::Ls),
            LirLeafOp::I32GeS => self.emit_compare32(args, results, Cond::Ge),
            LirLeafOp::I32GeU => self.emit_compare32(args, results, Cond::Hs),

            LirLeafOp::I64Eq => self.emit_compare64(args, results, Cond::Eq),
            LirLeafOp::I64Ne => self.emit_compare64(args, results, Cond::Ne),
            LirLeafOp::I64LtS => self.emit_compare64(args, results, Cond::Lt),
            LirLeafOp::I64LtU => self.emit_compare64(args, results, Cond::Lo),
            LirLeafOp::I64GtS => self.emit_compare64(args, results, Cond::Gt),
            LirLeafOp::I64GtU => self.emit_compare64(args, results, Cond::Hi),
            LirLeafOp::I64LeS => self.emit_compare64(args, results, Cond::Le),
            LirLeafOp::I64LeU => self.emit_compare64(args, results, Cond::Ls),
            LirLeafOp::I64GeS => self.emit_compare64(args, results, Cond::Ge),
            LirLeafOp::I64GeU => self.emit_compare64(args, results, Cond::Hs),

            LirLeafOp::I32Eqz => self.emit_eqz32(args, results),
            LirLeafOp::I32Clz => self.emit_unary_op32(args, results, enc::clz_32),
            LirLeafOp::I32Ctz => self.emit_ctz32(args, results),
            LirLeafOp::I64Eqz => self.emit_eqz64(args, results),
            LirLeafOp::I64Clz => self.emit_unary_op64(args, results, enc::clz_64),
            LirLeafOp::I64Ctz => self.emit_ctz64(args, results),
            LirLeafOp::I32WrapI64 => self.emit_unary_op32(args, results, enc::mov_reg_32),
            LirLeafOp::I64ExtendI32S => self.emit_unary_op64(args, results, enc::sxtw),
            LirLeafOp::I64ExtendI32U => self.emit_unary_op32_to_64(args, results),
            LirLeafOp::I32Extend8S => self.emit_unary_op32(args, results, enc::sxtb_32),
            LirLeafOp::I32Extend16S => self.emit_unary_op32(args, results, enc::sxth_32),
            LirLeafOp::I64Extend8S => self.emit_unary_op64(args, results, enc::sxtb_64),
            LirLeafOp::I64Extend16S => self.emit_unary_op64(args, results, enc::sxth_64),
            LirLeafOp::I64Extend32S => self.emit_unary_op64(args, results, enc::sxtw),
            _ => Err(WasmError::internal(alloc::format!(
                "arm64 direct lowering does not support leaf op {:?}",
                op
            ))),
        }
    }

    fn emit_constant(&mut self, value: u64, results: &[NativePlace]) -> Result<(), WasmError> {
        if let Some(&dst) = results.first() {
            self.emit_move(dst, NativeValue::Imm64(value))?;
        }
        Ok(())
    }

    fn emit_binary_op32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        encode: fn(Arm64Reg, Arm64Reg, Arm64Reg) -> u32,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], dst_reg)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(encode(dst_reg, dst_reg, REG_SCRATCH1));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_binary_op64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        encode: fn(Arm64Reg, Arm64Reg, Arm64Reg) -> u32,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], dst_reg)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(encode(dst_reg, dst_reg, REG_SCRATCH1));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_compare32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        cond: Cond,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(enc::cmp_reg_32(REG_SCRATCH0, REG_SCRATCH1));
        self.emitter.emit_u32(enc::cset_64(dst_reg, cond));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_compare64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        cond: Cond,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_SCRATCH0, REG_SCRATCH1));
        self.emitter.emit_u32(enc::cset_64(dst_reg, cond));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_unary_op32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        encode: fn(Arm64Reg, Arm64Reg) -> u32,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.emitter.emit_u32(encode(dst_reg, REG_SCRATCH0));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_unary_op64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        encode: fn(Arm64Reg, Arm64Reg) -> u32,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.emitter.emit_u32(encode(dst_reg, REG_SCRATCH0));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_unary_op32_to_64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.emitter.emit_u32(enc::mov_reg_32(dst_reg, REG_SCRATCH0));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_eqz32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.emitter.emit_u32(enc::cmp_imm_32(REG_SCRATCH0, 0));
        self.emitter.emit_u32(enc::cset_64(dst_reg, Cond::Eq));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_eqz64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.emitter.emit_u32(enc::cmp_imm_64(REG_SCRATCH0, 0));
        self.emitter.emit_u32(enc::cset_64(dst_reg, Cond::Eq));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_ctz32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.emitter.emit_u32(enc::rbit_32(REG_SCRATCH0, REG_SCRATCH0));
        self.emitter.emit_u32(enc::clz_32(dst_reg, REG_SCRATCH0));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_ctz64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.emitter.emit_u32(enc::rbit_64(REG_SCRATCH0, REG_SCRATCH0));
        self.emitter.emit_u32(enc::clz_64(dst_reg, REG_SCRATCH0));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_terminator(&mut self, term: &NativeTerminator) -> Result<(), WasmError> {
        match term {
            NativeTerminator::Goto(edge) => {
                self.emit_edge(edge)?;
                self.emit_branch_to_block(edge.target);
                Ok(())
            }
            NativeTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                if let Some(taken) = self.constant_branch(cond, then_edge, else_edge) {
                    self.emit_edge(taken)?;
                    self.emit_branch_to_block(taken.target);
                    return Ok(());
                }

                self.read_value_into(*cond, REG_SCRATCH0)?;
                let else_patch = self.emitter.len();
                self.emitter.emit_u32(enc::cbz_64(REG_SCRATCH0, 0));
                self.emit_edge(then_edge)?;
                self.emit_branch_to_block(then_edge.target);
                let else_offset = self.emitter.len();
                self.patch_cbz(else_patch, REG_SCRATCH0, else_offset);
                self.emit_edge(else_edge)?;
                self.emit_branch_to_block(else_edge.target);
                Ok(())
            }
            NativeTerminator::Return { values } => {
                for (index, value) in values.iter().copied().enumerate() {
                    self.emit_move(NativePlace::Frame(FrameSlot(index as u16)), value)?;
                }
                self.emit_epilogue();
                Ok(())
            }
            NativeTerminator::BrTable { .. } | NativeTerminator::TrapUnreachable => {
                Err(WasmError::internal(
                    "arm64 direct lowering does not support this terminator yet".into(),
                ))
            }
        }
    }

    fn constant_branch<'b>(
        &self,
        cond: &NativeValue,
        then_edge: &'b NativeEdge,
        else_edge: &'b NativeEdge,
    ) -> Option<&'b NativeEdge> {
        match cond {
            NativeValue::Imm64(value) => Some(if *value == 0 { else_edge } else { then_edge }),
            _ => None,
        }
    }

    fn emit_edge(&mut self, edge: &NativeEdge) -> Result<(), WasmError> {
        let mut pending: Vec<(NativePlace, CopySource)> = edge
            .copies
            .iter()
            .copied()
            .map(|mov| (mov.dst, CopySource::Value(mov.src)))
            .collect();

        while !pending.is_empty() {
            let mut progressed = false;
            let mut index = 0usize;
            while index < pending.len() {
                let dst = pending[index].0;
                let read_later = pending
                    .iter()
                    .enumerate()
                    .any(|(other_index, (_, src))| other_index != index && reads_place(*src, dst));
                if !read_later {
                    let (dst, src) = pending.remove(index);
                    self.emit_copy_source(dst, src)?;
                    progressed = true;
                } else {
                    index += 1;
                }
            }

            if progressed {
                continue;
            }

            let (dst, src) = pending.remove(0);
            let CopySource::Value(NativeValue::Place(saved_place)) = src else {
                return Err(WasmError::internal(
                    "arm64 copy cycle must start from a real place".into(),
                ));
            };
            self.read_place_into(saved_place, REG_COPY_TEMP)?;
            for (_, pending_src) in &mut pending {
                if *pending_src == CopySource::Value(NativeValue::Place(saved_place)) {
                    *pending_src = CopySource::Saved;
                }
            }
            pending.push((dst, CopySource::Saved));
        }

        Ok(())
    }

    fn emit_copy_source(&mut self, dst: NativePlace, src: CopySource) -> Result<(), WasmError> {
        match src {
            CopySource::Value(value) => self.emit_move(dst, value),
            CopySource::Saved => self.write_reg_to_place(dst, REG_COPY_TEMP),
        }
    }

    fn emit_move(&mut self, dst: NativePlace, src: NativeValue) -> Result<(), WasmError> {
        match dst {
            NativePlace::Location(loc) => self.move_value_into_reg(map_location(loc), src),
            _ => {
                self.read_value_into(src, REG_SCRATCH0)?;
                self.write_reg_to_place(dst, REG_SCRATCH0)
            }
        }
    }

    fn move_value_into_reg(&mut self, dst: Arm64Reg, src: NativeValue) -> Result<(), WasmError> {
        match src {
            NativeValue::Imm64(value) => {
                self.materialize_u64(dst, value);
                Ok(())
            }
            NativeValue::Place(place) => match place {
                NativePlace::Location(loc) => {
                    self.move_if_needed(dst, map_location(loc));
                    Ok(())
                }
                _ => self.read_place_into(place, dst),
            },
        }
    }

    fn read_value_into(&mut self, value: NativeValue, dst: Arm64Reg) -> Result<(), WasmError> {
        match value {
            NativeValue::Imm64(value) => {
                self.materialize_u64(dst, value);
                Ok(())
            }
            NativeValue::Place(place) => self.read_place_into(place, dst),
        }
    }

    fn read_place_into(&mut self, place: NativePlace, dst: Arm64Reg) -> Result<(), WasmError> {
        match place {
            NativePlace::Location(loc) => {
                self.move_if_needed(dst, map_location(loc));
                Ok(())
            }
            NativePlace::Frame(_) | NativePlace::Spill(_) => {
                self.emitter
                    .emit_u32(enc::ldr_64(dst, REG_FP, self.slot_index(place)?));
                Ok(())
            }
        }
    }

    fn write_reg_to_place(&mut self, dst: NativePlace, src: Arm64Reg) -> Result<(), WasmError> {
        match dst {
            NativePlace::Location(loc) => {
                self.move_if_needed(map_location(loc), src);
                Ok(())
            }
            NativePlace::Frame(_) | NativePlace::Spill(_) => {
                self.emitter
                    .emit_u32(enc::str_64(src, REG_FP, self.slot_index(dst)?));
                Ok(())
            }
        }
    }

    fn write_result(
        &mut self,
        result: Option<NativePlace>,
        src: Arm64Reg,
    ) -> Result<(), WasmError> {
        if let Some(place) = result {
            self.write_reg_to_place(place, src)?;
        }
        Ok(())
    }

    fn slot_index(&self, place: NativePlace) -> Result<u32, WasmError> {
        let slot = match place {
            NativePlace::Frame(slot) => slot.0 as u32,
            NativePlace::Spill(slot) => self.program.frame.operands.end().0 as u32 + slot.0 as u32,
            NativePlace::Location(_) => {
                return Err(WasmError::internal(
                    "slot_index called with register-backed place".into(),
                ))
            }
        };
        if slot >= 0x1000 {
            return Err(WasmError::internal(
                "arm64 slot index exceeds load/store immediate range".into(),
            ));
        }
        Ok(slot)
    }

    fn materialize_u64(&mut self, dst: Arm64Reg, value: u64) {
        let chunks = [
            ((value & 0xffff) as u16, 0),
            (((value >> 16) & 0xffff) as u16, 16),
            (((value >> 32) & 0xffff) as u16, 32),
            (((value >> 48) & 0xffff) as u16, 48),
        ];
        let mut first = true;
        for &(chunk, shift) in &chunks {
            if chunk != 0 || first {
                if first {
                    self.emitter.emit_u32(enc::movz_64(dst, chunk, shift));
                    first = false;
                } else {
                    self.emitter.emit_u32(enc::movk_64(dst, chunk, shift));
                }
            }
        }
    }

    fn move_if_needed(&mut self, dst: Arm64Reg, src: Arm64Reg) {
        if dst != src {
            self.emitter.emit_u32(enc::mov_reg_64(dst, src));
        }
    }

    fn emit_branch_to_block(&mut self, target: NativeBlockId) {
        let offset = self.emitter.len();
        self.emitter.emit_u32(enc::b(0));
        self.branch_patches
            .push(BlockBranchPatch { offset, target });
    }

    fn patch_cbz(&mut self, patch_offset: usize, reg: Arm64Reg, target_offset: usize) {
        let delta = ((target_offset as isize - patch_offset as isize) / 4) as i32;
        self.emitter
            .patch_u32(patch_offset, enc::cbz_64(reg, delta));
    }

    fn patch_block_branches(&mut self) -> Result<(), WasmError> {
        for patch in &self.branch_patches {
            let target = self
                .block_offsets
                .get(patch.target.as_usize())
                .and_then(|offset| *offset)
                .ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "arm64 lowering missing block offset for target {}",
                        patch.target.as_usize()
                    ))
                })?;
            let delta = ((target as isize - patch.offset as isize) / 4) as i32;
            self.emitter.patch_u32(patch.offset, enc::b(delta));
        }
        Ok(())
    }

    fn result_reg(&self, result: Option<NativePlace>) -> Arm64Reg {
        match result {
            Some(NativePlace::Location(loc)) => map_location(loc),
            _ => REG_TMP0,
        }
    }
}

fn map_location(loc: NativeLocation) -> Arm64Reg {
    match loc {
        NativeLocation::Ctx => REG_CTX,
        NativeLocation::Fp => REG_FP,
        NativeLocation::Hot(0) => REG_HOT0,
        NativeLocation::Hot(1) => REG_HOT1,
        NativeLocation::Hot(2) => REG_HOT2,
        NativeLocation::Hot(reg) => panic!("unsupported arm64 hot reg {}", reg),
        NativeLocation::Tos(0) => REG_TOS0,
        NativeLocation::Tos(1) => REG_TOS1,
        NativeLocation::Tos(2) => REG_TOS2,
        NativeLocation::Tos(3) => REG_TOS3,
        NativeLocation::Tos(lane) => panic!("unsupported arm64 tos lane {}", lane),
        NativeLocation::Tmp(0) => REG_TMP0,
        NativeLocation::Tmp(1) => REG_TMP1,
        NativeLocation::Tmp(2) => REG_TMP2,
        NativeLocation::Tmp(3) => REG_TMP3,
        NativeLocation::Tmp(reg) => panic!("unsupported arm64 tmp reg {}", reg),
    }
}

fn reads_place(src: CopySource, place: NativePlace) -> bool {
    matches!(src, CopySource::Value(NativeValue::Place(src_place)) if src_place == place)
}
