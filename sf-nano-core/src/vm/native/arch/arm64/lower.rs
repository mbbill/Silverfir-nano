//! Lower target-independent native IR into ARM64 code.
//!
//! This file keeps the ISA layer mechanical:
//! - map placed VM locations onto fixed ARM64 registers
//! - materialize frame/spill accesses
//! - emit block control flow
//!
use alloc::vec::Vec;

use crate::{
    constants::MAX_CALL_STACK_DEPTH,
    error::WasmError,
    vm::{
        lir::{leaf::LirLeafOp, slot::FrameSlot},
        native::{
            abi::{NativeLocation, NativePlace, NativeValue},
            code::DirectCallPatch,
            ir::{
                NativeBlockId, NativeBranchCond, NativeCompareKind, NativeEdge, NativeHelperEffect,
                NativeHelperInst, NativeInst, NativeInstKind, NativeProgram, NativeTerminator,
            },
            runtime::context::NativeContext,
        },
    },
};

use super::{
    emit::Arm64TextEmitter,
    enc::{self, Cond},
    reg::Arm64Reg,
};
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
const REG_MEMORY_LEN: Arm64Reg = Arm64Reg::X28;
const REG_MEMORY_BASE: Arm64Reg = Arm64Reg::X8;
const REG_TMP0: Arm64Reg = Arm64Reg::X9;
const REG_TMP1: Arm64Reg = Arm64Reg::X10;
const REG_TMP2: Arm64Reg = Arm64Reg::X11;
const REG_TMP3: Arm64Reg = Arm64Reg::X12;
const REG_SCRATCH0: Arm64Reg = Arm64Reg::X13;
const REG_SCRATCH1: Arm64Reg = Arm64Reg::X14;
const REG_GLOBALS_BASE: Arm64Reg = Arm64Reg::X15;
const REG_COPY_TEMP: Arm64Reg = Arm64Reg::X16;
const REG_CONT_FP: Arm64Reg = Arm64Reg::X16;
const REG_BRANCH_TARGET: Arm64Reg = Arm64Reg::X17;
const REG_SP: Arm64Reg = Arm64Reg::Xzr;
const SAVED_REGS: [Arm64Reg; 10] = [
    REG_CTX,
    REG_FP,
    REG_HOT0,
    REG_HOT1,
    REG_HOT2,
    REG_TOS0,
    REG_TOS1,
    REG_TOS2,
    REG_TOS3,
    REG_MEMORY_LEN,
];
const ROOT_LR_SLOT: u32 = SAVED_REGS.len() as u32;
const SAVE_BYTES: u32 = 96;
const MAX_ARM64_CALL_DEPTH: u64 = if MAX_CALL_STACK_DEPTH < 300 {
    MAX_CALL_STACK_DEPTH as u64
} else {
    300
};
const STACK_END_OFFSET: u32 = (core::mem::offset_of!(NativeContext, stack_end) / 8) as u32;
const MEMORY0_BASE_OFFSET: u32 = (core::mem::offset_of!(NativeContext, memory0_base) / 8) as u32;
const MEMORY0_LEN_OFFSET: u32 = (core::mem::offset_of!(NativeContext, memory0_len) / 8) as u32;
const GLOBALS_BASE_OFFSET: u32 = (core::mem::offset_of!(NativeContext, globals_base) / 8) as u32;
const CALL_DEPTH_OFFSET: u32 = (core::mem::offset_of!(NativeContext, call_depth) / 8) as u32;
const TERM_ENTRY_OFFSET: u32 = (core::mem::offset_of!(NativeContext, term_entry) / 8) as u32;
const GLOBAL_MUTABLE_OFFSET: u32 = core::mem::offset_of!(crate::vm::entities::GlobalInst, mutable) as u32;
const CALL_EXTERNAL_REQ_SLOTS: usize =
    core::mem::size_of::<super::entry::Arm64CallExternalRequest>() / 8;
const CALL_INDIRECT_REQ_SLOTS: usize =
    core::mem::size_of::<super::entry::Arm64CallIndirectRequest>() / 8;

#[inline]
const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

#[inline]
const fn helper_stack_bytes(args_len: usize, results_len: usize, req_slots: usize) -> usize {
    align_up((args_len + results_len + req_slots) * 8, 16)
}

/// Result of lowering one function to ARM64.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Arm64LoweredFunction {
    pub text: Vec<u8>,
    pub entry_offset: u32,
    pub internal_entry_offset: Option<u32>,
    pub local_literal_patches: Vec<Arm64LocalLiteralPatch>,
    pub direct_call_patches: Vec<DirectCallPatch>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm64LocalLiteralPatch {
    pub literal_offset: u32,
    pub target_offset: u32,
}

pub fn lower_arm64(program: &NativeProgram) -> Result<Arm64LoweredFunction, WasmError> {
    if program.blocks.is_empty() {
        return Err(WasmError::internal(
            "arm64 lowering requires at least one native block".into(),
        ));
    }
    Arm64Lowerer::new(program)?.lower()
}

pub(crate) fn lower_shared_entry() -> Arm64LoweredFunction {
    let mut emitter = Arm64TextEmitter::new();
    emitter.emit_tail_branch_literal(Arm64Reg::X16, shared_native_entry as usize as u64);
    Arm64LoweredFunction {
        text: emitter.finish(),
        entry_offset: 0,
        internal_entry_offset: None,
        local_literal_patches: Vec::new(),
        direct_call_patches: Vec::new(),
    }
}

pub(crate) fn supports_direct_lowering_base(program: &NativeProgram) -> bool {
    for block in &program.blocks {
        if block.entry_abi.tos_width > 4 {
            return false;
        }
        for inst in &block.ops {
            match &inst.kind {
                NativeInstKind::Move(_) => {}
                NativeInstKind::Leaf { op, results, .. } => {
                    if results.len() > 1 || !supported_direct_leaf(op) {
                        return false;
                    }
                }
                NativeInstKind::Helper(NativeHelperInst::Leaf {
                    effect,
                    op,
                    results,
                    ..
                }) => {
                    if *effect != NativeHelperEffect::Transparent
                        || results.len() > 1
                        || !supported_direct_leaf(op)
                    {
                        return false;
                    }
                }
                NativeInstKind::CallLocal { results, .. } => {
                    if results.len() > 4 {
                        return false;
                    }
                }
                NativeInstKind::CallExternal { args, results, .. } => {
                    if helper_stack_bytes(args.len(), results.len(), CALL_EXTERNAL_REQ_SLOTS)
                        >= 0x1000
                    {
                        return false;
                    }
                }
                NativeInstKind::CallIndirect { args, results, .. } => {
                    if helper_stack_bytes(args.len(), results.len(), CALL_INDIRECT_REQ_SLOTS)
                        >= 0x1000
                    {
                        return false;
                    }
                }
            }
        }
        match &block.terminator {
            NativeTerminator::Goto(_)
            | NativeTerminator::Return { .. }
            | NativeTerminator::BrTable { .. }
            | NativeTerminator::TrapUnreachable => {}
            NativeTerminator::Branch { cond, .. } => {
                if !supported_branch_cond(cond) {
                    return false;
                }
            }
        }
    }
    program.frame.call_scratch.map_or(0, |span| span.count) >= 2
}

fn supported_direct_leaf(op: &LirLeafOp) -> bool {
    matches!(
        op,
        LirLeafOp::I32Add
            | LirLeafOp::I32Mul
            | LirLeafOp::I32Sub
            | LirLeafOp::I32DivS
            | LirLeafOp::I32DivU
            | LirLeafOp::I32RemS
            | LirLeafOp::I32RemU
            | LirLeafOp::I32And
            | LirLeafOp::I32Or
            | LirLeafOp::I32Xor
            | LirLeafOp::I32Shl
            | LirLeafOp::I32ShrS
            | LirLeafOp::I32ShrU
            | LirLeafOp::I32Rotl
            | LirLeafOp::I32Rotr
            | LirLeafOp::I64Add
            | LirLeafOp::I64Mul
            | LirLeafOp::I64Sub
            | LirLeafOp::I64DivS
            | LirLeafOp::I64DivU
            | LirLeafOp::I64RemS
            | LirLeafOp::I64RemU
            | LirLeafOp::I64And
            | LirLeafOp::I64Or
            | LirLeafOp::I64Xor
            | LirLeafOp::I64Shl
            | LirLeafOp::I64ShrS
            | LirLeafOp::I64ShrU
            | LirLeafOp::I64Rotl
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
            | LirLeafOp::Select
            | LirLeafOp::GlobalGet { .. }
            | LirLeafOp::GlobalSet { .. }
            | LirLeafOp::F64Add
            | LirLeafOp::F64Sub
            | LirLeafOp::F64Mul
            | LirLeafOp::F64Div
            | LirLeafOp::F64Eq
            | LirLeafOp::F64Ne
            | LirLeafOp::F64Lt
            | LirLeafOp::F64Gt
            | LirLeafOp::F64Neg
            | LirLeafOp::F64ConvertI32S
            | LirLeafOp::F64ConvertI32U
            | LirLeafOp::F64ReinterpretI64
            | LirLeafOp::I64ReinterpretF64
            | LirLeafOp::I32TruncSatF64S
            | LirLeafOp::I32TruncSatF64U
            | LirLeafOp::MemoryFill { .. }
            | LirLeafOp::MemoryCopy { .. }
            | LirLeafOp::MemorySize { mem_idx: 0 }
            | LirLeafOp::I32Load { memidx: 0, .. }
            | LirLeafOp::I64Load { memidx: 0, .. }
            | LirLeafOp::F32Load { memidx: 0, .. }
            | LirLeafOp::F64Load { memidx: 0, .. }
            | LirLeafOp::I32Load8S { memidx: 0, .. }
            | LirLeafOp::I32Load8U { memidx: 0, .. }
            | LirLeafOp::I32Load16S { memidx: 0, .. }
            | LirLeafOp::I32Load16U { memidx: 0, .. }
            | LirLeafOp::I64Load8S { memidx: 0, .. }
            | LirLeafOp::I64Load8U { memidx: 0, .. }
            | LirLeafOp::I64Load16S { memidx: 0, .. }
            | LirLeafOp::I64Load16U { memidx: 0, .. }
            | LirLeafOp::I64Load32S { memidx: 0, .. }
            | LirLeafOp::I64Load32U { memidx: 0, .. }
            | LirLeafOp::I32Store { memidx: 0, .. }
            | LirLeafOp::I64Store { memidx: 0, .. }
            | LirLeafOp::F32Store { memidx: 0, .. }
            | LirLeafOp::F64Store { memidx: 0, .. }
            | LirLeafOp::I32Store8 { memidx: 0, .. }
            | LirLeafOp::I32Store16 { memidx: 0, .. }
            | LirLeafOp::I64Store8 { memidx: 0, .. }
            | LirLeafOp::I64Store16 { memidx: 0, .. }
            | LirLeafOp::I64Store32 { memidx: 0, .. }
            | LirLeafOp::I32Const { .. }
            | LirLeafOp::I64Const { .. }
            | LirLeafOp::F32Const { .. }
            | LirLeafOp::F64Const { .. }
            | LirLeafOp::Unreachable
            | LirLeafOp::Nop
            | LirLeafOp::Drop
    )
}

fn supported_branch_cond(cond: &NativeBranchCond) -> bool {
    matches!(
        cond,
        NativeBranchCond::Value(_)
            | NativeBranchCond::I32Eqz(_)
            | NativeBranchCond::I64Eqz(_)
            | NativeBranchCond::I32Compare { .. }
            | NativeBranchCond::I64Compare { .. }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryLoadKind {
    U32,
    U64,
    S8To32,
    U8,
    S16To32,
    U16,
    S8To64,
    S16To64,
    S32To64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryStoreKind {
    U8,
    U16,
    U32,
    U64,
}

impl MemoryLoadKind {
    #[inline]
    fn access_size(self) -> u32 {
        match self {
            MemoryLoadKind::U32 | MemoryLoadKind::S32To64 => 4,
            MemoryLoadKind::U64 => 8,
            MemoryLoadKind::S8To32 | MemoryLoadKind::U8 | MemoryLoadKind::S8To64 => 1,
            MemoryLoadKind::S16To32 | MemoryLoadKind::U16 | MemoryLoadKind::S16To64 => 2,
        }
    }
}

impl MemoryStoreKind {
    #[inline]
    fn access_size(self) -> u32 {
        match self {
            MemoryStoreKind::U8 => 1,
            MemoryStoreKind::U16 => 2,
            MemoryStoreKind::U32 => 4,
            MemoryStoreKind::U64 => 8,
        }
    }
}

struct Arm64Lowerer<'a> {
    program: &'a NativeProgram,
    emitter: Arm64TextEmitter,
    block_offsets: Vec<Option<usize>>,
    branch_patches: Vec<BlockBranchPatch>,
    local_literal_patches: Vec<Arm64LocalLiteralPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    current_frame_slots: u32,
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
            local_literal_patches: Vec::new(),
            direct_call_patches: Vec::new(),
            current_frame_slots: max_slot,
        })
    }

    fn lower(mut self) -> Result<Arm64LoweredFunction, WasmError> {
        let entry_offset = self.emitter.len();
        let root_exit_literal = self.emit_public_entry()?;
        let internal_entry_offset = self.emitter.len();
        self.emit_branch_to_block(self.program.entry);

        for block in &self.program.blocks {
            self.block_offsets[block.id.as_usize()] = Some(self.emitter.len());
            for inst in &block.ops {
                self.emit_inst(inst)?;
            }
            self.emit_terminator(&block.terminator)?;
        }

        let root_exit_offset = self.emit_root_exit();
        self.local_literal_patches.push(Arm64LocalLiteralPatch {
            literal_offset: root_exit_literal as u32,
            target_offset: root_exit_offset as u32,
        });
        self.patch_block_branches()?;

        Ok(Arm64LoweredFunction {
            text: self.emitter.finish(),
            entry_offset: entry_offset as u32,
            internal_entry_offset: Some(internal_entry_offset as u32),
            local_literal_patches: self.local_literal_patches,
            direct_call_patches: self.direct_call_patches,
        })
    }

    fn emit_public_entry(&mut self) -> Result<usize, WasmError> {
        self.emitter
            .emit_u32(enc::sub_imm_64(REG_SP, REG_SP, SAVE_BYTES));
        for (slot, reg) in SAVED_REGS.iter().copied().enumerate() {
            self.emitter.emit_u32(enc::str_64(reg, REG_SP, slot as u32));
        }
        self.emitter
            .emit_u32(enc::str_64(Arm64Reg::X30, REG_SP, ROOT_LR_SLOT));

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
        self.emit_refresh_cached_views();

        let root_exit_load = self.emitter.len();
        self.emitter.emit_u32(enc::ldr_lit_64(REG_BRANCH_TARGET, 0));
        self.emitter
            .emit_u32(enc::str_64(REG_BRANCH_TARGET, REG_CTX, TERM_ENTRY_OFFSET));
        let call_scratch = self
            .program
            .frame
            .call_scratch
            .ok_or_else(|| WasmError::internal("arm64 direct entry requires call_scratch".into()))?;
        self.emitter.emit_u32(enc::str_64(
            REG_BRANCH_TARGET,
            REG_FP,
            call_scratch.start.0 as u32,
        ));
        self.emitter
            .emit_u32(enc::str_64(Arm64Reg::Xzr, REG_FP, call_scratch.start.0 as u32 + 1));
        let body_branch = self.emitter.len();
        self.emitter.emit_u32(enc::b(0));
        let root_exit_literal = self.emitter.len();
        self.emitter.emit_u64(0);
        let root_exit_delta = ((root_exit_literal as isize - root_exit_load as isize) / 4) as i32;
        self.emitter
            .patch_u32(root_exit_load, enc::ldr_lit_64(REG_BRANCH_TARGET, root_exit_delta));
        let body_offset = self.emitter.len();
        let body_delta = ((body_offset as isize - body_branch as isize) / 4) as i32;
        self.emitter.patch_u32(body_branch, enc::b(body_delta));
        Ok(root_exit_literal)
    }

    fn emit_root_exit(&mut self) -> usize {
        let start = self.emitter.len();
        self.emitter
            .emit_u32(enc::ldr_64(Arm64Reg::X30, REG_SP, ROOT_LR_SLOT));
        for (slot, reg) in SAVED_REGS.iter().copied().enumerate().rev() {
            self.emitter.emit_u32(enc::ldr_64(reg, REG_SP, slot as u32));
        }
        self.emitter
            .emit_u32(enc::add_imm_64(REG_SP, REG_SP, SAVE_BYTES));
        self.emitter.emit_u32(enc::ret());
        start
    }

    fn emit_inst(&mut self, inst: &NativeInst) -> Result<(), WasmError> {
        match &inst.kind {
            NativeInstKind::Move(mov) => self.emit_move(mov.dst, mov.src),
            NativeInstKind::Leaf { op, args, results } => self.emit_leaf(op, args, results),
            NativeInstKind::Helper(NativeHelperInst::Leaf {
                effect,
                op,
                args,
                results,
            }) if *effect == NativeHelperEffect::Transparent && supported_direct_leaf(op) => {
                self.emit_leaf(op, args, results)
            }
            NativeInstKind::Helper(NativeHelperInst::Leaf { op, .. }) => Err(WasmError::internal(
                alloc::format!(
                    "helper leaf {:?} is not supported by direct arm64 lowering yet",
                    op
                ),
            )),
            NativeInstKind::CallLocal {
                callee,
                callee_params_count,
                callee_frame_size,
                args,
                results,
            } => self.emit_call_local(
                *callee,
                *callee_params_count,
                *callee_frame_size,
                args,
                results,
            ),
            NativeInstKind::CallExternal {
                func_idx,
                args,
                results,
            } => self.emit_call_external(*func_idx, args, results),
            NativeInstKind::CallIndirect {
                type_idx,
                table_idx,
                index,
                args,
                results,
            } => self.emit_call_indirect(*type_idx, *table_idx, *index, args, results),
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
            LirLeafOp::I32DivS => self.emit_div32(args, results, true),
            LirLeafOp::I32DivU => self.emit_div32(args, results, false),
            LirLeafOp::I32RemS => self.emit_rem32(args, results, true),
            LirLeafOp::I32RemU => self.emit_rem32(args, results, false),
            LirLeafOp::I32And => self.emit_binary_op32(args, results, enc::and_reg_32),
            LirLeafOp::I32Or => self.emit_binary_op32(args, results, enc::orr_reg_32),
            LirLeafOp::I32Xor => self.emit_binary_op32(args, results, enc::eor_reg_32),
            LirLeafOp::I32Shl => self.emit_binary_op32(args, results, enc::lslv_32),
            LirLeafOp::I32ShrS => self.emit_binary_op32(args, results, enc::asrv_32),
            LirLeafOp::I32ShrU => self.emit_binary_op32(args, results, enc::lsrv_32),
            LirLeafOp::I32Rotl => self.emit_rotl32(args, results),
            LirLeafOp::I32Rotr => self.emit_binary_op32(args, results, enc::rorv_32),

            LirLeafOp::I64Add => self.emit_binary_op64(args, results, enc::add_reg_64),
            LirLeafOp::I64Sub => self.emit_binary_op64(args, results, enc::sub_reg_64),
            LirLeafOp::I64Mul => self.emit_binary_op64(args, results, enc::mul_64),
            LirLeafOp::I64DivS => self.emit_div64(args, results, true),
            LirLeafOp::I64DivU => self.emit_div64(args, results, false),
            LirLeafOp::I64RemS => self.emit_rem64(args, results, true),
            LirLeafOp::I64RemU => self.emit_rem64(args, results, false),
            LirLeafOp::I64And => self.emit_binary_op64(args, results, enc::and_reg_64),
            LirLeafOp::I64Or => self.emit_binary_op64(args, results, enc::orr_reg_64),
            LirLeafOp::I64Xor => self.emit_binary_op64(args, results, enc::eor_reg_64),
            LirLeafOp::I64Shl => self.emit_binary_op64(args, results, enc::lslv_64),
            LirLeafOp::I64ShrS => self.emit_binary_op64(args, results, enc::asrv_64),
            LirLeafOp::I64ShrU => self.emit_binary_op64(args, results, enc::lsrv_64),
            LirLeafOp::I64Rotl => self.emit_rotl64(args, results),
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
            LirLeafOp::F64Add => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_add as usize as u64,
            ),
            LirLeafOp::F64Sub => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_sub as usize as u64,
            ),
            LirLeafOp::F64Mul => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_mul as usize as u64,
            ),
            LirLeafOp::F64Div => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_div as usize as u64,
            ),
            LirLeafOp::F64Eq => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_eq as usize as u64,
            ),
            LirLeafOp::F64Ne => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_ne as usize as u64,
            ),
            LirLeafOp::F64Lt => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_lt as usize as u64,
            ),
            LirLeafOp::F64Gt => self.emit_binary_helper_call(
                args,
                results,
                super::entry::arm64_f64_gt as usize as u64,
            ),
            LirLeafOp::F64Neg => self.emit_unary_helper_call(
                args,
                results,
                super::entry::arm64_f64_neg as usize as u64,
            ),
            LirLeafOp::F64ConvertI32S => self.emit_unary_helper_call(
                args,
                results,
                super::entry::arm64_f64_convert_i32_s as usize as u64,
            ),
            LirLeafOp::F64ConvertI32U => self.emit_unary_helper_call(
                args,
                results,
                super::entry::arm64_f64_convert_i32_u as usize as u64,
            ),
            LirLeafOp::F64ReinterpretI64 => self.emit_unary_op64(args, results, enc::mov_reg_64),
            LirLeafOp::I64ReinterpretF64 => self.emit_unary_op64(args, results, enc::mov_reg_64),
            LirLeafOp::I32TruncSatF64S => self.emit_unary_helper_call(
                args,
                results,
                super::entry::arm64_i32_trunc_sat_f64_s as usize as u64,
            ),
            LirLeafOp::I32TruncSatF64U => self.emit_unary_helper_call(
                args,
                results,
                super::entry::arm64_i32_trunc_sat_f64_u as usize as u64,
            ),
            LirLeafOp::MemoryFill { imm0, .. } => self.emit_status_helper_call(
                args,
                &[
                    super::entry::arm64_memory_fill_helper as usize as u64,
                    u64::from(*imm0),
                ],
            ),
            LirLeafOp::MemoryCopy { imm0, imm1 } => self.emit_status_helper_call(
                args,
                &[
                    super::entry::arm64_memory_copy_helper as usize as u64,
                    u64::from(*imm0),
                    u64::from(*imm1),
                ],
            ),
            LirLeafOp::Select => self.emit_select(args, results),
            LirLeafOp::GlobalGet { idx } => self.emit_global_get(*idx, results),
            LirLeafOp::GlobalSet { idx } => self.emit_global_set(*idx, args),
            LirLeafOp::MemorySize { mem_idx: 0 } => self.emit_memory_size(results),
            LirLeafOp::Unreachable => {
                self.emit_trap_inline(2);
                Ok(())
            }
            LirLeafOp::I32Load { offset, memidx: 0 }
            | LirLeafOp::F32Load { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::U32)
            }
            LirLeafOp::I64Load { offset, memidx: 0 }
            | LirLeafOp::F64Load { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::U64)
            }
            LirLeafOp::I32Load8S { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::S8To32)
            }
            LirLeafOp::I32Load8U { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::U8)
            }
            LirLeafOp::I32Load16S { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::S16To32)
            }
            LirLeafOp::I32Load16U { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::U16)
            }
            LirLeafOp::I64Load8S { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::S8To64)
            }
            LirLeafOp::I64Load8U { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::U8)
            }
            LirLeafOp::I64Load16S { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::S16To64)
            }
            LirLeafOp::I64Load16U { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::U16)
            }
            LirLeafOp::I64Load32S { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::S32To64)
            }
            LirLeafOp::I64Load32U { offset, memidx: 0 } => {
                self.emit_memory_load(args, results, *offset, MemoryLoadKind::U32)
            }
            LirLeafOp::I32Store { offset, memidx: 0 }
            | LirLeafOp::F32Store { offset, memidx: 0 } => {
                self.emit_memory_store(args, *offset, MemoryStoreKind::U32)
            }
            LirLeafOp::I64Store { offset, memidx: 0 }
            | LirLeafOp::F64Store { offset, memidx: 0 } => {
                self.emit_memory_store(args, *offset, MemoryStoreKind::U64)
            }
            LirLeafOp::I32Store8 { offset, memidx: 0 }
            | LirLeafOp::I64Store8 { offset, memidx: 0 } => {
                self.emit_memory_store(args, *offset, MemoryStoreKind::U8)
            }
            LirLeafOp::I32Store16 { offset, memidx: 0 }
            | LirLeafOp::I64Store16 { offset, memidx: 0 } => {
                self.emit_memory_store(args, *offset, MemoryStoreKind::U16)
            }
            LirLeafOp::I64Store32 { offset, memidx: 0 } => {
                self.emit_memory_store(args, *offset, MemoryStoreKind::U32)
            }
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
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(encode(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_binary_op64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        encode: fn(Arm64Reg, Arm64Reg, Arm64Reg) -> u32,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(encode(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_rotl32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(enc::sub_reg_32(REG_SCRATCH1, Arm64Reg::Xzr, REG_SCRATCH1));
        self.emitter
            .emit_u32(enc::rorv_32(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_rotl64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emitter
            .emit_u32(enc::sub_reg_64(REG_SCRATCH1, Arm64Reg::Xzr, REG_SCRATCH1));
        self.emitter
            .emit_u32(enc::rorv_64(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_div32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        signed: bool,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emit_div_zero_guard(false)?;
        if signed {
            self.emit_div_signed_overflow_guard(false)?;
            self.emitter
                .emit_u32(enc::sdiv_32(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        } else {
            self.emitter
                .emit_u32(enc::udiv_32(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        }
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_div64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        signed: bool,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emit_div_zero_guard(true)?;
        if signed {
            self.emit_div_signed_overflow_guard(true)?;
            self.emitter
                .emit_u32(enc::sdiv_64(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        } else {
            self.emitter
                .emit_u32(enc::udiv_64(dst_reg, REG_SCRATCH0, REG_SCRATCH1));
        }
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_rem32(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        signed: bool,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emit_div_zero_guard(false)?;
        if signed {
            self.emitter
                .emit_u32(enc::sdiv_32(REG_COPY_TEMP, REG_SCRATCH0, REG_SCRATCH1));
        } else {
            self.emitter
                .emit_u32(enc::udiv_32(REG_COPY_TEMP, REG_SCRATCH0, REG_SCRATCH1));
        }
        self.emitter
            .emit_u32(enc::msub_32(dst_reg, REG_COPY_TEMP, REG_SCRATCH1, REG_SCRATCH0));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_rem64(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        signed: bool,
    ) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(args[0], REG_SCRATCH0)?;
        self.read_value_into(args[1], REG_SCRATCH1)?;
        self.emit_div_zero_guard(true)?;
        if signed {
            self.emitter
                .emit_u32(enc::sdiv_64(REG_COPY_TEMP, REG_SCRATCH0, REG_SCRATCH1));
        } else {
            self.emitter
                .emit_u32(enc::udiv_64(REG_COPY_TEMP, REG_SCRATCH0, REG_SCRATCH1));
        }
        self.emitter
            .emit_u32(enc::msub_64(dst_reg, REG_COPY_TEMP, REG_SCRATCH1, REG_SCRATCH0));
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

    fn emit_unary_helper_call(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        helper_addr: u64,
    ) -> Result<(), WasmError> {
        let [arg] = args else {
            return Err(WasmError::internal(
                "arm64 unary helper expects exactly one arg".into(),
            ));
        };
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(*arg, Arm64Reg::X0)?;
        self.materialize_u64(REG_BRANCH_TARGET, helper_addr);
        self.emitter.emit_u32(enc::blr(REG_BRANCH_TARGET));
        self.emit_refresh_cached_views();
        self.write_result(results.first().copied(), Arm64Reg::X0)?;
        if dst_reg != Arm64Reg::X0 {
            return Ok(());
        }
        Ok(())
    }

    fn emit_binary_helper_call(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        helper_addr: u64,
    ) -> Result<(), WasmError> {
        let [lhs, rhs] = args else {
            return Err(WasmError::internal(
                "arm64 binary helper expects exactly two args".into(),
            ));
        };
        self.read_value_into(*lhs, Arm64Reg::X0)?;
        self.read_value_into(*rhs, Arm64Reg::X1)?;
        self.materialize_u64(REG_BRANCH_TARGET, helper_addr);
        self.emitter.emit_u32(enc::blr(REG_BRANCH_TARGET));
        self.emit_refresh_cached_views();
        self.write_result(results.first().copied(), Arm64Reg::X0)
    }

    fn emit_status_helper_call(
        &mut self,
        args: &[NativeValue],
        helper_and_imms: &[u64],
    ) -> Result<(), WasmError> {
        let (&helper_addr, imm_values) = helper_and_imms.split_first().ok_or_else(|| {
            WasmError::internal("arm64 status helper is missing helper address".into())
        })?;
        if args.len() + imm_values.len() > 7 {
            return Err(WasmError::internal(
                "arm64 status helper exceeds argument register budget".into(),
            ));
        }

        self.move_if_needed(Arm64Reg::X0, REG_CTX);
        for (index, arg) in args.iter().copied().enumerate() {
            self.read_value_into(arg, arg_reg((index + 1) as u8))?;
        }
        for (index, imm) in imm_values.iter().copied().enumerate() {
            self.materialize_u64(arg_reg((args.len() + index + 1) as u8), imm);
        }
        self.materialize_u64(REG_BRANCH_TARGET, helper_addr);
        self.emitter.emit_u32(enc::blr(REG_BRANCH_TARGET));
        self.emit_refresh_cached_views();
        let ok_patch = self.emitter.len();
        self.emitter.emit_u32(enc::cbz_64(Arm64Reg::X0, 0));
        self.emitter
            .emit_u32(enc::ldr_64(REG_BRANCH_TARGET, REG_CTX, TERM_ENTRY_OFFSET));
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));
        let ok_label = self.emitter.len();
        self.patch_cbz(ok_patch, Arm64Reg::X0, ok_label);
        Ok(())
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

    fn emit_select(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let [if_true, if_false, cond] = args else {
            return Err(WasmError::internal(
                "arm64 select expects exactly three args".into(),
            ));
        };
        let dst_reg = self.result_reg(results.first().copied());
        self.read_value_into(*if_true, REG_SCRATCH0)?;
        self.read_value_into(*if_false, REG_SCRATCH1)?;
        self.read_value_into(*cond, REG_COPY_TEMP)?;
        self.emitter.emit_u32(enc::cmp_imm_64(REG_COPY_TEMP, 0));
        self.emitter
            .emit_u32(enc::csel_64(dst_reg, REG_SCRATCH0, REG_SCRATCH1, Cond::Ne));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_global_get(&mut self, idx: u32, results: &[NativePlace]) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.load_global_addr(idx, REG_BRANCH_TARGET);
        self.emitter
            .emit_u32(enc::ldr_64(dst_reg, REG_BRANCH_TARGET, 0));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_global_set(&mut self, idx: u32, args: &[NativeValue]) -> Result<(), WasmError> {
        let [value] = args else {
            return Err(WasmError::internal(
                "arm64 global.set expects exactly one value arg".into(),
            ));
        };
        self.load_global_addr(idx, REG_BRANCH_TARGET);
        self.emitter
            .emit_u32(enc::ldrb(REG_SCRATCH1, REG_BRANCH_TARGET, GLOBAL_MUTABLE_OFFSET));
        self.emitter.emit_u32(enc::cmp_imm_32(REG_SCRATCH1, 0));
        let mutable_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Ne, 0));
        self.emit_trap_inline(1);
        let mutable_label = self.emitter.len();
        self.patch_b_cond(mutable_patch, Cond::Ne, mutable_label)?;
        self.read_value_into(*value, REG_COPY_TEMP)?;
        self.emitter
            .emit_u32(enc::str_64(REG_COPY_TEMP, REG_BRANCH_TARGET, 0));
        Ok(())
    }

    fn emit_memory_size(&mut self, results: &[NativePlace]) -> Result<(), WasmError> {
        let dst_reg = self.result_reg(results.first().copied());
        self.emitter
            .emit_u32(enc::lsr_imm_64(dst_reg, REG_MEMORY_LEN, 16));
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_memory_load(
        &mut self,
        args: &[NativeValue],
        results: &[NativePlace],
        offset: u32,
        kind: MemoryLoadKind,
    ) -> Result<(), WasmError> {
        let [address] = args else {
            return Err(WasmError::internal(
                "arm64 memory load expects exactly one address arg".into(),
            ));
        };
        let dst_reg = self.result_reg(results.first().copied());
        self.compute_memory0_offset(*address, offset)?;
        self.emit_memory0_bounds_guard(kind.access_size())?;
        match kind {
            MemoryLoadKind::U32 => {
                self.emitter
                    .emit_u32(enc::ldr_reg_32(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::U64 => {
                self.emitter
                    .emit_u32(enc::ldr_reg_64(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::S8To32 => {
                self.emitter
                    .emit_u32(enc::ldrsb_reg_32(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::U8 => {
                self.emitter
                    .emit_u32(enc::ldrb_reg(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::S16To32 => {
                self.emitter
                    .emit_u32(enc::ldrsh_reg_32(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::U16 => {
                self.emitter
                    .emit_u32(enc::ldrh_reg(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::S8To64 => {
                self.emitter
                    .emit_u32(enc::ldrsb_reg_64(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::S16To64 => {
                self.emitter
                    .emit_u32(enc::ldrsh_reg_64(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryLoadKind::S32To64 => {
                self.emitter
                    .emit_u32(enc::ldrsw_reg(dst_reg, REG_MEMORY_BASE, REG_SCRATCH0));
            }
        }
        self.write_result(results.first().copied(), dst_reg)
    }

    fn emit_memory_store(
        &mut self,
        args: &[NativeValue],
        offset: u32,
        kind: MemoryStoreKind,
    ) -> Result<(), WasmError> {
        let [address, value] = args else {
            return Err(WasmError::internal(
                "arm64 memory store expects address and value args".into(),
            ));
        };
        self.compute_memory0_offset(*address, offset)?;
        self.emit_memory0_bounds_guard(kind.access_size())?;
        self.read_value_into(*value, REG_COPY_TEMP)?;
        match kind {
            MemoryStoreKind::U8 => {
                self.emitter
                    .emit_u32(enc::strb_reg(REG_COPY_TEMP, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryStoreKind::U16 => {
                self.emitter
                    .emit_u32(enc::strh_reg(REG_COPY_TEMP, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryStoreKind::U32 => {
                self.emitter
                    .emit_u32(enc::str_reg_32(REG_COPY_TEMP, REG_MEMORY_BASE, REG_SCRATCH0));
            }
            MemoryStoreKind::U64 => {
                self.emitter
                    .emit_u32(enc::str_reg_64(REG_COPY_TEMP, REG_MEMORY_BASE, REG_SCRATCH0));
            }
        }
        Ok(())
    }

    fn load_global_addr(&mut self, idx: u32, dst: Arm64Reg) {
        self.move_if_needed(dst, REG_GLOBALS_BASE);
        let byte_offset = (idx as u64) * (core::mem::size_of::<crate::vm::entities::GlobalInst>() as u64);
        self.add_u64_immediate(dst, dst, byte_offset, REG_SCRATCH0);
    }

    fn compute_memory0_offset(&mut self, address: NativeValue, offset: u32) -> Result<(), WasmError> {
        self.read_value_into(address, REG_SCRATCH0)?;
        self.add_u64_immediate(REG_SCRATCH0, REG_SCRATCH0, u64::from(offset), REG_SCRATCH1);
        Ok(())
    }

    fn emit_memory0_bounds_guard(&mut self, access_size: u32) -> Result<(), WasmError> {
        let ok_cond = if access_size == 1 { Cond::Lo } else { Cond::Ls };
        if access_size == 1 {
            self.emitter
                .emit_u32(enc::cmp_reg_64(REG_SCRATCH0, REG_MEMORY_LEN));
        } else {
            self.add_u64_immediate(
                REG_COPY_TEMP,
                REG_SCRATCH0,
                u64::from(access_size),
                REG_BRANCH_TARGET,
            );
            self.emitter
                .emit_u32(enc::cmp_reg_64(REG_COPY_TEMP, REG_MEMORY_LEN));
        }
        let ok_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(ok_cond, 0));
        self.emit_trap_inline(0);
        let ok_label = self.emitter.len();
        self.patch_b_cond(ok_patch, ok_cond, ok_label)
    }

    fn emit_div_zero_guard(&mut self, is_64: bool) -> Result<(), WasmError> {
        if is_64 {
            self.emitter.emit_u32(enc::cmp_imm_64(REG_SCRATCH1, 0));
        } else {
            self.emitter.emit_u32(enc::cmp_imm_32(REG_SCRATCH1, 0));
        }
        let ok_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Ne, 0));
        self.emit_trap_inline(3);
        let ok_label = self.emitter.len();
        self.patch_b_cond(ok_patch, Cond::Ne, ok_label)
    }

    fn emit_div_signed_overflow_guard(&mut self, is_64: bool) -> Result<(), WasmError> {
        let (min_value, neg_one) = if is_64 {
            (1u64 << 63, u64::MAX)
        } else {
            (1u64 << 31, u32::MAX as u64)
        };
        self.materialize_u64(REG_COPY_TEMP, min_value);
        if is_64 {
            self.emitter
                .emit_u32(enc::cmp_reg_64(REG_SCRATCH0, REG_COPY_TEMP));
        } else {
            self.emitter
                .emit_u32(enc::cmp_reg_32(REG_SCRATCH0, REG_COPY_TEMP));
        }
        let lhs_ok_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Ne, 0));

        self.materialize_u64(REG_COPY_TEMP, neg_one);
        if is_64 {
            self.emitter
                .emit_u32(enc::cmp_reg_64(REG_SCRATCH1, REG_COPY_TEMP));
        } else {
            self.emitter
                .emit_u32(enc::cmp_reg_32(REG_SCRATCH1, REG_COPY_TEMP));
        }
        let rhs_ok_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Ne, 0));
        self.emit_trap_inline(4);
        let ok_label = self.emitter.len();
        self.patch_b_cond(lhs_ok_patch, Cond::Ne, ok_label)?;
        self.patch_b_cond(rhs_ok_patch, Cond::Ne, ok_label)
    }

    fn emit_trap_inline(&mut self, kind: u64) {
        self.move_if_needed(Arm64Reg::X0, REG_CTX);
        self.materialize_u64(Arm64Reg::X1, kind);
        self.materialize_u64(REG_BRANCH_TARGET, super::entry::arm64_raise_trap as usize as u64);
        self.emitter.emit_u32(enc::blr(REG_BRANCH_TARGET));
        self.emitter
            .emit_u32(enc::ldr_64(REG_BRANCH_TARGET, REG_CTX, TERM_ENTRY_OFFSET));
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));
    }

    fn emit_call_external(
        &mut self,
        func_idx: u32,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let stack_bytes = helper_stack_bytes(args.len(), results.len(), CALL_EXTERNAL_REQ_SLOTS);
        debug_assert!(stack_bytes < 0x1000);
        let results_base_slot = args.len();
        let req_base_slot = results_base_slot + results.len();
        self.emitter
            .emit_u32(enc::sub_imm_64(REG_SP, REG_SP, stack_bytes as u32));
        self.store_call_args(args, 0)?;

        self.materialize_u64(REG_SCRATCH0, u64::from(func_idx));
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32));
        self.emit_stack_addr(REG_SCRATCH0, 0);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 1));
        self.materialize_u64(REG_SCRATCH0, args.len() as u64);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 2));
        self.emit_stack_addr(REG_SCRATCH0, (results_base_slot * 8) as u32);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 3));
        self.materialize_u64(REG_SCRATCH0, results.len() as u64);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 4));

        self.emit_stack_addr(Arm64Reg::X2, (req_base_slot * 8) as u32);
        self.move_if_needed(Arm64Reg::X0, REG_CTX);
        self.move_if_needed(Arm64Reg::X1, REG_FP);
        self.materialize_u64(
            REG_BRANCH_TARGET,
            super::entry::arm64_call_external_helper as usize as u64,
        );
        self.emitter.emit_u32(enc::blr(REG_BRANCH_TARGET));

        let ok_patch = self.emitter.len();
        self.emitter.emit_u32(enc::cbz_64(Arm64Reg::X0, 0));
        self.emitter
            .emit_u32(enc::add_imm_64(REG_SP, REG_SP, stack_bytes as u32));
        self.emitter
            .emit_u32(enc::ldr_64(REG_BRANCH_TARGET, REG_CTX, TERM_ENTRY_OFFSET));
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));

        let ok_label = self.emitter.len();
        self.patch_cbz(ok_patch, Arm64Reg::X0, ok_label);
        self.emit_refresh_cached_views();
        self.load_call_results(results, results_base_slot)?;
        self.emitter
            .emit_u32(enc::add_imm_64(REG_SP, REG_SP, stack_bytes as u32));
        Ok(())
    }

    fn emit_call_indirect(
        &mut self,
        type_idx: u32,
        table_idx: u32,
        index: NativeValue,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let stack_bytes = helper_stack_bytes(args.len(), results.len(), CALL_INDIRECT_REQ_SLOTS);
        debug_assert!(stack_bytes < 0x1000);
        let results_base_slot = args.len();
        let req_base_slot = results_base_slot + results.len();
        self.emitter
            .emit_u32(enc::sub_imm_64(REG_SP, REG_SP, stack_bytes as u32));
        self.store_call_args(args, 0)?;

        self.materialize_u64(REG_SCRATCH0, self.current_frame_slots as u64);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32));
        self.materialize_u64(REG_SCRATCH0, u64::from(type_idx));
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 1));
        self.materialize_u64(REG_SCRATCH0, u64::from(table_idx));
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 2));
        self.read_value_into(index, REG_SCRATCH0)?;
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 3));
        self.emit_stack_addr(REG_SCRATCH0, 0);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 4));
        self.materialize_u64(REG_SCRATCH0, args.len() as u64);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 5));
        self.emit_stack_addr(REG_SCRATCH0, (results_base_slot * 8) as u32);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 6));
        self.materialize_u64(REG_SCRATCH0, results.len() as u64);
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, req_base_slot as u32 + 7));

        self.emit_stack_addr(Arm64Reg::X2, (req_base_slot * 8) as u32);
        self.move_if_needed(Arm64Reg::X0, REG_CTX);
        self.move_if_needed(Arm64Reg::X1, REG_FP);
        self.materialize_u64(
            REG_BRANCH_TARGET,
            super::entry::arm64_call_indirect_helper as usize as u64,
        );
        self.emitter.emit_u32(enc::blr(REG_BRANCH_TARGET));

        let ok_patch = self.emitter.len();
        self.emitter.emit_u32(enc::cbz_64(Arm64Reg::X0, 0));
        self.emitter
            .emit_u32(enc::add_imm_64(REG_SP, REG_SP, stack_bytes as u32));
        self.emitter
            .emit_u32(enc::ldr_64(REG_BRANCH_TARGET, REG_CTX, TERM_ENTRY_OFFSET));
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));

        let ok_label = self.emitter.len();
        self.patch_cbz(ok_patch, Arm64Reg::X0, ok_label);
        self.emit_refresh_cached_views();
        self.load_call_results(results, results_base_slot)?;
        self.emitter
            .emit_u32(enc::add_imm_64(REG_SP, REG_SP, stack_bytes as u32));
        Ok(())
    }

    fn store_call_args(&mut self, args: &[NativeValue], base_slot: usize) -> Result<(), WasmError> {
        for (index, value) in args.iter().copied().enumerate() {
            self.read_value_into(value, REG_SCRATCH0)?;
            self.emitter
                .emit_u32(enc::str_64(REG_SCRATCH0, REG_SP, (base_slot + index) as u32));
        }
        Ok(())
    }

    fn load_call_results(
        &mut self,
        results: &[NativePlace],
        base_slot: usize,
    ) -> Result<(), WasmError> {
        for (index, place) in results.iter().copied().enumerate() {
            self.emitter
                .emit_u32(enc::ldr_64(REG_SCRATCH0, REG_SP, (base_slot + index) as u32));
            self.write_reg_to_place(place, REG_SCRATCH0)?;
        }
        Ok(())
    }

    fn emit_stack_addr(&mut self, dst: Arm64Reg, byte_offset: u32) {
        self.emitter
            .emit_u32(enc::add_imm_64(dst, REG_SP, byte_offset));
    }

    fn emit_call_local(
        &mut self,
        callee: u32,
        callee_params_count: u16,
        callee_frame_size: u16,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        if args.len() != callee_params_count as usize {
            return Err(WasmError::internal(
                "arm64 call_local arg count does not match callee params".into(),
            ));
        }

        let callee_frame_slots_load = self.emitter.len();
        self.emitter.emit_u32(enc::ldr_lit_64(REG_SCRATCH1, 0));

        self.materialize_u64(REG_SCRATCH0, u64::from(self.current_frame_slots) * 8);
        self.emitter
            .emit_u32(enc::add_reg_64(REG_SCRATCH0, REG_FP, REG_SCRATCH0));
        self.emitter
            .emit_u32(enc::lsl_imm_64(REG_COPY_TEMP, REG_SCRATCH1, 3));
        self.emitter
            .emit_u32(enc::add_reg_64(REG_BRANCH_TARGET, REG_SCRATCH0, REG_COPY_TEMP));
        self.emitter
            .emit_u32(enc::ldr_64(REG_COPY_TEMP, REG_CTX, STACK_END_OFFSET));
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_BRANCH_TARGET, REG_COPY_TEMP));
        let stack_overflow_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Hi, 0));

        self.emitter
            .emit_u32(enc::ldr_64(REG_SCRATCH1, REG_CTX, CALL_DEPTH_OFFSET));
        self.materialize_u64(REG_COPY_TEMP, MAX_ARM64_CALL_DEPTH);
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_SCRATCH1, REG_COPY_TEMP));
        let call_depth_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Hs, 0));
        self.emitter
            .emit_u32(enc::add_imm_64(REG_SCRATCH1, REG_SCRATCH1, 1));
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH1, REG_CTX, CALL_DEPTH_OFFSET));

        for (slot, value) in args.iter().copied().enumerate() {
            self.read_value_into(value, REG_COPY_TEMP)?;
            self.emitter
                .emit_u32(enc::str_64(REG_COPY_TEMP, REG_SCRATCH0, slot as u32));
        }
        self.add_u64_immediate(
            REG_BRANCH_TARGET,
            REG_SCRATCH0,
            u64::from(callee_frame_size) * 8,
            REG_COPY_TEMP,
        );
        if callee_params_count == 0 {
            self.move_if_needed(REG_COPY_TEMP, REG_SCRATCH0);
        } else {
            self.materialize_u64(REG_COPY_TEMP, u64::from(callee_params_count) * 8);
            self.emitter
                .emit_u32(enc::add_reg_64(REG_COPY_TEMP, REG_SCRATCH0, REG_COPY_TEMP));
        }
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_COPY_TEMP, REG_BRANCH_TARGET));
        let zero_done_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Hs, 0));
        self.emitter
            .emit_u32(enc::sub_imm_64(REG_SCRATCH1, REG_BRANCH_TARGET, 8));
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_COPY_TEMP, REG_SCRATCH1));
        let zero_single_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Hs, 0));
        let zero_loop = self.emitter.len();
        self.emitter
            .emit_u32(enc::stp_64(Arm64Reg::Xzr, Arm64Reg::Xzr, REG_COPY_TEMP, 0));
        self.emitter
            .emit_u32(enc::add_imm_64(REG_COPY_TEMP, REG_COPY_TEMP, 16));
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_COPY_TEMP, REG_SCRATCH1));
        let zero_loop_branch = self.emitter.len();
        let zero_loop_delta = ((zero_loop as isize - zero_loop_branch as isize) / 4) as i32;
        self.emitter
            .emit_u32(enc::b_cond(Cond::Lo, zero_loop_delta));
        let zero_single = self.emitter.len();
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_COPY_TEMP, REG_BRANCH_TARGET));
        let zero_tail_done_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(Cond::Hs, 0));
        self.emitter
            .emit_u32(enc::str_64(Arm64Reg::Xzr, REG_COPY_TEMP, 0));
        let zero_done = self.emitter.len();

        let continuation_load = self.emitter.len();
        self.emitter
            .emit_u32(enc::ldr_lit_64(REG_BRANCH_TARGET, 0));
        if callee_frame_size <= 63 {
            self.emitter.emit_u32(enc::stp_64(
                REG_BRANCH_TARGET,
                REG_FP,
                REG_SCRATCH0,
                callee_frame_size as i32,
            ));
        } else {
            self.emitter.emit_u32(enc::str_64(
                REG_BRANCH_TARGET,
                REG_SCRATCH0,
                callee_frame_size as u32,
            ));
            self.emitter
                .emit_u32(enc::str_64(REG_FP, REG_SCRATCH0, callee_frame_size as u32 + 1));
        }
        self.move_if_needed(REG_FP, REG_SCRATCH0);

        let callee_entry_load = self.emitter.len();
        self.emitter
            .emit_u32(enc::ldr_lit_64(REG_BRANCH_TARGET, 0));
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));

        let continuation_literal = self.emitter.len();
        self.emitter.emit_u64(0);
        let callee_frame_slots_literal = self.emitter.len();
        self.emitter.emit_u64(0);
        let callee_entry_literal = self.emitter.len();
        self.emitter.emit_u64(0);

        let stack_overflow_label = self.emit_exhaustion_stub(0);
        let call_depth_label = self.emit_exhaustion_stub(1);
        let continuation_offset = self.emitter.len();
        self.emit_call_local_continuation(results)?;

        let continuation_delta =
            ((continuation_literal as isize - continuation_load as isize) / 4) as i32;
        self.emitter.patch_u32(
            continuation_load,
            enc::ldr_lit_64(REG_BRANCH_TARGET, continuation_delta),
        );
        let callee_frame_slots_delta =
            ((callee_frame_slots_literal as isize - callee_frame_slots_load as isize) / 4) as i32;
        self.emitter.patch_u32(
            callee_frame_slots_load,
            enc::ldr_lit_64(REG_SCRATCH1, callee_frame_slots_delta),
        );
        let callee_entry_delta =
            ((callee_entry_literal as isize - callee_entry_load as isize) / 4) as i32;
        self.emitter.patch_u32(
            callee_entry_load,
            enc::ldr_lit_64(REG_BRANCH_TARGET, callee_entry_delta),
        );
        let stack_overflow_delta =
            ((stack_overflow_label as isize - stack_overflow_patch as isize) / 4) as i32;
        self.emitter
            .patch_u32(stack_overflow_patch, enc::b_cond(Cond::Hi, stack_overflow_delta));
        let call_depth_delta =
            ((call_depth_label as isize - call_depth_patch as isize) / 4) as i32;
        self.emitter
            .patch_u32(call_depth_patch, enc::b_cond(Cond::Hs, call_depth_delta));
        let zero_done_delta = ((zero_done as isize - zero_done_patch as isize) / 4) as i32;
        self.emitter
            .patch_u32(zero_done_patch, enc::b_cond(Cond::Hs, zero_done_delta));
        let zero_single_delta = ((zero_single as isize - zero_single_patch as isize) / 4) as i32;
        self.emitter
            .patch_u32(zero_single_patch, enc::b_cond(Cond::Hs, zero_single_delta));
        let zero_tail_done_delta =
            ((zero_done as isize - zero_tail_done_patch as isize) / 4) as i32;
        self.emitter.patch_u32(
            zero_tail_done_patch,
            enc::b_cond(Cond::Hs, zero_tail_done_delta),
        );

        self.local_literal_patches.push(Arm64LocalLiteralPatch {
            literal_offset: continuation_literal as u32,
            target_offset: continuation_offset as u32,
        });
        self.direct_call_patches.push(DirectCallPatch {
            callee_func_idx: callee,
            entry_branch_inst_offset: callee_entry_load as u32,
            entry_branch_reg_offset: callee_entry_load as u32 + 4,
            entry_literal_offset: callee_entry_literal as u32,
            frame_slots_literal_offset: callee_frame_slots_literal as u32,
        });
        Ok(())
    }

    fn emit_call_local_continuation(&mut self, results: &[NativePlace]) -> Result<(), WasmError> {
        for (index, place) in results.iter().copied().enumerate() {
            self.emitter
                .emit_u32(enc::ldr_64(REG_SCRATCH0, REG_CONT_FP, index as u32));
            self.write_reg_to_place(place, REG_SCRATCH0)?;
        }
        self.emit_refresh_cached_views();
        self.emit_reload_hot_locals_from_frame()?;
        Ok(())
    }

    fn emit_exhaustion_stub(&mut self, kind: u64) -> usize {
        let start = self.emitter.len();
        self.move_if_needed(Arm64Reg::X0, REG_CTX);
        self.materialize_u64(Arm64Reg::X1, kind);
        self.materialize_u64(REG_BRANCH_TARGET, super::entry::arm64_raise_exhaustion as usize as u64);
        self.emitter.emit_u32(enc::blr(REG_BRANCH_TARGET));
        self.emitter
            .emit_u32(enc::ldr_64(REG_BRANCH_TARGET, REG_CTX, TERM_ENTRY_OFFSET));
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));
        start
    }

    fn emit_reload_hot_locals_from_frame(&mut self) -> Result<(), WasmError> {
        let Some(hot_locals) = &self.program.hot_locals else {
            return Ok(());
        };

        for (reg, local_idx) in hot_locals.mapping().iter().copied().enumerate() {
            let Some(local_idx) = local_idx else {
                continue;
            };
            let reg = map_location(NativeLocation::Hot(reg as u8));
            self.emitter
                .emit_u32(enc::ldr_64(reg, REG_FP, local_idx));
        }
        Ok(())
    }

    fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let call_scratch = self
            .program
            .frame
            .call_scratch
            .ok_or_else(|| WasmError::internal("arm64 return requires call_scratch".into()))?;
        let continuation_slot = call_scratch.start.0 as u32;

        if continuation_slot <= 63 {
            self.emitter.emit_u32(enc::ldp_64(
                REG_BRANCH_TARGET,
                REG_SCRATCH1,
                REG_FP,
                continuation_slot as i32,
            ));
        } else {
            self.emitter
                .emit_u32(enc::ldr_64(REG_BRANCH_TARGET, REG_FP, continuation_slot));
            self.emitter
                .emit_u32(enc::ldr_64(REG_SCRATCH1, REG_FP, continuation_slot + 1));
        }
        let root_return_patch = self.emitter.len();
        self.emitter.emit_u32(enc::cbz_64(REG_SCRATCH1, 0));

        self.emitter
            .emit_u32(enc::ldr_64(REG_SCRATCH0, REG_CTX, CALL_DEPTH_OFFSET));
        self.emitter
            .emit_u32(enc::sub_imm_64(REG_SCRATCH0, REG_SCRATCH0, 1));
        self.emitter
            .emit_u32(enc::str_64(REG_SCRATCH0, REG_CTX, CALL_DEPTH_OFFSET));
        self.move_if_needed(REG_CONT_FP, REG_FP);
        self.move_if_needed(REG_FP, REG_SCRATCH1);
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));

        let root_return_label = self.emitter.len();
        self.patch_cbz(root_return_patch, REG_SCRATCH1, root_return_label);
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));
        Ok(())
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
                self.emit_branch_cond(cond, then_edge, else_edge)
            }
            NativeTerminator::Return { values } => {
                let pending = values
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, value)| {
                        (
                            NativePlace::Frame(FrameSlot(index as u16)),
                            CopySource::Value(value),
                        )
                    })
                    .collect();
                self.emit_parallel_copies(pending)?;
                self.emit_return_sequence()?;
                Ok(())
            }
            NativeTerminator::BrTable { index, entries } => self.emit_br_table(*index, entries),
            NativeTerminator::TrapUnreachable => {
                self.emit_trap_inline(2);
                Ok(())
            }
        }
    }

    fn constant_branch<'b>(
        &self,
        cond: &NativeBranchCond,
        then_edge: &'b NativeEdge,
        else_edge: &'b NativeEdge,
    ) -> Option<&'b NativeEdge> {
        match cond {
            NativeBranchCond::Value(NativeValue::Imm64(value)) => {
                Some(if *value == 0 { else_edge } else { then_edge })
            }
            _ => None,
        }
    }

    fn emit_branch_cond(
        &mut self,
        cond: &NativeBranchCond,
        then_edge: &NativeEdge,
        else_edge: &NativeEdge,
    ) -> Result<(), WasmError> {
        match cond {
            NativeBranchCond::Value(value) => {
                self.read_value_into(*value, REG_SCRATCH0)?;
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
            NativeBranchCond::I32Eqz(value) => {
                self.read_value_into(*value, REG_SCRATCH0)?;
                self.emitter.emit_u32(enc::cmp_imm_32(REG_SCRATCH0, 0));
                self.emit_conditional_edges(Cond::Eq, then_edge, else_edge)
            }
            NativeBranchCond::I64Eqz(value) => {
                self.read_value_into(*value, REG_SCRATCH0)?;
                self.emitter.emit_u32(enc::cmp_imm_64(REG_SCRATCH0, 0));
                self.emit_conditional_edges(Cond::Eq, then_edge, else_edge)
            }
            NativeBranchCond::I32Compare { kind, lhs, rhs } => {
                self.read_value_into(*lhs, REG_SCRATCH0)?;
                self.read_value_into(*rhs, REG_SCRATCH1)?;
                self.emitter.emit_u32(enc::cmp_reg_32(REG_SCRATCH0, REG_SCRATCH1));
                self.emit_conditional_edges(map_compare_cond(*kind)?, then_edge, else_edge)
            }
            NativeBranchCond::I64Compare { kind, lhs, rhs } => {
                self.read_value_into(*lhs, REG_SCRATCH0)?;
                self.read_value_into(*rhs, REG_SCRATCH1)?;
                self.emitter.emit_u32(enc::cmp_reg_64(REG_SCRATCH0, REG_SCRATCH1));
                self.emit_conditional_edges(map_compare_cond(*kind)?, then_edge, else_edge)
            }
            NativeBranchCond::F32Compare { .. } | NativeBranchCond::F64Compare { .. } => {
                Err(WasmError::internal(
                    "float compare branches are not supported by direct arm64 lowering yet".into(),
                ))
            }
        }
    }

    fn emit_br_table(
        &mut self,
        index: NativeValue,
        entries: &[NativeEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal("br_table has no entries".into()));
        }
        if entries.len() == 1 {
            self.emit_edge(&entries[0])?;
            self.emit_branch_to_block(entries[0].target);
            return Ok(());
        }

        self.read_value_into(index, REG_SCRATCH0)?;
        self.materialize_u64(REG_SCRATCH1, entries.len().saturating_sub(1) as u64);
        self.emitter
            .emit_u32(enc::cmp_reg_64(REG_SCRATCH0, REG_SCRATCH1));
        self.emitter.emit_u32(enc::csel_64(
            REG_SCRATCH0,
            REG_SCRATCH0,
            REG_SCRATCH1,
            Cond::Lo,
        ));

        let table_base_load = self.emitter.len();
        self.emitter.emit_u32(enc::ldr_lit_64(REG_BRANCH_TARGET, 0));
        self.emitter
            .emit_u32(enc::lsl_imm_64(REG_SCRATCH1, REG_SCRATCH0, 3));
        self.emitter
            .emit_u32(enc::add_reg_64(REG_BRANCH_TARGET, REG_BRANCH_TARGET, REG_SCRATCH1));
        self.emitter
            .emit_u32(enc::ldr_64(REG_BRANCH_TARGET, REG_BRANCH_TARGET, 0));
        self.emitter.emit_u32(enc::br(REG_BRANCH_TARGET));

        let table_base_literal = self.emitter.len();
        self.emitter.emit_u64(0);

        let mut stub_offsets = Vec::with_capacity(entries.len());
        for edge in entries {
            stub_offsets.push(self.emitter.len());
            self.emit_edge(edge)?;
            self.emit_branch_to_block(edge.target);
        }

        let table_offset = self.emitter.len();
        for &stub_offset in &stub_offsets {
            let literal_offset = self.emitter.len();
            self.emitter.emit_u64(0);
            self.local_literal_patches.push(Arm64LocalLiteralPatch {
                literal_offset: literal_offset as u32,
                target_offset: stub_offset as u32,
            });
        }

        let table_base_delta =
            ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
        self.emitter.patch_u32(
            table_base_load,
            enc::ldr_lit_64(REG_BRANCH_TARGET, table_base_delta),
        );
        self.local_literal_patches.push(Arm64LocalLiteralPatch {
            literal_offset: table_base_literal as u32,
            target_offset: table_offset as u32,
        });
        Ok(())
    }

    fn emit_refresh_cached_views(&mut self) {
        self.emitter
            .emit_u32(enc::ldr_64(REG_MEMORY_BASE, REG_CTX, MEMORY0_BASE_OFFSET));
        self.emitter
            .emit_u32(enc::ldr_64(REG_MEMORY_LEN, REG_CTX, MEMORY0_LEN_OFFSET));
        self.emitter
            .emit_u32(enc::ldr_64(REG_GLOBALS_BASE, REG_CTX, GLOBALS_BASE_OFFSET));
    }

    fn emit_conditional_edges(
        &mut self,
        cond: Cond,
        then_edge: &NativeEdge,
        else_edge: &NativeEdge,
    ) -> Result<(), WasmError> {
        let else_patch = self.emitter.len();
        self.emitter.emit_u32(enc::b_cond(cond.invert(), 0));
        self.emit_edge(then_edge)?;
        self.emit_branch_to_block(then_edge.target);
        let else_offset = self.emitter.len();
        self.patch_b_cond(else_patch, cond.invert(), else_offset)?;
        self.emit_edge(else_edge)?;
        self.emit_branch_to_block(else_edge.target);
        Ok(())
    }

    fn emit_edge(&mut self, edge: &NativeEdge) -> Result<(), WasmError> {
        let pending: Vec<(NativePlace, CopySource)> = edge
            .copies
            .iter()
            .copied()
            .map(|mov| (mov.dst, CopySource::Value(mov.src)))
            .collect();
        self.emit_parallel_copies(pending)
    }

    fn emit_parallel_copies(
        &mut self,
        mut pending: Vec<(NativePlace, CopySource)>,
    ) -> Result<(), WasmError> {
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

    fn add_u64_immediate(
        &mut self,
        dst: Arm64Reg,
        src: Arm64Reg,
        value: u64,
        scratch: Arm64Reg,
    ) {
        if value == 0 {
            self.move_if_needed(dst, src);
        } else if value < 0x1000 {
            self.emitter.emit_u32(enc::add_imm_64(dst, src, value as u32));
        } else {
            self.materialize_u64(scratch, value);
            self.emitter.emit_u32(enc::add_reg_64(dst, src, scratch));
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

    fn patch_b_cond(
        &mut self,
        patch_offset: usize,
        cond: Cond,
        target_offset: usize,
    ) -> Result<(), WasmError> {
        let delta = ((target_offset as isize - patch_offset as isize) / 4) as i32;
        self.emitter
            .patch_u32(patch_offset, enc::b_cond(cond, delta));
        Ok(())
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

fn arg_reg(index: u8) -> Arm64Reg {
    match index {
        0 => Arm64Reg::X0,
        1 => Arm64Reg::X1,
        2 => Arm64Reg::X2,
        3 => Arm64Reg::X3,
        4 => Arm64Reg::X4,
        5 => Arm64Reg::X5,
        6 => Arm64Reg::X6,
        7 => Arm64Reg::X7,
        reg => panic!("unsupported arm64 arg reg {}", reg),
    }
}

fn map_compare_cond(kind: NativeCompareKind) -> Result<Cond, WasmError> {
    match kind {
        NativeCompareKind::Eq => Ok(Cond::Eq),
        NativeCompareKind::Ne => Ok(Cond::Ne),
        NativeCompareKind::LtS => Ok(Cond::Lt),
        NativeCompareKind::LtU => Ok(Cond::Lo),
        NativeCompareKind::GtS => Ok(Cond::Gt),
        NativeCompareKind::GtU => Ok(Cond::Hi),
        NativeCompareKind::LeS => Ok(Cond::Le),
        NativeCompareKind::LeU => Ok(Cond::Ls),
        NativeCompareKind::GeS => Ok(Cond::Ge),
        NativeCompareKind::GeU => Ok(Cond::Hs),
        NativeCompareKind::Lt | NativeCompareKind::Gt | NativeCompareKind::Le | NativeCompareKind::Ge => {
            Err(WasmError::internal(
                "arm64 direct lowering does not support float compare branches yet".into(),
            ))
        }
    }
}

fn reads_place(src: CopySource, place: NativePlace) -> bool {
    matches!(src, CopySource::Value(NativeValue::Place(src_place)) if src_place == place)
}
