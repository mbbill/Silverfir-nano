//! Wasm→IR lowering pass.
//!
//! Decodes Wasm opcodes and produces `Vec<IrOp>` with all stack management
//! resolved: TOS variants, spill/fill, hot local mapping, control flow.
//!
//! This replaces `dispatch.rs` as the primary decode driver in the IR pipeline.
//! The existing dispatch.rs path remains working alongside this one.

use alloc::vec::Vec;

use super::config::HOT_LOCAL_COUNT;
use super::context::CompileContext;
use super::ir::{self, BrTableEntry, IrOp, IrOpKind, OpIndex, SlotRef};
use super::stack::{BlockKind, StackTracker};
use crate::error::WasmError;
use crate::op_decoder::{Decoder, Immediate, OpcodeHandler, OpStream};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};

/// Lower Wasm function body to IR.
///
/// Returns `Vec<IrOp>` with all stack management resolved.
/// `hot_locals` are needed to emit the InitLocals prologue.
pub fn lower_to_ir<'a>(
    code: &'a [u8],
    ctx: &'a CompileContext<'a>,
    stack: &'a mut StackTracker,
    hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
) -> Result<Vec<IrOp>, WasmError> {
    let mut ir = IrLower {
        ctx,
        stack,
        ops: Vec::with_capacity(256),
    };

    // Emit InitLocals prologue (mirrors build_for_function's emitter.emit_init_locals)
    ir.emit(IrOpKind::InitLocals {
        k0: hot_locals[0].unwrap_or(0) as u16,
        k1: hot_locals[1].unwrap_or(1) as u16,
        k2: hot_locals[2].unwrap_or(2) as u16,
    }, 0);

    // Decode and dispatch
    let mut decoder = Decoder::new(code);
    let mut handler = LowerHandler { ir: &mut ir };
    decoder.add_handler(&mut handler);
    decoder.decode_function()?;

    Ok(ir.ops)
}

/// IR lowering state.
struct IrLower<'a> {
    ctx: &'a CompileContext<'a>,
    stack: &'a mut StackTracker,
    ops: Vec<IrOp>,
}

impl<'a> IrLower<'a> {
    /// Current IR index (next op will be at this index).
    #[inline]
    fn current_index(&self) -> OpIndex {
        OpIndex::new(self.ops.len())
    }

    /// Compute pre_height: the stack height BEFORE this op executes.
    ///
    /// Callers pop the stack before calling emit, so `self.stack.height()` is
    /// post-pop.  We reconstruct the pre-op height by adding back the pops
    /// from `ir::stack_effect`, since callers pop the stack before calling
    /// emit.  This ensures pre_height reflects the true pre-op height.
    #[inline]
    fn pre_op_height(&self, kind: &IrOpKind) -> u16 {
        let (pops, _) = ir::stack_effect(kind);
        (self.stack.height() + pops as usize) as u16
    }

    /// Emit an IR op with automatic fallthrough.
    fn emit(&mut self, kind: IrOpKind, variant: u8) -> OpIndex {
        let idx = OpIndex::new(self.ops.len());
        let pre_height = self.pre_op_height(&kind);
        self.ops.push(IrOp {
            kind,
            variant,
            pre_height,
            fallthrough: Some(idx.next()),
            alt_target: None,
            has_target: false,
        });
        idx
    }

    /// Emit an IR op without fallthrough (terminal).
    fn emit_terminal(&mut self, kind: IrOpKind, variant: u8) -> OpIndex {
        let idx = OpIndex::new(self.ops.len());
        let pre_height = self.pre_op_height(&kind);
        self.ops.push(IrOp {
            kind,
            variant,
            pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        });
        idx
    }

    /// Emit an IR op with has_target set.
    fn emit_with_target(&mut self, kind: IrOpKind, variant: u8) -> OpIndex {
        let idx = OpIndex::new(self.ops.len());
        let pre_height = self.pre_op_height(&kind);
        self.ops.push(IrOp {
            kind,
            variant,
            pre_height,
            fallthrough: Some(idx.next()),
            alt_target: None,
            has_target: true,
        });
        idx
    }

    /// Patch alt_target for an IR op.
    fn patch_alt(&mut self, idx: OpIndex, alt: OpIndex) {
        if let Some(op) = self.ops.get_mut(idx.as_usize()) {
            op.alt_target = Some(alt);
        }
    }

    /// Patch Br/BrIf data for forward branches.
    fn patch_br_data(&mut self, idx: OpIndex, stack_offset: u64, arity: u64) {
        if let Some(op) = self.ops.get_mut(idx.as_usize()) {
            match &mut op.kind {
                IrOpKind::Br { stack_drop, arity: a, .. } => {
                    *stack_drop = stack_offset as u32;
                    *a = arity as u16;
                }
                IrOpKind::BrIf { stack_drop, arity: a, .. } => {
                    *stack_drop = stack_offset as u32;
                    *a = arity as u16;
                }
                _ => {}
            }
        }
    }

    /// Patch br_table entry target.
    fn patch_br_table_target(&mut self, inst_idx: OpIndex, entry_idx: usize, target_idx: OpIndex) {
        if let Some(op) = self.ops.get_mut(inst_idx.as_usize()) {
            if let IrOpKind::BrTable { entries, .. } = &mut op.kind {
                if let Some(entry) = entries.get_mut(entry_idx) {
                    entry.target_idx = Some(target_idx);
                }
            }
        }
    }

    // =========================================================================
    // Variant computation helpers
    // =========================================================================

    /// Variant for push ops: post-push depth.
    #[inline]
    fn post_push_variant(&self) -> u8 {
        let post_depth = self.stack.depth() + 1;
        if post_depth == 0 {
            1
        } else {
            (((post_depth - 1) % self.stack.config().tos_register_count) + 1) as u8
        }
    }

    /// Variant for pop ops: pre-pop depth.
    #[inline]
    fn pre_pop_variant(&self) -> u8 {
        self.stack.depth_variant()
    }

    // =========================================================================
    // Spill/Fill helpers (mirrors dispatch.rs)
    // =========================================================================

    /// Emit spill if TOS cache is full before a push.
    fn emit_spill_if_needed(&mut self) {
        if self.stack.is_unreachable() { return; }
        if self.stack.needs_spill_before_push() {
            let spill_depth = self.stack.spill_depth();
            let slot = (self.stack.operand_base() + spill_depth) as u16;
            let variant = ((spill_depth % self.stack.config().tos_register_count) + 1) as u8;
            self.stack.record_spill(1);
            self.emit(IrOpKind::Spill { slot, count: 1 }, variant);
        }
    }

    /// Emit fill if operands are in memory.
    fn emit_fill_for_operands(&mut self, operand_count: usize) {
        if self.stack.is_unreachable() { return; }
        let spill_depth = self.stack.spill_depth();
        let height = self.stack.height();
        let min_spill_for_operands = height.saturating_sub(operand_count);
        if spill_depth > min_spill_for_operands {
            let fill_count = spill_depth - min_spill_for_operands;
            let slot = (self.stack.operand_base() + spill_depth - 1) as u16;
            let variant = (((spill_depth - 1) % self.stack.config().tos_register_count) + 1) as u8;
            self.stack.record_fill(fill_count);
            self.emit(IrOpKind::Fill { slot, count: fill_count as u8 }, variant);
        }
    }

    /// Emit fill for control flow normalization.
    fn emit_fill_if_needed(&mut self) {
        if self.stack.is_unreachable() { return; }
        let old_spill_depth = self.stack.spill_depth();
        let fill_count = self.stack.normalize_for_control_flow();
        if fill_count > 0 {
            let slot = (self.stack.operand_base() + old_spill_depth - 1) as u16;
            let variant = (((old_spill_depth - 1) % self.stack.config().tos_register_count) + 1) as u8;
            self.emit(IrOpKind::Fill { slot, count: fill_count as u8 }, variant);
        }
    }

    /// Spill all TOS values.
    fn emit_spill_all(&mut self) {
        if self.stack.is_unreachable() { return; }
        let tos_count = self.stack.tos_count();
        if tos_count > 0 {
            let height = self.stack.height();
            let slot = (self.stack.operand_base() + height - 1) as u16;
            let variant = (((height - 1) % self.stack.config().tos_register_count) + 1) as u8;
            self.stack.record_spill(tos_count);
            self.emit(IrOpKind::Spill { slot, count: tos_count as u8 }, variant);
        }
    }

    /// Spill all except top N.
    fn emit_spill_all_except_top(&mut self, keep_top: usize) {
        if self.stack.is_unreachable() { return; }
        let tos_count = self.stack.tos_count();
        let to_spill = tos_count.saturating_sub(keep_top);
        if to_spill > 0 {
            let spill_depth = self.stack.spill_depth();
            let slot = (self.stack.operand_base() + spill_depth + to_spill - 1) as u16;
            let variant = (((spill_depth + to_spill - 1) % self.stack.config().tos_register_count) + 1) as u8;
            self.stack.record_spill(to_spill);
            self.emit(IrOpKind::Spill { slot, count: to_spill as u8 }, variant);
        }
    }

    // =========================================================================
    // Opcode handlers
    // =========================================================================

    /// Handle a push-1 op (constant, global_get, etc.).
    fn handle_push_op(&mut self, kind: IrOpKind) {
        self.emit_spill_if_needed();
        let v = self.post_push_variant();
        self.emit(kind, v);
        self.stack.push();
    }

    /// Handle a binary op (pop 2, push 1).
    fn handle_binop(&mut self, kind: IrOpKind) {
        self.emit_fill_for_operands(2);
        let v = self.pre_pop_variant();
        self.stack.pop();
        self.stack.pop();
        self.emit(kind, v);
        self.stack.push();
    }

    /// Handle a unary op (pop 1, push 1).
    fn handle_unop(&mut self, kind: IrOpKind) {
        self.emit_fill_for_operands(1);
        let v = self.pre_pop_variant();
        self.stack.pop();
        self.emit(kind, v);
        self.stack.push();
    }

    /// Handle a ternary op (pop 3, push 0).
    fn handle_ternary(&mut self, kind: IrOpKind) {
        self.emit_fill_for_operands(3);
        let v = self.pre_pop_variant();
        self.stack.pop();
        self.stack.pop();
        self.stack.pop();
        self.emit(kind, v);
    }

    /// Dispatch a single opcode.
    fn dispatch(&mut self, op: WasmOpcode, imm: &Immediate) {
        use Opcode::*;
        use WasmOpcode::{FC, OP};

        match op {
            // =================================================================
            // Locals
            // =================================================================
            OP(LOCAL_GET) => {
                if let Immediate::LocalIndex(idx) = imm {
                    let remapped = self.stack.remap_local(*idx);
                    self.emit_spill_if_needed();
                    let v = self.post_push_variant();
                    let kind = if remapped == 0 && self.stack.has_l0() {
                        IrOpKind::LocalGetHot { reg: 0 }
                    } else if remapped == 1 && self.stack.has_l1() {
                        IrOpKind::LocalGetHot { reg: 1 }
                    } else if remapped == 2 && self.stack.has_hot_local(2) {
                        IrOpKind::LocalGetHot { reg: 2 }
                    } else {
                        IrOpKind::LocalGetFrame { idx: remapped as u16 }
                    };
                    self.emit(kind, v);
                    self.stack.push();
                }
            }
            OP(LOCAL_SET) => {
                if let Immediate::LocalIndex(idx) = imm {
                    let remapped = self.stack.remap_local(*idx);
                    self.emit_fill_for_operands(1);
                    let v = self.pre_pop_variant();
                    let kind = if remapped == 0 && self.stack.has_l0() {
                        IrOpKind::LocalSetHot { reg: 0 }
                    } else if remapped == 1 && self.stack.has_l1() {
                        IrOpKind::LocalSetHot { reg: 1 }
                    } else if remapped == 2 && self.stack.has_hot_local(2) {
                        IrOpKind::LocalSetHot { reg: 2 }
                    } else {
                        IrOpKind::LocalSetFrame { idx: remapped as u16 }
                    };
                    self.stack.pop();
                    self.emit(kind, v);
                }
            }
            OP(LOCAL_TEE) => {
                if let Immediate::LocalIndex(idx) = imm {
                    let remapped = self.stack.remap_local(*idx);
                    self.emit_fill_for_operands(1);
                    let v = self.pre_pop_variant();
                    let kind = if remapped == 0 && self.stack.has_l0() {
                        IrOpKind::LocalTeeHot { reg: 0 }
                    } else if remapped == 1 && self.stack.has_l1() {
                        IrOpKind::LocalTeeHot { reg: 1 }
                    } else if remapped == 2 && self.stack.has_hot_local(2) {
                        IrOpKind::LocalTeeHot { reg: 2 }
                    } else {
                        IrOpKind::LocalTeeFrame { idx: remapped as u16 }
                    };
                    self.emit(kind, v);
                }
            }

            // =================================================================
            // Constants
            // =================================================================
            OP(I32_CONST) => {
                if let Immediate::I32(v) = imm {
                    self.handle_push_op(IrOpKind::I32Const { value: *v as u32 });
                }
            }
            OP(I64_CONST) => {
                if let Immediate::I64(v) = imm {
                    self.handle_push_op(IrOpKind::I64Const { value: *v as u64 });
                }
            }
            OP(F32_CONST) => {
                if let Immediate::F32(v) = imm {
                    self.handle_push_op(IrOpKind::F32Const { value: v.to_bits() });
                }
            }
            OP(F64_CONST) => {
                if let Immediate::F64(v) = imm {
                    self.handle_push_op(IrOpKind::F64Const { value: v.to_bits() });
                }
            }

            // =================================================================
            // Binary ops (pop 2, push 1)
            // =================================================================
            OP(I32_ADD) => self.handle_binop(IrOpKind::I32Add),
            OP(I32_SUB) => self.handle_binop(IrOpKind::I32Sub),
            OP(I32_MUL) => self.handle_binop(IrOpKind::I32Mul),
            OP(I32_DIV_S) => self.handle_binop(IrOpKind::I32DivS),
            OP(I32_DIV_U) => self.handle_binop(IrOpKind::I32DivU),
            OP(I32_REM_S) => self.handle_binop(IrOpKind::I32RemS),
            OP(I32_REM_U) => self.handle_binop(IrOpKind::I32RemU),
            OP(I32_AND) => self.handle_binop(IrOpKind::I32And),
            OP(I32_OR) => self.handle_binop(IrOpKind::I32Or),
            OP(I32_XOR) => self.handle_binop(IrOpKind::I32Xor),
            OP(I32_SHL) => self.handle_binop(IrOpKind::I32Shl),
            OP(I32_SHR_S) => self.handle_binop(IrOpKind::I32ShrS),
            OP(I32_SHR_U) => self.handle_binop(IrOpKind::I32ShrU),
            OP(I32_ROTL) => self.handle_binop(IrOpKind::I32Rotl),
            OP(I32_ROTR) => self.handle_binop(IrOpKind::I32Rotr),

            OP(I64_ADD) => self.handle_binop(IrOpKind::I64Add),
            OP(I64_SUB) => self.handle_binop(IrOpKind::I64Sub),
            OP(I64_MUL) => self.handle_binop(IrOpKind::I64Mul),
            OP(I64_DIV_S) => self.handle_binop(IrOpKind::I64DivS),
            OP(I64_DIV_U) => self.handle_binop(IrOpKind::I64DivU),
            OP(I64_REM_S) => self.handle_binop(IrOpKind::I64RemS),
            OP(I64_REM_U) => self.handle_binop(IrOpKind::I64RemU),
            OP(I64_AND) => self.handle_binop(IrOpKind::I64And),
            OP(I64_OR) => self.handle_binop(IrOpKind::I64Or),
            OP(I64_XOR) => self.handle_binop(IrOpKind::I64Xor),
            OP(I64_SHL) => self.handle_binop(IrOpKind::I64Shl),
            OP(I64_SHR_S) => self.handle_binop(IrOpKind::I64ShrS),
            OP(I64_SHR_U) => self.handle_binop(IrOpKind::I64ShrU),
            OP(I64_ROTL) => self.handle_binop(IrOpKind::I64Rotl),
            OP(I64_ROTR) => self.handle_binop(IrOpKind::I64Rotr),

            OP(F32_ADD) => self.handle_binop(IrOpKind::F32Add),
            OP(F32_SUB) => self.handle_binop(IrOpKind::F32Sub),
            OP(F32_MUL) => self.handle_binop(IrOpKind::F32Mul),
            OP(F32_DIV) => self.handle_binop(IrOpKind::F32Div),
            OP(F32_MIN) => self.handle_binop(IrOpKind::F32Min),
            OP(F32_MAX) => self.handle_binop(IrOpKind::F32Max),
            OP(F32_COPYSIGN) => self.handle_binop(IrOpKind::F32Copysign),

            OP(F64_ADD) => self.handle_binop(IrOpKind::F64Add),
            OP(F64_SUB) => self.handle_binop(IrOpKind::F64Sub),
            OP(F64_MUL) => self.handle_binop(IrOpKind::F64Mul),
            OP(F64_DIV) => self.handle_binop(IrOpKind::F64Div),
            OP(F64_MIN) => self.handle_binop(IrOpKind::F64Min),
            OP(F64_MAX) => self.handle_binop(IrOpKind::F64Max),
            OP(F64_COPYSIGN) => self.handle_binop(IrOpKind::F64Copysign),

            // =================================================================
            // Comparisons (pop 2, push 1)
            // =================================================================
            OP(I32_EQ) => self.handle_binop(IrOpKind::I32Eq),
            OP(I32_NE) => self.handle_binop(IrOpKind::I32Ne),
            OP(I32_LT_S) => self.handle_binop(IrOpKind::I32LtS),
            OP(I32_LT_U) => self.handle_binop(IrOpKind::I32LtU),
            OP(I32_GT_S) => self.handle_binop(IrOpKind::I32GtS),
            OP(I32_GT_U) => self.handle_binop(IrOpKind::I32GtU),
            OP(I32_LE_S) => self.handle_binop(IrOpKind::I32LeS),
            OP(I32_LE_U) => self.handle_binop(IrOpKind::I32LeU),
            OP(I32_GE_S) => self.handle_binop(IrOpKind::I32GeS),
            OP(I32_GE_U) => self.handle_binop(IrOpKind::I32GeU),

            OP(I64_EQ) => self.handle_binop(IrOpKind::I64Eq),
            OP(I64_NE) => self.handle_binop(IrOpKind::I64Ne),
            OP(I64_LT_S) => self.handle_binop(IrOpKind::I64LtS),
            OP(I64_LT_U) => self.handle_binop(IrOpKind::I64LtU),
            OP(I64_GT_S) => self.handle_binop(IrOpKind::I64GtS),
            OP(I64_GT_U) => self.handle_binop(IrOpKind::I64GtU),
            OP(I64_LE_S) => self.handle_binop(IrOpKind::I64LeS),
            OP(I64_LE_U) => self.handle_binop(IrOpKind::I64LeU),
            OP(I64_GE_S) => self.handle_binop(IrOpKind::I64GeS),
            OP(I64_GE_U) => self.handle_binop(IrOpKind::I64GeU),

            OP(F32_EQ) => self.handle_binop(IrOpKind::F32Eq),
            OP(F32_NE) => self.handle_binop(IrOpKind::F32Ne),
            OP(F32_LT) => self.handle_binop(IrOpKind::F32Lt),
            OP(F32_GT) => self.handle_binop(IrOpKind::F32Gt),
            OP(F32_LE) => self.handle_binop(IrOpKind::F32Le),
            OP(F32_GE) => self.handle_binop(IrOpKind::F32Ge),

            OP(F64_EQ) => self.handle_binop(IrOpKind::F64Eq),
            OP(F64_NE) => self.handle_binop(IrOpKind::F64Ne),
            OP(F64_LT) => self.handle_binop(IrOpKind::F64Lt),
            OP(F64_GT) => self.handle_binop(IrOpKind::F64Gt),
            OP(F64_LE) => self.handle_binop(IrOpKind::F64Le),
            OP(F64_GE) => self.handle_binop(IrOpKind::F64Ge),

            // =================================================================
            // Unary ops (pop 1, push 1)
            // =================================================================
            OP(I32_EQZ) => self.handle_unop(IrOpKind::I32Eqz),
            OP(I32_CLZ) => self.handle_unop(IrOpKind::I32Clz),
            OP(I32_CTZ) => self.handle_unop(IrOpKind::I32Ctz),
            OP(I32_POPCNT) => self.handle_unop(IrOpKind::I32Popcnt),

            OP(I64_EQZ) => self.handle_unop(IrOpKind::I64Eqz),
            OP(I64_CLZ) => self.handle_unop(IrOpKind::I64Clz),
            OP(I64_CTZ) => self.handle_unop(IrOpKind::I64Ctz),
            OP(I64_POPCNT) => self.handle_unop(IrOpKind::I64Popcnt),

            OP(F32_ABS) => self.handle_unop(IrOpKind::F32Abs),
            OP(F32_NEG) => self.handle_unop(IrOpKind::F32Neg),
            OP(F32_CEIL) => self.handle_unop(IrOpKind::F32Ceil),
            OP(F32_FLOOR) => self.handle_unop(IrOpKind::F32Floor),
            OP(F32_TRUNC) => self.handle_unop(IrOpKind::F32Trunc),
            OP(F32_NEAREST) => self.handle_unop(IrOpKind::F32Nearest),
            OP(F32_SQRT) => self.handle_unop(IrOpKind::F32Sqrt),

            OP(F64_ABS) => self.handle_unop(IrOpKind::F64Abs),
            OP(F64_NEG) => self.handle_unop(IrOpKind::F64Neg),
            OP(F64_CEIL) => self.handle_unop(IrOpKind::F64Ceil),
            OP(F64_FLOOR) => self.handle_unop(IrOpKind::F64Floor),
            OP(F64_TRUNC) => self.handle_unop(IrOpKind::F64Trunc),
            OP(F64_NEAREST) => self.handle_unop(IrOpKind::F64Nearest),
            OP(F64_SQRT) => self.handle_unop(IrOpKind::F64Sqrt),

            // =================================================================
            // Conversions (pop 1, push 1)
            // =================================================================
            OP(I32_WRAP_I64) => self.handle_unop(IrOpKind::I32WrapI64),
            OP(I32_TRUNC_F32_S) => self.handle_unop(IrOpKind::I32TruncF32S),
            OP(I32_TRUNC_F32_U) => self.handle_unop(IrOpKind::I32TruncF32U),
            OP(I32_TRUNC_F64_S) => self.handle_unop(IrOpKind::I32TruncF64S),
            OP(I32_TRUNC_F64_U) => self.handle_unop(IrOpKind::I32TruncF64U),
            OP(I64_EXTEND_I32_S) => self.handle_unop(IrOpKind::I64ExtendI32S),
            OP(I64_EXTEND_I32_U) => self.handle_unop(IrOpKind::I64ExtendI32U),
            OP(I64_TRUNC_F32_S) => self.handle_unop(IrOpKind::I64TruncF32S),
            OP(I64_TRUNC_F32_U) => self.handle_unop(IrOpKind::I64TruncF32U),
            OP(I64_TRUNC_F64_S) => self.handle_unop(IrOpKind::I64TruncF64S),
            OP(I64_TRUNC_F64_U) => self.handle_unop(IrOpKind::I64TruncF64U),
            OP(F32_CONVERT_I32_S) => self.handle_unop(IrOpKind::F32ConvertI32S),
            OP(F32_CONVERT_I32_U) => self.handle_unop(IrOpKind::F32ConvertI32U),
            OP(F32_CONVERT_I64_S) => self.handle_unop(IrOpKind::F32ConvertI64S),
            OP(F32_CONVERT_I64_U) => self.handle_unop(IrOpKind::F32ConvertI64U),
            OP(F32_DEMOTE_F64) => self.handle_unop(IrOpKind::F32DemoteF64),
            OP(F64_CONVERT_I32_S) => self.handle_unop(IrOpKind::F64ConvertI32S),
            OP(F64_CONVERT_I32_U) => self.handle_unop(IrOpKind::F64ConvertI32U),
            OP(F64_CONVERT_I64_S) => self.handle_unop(IrOpKind::F64ConvertI64S),
            OP(F64_CONVERT_I64_U) => self.handle_unop(IrOpKind::F64ConvertI64U),
            OP(F64_PROMOTE_F32) => self.handle_unop(IrOpKind::F64PromoteF32),
            OP(I32_REINTERPRET_F32) => self.handle_unop(IrOpKind::I32ReinterpretF32),
            OP(I64_REINTERPRET_F64) => self.handle_unop(IrOpKind::I64ReinterpretF64),
            OP(F32_REINTERPRET_I32) => self.handle_unop(IrOpKind::F32ReinterpretI32),
            OP(F64_REINTERPRET_I64) => self.handle_unop(IrOpKind::F64ReinterpretI64),

            // Sign extension
            OP(I32_EXTEND8_S) => self.handle_unop(IrOpKind::I32Extend8S),
            OP(I32_EXTEND16_S) => self.handle_unop(IrOpKind::I32Extend16S),
            OP(I64_EXTEND8_S) => self.handle_unop(IrOpKind::I64Extend8S),
            OP(I64_EXTEND16_S) => self.handle_unop(IrOpKind::I64Extend16S),
            OP(I64_EXTEND32_S) => self.handle_unop(IrOpKind::I64Extend32S),

            // Saturating truncation
            FC(OpcodeFC::I32_TRUNC_SAT_F32_S) => self.handle_unop(IrOpKind::I32TruncSatF32S),
            FC(OpcodeFC::I32_TRUNC_SAT_F32_U) => self.handle_unop(IrOpKind::I32TruncSatF32U),
            FC(OpcodeFC::I32_TRUNC_SAT_F64_S) => self.handle_unop(IrOpKind::I32TruncSatF64S),
            FC(OpcodeFC::I32_TRUNC_SAT_F64_U) => self.handle_unop(IrOpKind::I32TruncSatF64U),
            FC(OpcodeFC::I64_TRUNC_SAT_F32_S) => self.handle_unop(IrOpKind::I64TruncSatF32S),
            FC(OpcodeFC::I64_TRUNC_SAT_F32_U) => self.handle_unop(IrOpKind::I64TruncSatF32U),
            FC(OpcodeFC::I64_TRUNC_SAT_F64_S) => self.handle_unop(IrOpKind::I64TruncSatF64S),
            FC(OpcodeFC::I64_TRUNC_SAT_F64_U) => self.handle_unop(IrOpKind::I64TruncSatF64U),

            // =================================================================
            // Memory loads (pop 1, push 1)
            // =================================================================
            OP(I32_LOAD) => self.handle_load(imm, |o, m| IrOpKind::I32Load { offset: o, memidx: m }),
            OP(I64_LOAD) => self.handle_load(imm, |o, m| IrOpKind::I64Load { offset: o, memidx: m }),
            OP(F32_LOAD) => self.handle_load(imm, |o, m| IrOpKind::F32Load { offset: o, memidx: m }),
            OP(F64_LOAD) => self.handle_load(imm, |o, m| IrOpKind::F64Load { offset: o, memidx: m }),
            OP(I32_LOAD8_S) => self.handle_load(imm, |o, m| IrOpKind::I32Load8S { offset: o, memidx: m }),
            OP(I32_LOAD8_U) => self.handle_load(imm, |o, m| IrOpKind::I32Load8U { offset: o, memidx: m }),
            OP(I32_LOAD16_S) => self.handle_load(imm, |o, m| IrOpKind::I32Load16S { offset: o, memidx: m }),
            OP(I32_LOAD16_U) => self.handle_load(imm, |o, m| IrOpKind::I32Load16U { offset: o, memidx: m }),
            OP(I64_LOAD8_S) => self.handle_load(imm, |o, m| IrOpKind::I64Load8S { offset: o, memidx: m }),
            OP(I64_LOAD8_U) => self.handle_load(imm, |o, m| IrOpKind::I64Load8U { offset: o, memidx: m }),
            OP(I64_LOAD16_S) => self.handle_load(imm, |o, m| IrOpKind::I64Load16S { offset: o, memidx: m }),
            OP(I64_LOAD16_U) => self.handle_load(imm, |o, m| IrOpKind::I64Load16U { offset: o, memidx: m }),
            OP(I64_LOAD32_S) => self.handle_load(imm, |o, m| IrOpKind::I64Load32S { offset: o, memidx: m }),
            OP(I64_LOAD32_U) => self.handle_load(imm, |o, m| IrOpKind::I64Load32U { offset: o, memidx: m }),

            // =================================================================
            // Memory stores (pop 2, push 0)
            // =================================================================
            OP(I32_STORE) => self.handle_store(imm, |o, m| IrOpKind::I32Store { offset: o, memidx: m }),
            OP(I64_STORE) => self.handle_store(imm, |o, m| IrOpKind::I64Store { offset: o, memidx: m }),
            OP(F32_STORE) => self.handle_store(imm, |o, m| IrOpKind::F32Store { offset: o, memidx: m }),
            OP(F64_STORE) => self.handle_store(imm, |o, m| IrOpKind::F64Store { offset: o, memidx: m }),
            OP(I32_STORE8) => self.handle_store(imm, |o, m| IrOpKind::I32Store8 { offset: o, memidx: m }),
            OP(I32_STORE16) => self.handle_store(imm, |o, m| IrOpKind::I32Store16 { offset: o, memidx: m }),
            OP(I64_STORE8) => self.handle_store(imm, |o, m| IrOpKind::I64Store8 { offset: o, memidx: m }),
            OP(I64_STORE16) => self.handle_store(imm, |o, m| IrOpKind::I64Store16 { offset: o, memidx: m }),
            OP(I64_STORE32) => self.handle_store(imm, |o, m| IrOpKind::I64Store32 { offset: o, memidx: m }),

            // =================================================================
            // Memory size/grow
            // =================================================================
            OP(MEMORY_SIZE) => {
                if let Immediate::MemoryIndex(mem_idx) = imm {
                    self.handle_push_op(IrOpKind::MemorySize { mem_idx: *mem_idx });
                }
            }
            OP(MEMORY_GROW) => {
                if let Immediate::MemoryIndex(mem_idx) = imm {
                    self.emit_fill_for_operands(1);
                    let v = self.pre_pop_variant();
                    self.stack.pop();
                    self.emit(IrOpKind::MemoryGrow { mem_idx: *mem_idx }, v);
                    self.stack.push();
                }
            }

            // =================================================================
            // Bulk memory
            // =================================================================
            FC(OpcodeFC::MEMORY_FILL) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_ternary(IrOpKind::MemoryFill { imm0, imm1 });
            }
            FC(OpcodeFC::MEMORY_COPY) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_ternary(IrOpKind::MemoryCopy { imm0, imm1 });
            }
            FC(OpcodeFC::MEMORY_INIT) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_ternary(IrOpKind::MemoryInit { imm0, imm1 });
            }
            FC(OpcodeFC::DATA_DROP) => {
                if let Immediate::DataIndex(data_idx) = imm {
                    self.emit(IrOpKind::DataDrop { data_idx: *data_idx }, 0);
                }
            }

            // =================================================================
            // Globals
            // =================================================================
            OP(GLOBAL_GET) => {
                if let Immediate::GlobalIndex(global_idx) = imm {
                    self.handle_push_op(IrOpKind::GlobalGet { idx: *global_idx });
                }
            }
            OP(GLOBAL_SET) => {
                if let Immediate::GlobalIndex(global_idx) = imm {
                    self.emit_fill_for_operands(1);
                    let v = self.pre_pop_variant();
                    self.stack.pop();
                    self.emit(IrOpKind::GlobalSet { idx: *global_idx }, v);
                }
            }

            // =================================================================
            // Tables
            // =================================================================
            OP(TABLE_GET) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    self.emit_fill_for_operands(1);
                    let v = self.pre_pop_variant();
                    self.stack.pop();
                    self.emit(IrOpKind::TableGet { table_idx: *table_idx }, v);
                    self.stack.push();
                }
            }
            OP(TABLE_SET) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    self.emit_fill_for_operands(2);
                    let v = self.pre_pop_variant();
                    self.stack.pop();
                    self.stack.pop();
                    self.emit(IrOpKind::TableSet { table_idx: *table_idx }, v);
                }
            }
            FC(OpcodeFC::TABLE_SIZE) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    self.handle_push_op(IrOpKind::TableSize { table_idx: *table_idx });
                }
            }
            FC(OpcodeFC::TABLE_GROW) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    self.emit_fill_for_operands(2);
                    let v = self.pre_pop_variant();
                    self.stack.pop();
                    self.stack.pop();
                    self.emit(IrOpKind::TableGrow { table_idx: *table_idx }, v);
                    self.stack.push();
                }
            }
            FC(OpcodeFC::TABLE_FILL) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_ternary(IrOpKind::TableFill { imm0, imm1 });
            }
            FC(OpcodeFC::TABLE_COPY) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_ternary(IrOpKind::TableCopy { imm0, imm1 });
            }
            FC(OpcodeFC::TABLE_INIT) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_ternary(IrOpKind::TableInit { imm0, imm1 });
            }
            FC(OpcodeFC::ELEM_DROP) => {
                if let Immediate::ElementIndex(elem_idx) = imm {
                    self.emit(IrOpKind::ElemDrop { elem_idx: *elem_idx }, 0);
                }
            }

            // =================================================================
            // References
            // =================================================================
            OP(REF_NULL) => {
                self.handle_push_op(IrOpKind::RefNull);
            }
            OP(REF_IS_NULL) => self.handle_unop(IrOpKind::RefIsNull),
            OP(REF_FUNC) => {
                if let Immediate::FunctionIndex(func_idx) = imm {
                    self.handle_push_op(IrOpKind::RefFunc { func_idx: *func_idx });
                }
            }

            // =================================================================
            // Stack ops
            // =================================================================
            OP(DROP) => {
                self.emit_fill_for_operands(1);
                let v = self.pre_pop_variant();
                self.stack.pop();
                self.emit(IrOpKind::Drop, v);
            }
            OP(SELECT) | OP(SELECT_T) => {
                self.emit_fill_for_operands(3);
                let v = self.pre_pop_variant();
                self.stack.pop(); // cond
                self.stack.pop(); // val2
                self.stack.pop(); // val1
                self.emit(IrOpKind::Select, v);
                self.stack.push();
            }

            // =================================================================
            // Control flow
            // =================================================================
            OP(BLOCK) => {
                let (params, results) = self.ctx.resolve_block_type_from_imm(imm);
                let idx = self.emit(IrOpKind::Block, 0);
                self.stack.enter_block(BlockKind::Block, params, results, idx);
            }
            OP(LOOP) => {
                let (params, results) = self.ctx.resolve_block_type_from_imm(imm);
                self.emit_spill_all();
                self.emit(IrOpKind::Loop, 0);
                let loop_target = self.current_index();
                self.emit_fill_if_needed();
                self.stack.enter_block(BlockKind::Loop, params, results, loop_target);
            }
            OP(IF) => {
                self.emit_fill_for_operands(1);
                self.emit_spill_all_except_top(1);
                let v = self.pre_pop_variant();
                self.stack.pop();
                let (params, results) = self.ctx.resolve_block_type_from_imm(imm);
                let idx = self.emit_with_target(IrOpKind::If, v);
                self.stack.enter_block(BlockKind::If, params, results, idx);
                self.stack.set_if_inst(idx);
            }
            OP(ELSE) => {
                self.emit_spill_all();
                let idx = self.emit_with_target(IrOpKind::Else, 0);
                self.stack.set_else_inst(idx);
                self.stack.enter_else();
                let else_body_start = self.current_index();
                if let Some(frame) = self.stack.current_frame() {
                    if let Some(if_idx) = frame.if_inst_idx {
                        self.patch_alt(if_idx, else_body_start);
                    }
                }
            }
            OP(END) => {
                let has_forward_branches = self.stack.current_frame()
                    .map(|f| !f.pending_fixups.is_empty())
                    .unwrap_or(false);
                let has_stack_drop_branch = self.stack.current_frame()
                    .map(|f| f.pending_fixups.iter().any(|fix| fix.stack_offset > 0))
                    .unwrap_or(false);
                let is_if_block = self.stack.current_frame()
                    .map(|f| f.kind == BlockKind::If)
                    .unwrap_or(false);
                let needs_sync = has_stack_drop_branch && self.stack.spill_depth() < self.stack.height();
                if needs_sync {
                    self.emit_spill_all();
                }
                let needs_fill = has_forward_branches || is_if_block;
                let target_idx = self.current_index();
                if needs_fill {
                    self.emit_fill_if_needed();
                }
                self.emit(IrOpKind::End, 0);
                if let Some((frame, _can_preserve)) = self.stack.exit_block() {
                    for fixup in &frame.pending_fixups {
                        if let Some(entry_idx) = fixup.br_table_entry {
                            self.patch_br_table_target(fixup.inst_idx, entry_idx, target_idx);
                        } else {
                            self.patch_alt(fixup.inst_idx, target_idx);
                            self.patch_br_data(fixup.inst_idx, fixup.stack_offset as u64, fixup.arity as u64);
                        }
                    }
                    if frame.kind == BlockKind::If && frame.else_inst_idx.is_none() {
                        if let Some(if_idx) = frame.if_inst_idx {
                            self.patch_alt(if_idx, target_idx);
                        }
                    }
                    if let Some(else_idx) = frame.else_inst_idx {
                        self.patch_alt(else_idx, target_idx);
                    }
                }
            }

            // =================================================================
            // Branches
            // =================================================================
            OP(BR) => {
                if let Immediate::LabelIndex(label) = imm {
                    self.emit_spill_all();
                    let arity = self.stack.branch_arity(*label);
                    let (stack_offset, target) = self.stack.branch_info(*label);
                    let current_height = self.stack.height();
                    let operand_base_offset = (self.stack.operand_base() * 8) as u32;
                    let idx = self.emit_with_target(
                        IrOpKind::Br {
                            stack_drop: stack_offset as u32,
                            arity: arity as u16,
                            height: current_height as u16,
                            operand_base_offset,
                        },
                        0,
                    );
                    if let Some(tgt) = target {
                        self.patch_alt(idx, tgt);
                    } else {
                        self.stack.register_forward_branch(*label, idx, None);
                    }
                    self.stack.set_unreachable();
                }
            }
            OP(BR_IF) => {
                if let Immediate::LabelIndex(label) = imm {
                    self.emit_fill_for_operands(1);
                    let tos_count_before_spill = self.stack.tos_count();
                    self.emit_spill_all_except_top(1);
                    let variant_idx = ((self.stack.height().saturating_sub(1)
                        % self.stack.config().tos_register_count)
                        + 1) as u8;
                    let pre_pop_height = self.stack.height();
                    self.stack.pop();
                    let arity = self.stack.branch_arity(*label);
                    let (stack_offset, target) = self.stack.branch_info(*label);
                    let idx = if arity == 0 && stack_offset == 0 {
                        self.emit_with_target(IrOpKind::BrIfSimple, variant_idx)
                    } else {
                        let operand_base_offset = (self.stack.operand_base() * 8) as u32;
                        self.emit_with_target(
                            IrOpKind::BrIf {
                                stack_drop: stack_offset as u32,
                                arity: arity as u16,
                                height: pre_pop_height as u16,
                                operand_base_offset,
                            },
                            variant_idx,
                        )
                    };
                    self.stack.record_fill(tos_count_before_spill.saturating_sub(1));
                    if let Some(tgt) = target {
                        self.patch_alt(idx, tgt);
                    } else {
                        self.stack.register_forward_branch(*label, idx, None);
                    }
                }
            }
            OP(BR_TABLE) => {
                if let Immediate::BrLabels(labels, default_label) = imm {
                    self.emit_fill_for_operands(1);
                    self.emit_spill_all_except_top(1);
                    let v = self.pre_pop_variant();
                    let pre_pop_height = self.stack.height();
                    self.stack.pop();
                    let effective_height = pre_pop_height;
                    let mut entries = Vec::with_capacity(labels.len() + 1);
                    let all_labels: Vec<u32> = labels.iter().copied().chain(core::iter::once(*default_label)).collect();
                    for &label in &all_labels {
                        let (stack_offset, target) = self.stack.branch_info(label);
                        let arity = self.stack.branch_arity(label);
                        entries.push(BrTableEntry {
                            target_idx: target,
                            stack_offset,
                            arity,
                        });
                    }
                    let operand_base_offset = (self.stack.operand_base() * 8) as u32;
                    let inst_idx = self.emit_with_target(
                        IrOpKind::BrTable {
                            entries,
                            entry_count: 0,
                            data_slot_count: 0,
                            height: effective_height as u16,
                            operand_base_offset,
                        },
                        v,
                    );
                    for (entry_idx, &label) in all_labels.iter().enumerate() {
                        let (_, target) = self.stack.branch_info(label);
                        if target.is_none() {
                            self.stack.register_forward_branch(label, inst_idx, Some(entry_idx));
                        }
                    }
                    self.stack.set_unreachable();
                }
            }

            // =================================================================
            // Return / Unreachable
            // =================================================================
            OP(RETURN) => {
                self.emit_spill_all();
                let arity = self.ctx.results_count();
                let frame_size = self.stack.frame_size();
                let current_height = self.stack.height();
                self.emit_return(arity, frame_size, current_height);
                self.stack.set_unreachable();
            }
            OP(UNREACHABLE) => {
                self.emit_terminal(IrOpKind::Unreachable, 0);
                self.stack.set_unreachable();
            }

            // =================================================================
            // Calls
            // =================================================================
            OP(CALL) => {
                if let Immediate::FunctionIndex(func_idx) = imm {
                    self.emit_spill_all();
                    let (params, results) = self.ctx.resolve_func_type(*func_idx as usize);
                    let delta = SlotRef::operand_relative((self.stack.height() - params) as u16);
                    if self.ctx.is_func_internal(*func_idx as usize) {
                        if let Some(callee_inst) = self.ctx.get_func_inst(*func_idx as usize) {
                            self.emit(
                                IrOpKind::CallInternal {
                                    callee: callee_inst as *const _ as u64,
                                    delta,
                                },
                                0,
                            );
                        } else {
                            self.emit(
                                IrOpKind::CallExternal { func_idx: *func_idx, delta },
                                0,
                            );
                        }
                    } else {
                        self.emit(
                            IrOpKind::CallExternal { func_idx: *func_idx, delta },
                            0,
                        );
                    }
                    self.stack.apply_call(params, results);
                    self.stack.reset_tos_state();
                }
            }
            OP(CALL_INDIRECT) => {
                if let Immediate::CallIndirectArgs { typeidx, tableidx } = imm {
                    self.emit_spill_all();
                    let (params, results) = self.ctx.resolve_type_index(*typeidx as usize);
                    let delta = SlotRef::operand_relative((self.stack.height() - params - 1) as u16);
                    let operand_base_offset = (self.stack.operand_base() * 8) as u32;
                    let height = self.stack.height() as u16;
                    self.stack.pop(); // Pop materialized index
                    self.emit(
                        IrOpKind::CallIndirect {
                            type_idx: *typeidx,
                            table_idx: *tableidx,
                            delta,
                            operand_base_offset,
                            height,
                        },
                        0,
                    );
                    self.stack.apply_call(params, results);
                    self.stack.reset_tos_state();
                }
            }

            // =================================================================
            // NOP
            // =================================================================
            OP(NOP) => {
                self.emit(IrOpKind::Nop, 0);
            }

            // =================================================================
            // Catch-all for unhandled opcodes
            // =================================================================
            _ => {
                self.emit_fill_if_needed();
                self.emit(IrOpKind::Nop, 0);
            }
        }
    }

    /// Handle load op: pop address, push value.
    fn handle_load(&mut self, imm: &Immediate, make_kind: impl FnOnce(u32, u32) -> IrOpKind) {
        if let Immediate::MemArg { memidx, offset, .. } = imm {
            self.emit_fill_for_operands(1);
            let v = self.pre_pop_variant();
            self.stack.pop();
            self.emit(make_kind(*offset as u32, *memidx), v);
            self.stack.push();
        }
    }

    /// Handle store op: pop value + address.
    fn handle_store(&mut self, imm: &Immediate, make_kind: impl FnOnce(u32, u32) -> IrOpKind) {
        if let Immediate::MemArg { memidx, offset, .. } = imm {
            self.emit_fill_for_operands(2);
            let v = self.pre_pop_variant();
            self.stack.pop();
            self.stack.pop();
            self.emit(make_kind(*offset as u32, *memidx), v);
        }
    }

    /// Emit return (specialized by arity).
    fn emit_return(&mut self, arity: usize, frame_size: usize, current_height: usize) {
        let operand_base_offset = self.stack.config().operand_base_offset(frame_size) as u32;
        match arity {
            0 => {
                self.emit_terminal(
                    IrOpKind::ReturnVoid { frame_size: frame_size as u16 },
                    0,
                );
            }
            1 => {
                self.emit_terminal(
                    IrOpKind::ReturnOne {
                        frame_size: frame_size as u16,
                        operand_base_offset,
                        height: current_height as u16,
                    },
                    0,
                );
            }
            _ => {
                self.emit_terminal(
                    IrOpKind::Return {
                        arity: arity as u16,
                        frame_size: frame_size as u16,
                        operand_base_offset,
                        height: current_height as u16,
                    },
                    0,
                );
            }
        }
    }
}

/// Extract (imm0, imm1) from an Immediate (for ternary ops).
fn extract_imm01(imm: &Immediate) -> (u32, u32) {
    match imm {
        Immediate::MemoryIndex(idx) => (*idx, 0),
        Immediate::TableIndex(idx) => (*idx, 0),
        Immediate::MemoryInitArgs { dataidx, memidx } => (*memidx, *dataidx),
        Immediate::MemoryCopyArgs { dstidx, srcidx } => (*dstidx, *srcidx),
        Immediate::TableInitArgs { elemidx, tableidx } => (*tableidx, *elemidx),
        Immediate::TableCopyArgs { dstidx, srcidx } => (*dstidx, *srcidx),
        _ => (0, 0),
    }
}

/// Decoder handler that drives the lowering pass.
struct LowerHandler<'a, 'b> {
    ir: &'a mut IrLower<'b>,
}

impl<'a, 'b> OpcodeHandler for LowerHandler<'a, 'b> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            self.ir.dispatch(decoded.wasm_op, &decoded.imm);
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        // Emit implicit RETURN at function end (if not unreachable)
        if !self.ir.stack.is_unreachable() {
            self.ir.emit_spill_all();
            let arity = self.ir.ctx.results_count();
            let frame_size = self.ir.stack.frame_size();
            let current_height = self.ir.stack.height();
            self.ir.emit_return(arity, frame_size, current_height);
        }
        Ok(())
    }
}
