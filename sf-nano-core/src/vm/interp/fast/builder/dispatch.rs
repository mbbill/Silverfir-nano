//! Opcode dispatch: decode and handle each opcode.

use super::super::handler_lookup;
use super::super::TOS_REGISTER_COUNT;

use super::context::CompileContext;
use super::emitter::CodeEmitter;
#[cfg(feature = "fusion")]
use super::{FusedOp, OpFuser};
#[cfg(feature = "fusion")]
use super::emit_fused;
use super::stack::{BlockKind, StackTracker};
use super::temp_inst::{BrTableEntry, Handler};
use crate::op_decoder::{Decoder, Immediate, OpcodeHandler, OpStream};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};
use crate::vm::interp::fast::handlers::full_set::*;
use crate::error::WasmError;

use alloc::vec::Vec;

#[cfg(feature = "tos-stats")]
use std::eprintln;

/// Marker base for operand stack during compilation.
/// Used to create placeholder values that are fixed up in finalizer.
const OPERAND_BASE: usize = 16384;

// =============================================================================
// TOS Spill/Fill Helpers
// =============================================================================

/// Emit spill if TOS cache is full before a push operation.
#[inline]
fn emit_spill_if_needed(stack: &mut StackTracker, emitter: &mut CodeEmitter) {
    if stack.is_unreachable() {
        return;
    }
    if stack.needs_spill_before_push() {
        let spill_depth = stack.spill_depth();
        let slot = (stack.operand_base() + spill_depth) as u16;
        let variant = (spill_depth % TOS_REGISTER_COUNT) as u8;

        stack.record_spill(1);
        emitter.emit_spill(slot, 1, variant);
        // Update JIT height after spill so the next TempInst gets the correct pre_height.
        #[cfg(feature = "micro-jit")]
        emitter.set_current_height(stack.height() as u16);
    }
}

/// Emit fill if there are spilled values that need to be in TOS for control flow.
#[inline]
fn emit_fill_if_needed(stack: &mut StackTracker, emitter: &mut CodeEmitter) {
    if stack.is_unreachable() {
        return;
    }
    let old_spill_depth = stack.spill_depth();
    let fill_count = stack.normalize_for_control_flow();
    if fill_count > 0 {
        let slot = (stack.operand_base() + old_spill_depth - 1) as u16;
        let variant = ((old_spill_depth - 1) % TOS_REGISTER_COUNT) as u8;
        emitter.emit_fill(slot, fill_count as u8, variant);
        #[cfg(feature = "micro-jit")]
        emitter.set_current_height(stack.height() as u16);
    }
}

/// Emit fill if operands for an operation are in memory (not in TOS).
#[inline]
fn emit_fill_for_operands(stack: &mut StackTracker, emitter: &mut CodeEmitter, operand_count: usize) {
    if stack.is_unreachable() {
        return;
    }
    let spill_depth = stack.spill_depth();

    let height = stack.height();
    let min_spill_for_operands = height.saturating_sub(operand_count);
    if spill_depth > min_spill_for_operands {
        let fill_count = spill_depth - min_spill_for_operands;

        let slot = (stack.operand_base() + spill_depth - 1) as u16;
        let variant = ((spill_depth - 1) % TOS_REGISTER_COUNT) as u8;

        stack.record_fill(fill_count);
        emitter.emit_fill(slot, fill_count as u8, variant);
        #[cfg(feature = "micro-jit")]
        emitter.set_current_height(stack.height() as u16);
    }
}

// =============================================================================
// TOS Depth Variant Selection Helpers
// =============================================================================

#[inline]
fn pre_op_variant_idx(stack: &StackTracker) -> usize {
    (stack.depth_variant() - 1) as usize
}

#[inline]
fn post_op_variant_idx(stack: &StackTracker) -> usize {
    let post_depth = stack.depth() + 1;
    if post_depth == 0 {
        0
    } else {
        (post_depth - 1) % TOS_REGISTER_COUNT
    }
}

/// Spill all TOS values before a call (callee expects args in memory).
#[inline]
fn emit_spill_all_for_call(stack: &mut StackTracker, emitter: &mut CodeEmitter) {
    if stack.is_unreachable() {
        return;
    }
    let tos_count = stack.tos_count();
    if tos_count > 0 {
        let height = stack.height();
        let slot = (stack.operand_base() + height - 1) as u16;
        let variant = ((height - 1) % TOS_REGISTER_COUNT) as u8;

        stack.record_spill(tos_count);
        emitter.emit_spill(slot, tos_count as u8, variant);
    }
}

/// Spill all TOS values EXCEPT the top `keep_top` values.
#[inline]
fn emit_spill_all_except_top(stack: &mut StackTracker, emitter: &mut CodeEmitter, keep_top: usize) {
    if stack.is_unreachable() {
        return;
    }
    let tos_count = stack.tos_count();
    let to_spill = tos_count.saturating_sub(keep_top);
    if to_spill > 0 {
        let spill_depth = stack.spill_depth();
        let slot = (stack.operand_base() + spill_depth + to_spill - 1) as u16;
        let variant = ((spill_depth + to_spill - 1) % TOS_REGISTER_COUNT) as u8;

        stack.record_spill(to_spill);
        emitter.emit_spill(slot, to_spill as u8, variant);
    }
}

/// Internal handler struct for decoding.
struct DispatchHandler<'a> {
    ctx: &'a CompileContext<'a>,
    stack: &'a mut StackTracker,
    emitter: &'a mut CodeEmitter,
}

impl<'a> OpcodeHandler for DispatchHandler<'a> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        #[cfg(feature = "fusion")]
        if !super::super::is_fusion_disabled() {
            let mut fuser = OpFuser::new(stream);
            while let Some(op) = fuser.next(self.stack)? {
                match op {
                    FusedOp::Single { wasm_op, imm } => {
                        #[cfg(feature = "tos-stats")]
                        {
                            let h = self.stack.height();
                            let tc = self.stack.tos_count();
                            let dv = self.stack.depth_variant();
                            eprintln!("TOS h={} tos={} D{} | {:?}", h, tc, dv, wasm_op);
                        }
                        dispatch_opcode(wasm_op, &imm, self.ctx, self.stack, self.emitter);
                    }
                    fused => {
                        #[cfg(feature = "tos-stats")]
                        {
                            let h = self.stack.height();
                            let tc = self.stack.tos_count();
                            let dv = self.stack.depth_variant();
                            eprintln!("TOS h={} tos={} D{} | fused:{:?}", h, tc, dv, fused);
                        }
                        emit_fused(fused, self.stack, self.emitter);
                    }
                }
            }
            return Ok(());
        }

        // No fusion: emit one handler per Wasm opcode.
        while let Some(decoded) = stream.next()? {
            #[cfg(feature = "tos-stats")]
            {
                let h = self.stack.height();
                let tc = self.stack.tos_count();
                let dv = self.stack.depth_variant();
                eprintln!("TOS h={} tos={} D{} | {:?}", h, tc, dv, decoded.wasm_op);
            }
            dispatch_opcode(decoded.wasm_op, &decoded.imm, self.ctx, self.stack, self.emitter);
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        // Emit explicit RETURN at function end (if not already unreachable)
        if !self.stack.is_unreachable() {
            emit_spill_all_for_call(self.stack, self.emitter);

            let arity = self.ctx.results_count();
            let frame_size = self.stack.frame_size();
            let current_height = self.stack.height();
            self.emitter.emit_return(arity, frame_size, current_height);
        }
        Ok(())
    }
}

/// Decode function body and dispatch each opcode.
pub fn decode_and_dispatch<'a>(
    code: &'a [u8],
    ctx: &'a CompileContext<'a>,
    stack: &'a mut StackTracker,
    emitter: &'a mut CodeEmitter,
) -> Result<(), WasmError> {
    #[cfg(feature = "tos-stats")]
    {
        static FUNC_COUNTER: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        let idx = FUNC_COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        eprintln!("=== FUNC {} ===", idx);
    }
    let mut decoder = Decoder::new(code);
    let mut handler = DispatchHandler { ctx, stack, emitter };
    decoder.add_handler(&mut handler);
    decoder.decode_function()
}

/// Dispatch a single opcode.
fn dispatch_opcode(
    op: WasmOpcode,
    imm: &Immediate,
    ctx: &CompileContext,
    stack: &mut StackTracker,
    emitter: &mut CodeEmitter,
) {
    use Opcode::*;
    use WasmOpcode::{FC, OP};

    #[cfg(feature = "micro-jit")]
    emitter.set_current_height(stack.height() as u16);

    match op {
        // =====================================================================
        // SP-based Local Operations: Emit actual instructions
        // =====================================================================
        OP(LOCAL_GET) => {
            if let Immediate::LocalIndex(idx) = imm {
                let remapped = stack.remap_local(*idx);
                emit_spill_if_needed(stack, emitter);
                let variant_idx = post_op_variant_idx(stack);
                let handler = if remapped == 0 && stack.has_l0() {
                    handler_lookup::LOCAL_GET_L0[variant_idx]
                } else if remapped == 1 && stack.has_l1() {
                    handler_lookup::LOCAL_GET_L1[variant_idx]
                } else if remapped == 2 && stack.has_hot_local(2) {
                    handler_lookup::LOCAL_GET_L2[variant_idx]
                } else {
                    handler_lookup::LOCAL_GET[variant_idx]
                };
                emitter.emit_local_get_variant(handler, remapped);
                stack.push();
            }
        }
        OP(LOCAL_SET) => {
            if let Immediate::LocalIndex(idx) = imm {
                let remapped = stack.remap_local(*idx);
                emit_fill_for_operands(stack, emitter, 1);
                let variant_idx = pre_op_variant_idx(stack);
                let handler = if remapped == 0 && stack.has_l0() {
                    handler_lookup::LOCAL_SET_L0[variant_idx]
                } else if remapped == 1 && stack.has_l1() {
                    handler_lookup::LOCAL_SET_L1[variant_idx]
                } else if remapped == 2 && stack.has_hot_local(2) {
                    handler_lookup::LOCAL_SET_L2[variant_idx]
                } else {
                    handler_lookup::LOCAL_SET[variant_idx]
                };
                stack.pop();
                emitter.emit_local_set_variant(handler, remapped);
            }
        }
        OP(LOCAL_TEE) => {
            if let Immediate::LocalIndex(idx) = imm {
                let remapped = stack.remap_local(*idx);
                emit_fill_for_operands(stack, emitter, 1);
                let variant_idx = pre_op_variant_idx(stack);
                let handler = if remapped == 0 && stack.has_l0() {
                    handler_lookup::LOCAL_TEE_L0[variant_idx]
                } else if remapped == 1 && stack.has_l1() {
                    handler_lookup::LOCAL_TEE_L1[variant_idx]
                } else if remapped == 2 && stack.has_hot_local(2) {
                    handler_lookup::LOCAL_TEE_L2[variant_idx]
                } else {
                    handler_lookup::LOCAL_TEE[variant_idx]
                };
                emitter.emit_local_tee_variant(handler, remapped);
            }
        }

        // =====================================================================
        // SP-based Constants: Emit actual instructions
        // =====================================================================
        OP(I32_CONST) => {
            if let Immediate::I32(v) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::I32_CONST[idx];
                emitter.emit_const_sp(handler, *v as u64, op);
                stack.push();
            }
        }
        OP(I64_CONST) => {
            if let Immediate::I64(v) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::I64_CONST[idx];
                emitter.emit_const_sp(handler, *v as u64, op);
                stack.push();
            }
        }
        OP(F32_CONST) => {
            if let Immediate::F32(v) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::F32_CONST[idx];
                emitter.emit_const_sp(handler, v.to_bits() as u64, op);
                stack.push();
            }
        }
        OP(F64_CONST) => {
            if let Immediate::F64(v) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::F64_CONST[idx];
                emitter.emit_const_sp(handler, v.to_bits(), op);
                stack.push();
            }
        }

        // =====================================================================
        // Binary Ops (pop 2, push 1)
        // =====================================================================
        OP(I32_ADD) => handle_binop(stack, emitter, &handler_lookup::I32_ADD, op),
        OP(I32_SUB) => handle_binop(stack, emitter, &handler_lookup::I32_SUB, op),
        OP(I32_MUL) => handle_binop(stack, emitter, &handler_lookup::I32_MUL, op),
        OP(I32_DIV_S) => handle_binop(stack, emitter, &handler_lookup::I32_DIV_S, op),
        OP(I32_DIV_U) => handle_binop(stack, emitter, &handler_lookup::I32_DIV_U, op),
        OP(I32_REM_S) => handle_binop(stack, emitter, &handler_lookup::I32_REM_S, op),
        OP(I32_REM_U) => handle_binop(stack, emitter, &handler_lookup::I32_REM_U, op),
        OP(I32_AND) => handle_binop(stack, emitter, &handler_lookup::I32_AND, op),
        OP(I32_OR) => handle_binop(stack, emitter, &handler_lookup::I32_OR, op),
        OP(I32_XOR) => handle_binop(stack, emitter, &handler_lookup::I32_XOR, op),
        OP(I32_SHL) => handle_binop(stack, emitter, &handler_lookup::I32_SHL, op),
        OP(I32_SHR_S) => handle_binop(stack, emitter, &handler_lookup::I32_SHR_S, op),
        OP(I32_SHR_U) => handle_binop(stack, emitter, &handler_lookup::I32_SHR_U, op),
        OP(I32_ROTL) => handle_binop(stack, emitter, &handler_lookup::I32_ROTL, op),
        OP(I32_ROTR) => handle_binop(stack, emitter, &handler_lookup::I32_ROTR, op),

        OP(I64_ADD) => handle_binop(stack, emitter, &handler_lookup::I64_ADD, op),
        OP(I64_SUB) => handle_binop(stack, emitter, &handler_lookup::I64_SUB, op),
        OP(I64_MUL) => handle_binop(stack, emitter, &handler_lookup::I64_MUL, op),
        OP(I64_DIV_S) => handle_binop(stack, emitter, &handler_lookup::I64_DIV_S, op),
        OP(I64_DIV_U) => handle_binop(stack, emitter, &handler_lookup::I64_DIV_U, op),
        OP(I64_REM_S) => handle_binop(stack, emitter, &handler_lookup::I64_REM_S, op),
        OP(I64_REM_U) => handle_binop(stack, emitter, &handler_lookup::I64_REM_U, op),
        OP(I64_AND) => handle_binop(stack, emitter, &handler_lookup::I64_AND, op),
        OP(I64_OR) => handle_binop(stack, emitter, &handler_lookup::I64_OR, op),
        OP(I64_XOR) => handle_binop(stack, emitter, &handler_lookup::I64_XOR, op),
        OP(I64_SHL) => handle_binop(stack, emitter, &handler_lookup::I64_SHL, op),
        OP(I64_SHR_S) => handle_binop(stack, emitter, &handler_lookup::I64_SHR_S, op),
        OP(I64_SHR_U) => handle_binop(stack, emitter, &handler_lookup::I64_SHR_U, op),
        OP(I64_ROTL) => handle_binop(stack, emitter, &handler_lookup::I64_ROTL, op),
        OP(I64_ROTR) => handle_binop(stack, emitter, &handler_lookup::I64_ROTR, op),

        OP(F32_ADD) => handle_binop(stack, emitter, &handler_lookup::F32_ADD, op),
        OP(F32_SUB) => handle_binop(stack, emitter, &handler_lookup::F32_SUB, op),
        OP(F32_MUL) => handle_binop(stack, emitter, &handler_lookup::F32_MUL, op),
        OP(F32_DIV) => handle_binop(stack, emitter, &handler_lookup::F32_DIV, op),
        OP(F32_MIN) => handle_binop(stack, emitter, &handler_lookup::F32_MIN, op),
        OP(F32_MAX) => handle_binop(stack, emitter, &handler_lookup::F32_MAX, op),
        OP(F32_COPYSIGN) => handle_binop(stack, emitter, &handler_lookup::F32_COPYSIGN, op),

        OP(F64_ADD) => handle_binop(stack, emitter, &handler_lookup::F64_ADD, op),
        OP(F64_SUB) => handle_binop(stack, emitter, &handler_lookup::F64_SUB, op),
        OP(F64_MUL) => handle_binop(stack, emitter, &handler_lookup::F64_MUL, op),
        OP(F64_DIV) => handle_binop(stack, emitter, &handler_lookup::F64_DIV, op),
        OP(F64_MIN) => handle_binop(stack, emitter, &handler_lookup::F64_MIN, op),
        OP(F64_MAX) => handle_binop(stack, emitter, &handler_lookup::F64_MAX, op),
        OP(F64_COPYSIGN) => handle_binop(stack, emitter, &handler_lookup::F64_COPYSIGN, op),

        // =====================================================================
        // Comparison Ops (pop 2, push 1)
        // =====================================================================
        OP(I32_EQ) => handle_binop(stack, emitter, &handler_lookup::I32_EQ, op),
        OP(I32_NE) => handle_binop(stack, emitter, &handler_lookup::I32_NE, op),
        OP(I32_LT_S) => handle_binop(stack, emitter, &handler_lookup::I32_LT_S, op),
        OP(I32_LT_U) => handle_binop(stack, emitter, &handler_lookup::I32_LT_U, op),
        OP(I32_GT_S) => handle_binop(stack, emitter, &handler_lookup::I32_GT_S, op),
        OP(I32_GT_U) => handle_binop(stack, emitter, &handler_lookup::I32_GT_U, op),
        OP(I32_LE_S) => handle_binop(stack, emitter, &handler_lookup::I32_LE_S, op),
        OP(I32_LE_U) => handle_binop(stack, emitter, &handler_lookup::I32_LE_U, op),
        OP(I32_GE_S) => handle_binop(stack, emitter, &handler_lookup::I32_GE_S, op),
        OP(I32_GE_U) => handle_binop(stack, emitter, &handler_lookup::I32_GE_U, op),

        OP(I64_EQ) => handle_binop(stack, emitter, &handler_lookup::I64_EQ, op),
        OP(I64_NE) => handle_binop(stack, emitter, &handler_lookup::I64_NE, op),
        OP(I64_LT_S) => handle_binop(stack, emitter, &handler_lookup::I64_LT_S, op),
        OP(I64_LT_U) => handle_binop(stack, emitter, &handler_lookup::I64_LT_U, op),
        OP(I64_GT_S) => handle_binop(stack, emitter, &handler_lookup::I64_GT_S, op),
        OP(I64_GT_U) => handle_binop(stack, emitter, &handler_lookup::I64_GT_U, op),
        OP(I64_LE_S) => handle_binop(stack, emitter, &handler_lookup::I64_LE_S, op),
        OP(I64_LE_U) => handle_binop(stack, emitter, &handler_lookup::I64_LE_U, op),
        OP(I64_GE_S) => handle_binop(stack, emitter, &handler_lookup::I64_GE_S, op),
        OP(I64_GE_U) => handle_binop(stack, emitter, &handler_lookup::I64_GE_U, op),

        OP(F32_EQ) => handle_binop(stack, emitter, &handler_lookup::F32_EQ, op),
        OP(F32_NE) => handle_binop(stack, emitter, &handler_lookup::F32_NE, op),
        OP(F32_LT) => handle_binop(stack, emitter, &handler_lookup::F32_LT, op),
        OP(F32_GT) => handle_binop(stack, emitter, &handler_lookup::F32_GT, op),
        OP(F32_LE) => handle_binop(stack, emitter, &handler_lookup::F32_LE, op),
        OP(F32_GE) => handle_binop(stack, emitter, &handler_lookup::F32_GE, op),

        OP(F64_EQ) => handle_binop(stack, emitter, &handler_lookup::F64_EQ, op),
        OP(F64_NE) => handle_binop(stack, emitter, &handler_lookup::F64_NE, op),
        OP(F64_LT) => handle_binop(stack, emitter, &handler_lookup::F64_LT, op),
        OP(F64_GT) => handle_binop(stack, emitter, &handler_lookup::F64_GT, op),
        OP(F64_LE) => handle_binop(stack, emitter, &handler_lookup::F64_LE, op),
        OP(F64_GE) => handle_binop(stack, emitter, &handler_lookup::F64_GE, op),

        // =====================================================================
        // Unary Ops (pop 1, push 1)
        // =====================================================================
        OP(I32_EQZ) => handle_unop(stack, emitter, &handler_lookup::I32_EQZ, op),
        OP(I32_CLZ) => handle_unop(stack, emitter, &handler_lookup::I32_CLZ, op),
        OP(I32_CTZ) => handle_unop(stack, emitter, &handler_lookup::I32_CTZ, op),
        OP(I32_POPCNT) => handle_unop(stack, emitter, &handler_lookup::I32_POPCNT, op),

        OP(I64_EQZ) => handle_unop(stack, emitter, &handler_lookup::I64_EQZ, op),
        OP(I64_CLZ) => handle_unop(stack, emitter, &handler_lookup::I64_CLZ, op),
        OP(I64_CTZ) => handle_unop(stack, emitter, &handler_lookup::I64_CTZ, op),
        OP(I64_POPCNT) => handle_unop(stack, emitter, &handler_lookup::I64_POPCNT, op),

        OP(F32_ABS) => handle_unop(stack, emitter, &handler_lookup::F32_ABS, op),
        OP(F32_NEG) => handle_unop(stack, emitter, &handler_lookup::F32_NEG, op),
        OP(F32_CEIL) => handle_unop(stack, emitter, &handler_lookup::F32_CEIL, op),
        OP(F32_FLOOR) => handle_unop(stack, emitter, &handler_lookup::F32_FLOOR, op),
        OP(F32_TRUNC) => handle_unop(stack, emitter, &handler_lookup::F32_TRUNC, op),
        OP(F32_NEAREST) => handle_unop(stack, emitter, &handler_lookup::F32_NEAREST, op),
        OP(F32_SQRT) => handle_unop(stack, emitter, &handler_lookup::F32_SQRT, op),

        OP(F64_ABS) => handle_unop(stack, emitter, &handler_lookup::F64_ABS, op),
        OP(F64_NEG) => handle_unop(stack, emitter, &handler_lookup::F64_NEG, op),
        OP(F64_CEIL) => handle_unop(stack, emitter, &handler_lookup::F64_CEIL, op),
        OP(F64_FLOOR) => handle_unop(stack, emitter, &handler_lookup::F64_FLOOR, op),
        OP(F64_TRUNC) => handle_unop(stack, emitter, &handler_lookup::F64_TRUNC, op),
        OP(F64_NEAREST) => handle_unop(stack, emitter, &handler_lookup::F64_NEAREST, op),
        OP(F64_SQRT) => handle_unop(stack, emitter, &handler_lookup::F64_SQRT, op),

        // =====================================================================
        // Conversions (pop 1, push 1)
        // =====================================================================
        OP(I32_WRAP_I64) => handle_unop(stack, emitter, &handler_lookup::I32_WRAP_I64, op),
        OP(I32_TRUNC_F32_S) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_F32_S, op),
        OP(I32_TRUNC_F32_U) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_F32_U, op),
        OP(I32_TRUNC_F64_S) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_F64_S, op),
        OP(I32_TRUNC_F64_U) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_F64_U, op),
        OP(I64_EXTEND_I32_S) => handle_unop(stack, emitter, &handler_lookup::I64_EXTEND_I32_S, op),
        OP(I64_EXTEND_I32_U) => handle_unop(stack, emitter, &handler_lookup::I64_EXTEND_I32_U, op),
        OP(I64_TRUNC_F32_S) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_F32_S, op),
        OP(I64_TRUNC_F32_U) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_F32_U, op),
        OP(I64_TRUNC_F64_S) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_F64_S, op),
        OP(I64_TRUNC_F64_U) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_F64_U, op),
        OP(F32_CONVERT_I32_S) => handle_unop(stack, emitter, &handler_lookup::F32_CONVERT_I32_S, op),
        OP(F32_CONVERT_I32_U) => handle_unop(stack, emitter, &handler_lookup::F32_CONVERT_I32_U, op),
        OP(F32_CONVERT_I64_S) => handle_unop(stack, emitter, &handler_lookup::F32_CONVERT_I64_S, op),
        OP(F32_CONVERT_I64_U) => handle_unop(stack, emitter, &handler_lookup::F32_CONVERT_I64_U, op),
        OP(F32_DEMOTE_F64) => handle_unop(stack, emitter, &handler_lookup::F32_DEMOTE_F64, op),
        OP(F64_CONVERT_I32_S) => handle_unop(stack, emitter, &handler_lookup::F64_CONVERT_I32_S, op),
        OP(F64_CONVERT_I32_U) => handle_unop(stack, emitter, &handler_lookup::F64_CONVERT_I32_U, op),
        OP(F64_CONVERT_I64_S) => handle_unop(stack, emitter, &handler_lookup::F64_CONVERT_I64_S, op),
        OP(F64_CONVERT_I64_U) => handle_unop(stack, emitter, &handler_lookup::F64_CONVERT_I64_U, op),
        OP(F64_PROMOTE_F32) => handle_unop(stack, emitter, &handler_lookup::F64_PROMOTE_F32, op),
        OP(I32_REINTERPRET_F32) => handle_unop(stack, emitter, &handler_lookup::I32_REINTERPRET_F32, op),
        OP(I64_REINTERPRET_F64) => handle_unop(stack, emitter, &handler_lookup::I64_REINTERPRET_F64, op),
        OP(F32_REINTERPRET_I32) => handle_unop(stack, emitter, &handler_lookup::F32_REINTERPRET_I32, op),
        OP(F64_REINTERPRET_I64) => handle_unop(stack, emitter, &handler_lookup::F64_REINTERPRET_I64, op),

        // Sign extension
        OP(I32_EXTEND8_S) => handle_unop(stack, emitter, &handler_lookup::I32_EXTEND8_S, op),
        OP(I32_EXTEND16_S) => handle_unop(stack, emitter, &handler_lookup::I32_EXTEND16_S, op),
        OP(I64_EXTEND8_S) => handle_unop(stack, emitter, &handler_lookup::I64_EXTEND8_S, op),
        OP(I64_EXTEND16_S) => handle_unop(stack, emitter, &handler_lookup::I64_EXTEND16_S, op),
        OP(I64_EXTEND32_S) => handle_unop(stack, emitter, &handler_lookup::I64_EXTEND32_S, op),

        // Saturating truncation
        FC(OpcodeFC::I32_TRUNC_SAT_F32_S) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_SAT_F32_S, op),
        FC(OpcodeFC::I32_TRUNC_SAT_F32_U) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_SAT_F32_U, op),
        FC(OpcodeFC::I32_TRUNC_SAT_F64_S) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_SAT_F64_S, op),
        FC(OpcodeFC::I32_TRUNC_SAT_F64_U) => handle_unop(stack, emitter, &handler_lookup::I32_TRUNC_SAT_F64_U, op),
        FC(OpcodeFC::I64_TRUNC_SAT_F32_S) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_SAT_F32_S, op),
        FC(OpcodeFC::I64_TRUNC_SAT_F32_U) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_SAT_F32_U, op),
        FC(OpcodeFC::I64_TRUNC_SAT_F64_S) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_SAT_F64_S, op),
        FC(OpcodeFC::I64_TRUNC_SAT_F64_U) => handle_unop(stack, emitter, &handler_lookup::I64_TRUNC_SAT_F64_U, op),

        // =====================================================================
        // Memory Loads (pop 1, push 1)
        // =====================================================================
        OP(I32_LOAD) => handle_load(stack, emitter, imm, &handler_lookup::I32_LOAD, &handler_lookup::I32_LOAD_MM, op),
        OP(I64_LOAD) => handle_load(stack, emitter, imm, &handler_lookup::I64_LOAD, &handler_lookup::I64_LOAD_MM, op),
        OP(F32_LOAD) => handle_load(stack, emitter, imm, &handler_lookup::F32_LOAD, &handler_lookup::F32_LOAD_MM, op),
        OP(F64_LOAD) => handle_load(stack, emitter, imm, &handler_lookup::F64_LOAD, &handler_lookup::F64_LOAD_MM, op),
        OP(I32_LOAD8_S) => handle_load(stack, emitter, imm, &handler_lookup::I32_LOAD8_S, &handler_lookup::I32_LOAD8_S_MM, op),
        OP(I32_LOAD8_U) => handle_load(stack, emitter, imm, &handler_lookup::I32_LOAD8_U, &handler_lookup::I32_LOAD8_U_MM, op),
        OP(I32_LOAD16_S) => handle_load(stack, emitter, imm, &handler_lookup::I32_LOAD16_S, &handler_lookup::I32_LOAD16_S_MM, op),
        OP(I32_LOAD16_U) => handle_load(stack, emitter, imm, &handler_lookup::I32_LOAD16_U, &handler_lookup::I32_LOAD16_U_MM, op),
        OP(I64_LOAD8_S) => handle_load(stack, emitter, imm, &handler_lookup::I64_LOAD8_S, &handler_lookup::I64_LOAD8_S_MM, op),
        OP(I64_LOAD8_U) => handle_load(stack, emitter, imm, &handler_lookup::I64_LOAD8_U, &handler_lookup::I64_LOAD8_U_MM, op),
        OP(I64_LOAD16_S) => handle_load(stack, emitter, imm, &handler_lookup::I64_LOAD16_S, &handler_lookup::I64_LOAD16_S_MM, op),
        OP(I64_LOAD16_U) => handle_load(stack, emitter, imm, &handler_lookup::I64_LOAD16_U, &handler_lookup::I64_LOAD16_U_MM, op),
        OP(I64_LOAD32_S) => handle_load(stack, emitter, imm, &handler_lookup::I64_LOAD32_S, &handler_lookup::I64_LOAD32_S_MM, op),
        OP(I64_LOAD32_U) => handle_load(stack, emitter, imm, &handler_lookup::I64_LOAD32_U, &handler_lookup::I64_LOAD32_U_MM, op),

        // =====================================================================
        // Memory Stores (pop 2, push 0)
        // =====================================================================
        OP(I32_STORE) => handle_store(stack, emitter, imm, &handler_lookup::I32_STORE, &handler_lookup::I32_STORE_MM, op),
        OP(I64_STORE) => handle_store(stack, emitter, imm, &handler_lookup::I64_STORE, &handler_lookup::I64_STORE_MM, op),
        OP(F32_STORE) => handle_store(stack, emitter, imm, &handler_lookup::F32_STORE, &handler_lookup::F32_STORE_MM, op),
        OP(F64_STORE) => handle_store(stack, emitter, imm, &handler_lookup::F64_STORE, &handler_lookup::F64_STORE_MM, op),
        OP(I32_STORE8) => handle_store(stack, emitter, imm, &handler_lookup::I32_STORE8, &handler_lookup::I32_STORE8_MM, op),
        OP(I32_STORE16) => handle_store(stack, emitter, imm, &handler_lookup::I32_STORE16, &handler_lookup::I32_STORE16_MM, op),
        OP(I64_STORE8) => handle_store(stack, emitter, imm, &handler_lookup::I64_STORE8, &handler_lookup::I64_STORE8_MM, op),
        OP(I64_STORE16) => handle_store(stack, emitter, imm, &handler_lookup::I64_STORE16, &handler_lookup::I64_STORE16_MM, op),
        OP(I64_STORE32) => handle_store(stack, emitter, imm, &handler_lookup::I64_STORE32, &handler_lookup::I64_STORE32_MM, op),

        // =====================================================================
        // Memory Size/Grow
        // =====================================================================
        OP(MEMORY_SIZE) => {
            if let Immediate::MemoryIndex(mem_idx) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::MEMORY_SIZE[idx];
                emitter.emit_memory_size(handler, *mem_idx);
                stack.push();
            }
        }
        OP(MEMORY_GROW) => {
            if let Immediate::MemoryIndex(mem_idx) = imm {
                emit_fill_for_operands(stack, emitter, 1);
                let idx = pre_op_variant_idx(stack);
                let handler = handler_lookup::MEMORY_GROW[idx];
                stack.pop();
                emitter.emit_memory_grow(handler, *mem_idx);
                stack.push();
            }
        }

        // =====================================================================
        // Bulk Memory
        // =====================================================================
        FC(OpcodeFC::MEMORY_FILL) => handle_ternary(stack, emitter, imm, &handler_lookup::MEMORY_FILL, op),
        FC(OpcodeFC::MEMORY_COPY) => handle_ternary(stack, emitter, imm, &handler_lookup::MEMORY_COPY, op),
        FC(OpcodeFC::MEMORY_INIT) => handle_ternary(stack, emitter, imm, &handler_lookup::MEMORY_INIT, op),
        FC(OpcodeFC::DATA_DROP) => {
            if let Immediate::DataIndex(data_idx) = imm {
                emitter.emit_data_drop(op_data_drop, *data_idx, op);
            }
        }

        // =====================================================================
        // Globals
        // =====================================================================
        OP(GLOBAL_GET) => {
            if let Immediate::GlobalIndex(global_idx) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::GLOBAL_GET[idx];
                emitter.emit_global_get(handler, *global_idx);
                stack.push();
            }
        }
        OP(GLOBAL_SET) => {
            if let Immediate::GlobalIndex(global_idx) = imm {
                emit_fill_for_operands(stack, emitter, 1);
                let idx = pre_op_variant_idx(stack);
                let handler = handler_lookup::GLOBAL_SET[idx];
                stack.pop();
                emitter.emit_global_set(handler, *global_idx);
            }
        }

        // =====================================================================
        // Tables
        // =====================================================================
        OP(TABLE_GET) => {
            if let Immediate::TableIndex(table_idx) = imm {
                emit_fill_for_operands(stack, emitter, 1);
                let idx = pre_op_variant_idx(stack);
                let handler = handler_lookup::TABLE_GET[idx];
                stack.pop();
                emitter.emit_table_get(handler, *table_idx, op);
                stack.push();
            }
        }
        OP(TABLE_SET) => {
            if let Immediate::TableIndex(table_idx) = imm {
                emit_fill_for_operands(stack, emitter, 2);
                let idx = pre_op_variant_idx(stack);
                let handler = handler_lookup::TABLE_SET[idx];
                stack.pop();
                stack.pop();
                emitter.emit_table_set(handler, *table_idx, op);
            }
        }
        FC(OpcodeFC::TABLE_SIZE) => {
            if let Immediate::TableIndex(table_idx) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::TABLE_SIZE[idx];
                emitter.emit_table_size(handler, *table_idx, op);
                stack.push();
            }
        }
        FC(OpcodeFC::TABLE_GROW) => {
            if let Immediate::TableIndex(table_idx) = imm {
                emit_fill_for_operands(stack, emitter, 2);
                let idx = pre_op_variant_idx(stack);
                let handler = handler_lookup::TABLE_GROW[idx];
                stack.pop();
                stack.pop();
                emitter.emit_table_grow(handler, *table_idx, op);
                stack.push();
            }
        }
        FC(OpcodeFC::TABLE_FILL) => handle_ternary(stack, emitter, imm, &handler_lookup::TABLE_FILL, op),
        FC(OpcodeFC::TABLE_COPY) => handle_ternary(stack, emitter, imm, &handler_lookup::TABLE_COPY, op),
        FC(OpcodeFC::TABLE_INIT) => handle_ternary(stack, emitter, imm, &handler_lookup::TABLE_INIT, op),
        FC(OpcodeFC::ELEM_DROP) => {
            if let Immediate::ElementIndex(elem_idx) = imm {
                emitter.emit_elem_drop(op_elem_drop, *elem_idx, op);
            }
        }

        // =====================================================================
        // References
        // =====================================================================
        OP(REF_NULL) => {
            emit_spill_if_needed(stack, emitter);
            let idx = post_op_variant_idx(stack);
            let handler = handler_lookup::REF_NULL[idx];
            emitter.emit_ref_null(handler, op);
            stack.push();
        }
        OP(REF_IS_NULL) => handle_unop(stack, emitter, &handler_lookup::REF_IS_NULL, op),
        OP(REF_FUNC) => {
            if let Immediate::FunctionIndex(func_idx) = imm {
                emit_spill_if_needed(stack, emitter);
                let idx = post_op_variant_idx(stack);
                let handler = handler_lookup::REF_FUNC[idx];
                emitter.emit_ref_func(handler, *func_idx, op);
                stack.push();
            }
        }

        // =====================================================================
        // Stack Ops (SP-based)
        // =====================================================================
        OP(DROP) => {
            emit_fill_for_operands(stack, emitter, 1);
            let idx = pre_op_variant_idx(stack);
            let handler = handler_lookup::DROP[idx];
            stack.pop();
            emitter.emit_drop_sp(handler);
        }
        OP(SELECT) | OP(SELECT_T) => {
            emit_fill_for_operands(stack, emitter, 3);
            let idx = pre_op_variant_idx(stack);
            let handler = handler_lookup::SELECT[idx];
            stack.pop(); // cond
            stack.pop(); // val2
            stack.pop(); // val1
            emitter.emit_select_sp(handler, op);
            stack.push();
        }

        // =====================================================================
        // Control Flow: Block/Loop/If
        // =====================================================================
        OP(BLOCK) => {
            let (params, results) = ctx.resolve_block_type_from_imm(imm);
            let idx = emitter.emit_block();
            stack.enter_block(BlockKind::Block, params, results, idx);
        }
        OP(LOOP) => {
            let (params, results) = ctx.resolve_block_type_from_imm(imm);

            emit_spill_all_for_call(stack, emitter);

            emitter.emit_loop();

            let loop_target = emitter.current_index();

            emit_fill_if_needed(stack, emitter);

            stack.enter_block(BlockKind::Loop, params, results, loop_target);
        }
        OP(IF) => {
            emit_fill_for_operands(stack, emitter, 1);
            emit_spill_all_except_top(stack, emitter, 1);
            let variant_idx = pre_op_variant_idx(stack);
            let handler = handler_lookup::IF_[variant_idx];
            stack.pop();
            let (params, results) = ctx.resolve_block_type_from_imm(imm);
            let idx = emitter.emit_if_variant(handler);
            stack.enter_block(BlockKind::If, params, results, idx);
            stack.set_if_inst(idx);
        }
        OP(ELSE) => {
            emit_spill_all_for_call(stack, emitter);

            let idx = emitter.emit_else();
            stack.set_else_inst(idx);
            stack.enter_else();

            let else_body_start = emitter.current_index();

            if let Some(frame) = stack.current_frame() {
                if let Some(if_idx) = frame.if_inst_idx {
                    emitter.patch_alt(if_idx, else_body_start);
                }
            }
        }
        OP(END) => {
            let has_forward_branches = stack.current_frame()
                .map(|f| !f.pending_fixups.is_empty())
                .unwrap_or(false);

            let has_stack_drop_branch = stack.current_frame()
                .map(|f| f.pending_fixups.iter().any(|fix| fix.stack_offset > 0))
                .unwrap_or(false);

            let is_if_block = stack.current_frame()
                .map(|f| f.kind == BlockKind::If)
                .unwrap_or(false);

            let needs_sync = has_stack_drop_branch && stack.spill_depth() < stack.height();

            if needs_sync {
                emit_spill_all_for_call(stack, emitter);
            }

            let needs_fill = has_forward_branches || is_if_block;

            let target_idx = emitter.current_index();

            if needs_fill {
                emit_fill_if_needed(stack, emitter);
            }

            emitter.emit_end();

            if let Some((frame, _can_preserve)) = stack.exit_block() {
                for fixup in &frame.pending_fixups {
                    if let Some(entry_idx) = fixup.br_table_entry {
                        emitter.patch_br_table_target(fixup.inst_idx, entry_idx, target_idx);
                    } else {
                        emitter.patch_alt(fixup.inst_idx, target_idx);
                        emitter.patch_br_data(fixup.inst_idx, fixup.stack_offset as u64, fixup.arity as u64);
                    }
                }
                if frame.kind == BlockKind::If && frame.else_inst_idx.is_none() {
                    if let Some(if_idx) = frame.if_inst_idx {
                        emitter.patch_alt(if_idx, target_idx);
                    }
                }
                if let Some(else_idx) = frame.else_inst_idx {
                    emitter.patch_alt(else_idx, target_idx);
                }
            }
        }

        // =====================================================================
        // Control Flow: Branches
        // =====================================================================
        OP(BR) => {
            if let Immediate::LabelIndex(label) = imm {
                emit_spill_all_for_call(stack, emitter);

                let arity = stack.branch_arity(*label);

                let (stack_offset, target) = stack.branch_info(*label);
                let current_height = stack.height();
                let operand_base_offset = (stack.operand_base() * 8) as u32;
                let idx = emitter.emit_br(stack_offset, arity, current_height, operand_base_offset);

                if let Some(tgt) = target {
                    emitter.patch_alt(idx, tgt);
                } else {
                    stack.register_forward_branch(*label, idx, None);
                }
                stack.set_unreachable();
            }
        }
        OP(BR_IF) => {
            if let Immediate::LabelIndex(label) = imm {
                emit_fill_for_operands(stack, emitter, 1);
                let tos_count_before_spill = stack.tos_count();
                emit_spill_all_except_top(stack, emitter, 1);
                let variant_idx = (stack.height().saturating_sub(1) % TOS_REGISTER_COUNT) as usize;

                let pre_pop_height = stack.height();

                stack.pop();
                let arity = stack.branch_arity(*label);

                let (stack_offset, target) = stack.branch_info(*label);

                let idx = if arity == 0 && stack_offset == 0 {
                    let handler = handler_lookup::BR_IF_SIMPLE[variant_idx];
                    emitter.emit_br_if_simple(handler)
                } else {
                    let handler = handler_lookup::BR_IF[variant_idx];
                    let operand_base_offset = (stack.operand_base() * 8) as u32;
                    emitter.emit_br_if_variant(handler, stack_offset, arity, pre_pop_height, operand_base_offset)
                };

                stack.record_fill(tos_count_before_spill.saturating_sub(1));

                if let Some(tgt) = target {
                    emitter.patch_alt(idx, tgt);
                } else {
                    stack.register_forward_branch(*label, idx, None);
                }
            }
        }
        OP(BR_TABLE) => {
            if let Immediate::BrLabels(labels, default_label) = imm {
                emit_fill_for_operands(stack, emitter, 1);
                emit_spill_all_except_top(stack, emitter, 1);
                let variant_idx = pre_op_variant_idx(stack);
                let handler = handler_lookup::BR_TABLE[variant_idx];

                let pre_pop_height = stack.height();

                stack.pop();

                let effective_height = pre_pop_height;

                let mut entries = Vec::with_capacity(labels.len() + 1);
                let all_labels: Vec<u32> = labels.iter().copied().chain(core::iter::once(*default_label)).collect();

                for &label in &all_labels {
                    let (stack_offset, target) = stack.branch_info(label);
                    let arity = stack.branch_arity(label);
                    entries.push(BrTableEntry {
                        target_idx: target,
                        stack_offset,
                        arity,
                    });
                }

                let operand_base_offset = (stack.operand_base() * 8) as u32;
                let inst_idx = emitter.emit_br_table_variant(handler, entries, effective_height, operand_base_offset);

                for (entry_idx, &label) in all_labels.iter().enumerate() {
                    let (_, target) = stack.branch_info(label);
                    if target.is_none() {
                        stack.register_forward_branch(label, inst_idx, Some(entry_idx));
                    }
                }

                stack.set_unreachable();
            }
        }
        OP(RETURN) => {
            emit_spill_all_for_call(stack, emitter);

            let arity = ctx.results_count();
            let frame_size = stack.frame_size();
            let current_height = stack.height();
            emitter.emit_return(arity, frame_size, current_height);
            stack.set_unreachable();
        }
        OP(UNREACHABLE) => {
            emitter.emit_unreachable();
            stack.set_unreachable();
        }

        // =====================================================================
        // Calls - All use precomputed delta
        // =====================================================================
        OP(CALL) => {
            if let Immediate::FunctionIndex(func_idx) = imm {
                emit_spill_all_for_call(stack, emitter);

                let (params, results) = ctx.resolve_func_type(*func_idx as usize);

                let delta = OPERAND_BASE + stack.height() - params;

                if ctx.is_func_internal(*func_idx as usize) {
                    if let Some(callee_inst) = ctx.get_func_inst(*func_idx as usize) {
                        emitter.emit_call_internal(callee_inst as *const _ as u64, delta);
                    } else {
                        emitter.emit_call_external(*func_idx, delta);
                    }
                } else {
                    emitter.emit_call_external(*func_idx, delta);
                }
                stack.apply_call(params, results);
                stack.reset_tos_state();
            }
        }
        OP(CALL_INDIRECT) => {
            if let Immediate::CallIndirectArgs { typeidx, tableidx } = imm {
                emit_spill_all_for_call(stack, emitter);

                let (params, results) = ctx.resolve_type_index(*typeidx as usize);

                let delta = OPERAND_BASE + stack.height() - params - 1;
                let operand_base_offset = (stack.operand_base() * 8) as u32;
                let height = stack.height() as u16;
                stack.pop(); // Pop the materialized index
                emitter.emit_call_indirect(*typeidx, *tableidx, delta, operand_base_offset, height);
                stack.apply_call(params, results);
                stack.reset_tos_state();
            }
        }

        // =====================================================================
        // NOP
        // =====================================================================
        OP(NOP) => {
            emitter.emit_nop();
        }

        // Catch-all for unhandled opcodes
        _ => {
            emit_fill_if_needed(stack, emitter);
            emitter.emit_nop();
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn handle_binop(
    stack: &mut StackTracker,
    emitter: &mut CodeEmitter,
    variants: &[Handler],
    op: WasmOpcode,
) {
    emit_fill_for_operands(stack, emitter, 2);

    let idx = pre_op_variant_idx(stack);
    let handler = variants[idx];

    stack.pop();
    stack.pop();
    emitter.emit_binop_sp(handler, op);
    stack.push();
}

fn handle_unop(
    stack: &mut StackTracker,
    emitter: &mut CodeEmitter,
    variants: &[Handler],
    op: WasmOpcode,
) {
    emit_fill_for_operands(stack, emitter, 1);

    let idx = pre_op_variant_idx(stack);
    let handler = variants[idx];

    stack.pop();
    emitter.emit_unop_sp(handler, op);
    stack.push();
}

/// Handle load operations (SP-based), selecting fast (C) or slow (Rust) handler based on mem_idx.
fn handle_load(
    stack: &mut StackTracker,
    emitter: &mut CodeEmitter,
    imm: &Immediate,
    fast_variants: &[Handler],
    slow_variants: &[Handler],
    op: WasmOpcode,
) {
    if let Immediate::MemArg { memidx, offset, .. } = imm {
        emit_fill_for_operands(stack, emitter, 1);
        let idx = pre_op_variant_idx(stack);
        let handler = if *memidx == 0 { fast_variants[idx] } else { slow_variants[idx] };

        stack.pop();
        emitter.emit_load(handler, *memidx, *offset as u32, op);
        stack.push();
    }
}

/// Handle store operations (SP-based), selecting fast (C) or slow (Rust) handler based on mem_idx.
fn handle_store(
    stack: &mut StackTracker,
    emitter: &mut CodeEmitter,
    imm: &Immediate,
    fast_variants: &[Handler],
    slow_variants: &[Handler],
    op: WasmOpcode,
) {
    if let Immediate::MemArg { memidx, offset, .. } = imm {
        emit_fill_for_operands(stack, emitter, 2);
        let idx = pre_op_variant_idx(stack);
        let handler = if *memidx == 0 { fast_variants[idx] } else { slow_variants[idx] };

        stack.pop();
        stack.pop();
        emitter.emit_store(handler, *memidx, *offset as u32, op);
    }
}

/// Handle ternary operations (memory.fill, memory.copy, table.fill, etc.)
fn handle_ternary(stack: &mut StackTracker, emitter: &mut CodeEmitter, imm: &Immediate, variants: &[Handler], op: WasmOpcode) {
    emit_fill_for_operands(stack, emitter, 3);
    let idx = pre_op_variant_idx(stack);
    let handler = variants[idx];

    stack.pop();
    stack.pop();
    stack.pop();
    let (imm0, imm1) = extract_imm01(imm);
    emitter.emit_ternary(handler, imm0 as u64, imm1 as u64, op);
}

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
