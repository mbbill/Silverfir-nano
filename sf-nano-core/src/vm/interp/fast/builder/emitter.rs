//! Code emitter: instruction emission.
//!
//! Emitter creates TempInst with logical PatternData values.
//! NO encoding happens here - encoding is done at finalization time.
//!
//! SP-based model: All operations use the stack pointer (sp) for operands.
//! No slot indices are needed - handlers read from sp[-n] and write to sp[0].

use alloc::vec::Vec;

use super::super::TOS_REGISTER_COUNT;
use super::super::handler_lookup;
use super::temp_inst::{BrTableEntry, Handler, TempInst};
use crate::opcodes::WasmOpcode;
use crate::vm::interp::fast::encoding::PatternData;
use crate::vm::interp::fast::frame_layout;
use crate::vm::interp::fast::handlers::full_set::*;
use crate::opcodes::Opcode::*;

/// Emitter for building instructions.
pub struct CodeEmitter {
    temps: Vec<TempInst>,
    /// Current TOS height, tracked for micro-JIT group compilation.
    #[cfg(feature = "micro-jit")]
    current_height: u16,
}

impl CodeEmitter {
    pub fn new() -> Self {
        Self {
            temps: Vec::with_capacity(256),
            #[cfg(feature = "micro-jit")]
            current_height: 0,
        }
    }

    /// Set the current TOS height (called before each opcode dispatch).
    #[cfg(feature = "micro-jit")]
    pub fn set_current_height(&mut self, h: u16) {
        self.current_height = h;
    }

    /// Current instruction index.
    #[inline]
    pub fn current_index(&self) -> usize {
        self.temps.len()
    }

    /// Take temps for finalization.
    pub fn take_temps(&mut self) -> Vec<TempInst> {
        core::mem::take(&mut self.temps)
    }

    // =========================================================================
    // Core Emission Helper
    // =========================================================================

    /// Emit a TempInst with automatic fallthrough setup.
    pub(super) fn emit(&mut self, mut inst: TempInst) -> usize {
        let idx = self.temps.len();
        inst.fallthrough_idx = Some(idx + 1);
        #[cfg(feature = "micro-jit")]
        { inst.pre_height = self.current_height; }
        self.temps.push(inst);
        idx
    }

    /// Emit a TempInst without fallthrough (for terminal instructions).
    fn emit_terminal(&mut self, mut inst: TempInst) -> usize {
        let idx = self.temps.len();
        #[cfg(feature = "micro-jit")]
        { inst.pre_height = self.current_height; }
        self.temps.push(inst);
        idx
    }


    // =========================================================================
    // SP-based Local Operations
    // =========================================================================

    /// Emit LOCAL_GET: push fp[idx] onto operand stack.
    /// SP-based: *sp++ = fp[idx]
    pub fn emit_local_get(&mut self, idx: u32) -> usize {
        self.emit(TempInst::new(
            op_local_get_D1,
            PatternData::LocalGet { idx: idx as u16 },
            WasmOpcode::OP(LOCAL_GET),
        ))
    }

    /// Emit LOCAL_GET with specific variant handler.
    pub fn emit_local_get_variant(&mut self, handler: Handler, idx: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::LocalGet { idx: idx as u16 },
            WasmOpcode::OP(LOCAL_GET),
        ))
    }

    /// Emit LOCAL_SET: pop operand stack to local.
    /// SP-based: fp[idx] = *--sp
    pub fn emit_local_set(&mut self, idx: u32) -> usize {
        self.emit(TempInst::new(
            op_local_set_D1,
            PatternData::LocalSet { idx: idx as u16 },
            WasmOpcode::OP(LOCAL_SET),
        ))
    }

    /// Emit LOCAL_SET with specific variant handler.
    pub fn emit_local_set_variant(&mut self, handler: Handler, idx: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::LocalSet { idx: idx as u16 },
            WasmOpcode::OP(LOCAL_SET),
        ))
    }

    /// Emit LOCAL_TEE: copy top of operand stack to local (no pop).
    /// SP-based: fp[idx] = sp[-1]
    pub fn emit_local_tee(&mut self, idx: u32) -> usize {
        self.emit(TempInst::new(
            op_local_tee_D1,
            PatternData::LocalTee { idx: idx as u16 },
            WasmOpcode::OP(LOCAL_TEE),
        ))
    }

    /// Emit LOCAL_TEE with specific variant handler.
    pub fn emit_local_tee_variant(&mut self, handler: Handler, idx: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::LocalTee { idx: idx as u16 },
            WasmOpcode::OP(LOCAL_TEE),
        ))
    }

    // =========================================================================
    // L0 Local Register Cache
    // =========================================================================

    /// Emit INIT_LOCALS: combined prologue to swap+fill all 3 hot locals in one dispatch.
    pub fn emit_init_locals(&mut self, k0: u32, k1: u32, k2: u32) -> usize {
        self.emit(TempInst::new(
            op_init_locals,
            PatternData::InitLocals {
                hot_local_idx_0: k0 as u16,
                hot_local_idx_1: k1 as u16,
                hot_local_idx_2: k2 as u16,
            },
            WasmOpcode::OP(NOP),
        ))
    }


    // =========================================================================
    // SP-based Arithmetic (handlers use sp[-1], sp[-2])
    // =========================================================================

    /// Emit SP-based binary operation: sp[-2] = op(sp[-2], sp[-1]); sp--
    /// Handler reads operands from sp. No encoding needed (stack-only).
    pub fn emit_binop_sp(&mut self, handler: Handler, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 },
            wasm_op,
        ))
    }

    /// Emit SP-based unary operation: sp[-1] = op(sp[-1])
    /// Handler reads operand from sp. No encoding needed (stack-only).
    pub fn emit_unop_sp(&mut self, handler: Handler, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 },
            wasm_op,
        ))
    }

    /// Emit SP-based constant push: *sp++ = value
    pub fn emit_const_sp(&mut self, handler: Handler, value: u64, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Const { value },
            wasm_op,
        ))
    }

    /// Emit SP-based select: sp[-3] = cond ? sp[-3] : sp[-2]; sp -= 2
    /// Handler reads 3 values from sp, no encoding needed (stack-only).
    pub fn emit_select_sp(&mut self, handler: Handler, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 },
            wasm_op,
        ))
    }

    /// Emit SP-based drop: sp--
    /// Accepts handler for D1-D4 variant selection.
    pub fn emit_drop_sp(&mut self, handler: Handler) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 },
            WasmOpcode::OP(DROP),
        ))
    }


    // =========================================================================
    // Memory Operations (SP-based)
    // =========================================================================

    /// Emit SP-based load: sp[-1] = mem[sp[-1] + offset]
    /// Address is popped from stack, result pushed (1→1, sp unchanged)
    pub fn emit_load(&mut self, handler: Handler, memidx: u32, offset: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Load { offset, memidx },
            wasm_op,
        ))
    }

    /// Emit SP-based store: mem[sp[-2] + offset] = sp[-1]; sp -= 2
    /// Address and value popped from stack (2→0)
    pub fn emit_store(&mut self, handler: Handler, memidx: u32, offset: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Store { offset, memidx },
            wasm_op,
        ))
    }

    // =========================================================================
    // Control Flow
    // =========================================================================

    /// Emit a NOP (will be removed by finalizer).
    pub fn emit_nop(&mut self) -> usize {
        self.emit(TempInst::new(op_nop, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, WasmOpcode::OP(NOP)))
    }

    /// Emit BLOCK (will be removed by finalizer).
    pub fn emit_block(&mut self) -> usize {
        self.emit(TempInst::new(op_block, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, WasmOpcode::OP(BLOCK)))
    }

    /// Emit LOOP (will be removed by finalizer).
    pub fn emit_loop(&mut self) -> usize {
        self.emit(TempInst::new(op_loop, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, WasmOpcode::OP(LOOP)))
    }

    /// Emit END (will be removed by finalizer).
    pub fn emit_end(&mut self) -> usize {
        self.emit(TempInst::new(op_end, PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 }, WasmOpcode::OP(END)))
    }

    /// Emit IF: condition read from TOS register (passed via variant).
    /// Jumps to alt_idx if zero.
    pub fn emit_if_variant(&mut self, handler: Handler) -> usize {
        // Note: After rebuild, PatternData::If will be a unit variant (only target).
        // Use Else for now since it has the same encoding structure.
        self.emit(TempInst::new(
            handler,
            PatternData::Else,
            WasmOpcode::OP(IF),
        ).with_target())
    }

    /// Emit ELSE: unconditional jump to alt_idx.
    pub fn emit_else(&mut self) -> usize {
        self.emit(TempInst::new(
            op_else_,
            PatternData::Else,
            WasmOpcode::OP(ELSE),
        ).with_target())
    }

    /// Emit BR: unconditional branch to alt_idx.
    /// `operand_base_offset` is (frame_size + METADATA_SLOTS) * 8 (bytes from fp to operand stack).
    pub fn emit_br(&mut self, stack_offset: usize, arity: usize, current_height: usize, operand_base_offset: u32) -> usize {
        self.emit(TempInst::new(
            op_br,
            PatternData::Br {
                stack_drop: stack_offset as u64,
                arity: arity as u16,
                height: current_height as u16,
                operand_base_offset,
            },
            WasmOpcode::OP(BR),
        ).with_target())
    }

    /// Emit BR_IF: condition read from TOS register (passed via variant).
    /// Conditional branch to alt_idx if nonzero.
    pub fn emit_br_if_variant(&mut self, handler: Handler, stack_offset: usize, arity: usize, effective_height: usize, operand_base_offset: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::BrIf {
                stack_drop: stack_offset as u64,
                arity: arity as u16,
                height: effective_height as u16,
                operand_base_offset,
            },
            WasmOpcode::OP(BR_IF),
        ).with_target())
    }

    /// Emit BR_IF_SIMPLE: specialized for arity=0, stack_drop=0 (common loop back-edges).
    /// Only encodes the branch target — no fixup overhead.
    pub fn emit_br_if_simple(&mut self, handler: Handler) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::BrIfSimple {},
            WasmOpcode::OP(BR_IF),
        ).with_target())
    }

    /// Emit BR_TABLE: index read from TOS register (passed via variant).
    /// Dispatches based on index value.
    pub fn emit_br_table_variant(&mut self, handler: Handler, entries: Vec<BrTableEntry>, effective_height: usize, operand_base_offset: u32) -> usize {
        let idx = self.temps.len();
        let inst = TempInst::new(
            handler,
            PatternData::BrTable {
                entry_count: 0,         // Filled by finalizer
                data_slot_count: 0,     // Filled by finalizer
                height: effective_height as u16,
                operand_base_offset,
            },
            WasmOpcode::OP(BR_TABLE),
        ).with_br_table_entries(entries).with_target();
        self.temps.push(inst);
        idx
    }

    /// Emit RETURN (specialized by arity for optimal codegen).
    /// `arity` is the number of return values.
    /// `frame_size` is params_count + locals_count (to find metadata on unified stack).
    pub fn emit_return(&mut self, arity: usize, frame_size: usize, current_height: usize) -> usize {
        let operand_base_offset = frame_layout::operand_base_offset(frame_size) as u32;
        match arity {
            0 => self.emit_terminal(TempInst::new(
                op_return_void,
                PatternData::ReturnVoid {
                    frame_size: frame_size as u16,
                },
                WasmOpcode::OP(RETURN),
            )),
            1 => self.emit_terminal(TempInst::new(
                op_return_one,
                PatternData::ReturnOne {
                    frame_size: frame_size as u16,
                    operand_base_offset,
                    height: current_height as u16,
                },
                WasmOpcode::OP(RETURN),
            )),
            _ => self.emit_terminal(TempInst::new(
                op_return,
                PatternData::Return {
                    arity: arity as u16,
                    frame_size: frame_size as u16,
                    operand_base_offset,
                    height: current_height as u16,
                },
                WasmOpcode::OP(RETURN),
            )),
        }
    }

    /// Emit UNREACHABLE.
    pub fn emit_unreachable(&mut self) -> usize {
        self.emit_terminal(TempInst::new(
            op_unreachable,
            PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 },
            WasmOpcode::OP(UNREACHABLE),
        ))
    }

    // =========================================================================
    // Calls
    // =========================================================================

    /// Emit CALL_EXTERNAL (external functions: imports, WASI, host functions).
    ///
    /// `delta` is the precomputed frame offset where callee args start.
    pub fn emit_call_external(&mut self, func_idx: u32, delta: usize) -> usize {
        self.emit(TempInst::new(
            op_call_external,
            PatternData::CallExternal {
                func_idx: func_idx as u64,
                delta: delta as u16,
            },
            WasmOpcode::OP(CALL),
        ))
    }

    /// Emit CALL_INTERNAL (internal call by callee FunctionInst pointer).
    ///
    /// `delta` is the precomputed frame offset where callee frame starts.
    /// Patched to CALL_LOCAL for same-module calls after compilation.
    pub fn emit_call_internal(&mut self, callee_func: u64, delta: usize) -> usize {
        self.emit(TempInst::new(
            op_call_internal,
            PatternData::CallInternal {
                callee_func,
                delta: delta as u16,
            },
            WasmOpcode::OP(CALL),
        ))
    }

    // NOTE: call_local is NOT emitted during compilation.
    // It is created by patching call_internal in precompile.rs.
    // The patching adds params_count and locals_count at runtime.

    /// Emit CALL_INDIRECT.
    ///
    /// `delta` is the precomputed frame offset where callee args start.
    pub fn emit_call_indirect(&mut self, type_idx: u32, table_idx: u32, delta: usize, operand_base_offset: u32, height: u16) -> usize {
        self.emit(TempInst::new(
            op_call_indirect,
            PatternData::CallIndirect {
                type_idx: type_idx as u64,
                table_idx: table_idx as u64,
                delta: delta as u16,
                operand_base_offset,
                height,
            },
            WasmOpcode::OP(CALL_INDIRECT),
        ))
    }

    // =========================================================================
    // Globals (SP-based)
    // =========================================================================

    /// Emit GLOBAL_GET: push global value onto stack.
    /// SP-based: *sp++ = globals[idx]
    /// Handler variant based on post-op stack depth.
    pub fn emit_global_get(&mut self, handler: Handler, global_idx: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Global {
                global_idx: global_idx as u64,
            },
            WasmOpcode::OP(GLOBAL_GET),
        ))
    }

    /// Emit GLOBAL_SET: pop value from stack to global.
    /// SP-based: globals[idx] = *--sp
    /// Handler variant based on pre-op stack depth.
    pub fn emit_global_set(&mut self, handler: Handler, global_idx: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Global {
                global_idx: global_idx as u64,
            },
            WasmOpcode::OP(GLOBAL_SET),
        ))
    }

    // =========================================================================
    // Memory Size/Grow (SP-based)
    // =========================================================================

    /// Emit MEMORY_SIZE: push memory size onto stack.
    /// SP-based: *sp++ = memory.size()
    /// Handler variant based on post-op stack depth.
    pub fn emit_memory_size(&mut self, handler: Handler, mem_idx: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::MemorySize {
                mem_idx: mem_idx as u64,
            },
            WasmOpcode::OP(MEMORY_SIZE),
        ))
    }

    /// Emit MEMORY_GROW: pop pages, grow memory, push result.
    /// SP-based: pages = *--sp; *sp++ = memory.grow(pages)
    pub fn emit_memory_grow(&mut self, handler: Handler, mem_idx: u32) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::MemoryGrow {
                mem_idx: mem_idx as u64,
            },
            WasmOpcode::OP(MEMORY_GROW),
        ))
    }

    // =========================================================================
    // Drop / Data / Element (SP-based)
    // =========================================================================

    /// Emit DATA_DROP.
    pub fn emit_data_drop(&mut self, handler: Handler, data_idx: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::DropOp {
                idx: data_idx as u64,
            },
            wasm_op,
        ))
    }

    /// Emit ELEM_DROP.
    pub fn emit_elem_drop(&mut self, handler: Handler, elem_idx: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::DropOp {
                idx: elem_idx as u64,
            },
            wasm_op,
        ))
    }

    // =========================================================================
    // Tables (SP-based)
    // =========================================================================

    /// Emit TABLE_GET: pop index, push value.
    /// SP-based: idx = *--sp; *sp++ = table[idx]
    pub fn emit_table_get(&mut self, handler: Handler, table_idx: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::TableGet {
                table_idx: table_idx as u64,
            },
            wasm_op,
        ))
    }

    /// Emit TABLE_SET: pop value, pop index, set table.
    /// SP-based: val = *--sp; idx = *--sp; table[idx] = val
    pub fn emit_table_set(&mut self, handler: Handler, table_idx: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::TableSet {
                table_idx: table_idx as u64,
            },
            wasm_op,
        ))
    }

    /// Emit TABLE_SIZE: push size.
    /// SP-based: *sp++ = table.size()
    pub fn emit_table_size(&mut self, handler: Handler, table_idx: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::TableSize {
                table_idx: table_idx as u64,
            },
            wasm_op,
        ))
    }

    /// Emit TABLE_GROW: pop delta, pop init, grow, push result.
    /// SP-based: delta = *--sp; init = *--sp; *sp++ = table.grow(init, delta)
    pub fn emit_table_grow(&mut self, handler: Handler, table_idx: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::TableGrow {
                table_idx: table_idx as u64,
            },
            wasm_op,
        ))
    }

    // =========================================================================
    // References (SP-based)
    // =========================================================================

    /// Emit REF_NULL: push null reference.
    /// SP-based: *sp++ = null. No encoding needed (stack-only).
    pub fn emit_ref_null(&mut self, handler: Handler, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Raw { imm0: 0, imm1: 0, imm2: 0 },
            wasm_op,
        ))
    }

    /// Emit REF_FUNC: push function reference.
    /// SP-based: *sp++ = func_ref
    pub fn emit_ref_func(&mut self, handler: Handler, func_idx: u32, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::RefFunc {
                func_idx: func_idx as u64,
            },
            wasm_op,
        ))
    }

    // =========================================================================
    // Ternary Operations (memory.copy, table.copy, etc.) - SP-based
    // =========================================================================

    /// Emit a ternary operation: pop 3 values from stack.
    /// SP-based: c = *--sp; b = *--sp; a = *--sp; op(a, b, c)
    pub fn emit_ternary(&mut self, handler: Handler, imm0: u64, imm1: u64, wasm_op: WasmOpcode) -> usize {
        self.emit(TempInst::new(
            handler,
            PatternData::Ternary {
                imm0,
                imm1,
            },
            wasm_op,
        ))
    }

    // =========================================================================
    // Patching
    // =========================================================================

    /// Patch alt_idx for an instruction (for branch targets).
    pub fn patch_alt(&mut self, idx: usize, alt_idx: usize) {
        if let Some(inst) = self.temps.get_mut(idx) {
            inst.alt_idx = Some(alt_idx);
        }
    }

    /// Patch BR/BR_IF instruction data for forward branches.
    /// Called when the branch target becomes known.
    pub fn patch_br_data(&mut self, idx: usize, stack_offset: u64, arity: u64) {
        if let Some(inst) = self.temps.get_mut(idx) {
            match &mut inst.data {
                PatternData::Br { stack_drop, arity: a, .. } => {
                    *stack_drop = stack_offset;
                    *a = arity as u16;
                }
                PatternData::BrIf { stack_drop, arity: a, .. } => {
                    *stack_drop = stack_offset;
                    *a = arity as u16;
                }
                _ => {}
            }
        }
    }

    /// Patch br_table entry target.
    pub fn patch_br_table_target(&mut self, inst_idx: usize, entry_idx: usize, target_idx: usize) {
        if let Some(inst) = self.temps.get_mut(inst_idx) {
            if let Some(entries) = &mut inst.br_table_entries {
                if let Some(entry) = entries.get_mut(entry_idx) {
                    entry.target_idx = Some(target_idx);
                }
            }
        }
    }

    // =========================================================================
    // TOS Spill/Fill (Phase 3 - FP-Based Addressing)
    // =========================================================================

    /// Emit a spill instruction to write TOS values to operand stack.
    /// Uses fp-based addressing: operand_base = fp + operand_base_offset
    ///
    /// Emit a spill instruction to write TOS to frame slots.
    ///
    /// - slot: frame slot index of top of operand stack (frame_size + height - 1)
    /// - count: number of TOS values to write (1-4)
    /// - variant: D1-D4 variant index (0-3), computed as (height - 1) % 4
    pub fn emit_spill(&mut self, slot: u16, count: u8, variant: u8) -> usize {
        debug_assert!(count >= 1 && count <= TOS_REGISTER_COUNT as u8, "count must be 1-TOS_REGISTER_COUNT");
        debug_assert!(variant <= (TOS_REGISTER_COUNT - 1) as u8, "variant must be 0-(TOS_REGISTER_COUNT-1)");

        // Select handler based on count, then index by variant
        let handlers = match count {
            1 => &handler_lookup::SPILL_1,
            2 => &handler_lookup::SPILL_2,
            3 => &handler_lookup::SPILL_3,
            4 => &handler_lookup::SPILL_4,
            _ => unreachable!(),
        };
        let handler = handlers[variant as usize];

        self.emit(TempInst::new(
            handler,
            PatternData::Spill1 { slot }, // All spill variants use same encoding
            WasmOpcode::OP(NOP),
        ))
    }

    /// Emit a fill instruction to read frame slots into TOS.
    ///
    /// - slot: frame slot index of top of operand stack (frame_size + height - 1)
    /// - count: number of TOS values to read (1-4)
    /// - variant: D1-D4 variant index (0-3), computed as (height - 1) % 4
    pub fn emit_fill(&mut self, slot: u16, count: u8, variant: u8) -> usize {
        debug_assert!(count >= 1 && count <= TOS_REGISTER_COUNT as u8, "count must be 1-TOS_REGISTER_COUNT");
        debug_assert!(variant <= (TOS_REGISTER_COUNT - 1) as u8, "variant must be 0-(TOS_REGISTER_COUNT-1)");

        // Select handler based on count, then index by variant
        let handlers = match count {
            1 => &handler_lookup::FILL_1,
            2 => &handler_lookup::FILL_2,
            3 => &handler_lookup::FILL_3,
            4 => &handler_lookup::FILL_4,
            _ => unreachable!(),
        };
        let handler = handlers[variant as usize];

        self.emit(TempInst::new(
            handler,
            PatternData::Fill1 { slot }, // All fill variants use same encoding
            WasmOpcode::OP(NOP),
        ))
    }
}

impl Default for CodeEmitter {
    fn default() -> Self {
        Self::new()
    }
}
