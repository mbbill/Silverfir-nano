//! Memory, global, and table lowering — loads, stores, bounds checks.

use crate::collections;
#[cfg(sf_has_simd)]
use crate::opcodes::OpcodeFD;

#[cfg(sf_has_simd)]
use crate::vm::jit::middle::ssa_ir::ir::DecodedOperand;
use crate::{
    error::WasmError,
    vm::{
        jit::machine::machine_ir::{
            machine_ptr_width, MachineAddr, MachineBlockId, MachineBranchCond, MachineCompareKind,
            MachineConvertOp, MachineEdge, MachineInst, MachineInstKind, MachineIntBinaryOp,
            MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner,
            MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
            MACHINE_MEM0_BASE_REG,
        },
        jit::middle::ssa_ir::ir::{SsaOperand, SsaValue},
        jit::wasm::primitive_op::PrimitiveOpKind,
        runtime::layout::native_runtime_abi_layout,
    },
};

use super::{
    lower_context::BlockLowerContext,
    lower_inst::LeafLowering,
    lower_regalloc::{canonical_value_mem_width_for_value, lir_value_storage_type},
    lower_util::{single_arg, single_result, three_args, two_args},
};

impl<'a> BlockLowerContext<'a> {
    #[inline]
    fn use_memory_index_word(
        &mut self,
        value: SsaValue,
    ) -> Result<(MachineReg, Option<MachineReg>), WasmError> {
        if self.gp_reg_width() == 4 {
            if let Some((_, Some(_))) = self.try_value_regs(value) {
                let (lo, hi) = self.use_i64_value_pair(value)?;
                return Ok((lo, Some(hi)));
            }
        }
        Ok((self.use_value(value)?, None))
    }

    #[inline]
    fn emit_memory_index_high_trap_if_nonzero(&mut self, hi: Option<MachineReg>) {
        let Some(hi) = hi else {
            return;
        };
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TrapIf {
                kind: MachineTrapKind::MemoryOutOfBounds,
                cond: MachineBranchCond::IntCompare {
                    width: self.gp_word_int_width(),
                    kind: MachineCompareKind::Ne,
                    sign: MachineSign::Unsigned,
                    lhs: MachineValue::Reg(hi),
                    rhs: MachineValue::Imm64(0),
                },
            },
        });
    }

    pub(super) fn lower_memory_size(
        &mut self,
        mem_idx: u32,
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let result = single_result(results)?;
        let pair_result = self.gp_reg_width() == 4
            && matches!(
                lir_value_storage_type(self.program(), result),
                MachineStorageType::GpI64
            );
        let (dst, dst_hi) = if pair_result {
            let (lo, hi) = self.alloc_i64_value_pair(result)?;
            (lo, Some(hi))
        } else {
            (self.alloc_result_value(result)?, None)
        };
        if mem_idx == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst,
                    src: MachineValue::Reg(self.mem0_size_reg()),
                },
            });
        } else {
            let runtime_layout = self.runtime_abi_layout();
            let temp = self.borrow_free_gp_dynamic_regs(1)?[0];
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: temp,
                    addr: self.runtime_addr(runtime_layout.context.memory_views_base_offset),
                    width: self.gp_word_mem_width(),
                    extension: MachineLoadExtension::None,
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst,
                    addr: self.indexed_addr(
                        temp,
                        mem_idx,
                        runtime_layout.pointer_len_view.stride as usize,
                        runtime_layout.pointer_len_view.len_offset,
                    )?,
                    width: self.gp_word_mem_width(),
                    extension: MachineLoadExtension::None,
                },
            });
        }
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::ShrU,
                dst,
                lhs: MachineValue::Reg(dst),
                rhs: MachineValue::Imm64(crate::constants::WASM_PAGE_SIZE.trailing_zeros() as u64),
            },
        });
        if let Some(dst_hi) = dst_hi {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    src: MachineValue::Imm64(0),
                },
            });
        }
        Ok(())
    }

    pub(super) fn lower_memory_grow(
        &mut self,
        mem_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let result = single_result(results)?;
        let pair_result = self.gp_reg_width() == 4
            && matches!(
                lir_value_storage_type(self.program(), result),
                MachineStorageType::GpI64
            );
        if pair_result {
            let delta_value = single_arg(args)?.unwrap_value();
            let (delta_lo, delta_hi) = self.use_memory_index_word(delta_value)?;
            let (dst_lo, dst_hi) =
                self.alloc_i64_value_pair_reusing_dead_inputs(result, &[delta_value])?;
            let delta = if delta_hi.is_some() {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Ne,
                        sign: MachineSign::Unsigned,
                        dst: dst_hi,
                        lhs: MachineValue::Reg(delta_hi.expect("checked above")),
                        rhs: MachineValue::Imm64(0),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Select {
                        ty: MachineStorageType::GpWord,
                        dst: dst_hi,
                        on_true: MachineValue::Imm64(u32::MAX as u64),
                        on_false: MachineValue::Reg(delta_lo),
                        cond: MachineValue::Reg(dst_hi),
                    },
                });
                MachineValue::Reg(dst_hi)
            } else {
                MachineValue::Reg(delta_lo)
            };
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::MemoryGrow {
                    mem_idx,
                    dst: dst_lo,
                    delta,
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntCompare {
                    width: self.gp_word_int_width(),
                    kind: MachineCompareKind::Eq,
                    sign: MachineSign::Unsigned,
                    dst: dst_hi,
                    lhs: MachineValue::Reg(dst_lo),
                    rhs: MachineValue::Imm64(u32::MAX as u64),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Select {
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    on_true: MachineValue::Imm64(u32::MAX as u64),
                    on_false: MachineValue::Imm64(0),
                    cond: MachineValue::Reg(dst_hi),
                },
            });
            self.emit_reload_mem0_cache_regs();
            return Ok(());
        }

        let delta = self.lower_operand(single_arg(args)?)?;
        let dst = self.alloc_result_value(result)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::MemoryGrow {
                mem_idx,
                dst,
                delta,
            },
        });
        self.emit_reload_mem0_cache_regs();
        Ok(())
    }

    pub(super) fn lower_memory_fill(
        &mut self,
        mem_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (dest, val, len) = three_args(args)?;
        let dest = self.lower_operand(dest)?;
        let val = self.lower_operand(val)?;
        let len = self.lower_operand(len)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::MemoryFill {
                mem_idx,
                dest,
                val,
                len,
            },
        });
        Ok(())
    }

    pub(super) fn lower_memory_copy(
        &mut self,
        dst_mem: u32,
        src_mem: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (dest, src, len) = three_args(args)?;
        let dest = self.lower_operand(dest)?;
        let src = self.lower_operand(src)?;
        let len = self.lower_operand(len)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::MemoryCopy {
                dst_mem,
                src_mem,
                dest,
                src,
                len,
            },
        });
        Ok(())
    }

    pub(super) fn lower_memory_init(
        &mut self,
        mem_idx: u32,
        data_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (dest, src, len) = three_args(args)?;
        let dest = self.lower_operand(dest)?;
        let src = self.lower_operand(src)?;
        let len = self.lower_operand(len)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::MemoryInit {
                mem_idx,
                data_idx,
                dest,
                src,
                len,
            },
        });
        Ok(())
    }

    pub(super) fn lower_data_drop(&mut self, data_idx: u32) -> Result<(), WasmError> {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::DataDrop { data_idx },
        });
        Ok(())
    }

    pub(super) fn lower_table_grow(
        &mut self,
        table_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let result = single_result(results)?;
        let pair_result = self.gp_reg_width() == 4
            && matches!(
                lir_value_storage_type(self.program(), result),
                MachineStorageType::GpI64
            );
        let (init_arg, delta_arg) = two_args(args)?;
        let init_val = self.lower_operand(init_arg)?;
        if pair_result {
            let delta_value = delta_arg.unwrap_value();
            let (delta_lo, delta_hi) = self.use_memory_index_word(delta_value)?;
            let (dst_lo, dst_hi) =
                self.alloc_i64_value_pair_reusing_dead_inputs(result, &[delta_value])?;
            let delta = if delta_hi.is_some() {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Ne,
                        sign: MachineSign::Unsigned,
                        dst: dst_hi,
                        lhs: MachineValue::Reg(delta_hi.expect("checked above")),
                        rhs: MachineValue::Imm64(0),
                    },
                });
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::Select {
                        ty: MachineStorageType::GpWord,
                        dst: dst_hi,
                        on_true: MachineValue::Imm64(u32::MAX as u64),
                        on_false: MachineValue::Reg(delta_lo),
                        cond: MachineValue::Reg(dst_hi),
                    },
                });
                MachineValue::Reg(dst_hi)
            } else {
                MachineValue::Reg(delta_lo)
            };
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TableGrow {
                    table_idx,
                    dst: dst_lo,
                    init_val,
                    delta,
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntCompare {
                    width: self.gp_word_int_width(),
                    kind: MachineCompareKind::Eq,
                    sign: MachineSign::Unsigned,
                    dst: dst_hi,
                    lhs: MachineValue::Reg(dst_lo),
                    rhs: MachineValue::Imm64(u32::MAX as u64),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Select {
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    on_true: MachineValue::Imm64(u32::MAX as u64),
                    on_false: MachineValue::Imm64(0),
                    cond: MachineValue::Reg(dst_hi),
                },
            });
            return Ok(());
        }

        let delta = self.lower_operand(delta_arg)?;
        let dst = self.alloc_result_value(result)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TableGrow {
                table_idx,
                dst,
                init_val,
                delta,
            },
        });
        Ok(())
    }

    pub(super) fn lower_table_fill(
        &mut self,
        table_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (start, val, len) = three_args(args)?;
        let start = self.lower_operand(start)?;
        let val = self.lower_operand(val)?;
        let len = self.lower_operand(len)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TableFill {
                table_idx,
                start,
                val,
                len,
            },
        });
        Ok(())
    }

    pub(super) fn lower_table_copy(
        &mut self,
        dst_tbl: u32,
        src_tbl: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (dest, src, len) = three_args(args)?;
        let dest = self.lower_operand(dest)?;
        let src = self.lower_operand(src)?;
        let len = self.lower_operand(len)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TableCopy {
                dst_tbl,
                src_tbl,
                dest,
                src,
                len,
            },
        });
        Ok(())
    }

    pub(super) fn lower_table_init(
        &mut self,
        table_idx: u32,
        elem_idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let (dest, src, len) = three_args(args)?;
        let dest = self.lower_operand(dest)?;
        let src = self.lower_operand(src)?;
        let len = self.lower_operand(len)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TableInit {
                table_idx,
                elem_idx,
                dest,
                src,
                len,
            },
        });
        Ok(())
    }

    pub(super) fn lower_elem_drop(&mut self, elem_idx: u32) -> Result<(), WasmError> {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::ElemDrop { elem_idx },
        });
        Ok(())
    }

    pub(super) fn lower_eh_alloc_exn_ref(
        &mut self,
        tag_idx: u32,
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let result = single_result(results)?;
        let dst = self.alloc_result_value(result)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::EhAllocExnRef { tag_idx, dst },
        });
        Ok(())
    }

    pub(super) fn lower_global_get(
        &mut self,
        idx: u32,
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let result = single_result(results)?;
        let ty = self.value_storage_type(result);
        #[cfg(sf_has_simd)]
        if matches!(ty, MachineStorageType::V128) {
            return self.lower_global_get_v128(idx, result);
        }
        if matches!(ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            return ops.emit_global_get_i64(self, idx, result);
        }
        self.lower_global_get_scalar(idx, result)
    }

    /// Scalar (non-i64-pair) global.get -- used by both Gp64Lowering and the
    /// non-i64 path above.
    ///
    /// Emits two loads: one indexed fetch of the precomputed raw_ptr from the
    /// context's inline tail, then a deref. No intermediate view-pointer load
    /// (the tail lives at a constant offset from `runtime_base`).
    pub(super) fn lower_global_get_scalar(
        &mut self,
        idx: u32,
        result: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = self.value_storage_type(result);
        let dst = self.alloc_result_value(result)?;
        let width = canonical_value_mem_width_for_value(self.program(), result);
        let base = self.borrow_free_gp_dynamic_regs(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: base,
                addr: self.globals_ptr_slot_addr(idx)?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty,
                dst,
                addr: MachineAddr { base, offset: 0 },
                width,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    #[cfg(sf_has_simd)]
    pub(super) fn lower_global_get_v128(
        &mut self,
        idx: u32,
        result: SsaValue,
    ) -> Result<(), WasmError> {
        self.ensure_v128_raw_abi()?;
        let dst = self.alloc_result_value(result)?;
        let base = self.borrow_free_gp_dynamic_regs(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: base,
                addr: self.globals_ptr_slot_addr(idx)?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: base,
                addr: MachineAddr { base, offset: 0 },
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_v128_from_raw_reg(dst, base, MachineRegOwner::LinearValue);
        Ok(())
    }

    /// 32-bit i64 pair global.get -- called from Gp32Lowering.
    pub(super) fn lower_global_get_i64_pair(
        &mut self,
        idx: u32,
        result: SsaValue,
    ) -> Result<(), WasmError> {
        let (dst_lo, dst_hi) = self.alloc_i64_value_pair(result)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: dst_hi,
                addr: self.globals_ptr_slot_addr(idx)?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        let lo_addr = MachineAddr {
            base: dst_hi,
            offset: 0,
        };
        let hi_addr = addr_with_byte_offset(lo_addr, 4)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: dst_lo,
                addr: lo_addr,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: dst_hi,
                addr: hi_addr,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    pub(super) fn lower_global_set(
        &mut self,
        idx: u32,
        args: &[SsaOperand],
    ) -> Result<(), WasmError> {
        let src_value = single_arg(args)?.unwrap_value();
        let ty = self.value_storage_type(src_value);
        #[cfg(sf_has_simd)]
        if matches!(ty, MachineStorageType::V128) {
            return self.lower_global_set_v128(idx, src_value);
        }
        if matches!(ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            return ops.emit_global_set_i64(self, idx, src_value);
        }
        self.lower_global_set_scalar(idx, src_value)
    }

    /// Scalar (non-i64-pair) global.set -- used by both Gp64Lowering and the
    /// non-i64 path above.
    pub(super) fn lower_global_set_scalar(
        &mut self,
        idx: u32,
        src_value: SsaValue,
    ) -> Result<(), WasmError> {
        let ty = self.value_storage_type(src_value);
        let src = self.use_value(src_value)?;
        let width = canonical_value_mem_width_for_value(self.program(), src_value);
        let base = self.borrow_free_gp_dynamic_regs(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: base,
                addr: self.globals_ptr_slot_addr(idx)?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty,
                addr: MachineAddr { base, offset: 0 },
                width,
                src: MachineValue::Reg(src),
            },
        });
        Ok(())
    }

    #[cfg(sf_has_simd)]
    pub(super) fn lower_global_set_v128(
        &mut self,
        idx: u32,
        src_value: SsaValue,
    ) -> Result<(), WasmError> {
        self.ensure_v128_raw_abi()?;
        let src = self.use_value(src_value)?;
        let temps = self.borrow_free_gp_dynamic_regs(2)?;
        let base = temps[0];
        let raw = temps[1];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: base,
                addr: self.globals_ptr_slot_addr(idx)?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_v128_to_raw_reg(raw, src);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: MachineAddr { base, offset: 0 },
                width: self.gp_word_mem_width(),
                src: MachineValue::Reg(raw),
            },
        });
        Ok(())
    }

    /// 32-bit i64 pair global.set -- called from Gp32Lowering.
    pub(super) fn lower_global_set_i64_pair(
        &mut self,
        idx: u32,
        src_value: SsaValue,
    ) -> Result<(), WasmError> {
        let (src_lo, src_hi) = self.use_i64_value_pair(src_value)?;
        let base = self.borrow_free_gp_dynamic_regs(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: base,
                addr: self.globals_ptr_slot_addr(idx)?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        let lo_addr = MachineAddr { base, offset: 0 };
        let hi_addr = addr_with_byte_offset(lo_addr, 4)?;
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: lo_addr,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(src_lo),
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: hi_addr,
                width: MachineMemWidth::U32,
                src: MachineValue::Reg(src_hi),
            },
        });
        Ok(())
    }

    pub(super) fn lower_table_size(
        &mut self,
        table_idx: u32,
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        let result = single_result(results)?;
        let pair_result = self.gp_reg_width() == 4
            && matches!(
                lir_value_storage_type(self.program(), result),
                MachineStorageType::GpI64
            );
        let (dst, dst_hi) = if pair_result {
            let (lo, hi) = self.alloc_i64_value_pair(result)?;
            (lo, Some(hi))
        } else {
            (self.alloc_result_value(result)?, None)
        };
        let table_views = self.borrow_free_gp_dynamic_regs(1)?[0];
        let runtime_layout = self.runtime_abi_layout();
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: table_views,
                addr: self.runtime_addr(runtime_layout.context.table_views_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst,
                addr: self.indexed_addr(
                    table_views,
                    table_idx,
                    runtime_layout.pointer_len_view.stride as usize,
                    runtime_layout.pointer_len_view.len_offset,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        if let Some(dst_hi) = dst_hi {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    src: MachineValue::Imm64(0),
                },
            });
        }
        Ok(())
    }

    pub(super) fn lower_table_get(
        &mut self,
        table_idx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        let index_value = single_arg(args)?.unwrap_value();
        let index = self.use_value(index_value)?;
        let dst = self.alloc_result_value(single_result(results)?)?;
        let (continuation_ops, terminator) = if self.remaining_use_count(index_value) == 0
            && dst != index
            && self.is_linear_value_reg(index)
        {
            let continuation_ops =
                self.lower_table_access_continuation_from_index64(table_idx, dst, index, dst)?;
            let terminator =
                self.lower_table_bounds_check(table_idx, index, dst, index, continuation, trap)?;
            (continuation_ops, terminator)
        } else {
            let index64 = dst;
            let table_len = self.borrow_free_gp_dynamic_regs(1)?[0];
            let continuation_ops = self.lower_table_access_continuation(
                table_idx,
                index,
                index64,
                table_len,
                Some(dst),
                None,
            )?;
            let terminator = self.lower_table_bounds_check(
                table_idx,
                index,
                index64,
                table_len,
                continuation,
                trap,
            )?;
            (continuation_ops, terminator)
        };
        Ok(LeafLowering::Split {
            continuation,
            trap,
            trap_kind: MachineTrapKind::TableOutOfBounds,
            terminator,
            continuation_ops,
        })
    }

    pub(super) fn lower_table_set(
        &mut self,
        table_idx: u32,
        args: &[SsaOperand],
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        let (index_value, src_value) = {
            let (a, b) = two_args(args)?;
            (a.unwrap_value(), b.unwrap_value())
        };
        let index = self.use_value(index_value)?;
        let src = self.use_value(src_value)?;
        let (index64, table_len) = if let Some(index64) = self.dead_value_reg(index_value) {
            (index64, self.borrow_free_gp_dynamic_regs(1)?[0])
        } else {
            let scratch = self.borrow_free_gp_dynamic_regs(2)?;
            (scratch[0], scratch[1])
        };
        let continuation_ops = self.lower_table_access_continuation(
            table_idx,
            index,
            index64,
            table_len,
            None,
            Some(src),
        )?;
        let terminator = self.lower_table_bounds_check(
            table_idx,
            index,
            index64,
            table_len,
            continuation,
            trap,
        )?;
        Ok(LeafLowering::Split {
            continuation,
            trap,
            trap_kind: MachineTrapKind::TableOutOfBounds,
            terminator,
            continuation_ops,
        })
    }

    pub(super) fn lower_memory_load(
        &mut self,
        spec: MemoryLoadSpec,
        args: &[SsaOperand],
        results: &[SsaValue],
        _continuation: MachineBlockId,
        _trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        if matches!(spec.ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            return ops.emit_memory_load_i64(self, spec, args, results);
        }
        self.lower_memory_load_scalar(spec, args, results)
    }

    /// Scalar (non-i64-pair) memory load -- used by both Gp64Lowering and
    /// the non-i64 path above.
    pub(super) fn lower_memory_load_scalar(
        &mut self,
        spec: MemoryLoadSpec,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        let addr_value = single_arg(args)?.unwrap_value();
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let fp_load_usable = spec.ty.float_width().is_some()
            && (spec.memidx != 0
                || self.dead_value_reg(addr_value).is_some()
                || self.borrow_free_gp_dynamic_regs(1).is_ok());
        let dst = if let Some(width) = spec.ty.float_width().filter(|_| fp_load_usable) {
            self.alloc_float_value(single_result(results)?, width)?
        } else {
            self.alloc_result_value_reusing_dead_inputs(single_result(results)?, &[addr_value])?
        };
        let access_bytes = spec.access_bytes();
        if spec.memidx == 0 {
            let addr32 = if self.is_fp_reg(dst) {
                if let Some(addr32) = self.dead_value_reg(addr_value) {
                    addr32
                } else {
                    self.borrow_free_gp_dynamic_regs(1)?[0]
                }
            } else {
                dst
            };
            let residual =
                self.emit_mem0_bounds_trap_if(spec.offset, access_bytes, addr, addr32)?;
            self.emit_machine_ops(self.lower_mem0_load_continuation(
                addr32,
                residual,
                dst,
                spec.ty,
                spec.width,
                spec.extension,
            )?);
        } else {
            let scratch = self.borrow_free_gp_dynamic_regs(2)?;
            let addr32 = scratch[0];
            let memory_view = scratch[1];
            let residual = self.emit_memory_bounds_trap_if(
                spec.memidx,
                spec.offset,
                access_bytes,
                addr,
                addr32,
                memory_view,
            )?;
            self.emit_machine_ops(self.lower_memory_continuation(
                spec.memidx,
                addr32,
                memory_view,
                residual,
                Some((dst, spec.ty, spec.width, spec.extension)),
                None,
            )?);
        }
        Ok(LeafLowering::InPlace)
    }

    #[cfg(sf_has_simd)]
    pub(super) fn lower_v128_load(
        &mut self,
        offset: u32,
        memidx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        self.lower_simd_load_common(OpcodeFD::V128_LOAD as u32, offset, memidx, args, results)
    }

    #[cfg(sf_has_simd)]
    pub(super) fn lower_simd_mem_load(
        &mut self,
        opcode: u32,
        offset: u32,
        memidx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        self.lower_simd_load_common(opcode, offset, memidx, args, results)
    }

    #[cfg(sf_has_simd)]
    pub(super) fn lower_v128_store(
        &mut self,
        offset: u32,
        memidx: u32,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError> {
        self.lower_simd_store_common(OpcodeFD::V128_STORE as u32, offset, memidx, args)
    }

    #[cfg(sf_has_simd)]
    pub(super) fn lower_simd_mem_load_lane(
        &mut self,
        opcode: u32,
        lane: u8,
        offset: u32,
        memidx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        let (addr_arg, vector_arg) = two_args(args)?;
        let addr_value = addr_arg.unwrap_value();
        let vector_value = vector_arg.unwrap_value();
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let vector = self.use_value(vector_value)?;
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|arg| match arg.decode() {
                DecodedOperand::Value(value) => Some(value),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_result_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        let access_bytes = simd_lane_access_bytes(opcode)?;
        if memidx == 0 {
            let addr32 = if let Some(reg) = self.dead_value_reg(addr_value) {
                reg
            } else {
                self.borrow_free_gp_dynamic_regs(1)?[0]
            };
            let residual = self.emit_mem0_bounds_trap_if(offset, access_bytes, addr, addr32)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(self.mem0_base_reg()),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdLoadLane {
                    opcode,
                    lane,
                    dst,
                    addr: MachineAddr {
                        base: addr32,
                        offset: 0,
                    },
                    vector: MachineValue::Reg(vector),
                },
            });
        } else {
            let scratch = self.borrow_free_gp_dynamic_regs(2)?;
            let addr32 = scratch[0];
            let base = scratch[1];
            let residual =
                self.emit_memory_bounds_trap_if(memidx, offset, access_bytes, addr, addr32, base)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            let mut ops = collections::Vec::new();
            emit_memory_base_load_ops(
                &mut ops,
                self.runtime_base_reg(),
                memidx,
                base,
                self.gp_reg_width(),
            )?;
            self.emit_machine_ops(ops);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: base,
                    lhs: MachineValue::Reg(base),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdLoadLane {
                    opcode,
                    lane,
                    dst,
                    addr: MachineAddr { base, offset: 0 },
                    vector: MachineValue::Reg(vector),
                },
            });
        }
        Ok(LeafLowering::InPlace)
    }

    #[cfg(sf_has_simd)]
    pub(super) fn lower_simd_mem_store_lane(
        &mut self,
        opcode: u32,
        lane: u8,
        offset: u32,
        memidx: u32,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError> {
        let (addr_arg, vector_arg) = two_args(args)?;
        let addr_value = addr_arg.unwrap_value();
        let vector_value = vector_arg.unwrap_value();
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let vector = self.use_value(vector_value)?;
        let access_bytes = simd_lane_access_bytes(opcode)?;
        if memidx == 0 {
            let addr32 = if let Some(reg) = self.dead_value_reg(addr_value) {
                reg
            } else {
                self.borrow_free_gp_dynamic_regs(1)?[0]
            };
            let residual = self.emit_mem0_bounds_trap_if(offset, access_bytes, addr, addr32)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(self.mem0_base_reg()),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdStoreLane {
                    opcode,
                    lane,
                    addr: MachineAddr {
                        base: addr32,
                        offset: 0,
                    },
                    vector: MachineValue::Reg(vector),
                },
            });
        } else {
            let scratch = self.borrow_free_gp_dynamic_regs(2)?;
            let addr32 = scratch[0];
            let base = scratch[1];
            let residual =
                self.emit_memory_bounds_trap_if(memidx, offset, access_bytes, addr, addr32, base)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            let mut ops = collections::Vec::new();
            emit_memory_base_load_ops(
                &mut ops,
                self.runtime_base_reg(),
                memidx,
                base,
                self.gp_reg_width(),
            )?;
            self.emit_machine_ops(ops);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: base,
                    lhs: MachineValue::Reg(base),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdStoreLane {
                    opcode,
                    lane,
                    addr: MachineAddr { base, offset: 0 },
                    vector: MachineValue::Reg(vector),
                },
            });
        }
        Ok(LeafLowering::InPlace)
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_load_common(
        &mut self,
        opcode: u32,
        offset: u32,
        memidx: u32,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        let addr_value = single_arg(args)?.unwrap_value();
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let dead: collections::Vec<SsaValue> = args
            .iter()
            .filter_map(|arg| match arg.decode() {
                DecodedOperand::Value(value) => Some(value),
                DecodedOperand::Const(_) | DecodedOperand::None => None,
            })
            .collect();
        let dst = self.alloc_result_value_reusing_dead_inputs(single_result(results)?, &dead)?;
        let access_bytes = simd_load_access_bytes(opcode)?;
        if memidx == 0 {
            let addr32 = if let Some(reg) = self.dead_value_reg(addr_value) {
                reg
            } else {
                self.borrow_free_gp_dynamic_regs(1)?[0]
            };
            let residual = self.emit_mem0_bounds_trap_if(offset, access_bytes, addr, addr32)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(self.mem0_base_reg()),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdLoad {
                    opcode,
                    dst,
                    addr: MachineAddr {
                        base: addr32,
                        offset: 0,
                    },
                },
            });
        } else {
            let scratch = self.borrow_free_gp_dynamic_regs(2)?;
            let addr32 = scratch[0];
            let base = scratch[1];
            let residual =
                self.emit_memory_bounds_trap_if(memidx, offset, access_bytes, addr, addr32, base)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            let mut ops = collections::Vec::new();
            emit_memory_base_load_ops(
                &mut ops,
                self.runtime_base_reg(),
                memidx,
                base,
                self.gp_reg_width(),
            )?;
            self.emit_machine_ops(ops);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: base,
                    lhs: MachineValue::Reg(base),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdLoad {
                    opcode,
                    dst,
                    addr: MachineAddr { base, offset: 0 },
                },
            });
        }
        Ok(LeafLowering::InPlace)
    }

    #[cfg(sf_has_simd)]
    fn lower_simd_store_common(
        &mut self,
        opcode: u32,
        offset: u32,
        memidx: u32,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError> {
        let (addr_arg, src_arg) = two_args(args)?;
        let addr_value = addr_arg.unwrap_value();
        let src_value = src_arg.unwrap_value();
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let src = self.use_value(src_value)?;
        let access_bytes = simd_load_access_bytes(opcode)?;
        if memidx == 0 {
            let addr32 = if let Some(reg) = self.dead_value_reg(addr_value) {
                reg
            } else {
                self.borrow_free_gp_dynamic_regs(1)?[0]
            };
            let residual = self.emit_mem0_bounds_trap_if(offset, access_bytes, addr, addr32)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(self.mem0_base_reg()),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdStore {
                    opcode,
                    addr: MachineAddr {
                        base: addr32,
                        offset: 0,
                    },
                    src: MachineValue::Reg(src),
                },
            });
        } else {
            let scratch = self.borrow_free_gp_dynamic_regs(2)?;
            let addr32 = scratch[0];
            let base = scratch[1];
            let residual =
                self.emit_memory_bounds_trap_if(memidx, offset, access_bytes, addr, addr32, base)?;
            if residual != 0 {
                self.emit_machine_inst(MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: self.gp_word_int_width(),
                        op: MachineIntBinaryOp::Sub,
                        dst: addr32,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Imm64(residual as u64),
                    },
                });
            }
            let mut ops = collections::Vec::new();
            emit_memory_base_load_ops(
                &mut ops,
                self.runtime_base_reg(),
                memidx,
                base,
                self.gp_reg_width(),
            )?;
            self.emit_machine_ops(ops);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: base,
                    lhs: MachineValue::Reg(base),
                    rhs: MachineValue::Reg(addr32),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::SimdStore {
                    opcode,
                    addr: MachineAddr { base, offset: 0 },
                    src: MachineValue::Reg(src),
                },
            });
        }
        Ok(LeafLowering::InPlace)
    }

    pub(super) fn lower_memory_store(
        &mut self,
        spec: MemoryStoreSpec,
        args: &[SsaOperand],
        _continuation: MachineBlockId,
        _trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        if matches!(spec.ty, MachineStorageType::GpI64) {
            let ops = self.i64_ops();
            return ops.emit_memory_store_i64(self, spec, args);
        }
        self.lower_memory_store_scalar(spec, args)
    }

    /// Scalar (non-i64-pair) memory store -- used by both Gp64Lowering and
    /// the non-i64 path above.
    pub(super) fn lower_memory_store_scalar(
        &mut self,
        spec: MemoryStoreSpec,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError> {
        let (addr_value, src_value) = {
            let (a, b) = two_args(args)?;
            (a.unwrap_value(), b.unwrap_value())
        };
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let access_bytes = spec.access_bytes();
        if spec.memidx == 0 {
            let addr32 = if let Some(addr32) = self.dead_value_reg(addr_value) {
                addr32
            } else {
                self.borrow_free_gp_dynamic_regs(1)?[0]
            };
            let residual =
                self.emit_mem0_bounds_trap_if(spec.offset, access_bytes, addr, addr32)?;
            let src = self.use_value(src_value)?;
            self.emit_machine_ops(
                self.lower_mem0_store_continuation(addr32, residual, src, spec.ty, spec.width)?,
            );
        } else {
            let scratch = self.borrow_free_gp_dynamic_regs(2)?;
            let addr32 = scratch[0];
            let memory_view = scratch[1];
            let residual = self.emit_memory_bounds_trap_if(
                spec.memidx,
                spec.offset,
                access_bytes,
                addr,
                addr32,
                memory_view,
            )?;
            let src = self.use_value(src_value)?;
            self.emit_machine_ops(self.lower_memory_continuation(
                spec.memidx,
                addr32,
                memory_view,
                residual,
                None,
                Some((src, spec.ty, spec.width)),
            )?);
        }
        Ok(LeafLowering::InPlace)
    }

    pub(super) fn lower_i64_memory_load(
        &mut self,
        spec: MemoryLoadSpec,
        args: &[SsaOperand],
        results: &[SsaValue],
    ) -> Result<LeafLowering, WasmError> {
        let addr_value = single_arg(args)?.unwrap_value();
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let (dst_lo, dst_hi) =
            self.alloc_i64_value_pair_reusing_dead_inputs(single_result(results)?, &[addr_value])?;
        let access_bytes = spec.access_bytes();
        if spec.memidx == 0 {
            let addr32 = dst_lo;
            let residual =
                self.emit_mem0_bounds_trap_if(spec.offset, access_bytes, addr, addr32)?;
            self.emit_machine_ops(self.lower_mem0_i64_load_continuation(
                addr32,
                residual,
                dst_lo,
                dst_hi,
                spec.width,
                spec.extension,
            )?);
        } else {
            let addr32 = dst_lo;
            let base = dst_hi;
            let residual = self.emit_memory_bounds_trap_if(
                spec.memidx,
                spec.offset,
                access_bytes,
                addr,
                addr32,
                base,
            )?;
            self.emit_machine_ops(self.lower_memory_i64_load_continuation(
                spec.memidx,
                addr32,
                base,
                residual,
                dst_lo,
                dst_hi,
                spec.width,
                spec.extension,
            )?);
        }
        Ok(LeafLowering::InPlace)
    }

    pub(super) fn lower_i64_memory_store(
        &mut self,
        spec: MemoryStoreSpec,
        args: &[SsaOperand],
    ) -> Result<LeafLowering, WasmError> {
        let (addr_value, src_value) = {
            let (a, b) = two_args(args)?;
            (a.unwrap_value(), b.unwrap_value())
        };
        let (addr, addr_hi) = self.use_memory_index_word(addr_value)?;
        self.emit_memory_index_high_trap_if_nonzero(addr_hi);
        let access_bytes = spec.access_bytes();
        if spec.memidx == 0 {
            let addr32 = if let Some(addr32) = self.dead_value_reg(addr_value) {
                addr32
            } else {
                self.borrow_free_gp_dynamic_regs(1)?[0]
            };
            let residual =
                self.emit_mem0_bounds_trap_if(spec.offset, access_bytes, addr, addr32)?;
            let (src_lo, src_hi) = self.use_i64_value_pair(src_value)?;
            self.emit_machine_ops(
                self.lower_mem0_i64_store_continuation(
                    addr32, residual, src_lo, src_hi, spec.width,
                )?,
            );
        } else {
            let (addr32, base) = if let Some(addr32) = self.dead_value_reg(addr_value) {
                (addr32, self.borrow_free_gp_dynamic_regs(1)?[0])
            } else {
                let scratch = self.borrow_free_gp_dynamic_regs(2)?;
                (scratch[0], scratch[1])
            };
            let residual = self.emit_memory_bounds_trap_if(
                spec.memidx,
                spec.offset,
                access_bytes,
                addr,
                addr32,
                base,
            )?;
            let (src_lo, src_hi) = self.use_i64_value_pair(src_value)?;
            self.emit_machine_ops(self.lower_memory_i64_store_continuation(
                spec.memidx,
                addr32,
                base,
                residual,
                src_lo,
                src_hi,
                spec.width,
            )?);
        }
        Ok(LeafLowering::InPlace)
    }

    /// Emit a non-mem0 bounds check. Returns residual access_bytes embedded
    /// in addr32 (0 when a scratch was available, access_bytes otherwise).
    fn emit_memory_bounds_trap_if(
        &mut self,
        memidx: u32,
        offset: u32,
        access_bytes: u32,
        addr: MachineReg,
        addr32: MachineReg,
        scratch: MachineReg,
    ) -> Result<u32, WasmError> {
        self.emit_effective_addr(offset, addr, addr32)?;
        self.emit_word_add_immediate_wrap_trap_if(addr32, offset);
        self.emit_memory_len_load(memidx, scratch)?;
        if access_bytes == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Reg(scratch),
                    },
                },
            });
            return Ok(0);
        }
        // Guard against aliasing the borrowed check register with either the
        // live effective-address register or the memory-length scratch. Those
        // values may still appear "free" to the allocator when they came from
        // dead-value reuse, but clobbering either one while still reporting a
        // zero residual silently shifts the final memory address (for example,
        // `load8` from memidx!=0 starts reading at `addr+1`).
        if let Some(check_reg) = self
            .borrow_free_gp_dynamic_regs(1)
            .ok()
            .map(|s| s[0])
            .filter(|r| *r != addr32 && *r != scratch)
        {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: check_reg,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
            self.emit_word_add_immediate_wrap_trap_if(check_reg, access_bytes);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(check_reg),
                        rhs: MachineValue::Reg(scratch),
                    },
                },
            });
            Ok(0)
        } else {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
            self.emit_word_add_immediate_wrap_trap_if(addr32, access_bytes);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Reg(scratch),
                    },
                },
            });
            Ok(access_bytes)
        }
    }

    fn emit_mem0_bounds_trap_if(
        &mut self,
        offset: u32,
        access_bytes: u32,
        addr: MachineReg,
        addr32: MachineReg,
    ) -> Result<u32, WasmError> {
        self.emit_effective_addr(offset, addr, addr32)?;
        self.emit_word_add_immediate_wrap_trap_if(addr32, offset);
        #[cfg(sf_has_guard_pages)]
        if self.use_guard_pages() && !self.needs_explicit_multiword_gp_bounds_check(access_bytes) {
            return Ok(0);
        }
        if access_bytes == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Reg(self.mem0_size_reg()),
                    },
                },
            });
            return Ok(0);
        }
        // Try to use a separate scratch register for the bounds-check
        // addition so that addr32 stays unmodified (residual = 0).
        //
        // Guard: the borrowed register must differ from addr32.  When
        // addr32 came from dead_value_reg it is back in the free pool,
        // so borrow_free_gp_dynamic_regs can hand it out again. If that
        // happens, `check_reg = addr32 + access_bytes` silently corrupts
        // addr32 while the caller believes residual is 0 (untouched).
        // Filtering it out forces the in-place fallback path below,
        // which correctly reports the residual for later subtraction.
        if let Some(check_reg) = self
            .borrow_free_gp_dynamic_regs(1)
            .ok()
            .map(|s| s[0])
            .filter(|r| *r != addr32)
        {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: check_reg,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
            self.emit_word_add_immediate_wrap_trap_if(check_reg, access_bytes);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(check_reg),
                        rhs: MachineValue::Reg(self.mem0_size_reg()),
                    },
                },
            });
            Ok(0)
        } else {
            let check_addend = access_bytes as u64;
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(check_addend),
                },
            });
            self.emit_word_add_immediate_wrap_trap_if(addr32, access_bytes);
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: MachineSign::Unsigned,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Reg(self.mem0_size_reg()),
                    },
                },
            });
            Ok(access_bytes)
        }
    }

    fn lower_memory_continuation(
        &self,
        memidx: u32,
        addr32: MachineReg,
        scratch: MachineReg,
        access_bytes: u32,
        load_dst: Option<(
            MachineReg,
            MachineStorageType,
            MachineMemWidth,
            MachineLoadExtension,
        )>,
        store_src: Option<(MachineReg, MachineStorageType, MachineMemWidth)>,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        emit_memory_base_load_ops(
            &mut ops,
            self.runtime_base_reg(),
            memidx,
            scratch,
            self.gp_reg_width(),
        )?;
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: scratch,
                lhs: MachineValue::Reg(scratch),
                rhs: MachineValue::Reg(addr32),
            },
        });
        if let Some((dst, ty, width, extension)) = load_dst {
            ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty,
                    dst,
                    addr: MachineAddr {
                        base: scratch,
                        offset: 0,
                    },
                    width,
                    extension,
                },
            });
        }
        if let Some((src, ty, width)) = store_src {
            ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    ty,
                    addr: MachineAddr {
                        base: scratch,
                        offset: 0,
                    },
                    width,
                    src: MachineValue::Reg(src),
                },
            });
        }
        Ok(ops)
    }

    fn lower_memory_i64_load_continuation(
        &self,
        memidx: u32,
        addr32: MachineReg,
        base: MachineReg,
        access_bytes: u32,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        emit_memory_base_load_ops(
            &mut ops,
            self.runtime_base_reg(),
            memidx,
            base,
            self.gp_reg_width(),
        )?;
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: base,
                lhs: MachineValue::Reg(base),
                rhs: MachineValue::Reg(addr32),
            },
        });
        append_i64_load_ops(
            &mut ops,
            self.gp_word_int_width(),
            base,
            dst_lo,
            dst_hi,
            width,
            extension,
        );
        Ok(ops)
    }

    fn lower_mem0_load_continuation(
        &self,
        addr32: MachineReg,
        access_bytes: u32,
        dst: MachineReg,
        ty: MachineStorageType,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: addr32,
                lhs: MachineValue::Reg(self.mem0_base_reg()),
                rhs: MachineValue::Reg(addr32),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty,
                dst,
                addr: MachineAddr {
                    base: addr32,
                    offset: 0,
                },
                width,
                extension,
            },
        });
        Ok(ops)
    }

    fn lower_mem0_i64_load_continuation(
        &self,
        addr32: MachineReg,
        access_bytes: u32,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: addr32,
                lhs: MachineValue::Reg(self.mem0_base_reg()),
                rhs: MachineValue::Reg(addr32),
            },
        });
        append_i64_load_ops(
            &mut ops,
            self.gp_word_int_width(),
            addr32,
            dst_lo,
            dst_hi,
            width,
            extension,
        );
        Ok(ops)
    }

    fn lower_mem0_store_continuation(
        &self,
        addr32: MachineReg,
        access_bytes: u32,
        src: MachineReg,
        ty: MachineStorageType,
        width: MachineMemWidth,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: addr32,
                lhs: MachineValue::Reg(self.mem0_base_reg()),
                rhs: MachineValue::Reg(addr32),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Store {
                ty,
                addr: MachineAddr {
                    base: addr32,
                    offset: 0,
                },
                width,
                src: MachineValue::Reg(src),
            },
        });
        Ok(ops)
    }

    fn lower_memory_i64_store_continuation(
        &self,
        memidx: u32,
        addr32: MachineReg,
        base: MachineReg,
        access_bytes: u32,
        src_lo: MachineReg,
        src_hi: MachineReg,
        width: MachineMemWidth,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        emit_memory_base_load_ops(
            &mut ops,
            self.runtime_base_reg(),
            memidx,
            base,
            self.gp_reg_width(),
        )?;
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: base,
                lhs: MachineValue::Reg(base),
                rhs: MachineValue::Reg(addr32),
            },
        });
        append_i64_store_ops(&mut ops, base, src_lo, src_hi, width)?;
        Ok(ops)
    }

    fn lower_mem0_i64_store_continuation(
        &self,
        addr32: MachineReg,
        access_bytes: u32,
        src_lo: MachineReg,
        src_hi: MachineReg,
        width: MachineMemWidth,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: addr32,
                lhs: MachineValue::Reg(self.mem0_base_reg()),
                rhs: MachineValue::Reg(addr32),
            },
        });
        append_i64_store_ops(&mut ops, addr32, src_lo, src_hi, width)?;
        Ok(ops)
    }

    fn lower_table_bounds_check(
        &mut self,
        table_idx: u32,
        index: MachineReg,
        index64: MachineReg,
        table_len: MachineReg,
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<MachineTerminator, WasmError> {
        if self.gp_reg_width() == 8 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Convert {
                    op: MachineConvertOp::I64ExtendI32U,
                    dst: index64,
                    src: MachineValue::Reg(index),
                },
            });
        } else if index64 != index {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: index64,
                    src: MachineValue::Reg(index),
                },
            });
        }
        let runtime_layout = self.runtime_abi_layout();
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: table_len,
                addr: self.runtime_addr(runtime_layout.context.table_views_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: table_len,
                addr: self.indexed_addr(
                    table_len,
                    table_idx,
                    runtime_layout.pointer_len_view.stride as usize,
                    runtime_layout.pointer_len_view.len_offset,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        Ok(MachineTerminator::Branch {
            cond: MachineBranchCond::IntCompare {
                width: self.gp_word_int_width(),
                kind: MachineCompareKind::Ge,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(index64),
                rhs: MachineValue::Reg(table_len),
            },
            then_edge: MachineEdge {
                target: trap,
                args: collections::Vec::new(),
            },
            else_edge: MachineEdge {
                target: continuation,
                args: collections::Vec::new(),
            },
        })
    }

    fn lower_table_access_continuation(
        &self,
        table_idx: u32,
        index: MachineReg,
        index64: MachineReg,
        scratch: MachineReg,
        load_dst: Option<MachineReg>,
        store_src: Option<MachineReg>,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        if self.gp_reg_width() == 8 {
            ops.push(MachineInst {
                kind: MachineInstKind::Convert {
                    op: MachineConvertOp::I64ExtendI32U,
                    dst: index64,
                    src: MachineValue::Reg(index),
                },
            });
        } else if index64 != index {
            ops.push(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: index64,
                    src: MachineValue::Reg(index),
                },
            });
        }
        let runtime_layout = self.runtime_abi_layout();
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: scratch,
                addr: self.runtime_addr(runtime_layout.context.table_views_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: scratch,
                addr: self.indexed_addr(
                    scratch,
                    table_idx,
                    runtime_layout.pointer_len_view.stride as usize,
                    runtime_layout.pointer_len_view.base_offset,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Mul,
                dst: index64,
                lhs: MachineValue::Reg(index64),
                rhs: MachineValue::Imm64(u64::from(runtime_layout.ref_handle_stride)),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: scratch,
                lhs: MachineValue::Reg(scratch),
                rhs: MachineValue::Reg(index64),
            },
        });
        if let Some(dst) = load_dst {
            ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst,
                    addr: MachineAddr {
                        base: scratch,
                        offset: 0,
                    },
                    width: self.gp_word_mem_width(),
                    extension: MachineLoadExtension::None,
                },
            });
        }
        if let Some(src) = store_src {
            ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr: MachineAddr {
                        base: scratch,
                        offset: 0,
                    },
                    width: self.gp_word_mem_width(),
                    src: MachineValue::Reg(src),
                },
            });
        }
        Ok(ops)
    }

    fn lower_table_access_continuation_from_index64(
        &self,
        table_idx: u32,
        index64: MachineReg,
        scratch: MachineReg,
        load_dst: MachineReg,
    ) -> Result<collections::Vec<MachineInst>, WasmError> {
        let mut ops = collections::Vec::new();
        let runtime_layout = self.runtime_abi_layout();
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: scratch,
                addr: self.runtime_addr(runtime_layout.context.table_views_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: scratch,
                addr: self.indexed_addr(
                    scratch,
                    table_idx,
                    runtime_layout.pointer_len_view.stride as usize,
                    runtime_layout.pointer_len_view.base_offset,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Mul,
                dst: index64,
                lhs: MachineValue::Reg(index64),
                rhs: MachineValue::Imm64(u64::from(runtime_layout.ref_handle_stride)),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: MachineIntBinaryOp::Add,
                dst: scratch,
                lhs: MachineValue::Reg(scratch),
                rhs: MachineValue::Reg(index64),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: load_dst,
                addr: MachineAddr {
                    base: scratch,
                    offset: 0,
                },
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        Ok(ops)
    }

    fn emit_effective_addr(
        &mut self,
        offset: u32,
        addr: MachineReg,
        addr32: MachineReg,
    ) -> Result<(), WasmError> {
        if self.gp_reg_width() == 8 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Convert {
                    op: MachineConvertOp::I64ExtendI32U,
                    dst: addr32,
                    src: MachineValue::Reg(addr),
                },
            });
        } else if addr32 != addr {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: addr32,
                    src: MachineValue::Reg(addr),
                },
            });
        }
        if offset != 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(offset as u64),
                },
            });
        }
        Ok(())
    }

    #[cfg(sf_has_guard_pages)]
    #[inline]
    fn needs_explicit_multiword_gp_bounds_check(&self, access_bytes: u32) -> bool {
        self.gp_reg_width() == 4 && access_bytes > u32::from(self.gp_reg_width())
    }

    fn emit_word_add_immediate_wrap_trap_if(&mut self, sum: MachineReg, addend: u32) {
        if self.gp_reg_width() != 4 || addend == 0 {
            return;
        }
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::TrapIf {
                kind: MachineTrapKind::MemoryOutOfBounds,
                cond: MachineBranchCond::IntCompare {
                    width: self.gp_word_int_width(),
                    kind: MachineCompareKind::Lt,
                    sign: MachineSign::Unsigned,
                    lhs: MachineValue::Reg(sum),
                    rhs: MachineValue::Imm64(addend as u64),
                },
            },
        });
    }

    fn emit_memory_len_load(&mut self, memidx: u32, dst: MachineReg) -> Result<(), WasmError> {
        if memidx == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst,
                    src: MachineValue::Reg(self.mem0_size_reg()),
                },
            });
            return Ok(());
        }

        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst,
                addr: self.runtime_addr(self.runtime_abi_layout().context.memory_views_base_offset),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst,
                addr: self.indexed_addr(
                    dst,
                    memidx,
                    self.runtime_abi_layout().pointer_len_view.stride as usize,
                    self.runtime_abi_layout().pointer_len_view.len_offset,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    pub(super) fn indexed_addr(
        &self,
        base: MachineReg,
        index: u32,
        stride: usize,
        field_offset: u32,
    ) -> Result<MachineAddr, WasmError> {
        let scaled = (index as u64)
            .checked_mul(stride as u64)
            .and_then(|value| value.checked_add(field_offset as u64))
            .ok_or_else(|| WasmError::internal("runtime view byte offset overflow"))?;
        let offset = i32::try_from(scaled)
            .map_err(|_| WasmError::internal("runtime view byte offset exceeds i32"))?;
        Ok(MachineAddr { base, offset })
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(super) struct MemoryLoadSpec {
    memidx: u32,
    offset: u32,
    ty: MachineStorageType,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
}

impl MemoryLoadSpec {
    #[inline]
    fn access_bytes(self) -> u32 {
        self.width.bytes()
    }
}

#[derive(Clone, Copy)]
pub(super) struct MemoryStoreSpec {
    memidx: u32,
    offset: u32,
    ty: MachineStorageType,
    width: MachineMemWidth,
}

impl MemoryStoreSpec {
    #[inline]
    fn access_bytes(self) -> u32 {
        self.width.bytes()
    }
}

pub(super) fn machine_load(primitive: &PrimitiveOpKind) -> Option<MemoryLoadSpec> {
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
        },
        P::I64Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        },
        P::F32Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::Fp32,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
        },
        P::F64Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::Fp64,
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
        },
        P::I32Load8S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::SignExtend,
        },
        P::I32Load8U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::ZeroExtend,
        },
        P::I32Load16S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::SignExtend,
        },
        P::I32Load16U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::ZeroExtend,
        },
        P::I64Load8S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::SignExtend,
        },
        P::I64Load8U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::ZeroExtend,
        },
        P::I64Load16S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::SignExtend,
        },
        P::I64Load16U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::ZeroExtend,
        },
        P::I64Load32S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::SignExtend,
        },
        P::I64Load32U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
        },
        _ => return None,
    })
}

pub(super) fn machine_store(primitive: &PrimitiveOpKind) -> Option<MemoryStoreSpec> {
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U32,
        },
        P::I64Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U64,
        },
        P::F32Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::Fp32,
            width: MachineMemWidth::U32,
        },
        P::F64Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::Fp64,
            width: MachineMemWidth::U64,
        },
        P::I32Store8 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U8,
        },
        P::I32Store16 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpWord,
            width: MachineMemWidth::U16,
        },
        P::I64Store8 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U8,
        },
        P::I64Store16 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U16,
        },
        P::I64Store32 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            ty: MachineStorageType::GpI64,
            width: MachineMemWidth::U32,
        },
        _ => return None,
    })
}

#[cfg(sf_has_simd)]
fn simd_load_access_bytes(opcode: u32) -> Result<u32, WasmError> {
    let opcode: OpcodeFD = opcode
        .try_into()
        .map_err(|_| WasmError::internal("unknown SIMD memory opcode"))?;
    Ok(match opcode {
        OpcodeFD::V128_LOAD | OpcodeFD::V128_STORE => 16,
        OpcodeFD::V128_LOAD8X8_S
        | OpcodeFD::V128_LOAD8X8_U
        | OpcodeFD::V128_LOAD16X4_S
        | OpcodeFD::V128_LOAD16X4_U
        | OpcodeFD::V128_LOAD32X2_S
        | OpcodeFD::V128_LOAD32X2_U => 8,
        OpcodeFD::V128_LOAD8_SPLAT => 1,
        OpcodeFD::V128_LOAD16_SPLAT => 2,
        OpcodeFD::V128_LOAD32_SPLAT | OpcodeFD::V128_LOAD32_ZERO => 4,
        OpcodeFD::V128_LOAD64_SPLAT | OpcodeFD::V128_LOAD64_ZERO => 8,
        _ => {
            return Err(WasmError::internal(
                "unexpected SIMD memory opcode in native lowering".into(),
            ));
        }
    })
}

#[cfg(sf_has_simd)]
fn simd_lane_access_bytes(opcode: u32) -> Result<u32, WasmError> {
    let opcode: OpcodeFD = opcode
        .try_into()
        .map_err(|_| WasmError::internal("unknown SIMD lane memory opcode"))?;
    Ok(match opcode {
        OpcodeFD::V128_LOAD8_LANE | OpcodeFD::V128_STORE8_LANE => 1,
        OpcodeFD::V128_LOAD16_LANE | OpcodeFD::V128_STORE16_LANE => 2,
        OpcodeFD::V128_LOAD32_LANE | OpcodeFD::V128_STORE32_LANE => 4,
        OpcodeFD::V128_LOAD64_LANE | OpcodeFD::V128_STORE64_LANE => 8,
        _ => {
            return Err(WasmError::internal(
                "unexpected SIMD lane memory opcode in native lowering".into(),
            ));
        }
    })
}

pub(super) fn append_i64_load_ops(
    ops: &mut collections::Vec<MachineInst>,
    gp_word_int_width: MachineIntWidth,
    base: MachineReg,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
) {
    match width {
        MachineMemWidth::U64 => {
            let load_lo_first = base == dst_hi;
            let first = if load_lo_first {
                (dst_lo, 0)
            } else {
                (dst_hi, 4)
            };
            let second = if load_lo_first {
                (dst_hi, 4)
            } else {
                (dst_lo, 0)
            };
            for (dst, offset) in [first, second] {
                ops.push(MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst,
                        addr: MachineAddr { base, offset },
                        width: MachineMemWidth::U32,
                        extension: MachineLoadExtension::None,
                    },
                });
            }
        }
        MachineMemWidth::U8 | MachineMemWidth::U16 => {
            ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_lo,
                    addr: MachineAddr { base, offset: 0 },
                    width,
                    extension,
                },
            });
            append_i64_load_hi_fill_ops(ops, gp_word_int_width, dst_lo, dst_hi, extension);
        }
        MachineMemWidth::U32 => {
            ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_lo,
                    addr: MachineAddr { base, offset: 0 },
                    width,
                    extension: MachineLoadExtension::None,
                },
            });
            append_i64_load_hi_fill_ops(ops, gp_word_int_width, dst_lo, dst_hi, extension);
        }
    }
}

fn append_i64_load_hi_fill_ops(
    ops: &mut collections::Vec<MachineInst>,
    gp_word_int_width: MachineIntWidth,
    dst_lo: MachineReg,
    dst_hi: MachineReg,
    extension: MachineLoadExtension,
) {
    match extension {
        MachineLoadExtension::SignExtend => {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: gp_word_int_width,
                    op: MachineIntBinaryOp::ShrS,
                    dst: dst_hi,
                    lhs: MachineValue::Reg(dst_lo),
                    rhs: MachineValue::Imm64(31),
                },
            });
        }
        MachineLoadExtension::ZeroExtend | MachineLoadExtension::None => {
            ops.push(MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpWord,
                    dst: dst_hi,
                    src: MachineValue::Imm64(0),
                },
            });
        }
    }
}

fn append_i64_store_ops(
    ops: &mut collections::Vec<MachineInst>,
    base: MachineReg,
    src_lo: MachineReg,
    src_hi: MachineReg,
    width: MachineMemWidth,
) -> Result<(), WasmError> {
    match width {
        MachineMemWidth::U64 => {
            ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr: MachineAddr { base, offset: 0 },
                    width: MachineMemWidth::U32,
                    src: MachineValue::Reg(src_lo),
                },
            });
            ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr: MachineAddr { base, offset: 4 },
                    width: MachineMemWidth::U32,
                    src: MachineValue::Reg(src_hi),
                },
            });
        }
        MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32 => {
            ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    ty: MachineStorageType::GpWord,
                    addr: MachineAddr { base, offset: 0 },
                    width,
                    src: MachineValue::Reg(src_lo),
                },
            });
        }
    }
    Ok(())
}

fn emit_memory_base_load_ops(
    ops: &mut collections::Vec<MachineInst>,
    runtime_base: MachineReg,
    memidx: u32,
    dst: MachineReg,
    gp_reg_width: u8,
) -> Result<(), WasmError> {
    let runtime_layout = native_runtime_abi_layout(gp_reg_width);
    if memidx == 0 {
        ops.push(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst,
                src: MachineValue::Reg(MACHINE_MEM0_BASE_REG),
            },
        });
        return Ok(());
    }

    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst,
            addr: MachineAddr {
                base: runtime_base,
                offset: runtime_layout.context.memory_views_base_offset as i32,
            },
            width: machine_ptr_width(gp_reg_width),
            extension: MachineLoadExtension::None,
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst,
            addr: MachineAddr {
                base: dst,
                offset: indexed_field_offset(
                    memidx,
                    runtime_layout.pointer_len_view.stride as usize,
                    runtime_layout.pointer_len_view.base_offset,
                )?,
            },
            width: machine_ptr_width(gp_reg_width),
            extension: MachineLoadExtension::None,
        },
    });
    Ok(())
}

fn indexed_field_offset(index: u32, stride: usize, field_offset: u32) -> Result<i32, WasmError> {
    let scaled = (index as u64)
        .checked_mul(stride as u64)
        .and_then(|value| value.checked_add(field_offset as u64))
        .ok_or_else(|| WasmError::internal("runtime view byte offset overflow"))?;
    i32::try_from(scaled).map_err(|_| WasmError::internal("runtime view byte offset exceeds i32"))
}

pub(super) fn addr_with_byte_offset(
    mut addr: MachineAddr,
    byte_offset: i32,
) -> Result<MachineAddr, WasmError> {
    addr.offset = addr
        .offset
        .checked_add(byte_offset)
        .ok_or_else(|| WasmError::internal("machine address offset overflow"))?;
    Ok(addr)
}
