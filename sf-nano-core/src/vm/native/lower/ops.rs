use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        entities::global_offset,
        lir::{ir::LirValue, leaf::LirLeafOp},
        native::{
            ir::machine::{
                machine_ptr_width, MachineBlockId, MachineBranchCond, MachineCompareKind,
                MachineConvertOp, MachineFloatWidth, MachineInst, MachineInstKind,
                MachineLoadExtension, MachineMemWidth, MachineTerminator, MachineTrapKind,
                MachineValue,
            },
            runtime::context::{
                ctx_offset, globals_view_offset, memory_view_offset, table_view_offset,
                NativeMemoryView, NativeTableView,
            },
        },
        wasm::primitive_op::PrimitiveOpKind,
    },
};

use super::{
    context::BlockLowerContext,
    util::{single_arg, single_result, two_args},
};

pub(super) enum LeafLowering {
    InPlace,
    Split {
        continuation: MachineBlockId,
        trap: MachineBlockId,
        trap_kind: MachineTrapKind,
        terminator: MachineTerminator,
        continuation_ops: Vec<MachineInst>,
    },
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn lower_special_leaf(
        &mut self,
        op: &LirLeafOp,
        args: &[LirValue],
        results: &[LirValue],
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<Option<LeafLowering>, WasmError> {
        use PrimitiveOpKind as P;

        let lowered = match op.primitive() {
            P::MemorySize { mem_idx } => {
                self.lower_memory_size(*mem_idx, results)?;
                LeafLowering::InPlace
            }
            P::GlobalGet { idx } => {
                self.lower_global_get(*idx, results)?;
                LeafLowering::InPlace
            }
            P::GlobalSet { idx } => {
                self.lower_global_set(*idx, args)?;
                LeafLowering::InPlace
            }
            P::TableSize { table_idx } => {
                self.lower_table_size(*table_idx, results)?;
                LeafLowering::InPlace
            }
            P::TableGet { table_idx } => {
                self.lower_table_get(*table_idx, args, results, continuation, trap)?
            }
            P::TableSet { table_idx } => {
                self.lower_table_set(*table_idx, args, continuation, trap)?
            }
            primitive => {
                if let Some(spec) = machine_load(primitive) {
                    return Ok(Some(self.lower_memory_load(
                        spec,
                        args,
                        results,
                        continuation,
                        trap,
                    )?));
                }
                if let Some(spec) = machine_store(primitive) {
                    return Ok(Some(self.lower_memory_store(
                        spec,
                        args,
                        continuation,
                        trap,
                    )?));
                }
                return Ok(None);
            }
        };
        Ok(Some(lowered))
    }

    pub(super) fn lower_leaf(
        &mut self,
        op: &LirLeafOp,
        args: &[LirValue],
        results: &[LirValue],
    ) -> Result<(), WasmError> {
        use PrimitiveOpKind as P;
        let primitive = op.primitive();

        match primitive {
            P::Drop | P::Nop => {
                for arg in args {
                    let _ = self.use_value(*arg)?;
                }
                Ok(())
            }
            P::I32Const { value } => self.lower_const(results, *value as u64),
            P::I64Const { value } => self.lower_const(results, *value),
            P::F32Const { value } => {
                self.lower_float_const(results, MachineFloatWidth::F32, u64::from(*value))
            }
            P::F64Const { value } => {
                self.lower_float_const(results, MachineFloatWidth::F64, *value)
            }
            P::RefNull => self.lower_const(results, self.gp_word_max_imm()),
            P::RefFunc { func_idx } => self.lower_const(results, *func_idx as u64),
            P::RefIsNull => self.lower_ref_is_null(args, results),
            P::Select => self.lower_select(args, results),
            primitive => {
                if let Some((width, op)) = machine_int_binary(primitive) {
                    return self.lower_int_binary(args, results, width, op);
                }
                if let Some((width, kind, sign)) = machine_int_compare(primitive) {
                    return self.lower_int_compare(args, results, width, kind, sign);
                }
                if let Some((width, op)) = machine_int_unary(primitive) {
                    return self.lower_int_unary(args, results, width, op);
                }
                if let Some((width, op)) = machine_float_binary(primitive) {
                    return self.lower_float_binary(args, results, width, op);
                }
                if let Some((width, kind)) = machine_float_compare(primitive) {
                    return self.lower_float_compare(args, results, width, kind);
                }
                if let Some((width, op)) = machine_float_unary(primitive) {
                    return self.lower_float_unary(args, results, width, op);
                }
                if let Some(op) = machine_convert(primitive) {
                    return self.lower_convert(args, results, op);
                }

                Err(WasmError::internal(alloc::format!(
                    "primitive {:?} is not lowered to MachineIR yet",
                    primitive
                )))
            }
        }
    }

    fn lower_memory_size(&mut self, mem_idx: u32, results: &[LirValue]) -> Result<(), WasmError> {
        let dst = self.alloc_result_value(single_result(results)?)?;
        if mem_idx == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    ty: crate::vm::native::ir::machine::MachineStorageType::GpWord,
                    dst,
                    src: MachineValue::Reg(self.mem0_size_reg()),
                },
            });
        } else {
            let temp = self.borrow_free_transients(1)?[0];
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    dst: temp,
                    addr: self.runtime_addr(ctx_offset::MEMORY_VIEWS_BASE),
                    width: self.gp_word_mem_width(),
                    extension: MachineLoadExtension::None,
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Load {
                    dst,
                    addr: self.indexed_addr(
                        temp,
                        mem_idx,
                        core::mem::size_of::<NativeMemoryView>(),
                        memory_view_offset::LEN,
                    )?,
                    width: self.gp_word_mem_width(),
                    extension: MachineLoadExtension::None,
                },
            });
        }
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::ShrU,
                dst,
                lhs: MachineValue::Reg(dst),
                rhs: MachineValue::Imm64(crate::constants::WASM_PAGE_SIZE.trailing_zeros() as u64),
            },
        });
        Ok(())
    }

    fn lower_global_get(&mut self, idx: u32, results: &[LirValue]) -> Result<(), WasmError> {
        let result = single_result(results)?;
        let dst = self.alloc_result_value(result)?;
        let width = self.slot_mem_width_for_value(result);
        let base = self.borrow_free_transients(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst: base,
                addr: self.runtime_addr(ctx_offset::GLOBALS_VIEW + globals_view_offset::BASE),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst,
                addr: self.indexed_addr(
                    base,
                    idx,
                    core::mem::size_of::<crate::vm::entities::GlobalInst>(),
                    global_offset::RAW,
                )?,
                width,
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn lower_global_set(&mut self, idx: u32, args: &[LirValue]) -> Result<(), WasmError> {
        let src_value = single_arg(args)?;
        let src = self.use_value(src_value)?;
        let width = self.slot_mem_width_for_value(src_value);
        let base = self.borrow_free_transients(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst: base,
                addr: self.runtime_addr(ctx_offset::GLOBALS_VIEW + globals_view_offset::BASE),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                addr: self.indexed_addr(
                    base,
                    idx,
                    core::mem::size_of::<crate::vm::entities::GlobalInst>(),
                    global_offset::RAW,
                )?,
                width,
                src: MachineValue::Reg(src),
            },
        });
        Ok(())
    }

    fn lower_table_size(&mut self, table_idx: u32, results: &[LirValue]) -> Result<(), WasmError> {
        let dst = self.alloc_result_value(single_result(results)?)?;
        let table_views = self.borrow_free_transients(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst: table_views,
                addr: self.runtime_addr(ctx_offset::TABLE_VIEWS_BASE),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst,
                addr: self.indexed_addr(
                    table_views,
                    table_idx,
                    core::mem::size_of::<NativeTableView>(),
                    table_view_offset::ELEMENTS_LEN,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn lower_table_get(
        &mut self,
        table_idx: u32,
        args: &[LirValue],
        results: &[LirValue],
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        let index = self.use_value(single_arg(args)?)?;
        let dst = self.alloc_result_value(single_result(results)?)?;
        let index64 = dst;
        let table_len = self.borrow_free_transients(1)?[0];
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
        Ok(LeafLowering::Split {
            continuation,
            trap,
            trap_kind: MachineTrapKind::TableOutOfBounds,
            terminator,
            continuation_ops,
        })
    }

    fn lower_table_set(
        &mut self,
        table_idx: u32,
        args: &[LirValue],
        continuation: MachineBlockId,
        trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        let (index_value, src_value) = two_args(args)?;
        let index = self.use_value(index_value)?;
        let src = self.use_value(src_value)?;
        let (index64, table_len) = if let Some(index64) = self.dead_value_reg(index_value) {
            (index64, self.borrow_free_transients(1)?[0])
        } else {
            let scratch = self.borrow_free_transients(2)?;
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

    fn lower_memory_load(
        &mut self,
        spec: MemoryLoadSpec,
        args: &[LirValue],
        results: &[LirValue],
        _continuation: MachineBlockId,
        _trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        let addr_value = single_arg(args)?;
        let addr = self.use_value(addr_value)?;
        let fp_load_usable = spec.float_width.is_some()
            && (spec.memidx != 0
                || self.dead_value_reg(addr_value).is_some()
                || self.borrow_free_transients(1).is_ok());
        let dst = if let Some(width) = spec.float_width.filter(|_| fp_load_usable) {
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
                    self.borrow_free_transients(1)?[0]
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
                spec.width,
                spec.extension,
            )?);
        } else {
            let scratch = self.borrow_free_transients(2)?;
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
                Some((dst, spec.width, spec.extension)),
                None,
            )?);
        }
        Ok(LeafLowering::InPlace)
    }

    fn lower_memory_store(
        &mut self,
        spec: MemoryStoreSpec,
        args: &[LirValue],
        _continuation: MachineBlockId,
        _trap: MachineBlockId,
    ) -> Result<LeafLowering, WasmError> {
        let (addr_value, src_value) = two_args(args)?;
        let addr = self.use_value(addr_value)?;
        let src = self.use_value(src_value)?;
        let access_bytes = spec.access_bytes();
        if spec.memidx == 0 {
            let addr32 = if let Some(addr32) = self.dead_value_reg(addr_value) {
                addr32
            } else {
                self.borrow_free_transients(1)?[0]
            };
            let residual =
                self.emit_mem0_bounds_trap_if(spec.offset, access_bytes, addr, addr32)?;
            self.emit_machine_ops(
                self.lower_mem0_store_continuation(addr32, residual, src, spec.width)?,
            );
        } else {
            let scratch = self.borrow_free_transients(2)?;
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
                None,
                Some((src, spec.width)),
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
        addr: crate::vm::native::ir::machine::MachineReg,
        addr32: crate::vm::native::ir::machine::MachineReg,
        scratch: crate::vm::native::ir::machine::MachineReg,
    ) -> Result<u32, WasmError> {
        self.emit_effective_addr(offset, addr, addr32)?;
        self.emit_memory_len_load(memidx, scratch)?;
        if access_bytes == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Reg(scratch),
                    },
                },
            });
            return Ok(0);
        }
        if let Ok(free) = self.borrow_free_transients(1) {
            let check_reg = free[0];
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                    dst: check_reg,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
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
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Reg(scratch),
                    },
                },
            });
            Ok(access_bytes)
        }
    }

    /// Emit a mem0 bounds check. Returns the number of `access_bytes` still
    /// embedded in `addr32` that the continuation must subtract. When a scratch
    /// transient is available this is 0 (addr32 = zext(addr) + offset). When
    /// register pressure is too high, falls back to adding offset+access_bytes
    /// into addr32 and returns `access_bytes` so the continuation can undo it.
    fn emit_mem0_bounds_trap_if(
        &mut self,
        offset: u32,
        access_bytes: u32,
        addr: crate::vm::native::ir::machine::MachineReg,
        addr32: crate::vm::native::ir::machine::MachineReg,
    ) -> Result<u32, WasmError> {
        self.emit_effective_addr(offset, addr, addr32)?;
        #[cfg(has_guard_pages)]
        if self.use_guard_pages() {
            return Ok(0);
        }
        if access_bytes == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                        lhs: MachineValue::Reg(addr32),
                        rhs: MachineValue::Reg(self.mem0_size_reg()),
                    },
                },
            });
            return Ok(0);
        }
        if let Ok(scratch) = self.borrow_free_transients(1) {
            let check_reg = scratch[0];
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                    dst: check_reg,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                        lhs: MachineValue::Reg(check_reg),
                        rhs: MachineValue::Reg(self.mem0_size_reg()),
                    },
                },
            });
            Ok(0)
        } else {
            // Fallback: add access_bytes into addr32; continuation must sub it back.
            let check_addend = access_bytes as u64;
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(check_addend),
                },
            });
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: self.gp_word_int_width(),
                        kind: MachineCompareKind::Gt,
                        sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
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
        addr32: crate::vm::native::ir::machine::MachineReg,
        scratch: crate::vm::native::ir::machine::MachineReg,
        access_bytes: u32,
        load_dst: Option<(
            crate::vm::native::ir::machine::MachineReg,
            MachineMemWidth,
            MachineLoadExtension,
        )>,
        store_src: Option<(crate::vm::native::ir::machine::MachineReg, MachineMemWidth)>,
    ) -> Result<Vec<MachineInst>, WasmError> {
        let mut ops = Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Sub,
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
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                dst: scratch,
                lhs: MachineValue::Reg(scratch),
                rhs: MachineValue::Reg(addr32),
            },
        });
        if let Some((dst, width, extension)) = load_dst {
            ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    dst,
                    addr: crate::vm::native::ir::machine::MachineAddr {
                        base: scratch,
                        offset: 0,
                    },
                    width,
                    extension,
                },
            });
        }
        if let Some((src, width)) = store_src {
            ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    addr: crate::vm::native::ir::machine::MachineAddr {
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

    fn lower_mem0_load_continuation(
        &self,
        addr32: crate::vm::native::ir::machine::MachineReg,
        access_bytes: u32,
        dst: crate::vm::native::ir::machine::MachineReg,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<Vec<MachineInst>, WasmError> {
        let mut ops = Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                dst: addr32,
                lhs: MachineValue::Reg(self.mem0_base_reg()),
                rhs: MachineValue::Reg(addr32),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                dst,
                addr: crate::vm::native::ir::machine::MachineAddr {
                    base: addr32,
                    offset: 0,
                },
                width,
                extension,
            },
        });
        Ok(ops)
    }

    fn lower_mem0_store_continuation(
        &self,
        addr32: crate::vm::native::ir::machine::MachineReg,
        access_bytes: u32,
        src: crate::vm::native::ir::machine::MachineReg,
        width: MachineMemWidth,
    ) -> Result<Vec<MachineInst>, WasmError> {
        let mut ops = Vec::new();
        if access_bytes != 0 {
            ops.push(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Sub,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(access_bytes as u64),
                },
            });
        }
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                dst: addr32,
                lhs: MachineValue::Reg(self.mem0_base_reg()),
                rhs: MachineValue::Reg(addr32),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Store {
                addr: crate::vm::native::ir::machine::MachineAddr {
                    base: addr32,
                    offset: 0,
                },
                width,
                src: MachineValue::Reg(src),
            },
        });
        Ok(ops)
    }

    fn lower_table_bounds_check(
        &mut self,
        table_idx: u32,
        index: crate::vm::native::ir::machine::MachineReg,
        index64: crate::vm::native::ir::machine::MachineReg,
        table_len: crate::vm::native::ir::machine::MachineReg,
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
                    ty: crate::vm::native::ir::machine::MachineStorageType::GpWord,
                    dst: index64,
                    src: MachineValue::Reg(index),
                },
            });
        }
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst: table_len,
                addr: self.runtime_addr(ctx_offset::TABLE_VIEWS_BASE),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst: table_len,
                addr: self.indexed_addr(
                    table_len,
                    table_idx,
                    core::mem::size_of::<NativeTableView>(),
                    table_view_offset::ELEMENTS_LEN,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        Ok(MachineTerminator::Branch {
            cond: MachineBranchCond::IntCompare {
                width: self.gp_word_int_width(),
                kind: MachineCompareKind::Ge,
                sign: crate::vm::native::ir::machine::MachineSign::Unsigned,
                lhs: MachineValue::Reg(index64),
                rhs: MachineValue::Reg(table_len),
            },
            then_edge: crate::vm::native::ir::machine::MachineEdge {
                target: trap,
                args: Vec::new(),
            },
            else_edge: crate::vm::native::ir::machine::MachineEdge {
                target: continuation,
                args: Vec::new(),
            },
        })
    }

    fn lower_table_access_continuation(
        &self,
        table_idx: u32,
        index: crate::vm::native::ir::machine::MachineReg,
        index64: crate::vm::native::ir::machine::MachineReg,
        scratch: crate::vm::native::ir::machine::MachineReg,
        load_dst: Option<crate::vm::native::ir::machine::MachineReg>,
        store_src: Option<crate::vm::native::ir::machine::MachineReg>,
    ) -> Result<Vec<MachineInst>, WasmError> {
        let mut ops = Vec::new();
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
                    ty: crate::vm::native::ir::machine::MachineStorageType::GpWord,
                    dst: index64,
                    src: MachineValue::Reg(index),
                },
            });
        }
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                dst: scratch,
                addr: self.runtime_addr(ctx_offset::TABLE_VIEWS_BASE),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::Load {
                dst: scratch,
                addr: self.indexed_addr(
                    scratch,
                    table_idx,
                    core::mem::size_of::<NativeTableView>(),
                    table_view_offset::ELEMENTS_BASE,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Mul,
                dst: index64,
                lhs: MachineValue::Reg(index64),
                rhs: MachineValue::Imm64(core::mem::size_of::<crate::vm::value::RefHandle>() as u64),
            },
        });
        ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: self.gp_word_int_width(),
                op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                dst: scratch,
                lhs: MachineValue::Reg(scratch),
                rhs: MachineValue::Reg(index64),
            },
        });
        if let Some(dst) = load_dst {
            ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    dst,
                    addr: crate::vm::native::ir::machine::MachineAddr {
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
                    addr: crate::vm::native::ir::machine::MachineAddr {
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

    fn emit_effective_addr(
        &mut self,
        offset: u32,
        addr: crate::vm::native::ir::machine::MachineReg,
        addr32: crate::vm::native::ir::machine::MachineReg,
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
                    ty: crate::vm::native::ir::machine::MachineStorageType::GpWord,
                    dst: addr32,
                    src: MachineValue::Reg(addr),
                },
            });
        }
        if offset != 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: self.gp_word_int_width(),
                    op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
                    dst: addr32,
                    lhs: MachineValue::Reg(addr32),
                    rhs: MachineValue::Imm64(offset as u64),
                },
            });
        }
        Ok(())
    }

    fn emit_memory_len_load(
        &mut self,
        memidx: u32,
        dst: crate::vm::native::ir::machine::MachineReg,
    ) -> Result<(), WasmError> {
        if memidx == 0 {
            self.emit_machine_inst(MachineInst {
                kind: MachineInstKind::Move {
                    ty: crate::vm::native::ir::machine::MachineStorageType::GpWord,
                    dst,
                    src: MachineValue::Reg(self.mem0_size_reg()),
                },
            });
            return Ok(());
        }

        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst,
                addr: self.runtime_addr(ctx_offset::MEMORY_VIEWS_BASE),
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                dst,
                addr: self.indexed_addr(
                    dst,
                    memidx,
                    core::mem::size_of::<NativeMemoryView>(),
                    memory_view_offset::LEN,
                )?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        Ok(())
    }

    fn indexed_addr(
        &self,
        base: crate::vm::native::ir::machine::MachineReg,
        index: u32,
        stride: usize,
        field_offset: u32,
    ) -> Result<crate::vm::native::ir::machine::MachineAddr, WasmError> {
        let scaled = (index as u64)
            .checked_mul(stride as u64)
            .and_then(|value| value.checked_add(field_offset as u64))
            .ok_or_else(|| WasmError::internal("runtime view byte offset overflow".into()))?;
        let offset = i32::try_from(scaled)
            .map_err(|_| WasmError::internal("runtime view byte offset exceeds i32".into()))?;
        Ok(crate::vm::native::ir::machine::MachineAddr { base, offset })
    }
}

fn machine_int_binary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineIntWidth,
    crate::vm::native::ir::machine::MachineIntBinaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineIntBinaryOp as Op, MachineIntWidth as W};
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Add => (W::I32, Op::Add),
        P::I32Sub => (W::I32, Op::Sub),
        P::I32Mul => (W::I32, Op::Mul),
        P::I32DivS => (W::I32, Op::DivS),
        P::I32DivU => (W::I32, Op::DivU),
        P::I32RemS => (W::I32, Op::RemS),
        P::I32RemU => (W::I32, Op::RemU),
        P::I32And => (W::I32, Op::And),
        P::I32Or => (W::I32, Op::Or),
        P::I32Xor => (W::I32, Op::Xor),
        P::I32Shl => (W::I32, Op::Shl),
        P::I32ShrS => (W::I32, Op::ShrS),
        P::I32ShrU => (W::I32, Op::ShrU),
        P::I32Rotl => (W::I32, Op::Rotl),
        P::I32Rotr => (W::I32, Op::Rotr),
        P::I64Add => (W::I64, Op::Add),
        P::I64Sub => (W::I64, Op::Sub),
        P::I64Mul => (W::I64, Op::Mul),
        P::I64DivS => (W::I64, Op::DivS),
        P::I64DivU => (W::I64, Op::DivU),
        P::I64RemS => (W::I64, Op::RemS),
        P::I64RemU => (W::I64, Op::RemU),
        P::I64And => (W::I64, Op::And),
        P::I64Or => (W::I64, Op::Or),
        P::I64Xor => (W::I64, Op::Xor),
        P::I64Shl => (W::I64, Op::Shl),
        P::I64ShrS => (W::I64, Op::ShrS),
        P::I64ShrU => (W::I64, Op::ShrU),
        P::I64Rotl => (W::I64, Op::Rotl),
        P::I64Rotr => (W::I64, Op::Rotr),
        _ => return None,
    })
}

fn machine_int_compare(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineIntWidth,
    crate::vm::native::ir::machine::MachineCompareKind,
    crate::vm::native::ir::machine::MachineSign,
)> {
    use crate::vm::native::ir::machine::{
        MachineCompareKind as K, MachineIntWidth as W, MachineSign as S,
    };
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Eq => (W::I32, K::Eq, S::Unsigned),
        P::I32Ne => (W::I32, K::Ne, S::Unsigned),
        P::I32LtS => (W::I32, K::Lt, S::Signed),
        P::I32LtU => (W::I32, K::Lt, S::Unsigned),
        P::I32GtS => (W::I32, K::Gt, S::Signed),
        P::I32GtU => (W::I32, K::Gt, S::Unsigned),
        P::I32LeS => (W::I32, K::Le, S::Signed),
        P::I32LeU => (W::I32, K::Le, S::Unsigned),
        P::I32GeS => (W::I32, K::Ge, S::Signed),
        P::I32GeU => (W::I32, K::Ge, S::Unsigned),
        P::I64Eq => (W::I64, K::Eq, S::Unsigned),
        P::I64Ne => (W::I64, K::Ne, S::Unsigned),
        P::I64LtS => (W::I64, K::Lt, S::Signed),
        P::I64LtU => (W::I64, K::Lt, S::Unsigned),
        P::I64GtS => (W::I64, K::Gt, S::Signed),
        P::I64GtU => (W::I64, K::Gt, S::Unsigned),
        P::I64LeS => (W::I64, K::Le, S::Signed),
        P::I64LeU => (W::I64, K::Le, S::Unsigned),
        P::I64GeS => (W::I64, K::Ge, S::Signed),
        P::I64GeU => (W::I64, K::Ge, S::Unsigned),
        _ => return None,
    })
}

fn machine_int_unary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineIntWidth,
    crate::vm::native::ir::machine::MachineIntUnaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineIntUnaryOp as Op, MachineIntWidth as W};
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Eqz => (W::I32, Op::Eqz),
        P::I32Clz => (W::I32, Op::Clz),
        P::I32Ctz => (W::I32, Op::Ctz),
        P::I32Popcnt => (W::I32, Op::Popcnt),
        P::I64Eqz => (W::I64, Op::Eqz),
        P::I64Clz => (W::I64, Op::Clz),
        P::I64Ctz => (W::I64, Op::Ctz),
        P::I64Popcnt => (W::I64, Op::Popcnt),
        P::I32Extend8S => (W::I32, Op::Extend8S),
        P::I32Extend16S => (W::I32, Op::Extend16S),
        P::I64Extend8S => (W::I64, Op::Extend8S),
        P::I64Extend16S => (W::I64, Op::Extend16S),
        P::I64Extend32S => (W::I64, Op::Extend32S),
        _ => return None,
    })
}

fn machine_float_binary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineFloatWidth,
    crate::vm::native::ir::machine::MachineFloatBinaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineFloatBinaryOp as Op, MachineFloatWidth as W};
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::F32Add => (W::F32, Op::Add),
        P::F32Sub => (W::F32, Op::Sub),
        P::F32Mul => (W::F32, Op::Mul),
        P::F32Div => (W::F32, Op::Div),
        P::F32Min => (W::F32, Op::Min),
        P::F32Max => (W::F32, Op::Max),
        P::F32Copysign => (W::F32, Op::Copysign),
        P::F64Add => (W::F64, Op::Add),
        P::F64Sub => (W::F64, Op::Sub),
        P::F64Mul => (W::F64, Op::Mul),
        P::F64Div => (W::F64, Op::Div),
        P::F64Min => (W::F64, Op::Min),
        P::F64Max => (W::F64, Op::Max),
        P::F64Copysign => (W::F64, Op::Copysign),
        _ => return None,
    })
}

fn machine_float_compare(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineFloatWidth,
    crate::vm::native::ir::machine::MachineCompareKind,
)> {
    use crate::vm::native::ir::machine::{MachineCompareKind as K, MachineFloatWidth as W};
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::F32Eq => (W::F32, K::Eq),
        P::F32Ne => (W::F32, K::Ne),
        P::F32Lt => (W::F32, K::Lt),
        P::F32Gt => (W::F32, K::Gt),
        P::F32Le => (W::F32, K::Le),
        P::F32Ge => (W::F32, K::Ge),
        P::F64Eq => (W::F64, K::Eq),
        P::F64Ne => (W::F64, K::Ne),
        P::F64Lt => (W::F64, K::Lt),
        P::F64Gt => (W::F64, K::Gt),
        P::F64Le => (W::F64, K::Le),
        P::F64Ge => (W::F64, K::Ge),
        _ => return None,
    })
}

fn machine_float_unary(
    primitive: &PrimitiveOpKind,
) -> Option<(
    crate::vm::native::ir::machine::MachineFloatWidth,
    crate::vm::native::ir::machine::MachineFloatUnaryOp,
)> {
    use crate::vm::native::ir::machine::{MachineFloatUnaryOp as Op, MachineFloatWidth as W};
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::F32Abs => (W::F32, Op::Abs),
        P::F32Neg => (W::F32, Op::Neg),
        P::F32Ceil => (W::F32, Op::Ceil),
        P::F32Floor => (W::F32, Op::Floor),
        P::F32Trunc => (W::F32, Op::Trunc),
        P::F32Nearest => (W::F32, Op::Nearest),
        P::F32Sqrt => (W::F32, Op::Sqrt),
        P::F64Abs => (W::F64, Op::Abs),
        P::F64Neg => (W::F64, Op::Neg),
        P::F64Ceil => (W::F64, Op::Ceil),
        P::F64Floor => (W::F64, Op::Floor),
        P::F64Trunc => (W::F64, Op::Trunc),
        P::F64Nearest => (W::F64, Op::Nearest),
        P::F64Sqrt => (W::F64, Op::Sqrt),
        _ => return None,
    })
}

fn machine_convert(
    primitive: &PrimitiveOpKind,
) -> Option<crate::vm::native::ir::machine::MachineConvertOp> {
    use crate::vm::native::ir::machine::MachineConvertOp as Op;
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32WrapI64 => Op::I32WrapI64,
        P::I64ExtendI32S => Op::I64ExtendI32S,
        P::I64ExtendI32U => Op::I64ExtendI32U,
        P::I32TruncF32S => Op::I32TruncF32S,
        P::I32TruncF32U => Op::I32TruncF32U,
        P::I32TruncF64S => Op::I32TruncF64S,
        P::I32TruncF64U => Op::I32TruncF64U,
        P::I64TruncF32S => Op::I64TruncF32S,
        P::I64TruncF32U => Op::I64TruncF32U,
        P::I64TruncF64S => Op::I64TruncF64S,
        P::I64TruncF64U => Op::I64TruncF64U,
        P::I32TruncSatF32S => Op::I32TruncSatF32S,
        P::I32TruncSatF32U => Op::I32TruncSatF32U,
        P::I32TruncSatF64S => Op::I32TruncSatF64S,
        P::I32TruncSatF64U => Op::I32TruncSatF64U,
        P::I64TruncSatF32S => Op::I64TruncSatF32S,
        P::I64TruncSatF32U => Op::I64TruncSatF32U,
        P::I64TruncSatF64S => Op::I64TruncSatF64S,
        P::I64TruncSatF64U => Op::I64TruncSatF64U,
        P::F32ConvertI32S => Op::F32ConvertI32S,
        P::F32ConvertI32U => Op::F32ConvertI32U,
        P::F32ConvertI64S => Op::F32ConvertI64S,
        P::F32ConvertI64U => Op::F32ConvertI64U,
        P::F64ConvertI32S => Op::F64ConvertI32S,
        P::F64ConvertI32U => Op::F64ConvertI32U,
        P::F64ConvertI64S => Op::F64ConvertI64S,
        P::F64ConvertI64U => Op::F64ConvertI64U,
        P::F32DemoteF64 => Op::F32DemoteF64,
        P::F64PromoteF32 => Op::F64PromoteF32,
        P::I32ReinterpretF32 => Op::I32ReinterpretF32,
        P::I64ReinterpretF64 => Op::I64ReinterpretF64,
        P::F32ReinterpretI32 => Op::F32ReinterpretI32,
        P::F64ReinterpretI64 => Op::F64ReinterpretI64,
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct MemoryLoadSpec {
    memidx: u32,
    offset: u32,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
    float_width: Option<MachineFloatWidth>,
}

impl MemoryLoadSpec {
    #[inline]
    fn access_bytes(self) -> u32 {
        mem_width_bytes(self.width)
    }
}

#[derive(Clone, Copy)]
struct MemoryStoreSpec {
    memidx: u32,
    offset: u32,
    width: MachineMemWidth,
}

impl MemoryStoreSpec {
    #[inline]
    fn access_bytes(self) -> u32 {
        mem_width_bytes(self.width)
    }
}

fn machine_load(primitive: &PrimitiveOpKind) -> Option<MemoryLoadSpec> {
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
            float_width: None,
        },
        P::I64Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
            float_width: None,
        },
        P::F32Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
            float_width: Some(MachineFloatWidth::F32),
        },
        P::F64Load { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U64,
            extension: MachineLoadExtension::None,
            float_width: Some(MachineFloatWidth::F64),
        },
        P::I32Load8S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::SignExtend,
            float_width: None,
        },
        P::I32Load8U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::ZeroExtend,
            float_width: None,
        },
        P::I32Load16S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::SignExtend,
            float_width: None,
        },
        P::I32Load16U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::ZeroExtend,
            float_width: None,
        },
        P::I64Load8S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::SignExtend,
            float_width: None,
        },
        P::I64Load8U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::ZeroExtend,
            float_width: None,
        },
        P::I64Load16S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::SignExtend,
            float_width: None,
        },
        P::I64Load16U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U16,
            extension: MachineLoadExtension::ZeroExtend,
            float_width: None,
        },
        P::I64Load32S { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::SignExtend,
            float_width: None,
        },
        P::I64Load32U { offset, memidx } => MemoryLoadSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U32,
            extension: MachineLoadExtension::ZeroExtend,
            float_width: None,
        },
        _ => return None,
    })
}

fn machine_store(primitive: &PrimitiveOpKind) -> Option<MemoryStoreSpec> {
    use PrimitiveOpKind as P;

    Some(match primitive {
        P::I32Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U32,
        },
        P::I64Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U64,
        },
        P::F32Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U32,
        },
        P::F64Store { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U64,
        },
        P::I32Store8 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U8,
        },
        P::I32Store16 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U16,
        },
        P::I64Store8 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U8,
        },
        P::I64Store16 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U16,
        },
        P::I64Store32 { offset, memidx } => MemoryStoreSpec {
            memidx: *memidx,
            offset: *offset,
            width: MachineMemWidth::U32,
        },
        _ => return None,
    })
}

fn mem_width_bytes(width: MachineMemWidth) -> u32 {
    match width {
        MachineMemWidth::U8 => 1,
        MachineMemWidth::U16 => 2,
        MachineMemWidth::U32 => 4,
        MachineMemWidth::U64 => 8,
    }
}

fn emit_effective_addr_ops(
    ops: &mut Vec<MachineInst>,
    offset: u32,
    addr: crate::vm::native::ir::machine::MachineReg,
    addr32: crate::vm::native::ir::machine::MachineReg,
    gp_reg_width: u8,
) {
    if gp_reg_width == 8 {
        ops.push(MachineInst {
            kind: MachineInstKind::Convert {
                op: MachineConvertOp::I64ExtendI32U,
                dst: addr32,
                src: MachineValue::Reg(addr),
            },
        });
    } else if addr32 != addr {
        ops.push(MachineInst {
            kind: MachineInstKind::Move {
                ty: crate::vm::native::ir::machine::MachineStorageType::GpWord,
                dst: addr32,
                src: MachineValue::Reg(addr),
            },
        });
    }
    ops.push(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: crate::vm::native::ir::machine::machine_word_int_width(gp_reg_width),
            op: crate::vm::native::ir::machine::MachineIntBinaryOp::Add,
            dst: addr32,
            lhs: MachineValue::Reg(addr32),
            rhs: MachineValue::Imm64(offset as u64),
        },
    });
}

fn emit_memory_base_load_ops(
    ops: &mut Vec<MachineInst>,
    runtime_base: crate::vm::native::ir::machine::MachineReg,
    memidx: u32,
    dst: crate::vm::native::ir::machine::MachineReg,
    gp_reg_width: u8,
) -> Result<(), WasmError> {
    if memidx == 0 {
        ops.push(MachineInst {
            kind: MachineInstKind::Move {
                ty: crate::vm::native::ir::machine::MachineStorageType::GpWord,
                dst,
                src: MachineValue::Reg(crate::vm::native::ir::machine::MACHINE_MEM0_BASE_REG),
            },
        });
        return Ok(());
    }

    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            dst,
            addr: crate::vm::native::ir::machine::MachineAddr {
                base: runtime_base,
                offset: ctx_offset::MEMORY_VIEWS_BASE as i32,
            },
            width: machine_ptr_width(gp_reg_width),
            extension: MachineLoadExtension::None,
        },
    });
    ops.push(MachineInst {
        kind: MachineInstKind::Load {
            dst,
            addr: crate::vm::native::ir::machine::MachineAddr {
                base: dst,
                offset: indexed_field_offset(
                    memidx,
                    core::mem::size_of::<NativeMemoryView>(),
                    memory_view_offset::BASE,
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
        .ok_or_else(|| WasmError::internal("runtime view byte offset overflow".into()))?;
    i32::try_from(scaled)
        .map_err(|_| WasmError::internal("runtime view byte offset exceeds i32".into()))
}
