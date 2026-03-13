use alloc::vec::Vec;
use core::mem;

use crate::{
    error::WasmError,
    vm::{
        lir::{
            ir::{
                LirBlock, LirBoundaryOp, LirEdge, LirInst, LirInstKind, LirLocalCachePrefs,
                LirProgram, LirTerminator, LirValue,
            },
            leaf::LirLeafOp,
            slot::FrameSlot,
        },
        native::{
            helper::meta::{
                CallExternalMeta, DataDropMeta, ElemDropMeta, HelperFrameRegion, MemoryGrowMeta,
                MemoryInitMeta, TableGrowMeta, TableInitMeta,
            },
            ir::machine::{
                MachineAddr, MachineBlockId, MachineBranchCond, MachineEdge, MachineFuncId,
                MachineInst, MachineInstKind, MachineTerminator, MachineTrapKind,
                MachineValue, MachineMemWidth, MachineLoadExtension, MachineReg,
            },
            ir::runtime::{
                MachineCallLinkLayout, MachineFrameRegion, MachineFunctionRuntime,
                MachineHelperSymbol,
            },
            lower::{slot_offset_bytes, target_param_regs},
        },
        plan::frame::{FrameLayoutPlan, FrameSpan},
    },
};

use super::{regfile::MachineRegFile, sidecar::SidecarBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CachedLocal {
    slot: FrameSlot,
    reg: MachineReg,
    dirty: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueLocation {
    value: LirValue,
    reg: MachineReg,
}

pub(super) struct BlockLowerContext<'a> {
    regfile: &'a MachineRegFile,
    frame: FrameLayoutPlan,
    program: &'a LirProgram,
    block: &'a LirBlock,
    runtime: MachineFunctionRuntime,
    all_runtime: &'a [MachineFunctionRuntime],
    call_link: MachineCallLinkLayout,
    machine_params: Vec<MachineReg>,
    ops: Vec<MachineInst>,
    cached_locals: Vec<CachedLocal>,
    values: Vec<ValueLocation>,
    remaining_uses: alloc::collections::BTreeMap<LirValue, u32>,
    transient_owner: Vec<Option<LirValue>>,
}

impl<'a> BlockLowerContext<'a> {
    pub(super) fn new(
        regfile: &'a MachineRegFile,
        frame: FrameLayoutPlan,
        program: &'a LirProgram,
        cache_prefs: &LirLocalCachePrefs,
        block: &'a LirBlock,
        runtime: MachineFunctionRuntime,
        all_runtime: &'a [MachineFunctionRuntime],
        call_link: MachineCallLinkLayout,
        is_entry: bool,
    ) -> Result<Self, WasmError> {
        let machine_params = target_param_regs(block.params.len(), regfile)?;
        let mut cached_locals = Vec::new();
        for (index, slot) in cache_prefs.preferred_slots.iter().copied().enumerate() {
            let Some(reg) = regfile.local_cache(index) else {
                break;
            };
            cached_locals.push(CachedLocal {
                slot,
                reg,
                dirty: false,
            });
        }

        let mut lower = Self {
            regfile,
            frame,
            program,
            block,
            runtime,
            all_runtime,
            call_link,
            machine_params,
            ops: Vec::new(),
            cached_locals,
            values: Vec::new(),
            remaining_uses: compute_remaining_uses(block),
            transient_owner: alloc::vec![None; regfile.transient_count()],
        };

        let machine_params = lower.machine_params.clone();
        for (param, reg) in block.params.iter().copied().zip(machine_params.into_iter()) {
            lower.values.push(ValueLocation { value: param, reg });
            lower.mark_transient(reg, Some(param))?;
        }
        lower.release_dead_values()?;

        if is_entry {
            lower.emit_reload_cached_locals()?;
        }

        Ok(lower)
    }

    pub(super) fn machine_params(&self) -> &[MachineReg] {
        &self.machine_params
    }

    pub(super) fn take_ops(&mut self) -> Vec<MachineInst> {
        mem::take(&mut self.ops)
    }

    pub(super) fn lower_ops(&mut self) -> Result<(), WasmError> {
        for inst in &self.block.ops {
            self.lower_inst(inst)?;
        }
        Ok(())
    }

    pub(super) fn lower_terminator(&mut self) -> Result<MachineTerminator, WasmError> {
        match &self.block.terminator {
            LirTerminator::Goto(edge) => Ok(MachineTerminator::Jump(self.lower_edge(edge)?)),
            LirTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                let cond = self.use_value(*cond)?;
                self.release_dead_values()?;
                Ok(MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Reg(cond)),
                    then_edge: self.lower_edge(then_edge)?,
                    else_edge: self.lower_edge(else_edge)?,
                })
            }
            LirTerminator::BrTable { index, entries } => {
                let index = self.use_value(*index)?;
                self.release_dead_values()?;
                let mut lowered = Vec::with_capacity(entries.len());
                for edge in entries {
                    lowered.push(self.lower_edge(edge)?);
                }
                Ok(MachineTerminator::JumpTable {
                    index: MachineValue::Reg(index),
                    entries: lowered,
                })
            }
            LirTerminator::Return { .. } => {
                if !self.values.is_empty() {
                    return Err(WasmError::internal(
                        "LIR return reached native lowering with live transient SSA values; results must be published before return".into(),
                    ));
                }
                Ok(MachineTerminator::Return)
            }
            LirTerminator::TrapUnreachable => Ok(MachineTerminator::Trap {
                kind: MachineTrapKind::Unreachable,
            }),
        }
    }

    pub(super) fn begin_continuation_block(&mut self) -> Result<(), WasmError> {
        self.emit_reload_cached_locals()
    }

    pub(super) fn lower_inst(&mut self, inst: &LirInst) -> Result<(), WasmError> {
        match &inst.kind {
            LirInstKind::LoadSlot { slot, dst } => {
                let dst_reg = self.alloc_value(*dst)?;
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    self.ops.push(MachineInst {
                        kind: MachineInstKind::Move {
                            dst: dst_reg,
                            src: MachineValue::Reg(self.cached_locals[cached_index].reg),
                        },
                    });
                } else {
                    self.ops.push(MachineInst {
                        kind: MachineInstKind::Load {
                            dst: dst_reg,
                            addr: self.frame_addr(*slot)?,
                            width: MachineMemWidth::U64,
                            extension: MachineLoadExtension::None,
                        },
                    });
                }
            }
            LirInstKind::StoreSlot { slot, src } => {
                let src_reg = self.use_value(*src)?;
                if let Some(cached_index) = self.cached_local_index(*slot) {
                    self.ops.push(MachineInst {
                        kind: MachineInstKind::Move {
                            dst: self.cached_locals[cached_index].reg,
                            src: MachineValue::Reg(src_reg),
                        },
                    });
                    self.cached_locals[cached_index].dirty = true;
                } else {
                    self.ops.push(MachineInst {
                        kind: MachineInstKind::Store {
                            addr: self.frame_addr(*slot)?,
                            width: MachineMemWidth::U64,
                            src: MachineValue::Reg(src_reg),
                        },
                    });
                }
                self.release_dead_values()?;
            }
            LirInstKind::Value { op, args, results } => {
                self.lower_leaf(op, args, results)?;
                self.release_dead_values()?;
            }
            LirInstKind::Boundary(boundary) => {
                return Err(WasmError::internal(alloc::format!(
                    "boundary op {:?} must be lowered through its specialized native path",
                    boundary
                )));
            }
        }
        Ok(())
    }

    pub(super) fn lower_call_internal(
        &mut self,
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
        continuation: MachineBlockId,
    ) -> Result<MachineTerminator, WasmError> {
        if !self.values.is_empty() {
            return Err(WasmError::internal(
                "prepared LIR call reached native lowering with live transient SSA values; values must be published before the call".into(),
            ));
        }

        let callee_id = MachineFuncId(callee);
        let callee_runtime = self.runtime_for_func(callee_id)?;
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("direct local call requires callee call scratch".into())
        })?;
        if call_scratch.slots < self.call_link.slot_count {
            return Err(WasmError::internal(
                "callee call scratch is smaller than the machine call-link layout".into(),
            ));
        }
        if args.count > callee_runtime.frame_prefix_slots {
            return Err(WasmError::internal(
                "direct local call passes more arguments than fit in the callee local prefix".into(),
            ));
        }
        let callee_results = callee_runtime
            .return_results
            .map(|region| region.slots)
            .unwrap_or(0);
        if callee_results != results.count {
            return Err(WasmError::internal(
                "direct local call result span does not match the callee return-result contract".into(),
            ));
        }

        self.emit_flush_dirty_cached_locals()?;

        let callee_frame_base = self
            .regfile
            .temp(0)
            .ok_or_else(|| WasmError::internal("direct local call requires one temp register".into()))?;
        let copy_reg = self
            .regfile
            .transient(0)
            .ok_or_else(|| WasmError::internal("direct local call requires one transient register".into()))?;
        if self.transient_owner.first().copied().flatten().is_some() {
            return Err(WasmError::internal(
                "direct local call requires a free transient register for argument copies".into(),
            ));
        }

        self.ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: super::super::ir::machine::MachineIntWidth::I64,
                op: super::super::ir::machine::MachineIntBinaryOp::Add,
                dst: callee_frame_base,
                lhs: MachineValue::Reg(self.regfile.frame_base()),
                rhs: MachineValue::Imm64(
                    slot_offset_bytes(FrameSlot(self.runtime.total_frame_slots))? as u64,
                ),
            },
        });

        for offset in 0..args.count as usize {
            let src_slot = args.start.advance(offset as u16);
            let dst_slot = FrameSlot(offset as u16);
            self.ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    dst: copy_reg,
                    addr: self.frame_addr(src_slot)?,
                    width: MachineMemWidth::U64,
                    extension: MachineLoadExtension::None,
                },
            });
            self.ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    addr: self.frame_addr_from(callee_frame_base, dst_slot)?,
                    width: MachineMemWidth::U64,
                    src: MachineValue::Reg(copy_reg),
                },
            });
        }

        for slot in args.count..callee_runtime.frame_prefix_slots {
            self.ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    addr: self.frame_addr_from(callee_frame_base, FrameSlot(slot))?,
                    width: MachineMemWidth::U64,
                    src: MachineValue::Imm64(0),
                },
            });
        }

        self.store_call_link(callee_frame_base, call_scratch, continuation, results)?;

        Ok(MachineTerminator::CallDirect {
            callee: callee_id,
            callee_frame_base,
            continuation,
        })
    }

    pub(super) fn lower_call_external(
        &mut self,
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
        sidecar: &mut SidecarBuilder,
    ) -> Result<(), WasmError> {
        if !self.values.is_empty() {
            return Err(WasmError::internal(
                "prepared LIR external call reached native lowering with live transient SSA values; values must be published before the call".into(),
            ));
        }

        let target = sidecar.extern_target(MachineHelperSymbol::CallExternal);
        let metadata = sidecar.call_external_meta(CallExternalMeta {
            func_idx,
            args: args.into(),
            results: results.into(),
        });
        self.ops.push(MachineInst {
            kind: MachineInstKind::CallHelper(
                crate::vm::native::ir::machine::MachineHelperCall { target, metadata },
            ),
        });
        Ok(())
    }

    pub(super) fn lower_runtime(
        &mut self,
        boundary: &LirBoundaryOp,
        sidecar: &mut SidecarBuilder,
    ) -> Result<(), WasmError> {
        if !self.values.is_empty() {
            return Err(WasmError::internal(
                "prepared LIR runtime boundary reached native lowering with live transient SSA values; values must be published before the boundary".into(),
            ));
        }

        let (target, metadata) = self.runtime_call_site(boundary, sidecar)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::CallHelper(
                crate::vm::native::ir::machine::MachineHelperCall { target, metadata },
            ),
        });
        Ok(())
    }

    fn lower_leaf(
        &mut self,
        op: &LirLeafOp,
        args: &[LirValue],
        results: &[LirValue],
    ) -> Result<(), WasmError> {
        use crate::vm::wasm::primitive_op::PrimitiveOpKind as P;
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
            P::F32Const { value } => self.lower_const(results, *value as u64),
            P::F64Const { value } => self.lower_const(results, *value),
            P::RefNull => self.lower_const(results, 0),
            P::RefFunc { func_idx } => self.lower_const(results, *func_idx as u64),
            P::I32Add => self.lower_int_binary(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineIntBinaryOp::Add),
            P::I32Sub => self.lower_int_binary(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineIntBinaryOp::Sub),
            P::I32Mul => self.lower_int_binary(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineIntBinaryOp::Mul),
            P::I64Add => self.lower_int_binary(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineIntBinaryOp::Add),
            P::I64Sub => self.lower_int_binary(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineIntBinaryOp::Sub),
            P::I64Mul => self.lower_int_binary(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineIntBinaryOp::Mul),
            P::F32Add => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineFloatBinaryOp::Add),
            P::F32Sub => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineFloatBinaryOp::Sub),
            P::F32Mul => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineFloatBinaryOp::Mul),
            P::F32Div => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineFloatBinaryOp::Div),
            P::F64Add => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineFloatBinaryOp::Add),
            P::F64Sub => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineFloatBinaryOp::Sub),
            P::F64Mul => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineFloatBinaryOp::Mul),
            P::F64Div => self.lower_float_binary(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineFloatBinaryOp::Div),
            P::I32Eq => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Eq, super::super::ir::machine::MachineSign::Unsigned),
            P::I32Ne => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Ne, super::super::ir::machine::MachineSign::Unsigned),
            P::I32LtS => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Lt, super::super::ir::machine::MachineSign::Signed),
            P::I32LtU => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Lt, super::super::ir::machine::MachineSign::Unsigned),
            P::I32GtS => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Gt, super::super::ir::machine::MachineSign::Signed),
            P::I32GtU => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Gt, super::super::ir::machine::MachineSign::Unsigned),
            P::I32LeS => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Le, super::super::ir::machine::MachineSign::Signed),
            P::I32LeU => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Le, super::super::ir::machine::MachineSign::Unsigned),
            P::I32GeS => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Ge, super::super::ir::machine::MachineSign::Signed),
            P::I32GeU => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineCompareKind::Ge, super::super::ir::machine::MachineSign::Unsigned),
            P::I64Eq => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineCompareKind::Eq, super::super::ir::machine::MachineSign::Unsigned),
            P::I64Ne => self.lower_int_compare(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineCompareKind::Ne, super::super::ir::machine::MachineSign::Unsigned),
            P::F32Eq => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineCompareKind::Eq),
            P::F32Ne => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineCompareKind::Ne),
            P::F32Lt => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineCompareKind::Lt),
            P::F32Gt => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineCompareKind::Gt),
            P::F32Le => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineCompareKind::Le),
            P::F32Ge => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F32, super::super::ir::machine::MachineCompareKind::Ge),
            P::F64Eq => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineCompareKind::Eq),
            P::F64Ne => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineCompareKind::Ne),
            P::F64Lt => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineCompareKind::Lt),
            P::F64Gt => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineCompareKind::Gt),
            P::F64Le => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineCompareKind::Le),
            P::F64Ge => self.lower_float_compare(args, results, super::super::ir::machine::MachineFloatWidth::F64, super::super::ir::machine::MachineCompareKind::Ge),
            P::I32Eqz => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineIntUnaryOp::Eqz),
            P::I64Eqz => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineIntUnaryOp::Eqz),
            P::I32Clz => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineIntUnaryOp::Clz),
            P::I32Ctz => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineIntUnaryOp::Ctz),
            P::I32Popcnt => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I32, super::super::ir::machine::MachineIntUnaryOp::Popcnt),
            P::I64Clz => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineIntUnaryOp::Clz),
            P::I64Ctz => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineIntUnaryOp::Ctz),
            P::I64Popcnt => self.lower_int_unary(args, results, super::super::ir::machine::MachineIntWidth::I64, super::super::ir::machine::MachineIntUnaryOp::Popcnt),
            P::Select => self.lower_select(args, results),
            P::I32WrapI64 => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I32WrapI64),
            P::I64ExtendI32S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I64ExtendI32S),
            P::I64ExtendI32U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I64ExtendI32U),
            P::I32TruncF32S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I32TruncF32S),
            P::I32TruncF32U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I32TruncF32U),
            P::I32TruncF64S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I32TruncF64S),
            P::I32TruncF64U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I32TruncF64U),
            P::I64TruncF32S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I64TruncF32S),
            P::I64TruncF32U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I64TruncF32U),
            P::I64TruncF64S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I64TruncF64S),
            P::I64TruncF64U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I64TruncF64U),
            P::F32ConvertI32S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F32ConvertI32S),
            P::F32ConvertI32U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F32ConvertI32U),
            P::F32ConvertI64S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F32ConvertI64S),
            P::F32ConvertI64U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F32ConvertI64U),
            P::F64ConvertI32S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F64ConvertI32S),
            P::F64ConvertI32U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F64ConvertI32U),
            P::F64ConvertI64S => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F64ConvertI64S),
            P::F64ConvertI64U => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F64ConvertI64U),
            P::F32DemoteF64 => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F32DemoteF64),
            P::F64PromoteF32 => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F64PromoteF32),
            P::I32ReinterpretF32 => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I32ReinterpretF32),
            P::I64ReinterpretF64 => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::I64ReinterpretF64),
            P::F32ReinterpretI32 => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F32ReinterpretI32),
            P::F64ReinterpretI64 => self.lower_convert(args, results, super::super::ir::machine::MachineConvertOp::F64ReinterpretI64),
            unsupported => Err(WasmError::internal(alloc::format!(
                "primitive {:?} is not lowered to MachineIR yet",
                unsupported
            ))),
        }
    }

    fn runtime_call_site(
        &self,
        boundary: &LirBoundaryOp,
        sidecar: &mut SidecarBuilder,
    ) -> Result<(crate::vm::native::ir::machine::MachineExternId, crate::vm::native::ir::machine::MachineConstId), WasmError> {
        use MachineHelperSymbol as H;

        let pair = match boundary {
            LirBoundaryOp::MemoryGrow { mem_idx, io } => {
                (
                    sidecar.extern_target(H::MemoryGrow),
                    sidecar.memory_grow_meta(MemoryGrowMeta {
                        mem_idx: *mem_idx,
                        io: (*io).into(),
                    }),
                )
            }
            LirBoundaryOp::TableGrow {
                table_idx,
                args,
                results,
            } => {
                (
                    sidecar.extern_target(H::TableGrow),
                    sidecar.table_grow_meta(TableGrowMeta {
                        table_idx: *table_idx,
                        args: span_region_with_slots(*args, 2, "table.grow args")?,
                        results: span_region_with_slots(*results, 1, "table.grow results")?,
                    }),
                )
            }
            LirBoundaryOp::MemoryInit {
                data_idx,
                mem_idx,
                args,
            } => {
                (
                    sidecar.extern_target(H::MemoryInit),
                    sidecar.memory_init_meta(MemoryInitMeta {
                        data_idx: *data_idx,
                        mem_idx: *mem_idx,
                        args: span_region_with_slots(*args, 3, "memory.init args")?,
                    }),
                )
            }
            LirBoundaryOp::DataDrop { data_idx } => (
                sidecar.extern_target(H::DataDrop),
                sidecar.data_drop_meta(DataDropMeta { data_idx: *data_idx }),
            ),
            LirBoundaryOp::TableInit {
                elem_idx,
                table_idx,
                args,
            } => {
                (
                    sidecar.extern_target(H::TableInit),
                    sidecar.table_init_meta(TableInitMeta {
                        elem_idx: *elem_idx,
                        table_idx: *table_idx,
                        args: span_region_with_slots(*args, 3, "table.init args")?,
                    }),
                )
            }
            LirBoundaryOp::ElemDrop { elem_idx } => (
                sidecar.extern_target(H::ElemDrop),
                sidecar.elem_drop_meta(ElemDropMeta { elem_idx: *elem_idx }),
            ),
            _ => {
                return Err(WasmError::internal(
                    "non-runtime boundary reached runtime helper lowering".into(),
                ))
            }
        };
        Ok(pair)
    }

    fn lower_const(&mut self, results: &[LirValue], imm: u64) -> Result<(), WasmError> {
        let dst = single_result(results)?;
        let dst_reg = self.alloc_value(dst)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::Move {
                dst: dst_reg,
                src: MachineValue::Imm64(imm),
            },
        });
        Ok(())
    }

    fn lower_int_unary(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineIntWidth,
        op: super::super::ir::machine::MachineIntUnaryOp,
    ) -> Result<(), WasmError> {
        let src = self.use_value(single_arg(args)?)?;
        let dst = self.alloc_value(single_result(results)?)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src: MachineValue::Reg(src),
            },
        });
        Ok(())
    }

    fn lower_int_binary(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineIntWidth,
        op: super::super::ir::machine::MachineIntBinaryOp,
    ) -> Result<(), WasmError> {
        let lhs = self.use_value(two_args(args)?.0)?;
        let rhs = self.use_value(two_args(args)?.1)?;
        let dst = self.alloc_value(single_result(results)?)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    fn lower_int_compare(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineIntWidth,
        kind: super::super::ir::machine::MachineCompareKind,
        sign: super::super::ir::machine::MachineSign,
    ) -> Result<(), WasmError> {
        let (lhs_value, rhs_value) = two_args(args)?;
        let lhs = self.use_value(lhs_value)?;
        let rhs = self.use_value(rhs_value)?;
        let dst = self.alloc_value(single_result(results)?)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    fn lower_float_binary(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineFloatWidth,
        op: super::super::ir::machine::MachineFloatBinaryOp,
    ) -> Result<(), WasmError> {
        let (lhs_value, rhs_value) = two_args(args)?;
        let lhs = self.use_value(lhs_value)?;
        let rhs = self.use_value(rhs_value)?;
        let dst = self.alloc_value(single_result(results)?)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    fn lower_float_compare(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        width: super::super::ir::machine::MachineFloatWidth,
        kind: super::super::ir::machine::MachineCompareKind,
    ) -> Result<(), WasmError> {
        let (lhs_value, rhs_value) = two_args(args)?;
        let lhs = self.use_value(lhs_value)?;
        let rhs = self.use_value(rhs_value)?;
        let dst = self.alloc_value(single_result(results)?)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs: MachineValue::Reg(rhs),
            },
        });
        Ok(())
    }

    fn lower_convert(
        &mut self,
        args: &[LirValue],
        results: &[LirValue],
        op: super::super::ir::machine::MachineConvertOp,
    ) -> Result<(), WasmError> {
        let src = self.use_value(single_arg(args)?)?;
        let dst = self.alloc_value(single_result(results)?)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::Convert {
                op,
                dst,
                src: MachineValue::Reg(src),
            },
        });
        Ok(())
    }

    fn lower_select(&mut self, args: &[LirValue], results: &[LirValue]) -> Result<(), WasmError> {
        if args.len() != 3 {
            return Err(WasmError::internal("select expects three arguments".into()));
        }
        let on_false = self.use_value(args[0])?;
        let on_true = self.use_value(args[1])?;
        let cond = self.use_value(args[2])?;
        let dst = self.alloc_value(single_result(results)?)?;
        self.ops.push(MachineInst {
            kind: MachineInstKind::Select {
                dst,
                on_true: MachineValue::Reg(on_true),
                on_false: MachineValue::Reg(on_false),
                cond: MachineValue::Reg(cond),
            },
        });
        Ok(())
    }

    fn lower_edge(&self, edge: &LirEdge) -> Result<MachineEdge, WasmError> {
        let target_block = MachineBlockId(edge.target.as_u32());
        let target = self
            .program
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| WasmError::internal("edge target is out of range during native lowering".into()))?;
        let mut args = Vec::with_capacity(target.params.len());
        for target_param in &target.params {
            let binding = edge
                .bindings
                .iter()
                .find(|binding| binding.param == *target_param)
                .ok_or_else(|| WasmError::internal("missing LIR edge binding for target param during native lowering".into()))?;
            args.push(MachineValue::Reg(self.value_reg(binding.value)?));
        }
        Ok(MachineEdge {
            target: target_block,
            args,
        })
    }

    fn emit_reload_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals.len() {
            let reg = self.cached_locals[index].reg;
            let slot = self.cached_locals[index].slot;
            self.ops.push(MachineInst {
                kind: MachineInstKind::Load {
                    dst: reg,
                    addr: self.frame_addr(slot)?,
                    width: MachineMemWidth::U64,
                    extension: MachineLoadExtension::None,
                },
            });
            self.cached_locals[index].dirty = false;
        }
        Ok(())
    }

    fn emit_flush_dirty_cached_locals(&mut self) -> Result<(), WasmError> {
        for index in 0..self.cached_locals.len() {
            if !self.cached_locals[index].dirty {
                continue;
            }
            let cached = self.cached_locals[index];
            self.ops.push(MachineInst {
                kind: MachineInstKind::Store {
                    addr: self.frame_addr(cached.slot)?,
                    width: MachineMemWidth::U64,
                    src: MachineValue::Reg(cached.reg),
                },
            });
            self.cached_locals[index].dirty = false;
        }
        Ok(())
    }

    fn cached_local_index(&self, slot: FrameSlot) -> Option<usize> {
        self.cached_locals
            .iter()
            .position(|cached| cached.slot == slot)
    }

    fn frame_addr(&self, slot: FrameSlot) -> Result<MachineAddr, WasmError> {
        self.frame_addr_from(self.regfile.frame_base(), slot)
    }

    fn frame_addr_from(&self, base: MachineReg, slot: FrameSlot) -> Result<MachineAddr, WasmError> {
        Ok(MachineAddr {
            base,
            offset: slot_offset_bytes(slot)?,
        })
    }

    fn frame_region_addr(
        &self,
        base: MachineReg,
        region: MachineFrameRegion,
        byte_offset: i32,
    ) -> Result<MachineAddr, WasmError> {
        let region_offset = slot_offset_bytes(FrameSlot(region.base_slot))?;
        let offset = region_offset
            .checked_add(byte_offset)
            .ok_or_else(|| WasmError::internal("frame region byte offset overflow".into()))?;
        Ok(MachineAddr { base, offset })
    }

    fn runtime_for_func(&self, func: MachineFuncId) -> Result<MachineFunctionRuntime, WasmError> {
        self.all_runtime
            .get(func.0 as usize)
            .copied()
            .ok_or_else(|| WasmError::internal("machine runtime metadata missing for callee".into()))
    }

    fn store_call_link(
        &mut self,
        callee_frame_base: MachineReg,
        call_scratch: MachineFrameRegion,
        continuation: MachineBlockId,
        results: FrameSpan,
    ) -> Result<(), WasmError> {
        let caller_result_base = slot_offset_bytes(results.start)? as u64;
        self.ops.push(MachineInst {
            kind: MachineInstKind::Store {
                addr: self.frame_region_addr(
                    callee_frame_base,
                    call_scratch,
                    self.call_link.continuation_offset,
                )?,
                width: MachineMemWidth::U64,
                src: MachineValue::Imm64(continuation.0 as u64),
            },
        });
        self.ops.push(MachineInst {
            kind: MachineInstKind::Store {
                addr: self.frame_region_addr(
                    callee_frame_base,
                    call_scratch,
                    self.call_link.caller_frame_offset,
                )?,
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(self.regfile.frame_base()),
            },
        });
        self.ops.push(MachineInst {
            kind: MachineInstKind::Store {
                addr: self.frame_region_addr(
                    callee_frame_base,
                    call_scratch,
                    self.call_link.caller_result_base_offset,
                )?,
                width: MachineMemWidth::U64,
                src: MachineValue::Imm64(caller_result_base),
            },
        });
        Ok(())
    }

    fn alloc_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        if let Some(reg) = self.try_value_reg(value) {
            return Ok(reg);
        }
        let reg = self
            .first_free_transient()
            .ok_or_else(|| WasmError::internal("prepared LIR exceeded transient register budget during native lowering".into()))?;
        self.values.push(ValueLocation { value, reg });
        self.mark_transient(reg, Some(value))?;
        Ok(reg)
    }

    fn use_value(&mut self, value: LirValue) -> Result<MachineReg, WasmError> {
        let reg = self.value_reg(value)?;
        if let Some(remaining) = self.remaining_uses.get_mut(&value) {
            *remaining = remaining.saturating_sub(1);
        }
        Ok(reg)
    }

    pub(super) fn release_dead_values(&mut self) -> Result<(), WasmError> {
        let mut index = 0;
        while index < self.values.len() {
            let value = self.values[index].value;
            let remaining = self.remaining_uses.get(&value).copied().unwrap_or(0);
            if remaining == 0 {
                let reg = self.values[index].reg;
                self.values.swap_remove(index);
                self.mark_transient(reg, None)?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    fn try_value_reg(&self, value: LirValue) -> Option<MachineReg> {
        self.values
            .iter()
            .find(|entry| entry.value == value)
            .map(|entry| entry.reg)
    }

    fn value_reg(&self, value: LirValue) -> Result<MachineReg, WasmError> {
        self.try_value_reg(value)
            .ok_or_else(|| WasmError::internal(alloc::format!("no machine register assigned for LIR value {:?}", value)))
    }

    fn first_free_transient(&self) -> Option<MachineReg> {
        for index in 0..self.transient_owner.len() {
            if self.transient_owner[index].is_none() {
                return self.regfile.transient(index);
            }
        }
        None
    }

    fn mark_transient(&mut self, reg: MachineReg, value: Option<LirValue>) -> Result<(), WasmError> {
        let index = reg
            .0
            .checked_sub(
                self.regfile
                    .transient(0)
                    .ok_or_else(|| WasmError::internal("native lowering has no transient registers".into()))?
                    .0,
            )
            .ok_or_else(|| WasmError::internal("machine register is not in transient partition".into()))? as usize;
        let slot = self
            .transient_owner
            .get_mut(index)
            .ok_or_else(|| WasmError::internal("transient register index is out of range".into()))?;
        *slot = value;
        Ok(())
    }
}

fn compute_remaining_uses(block: &LirBlock) -> alloc::collections::BTreeMap<LirValue, u32> {
    use alloc::collections::BTreeMap;

    let mut uses = BTreeMap::new();
    for inst in &block.ops {
        match &inst.kind {
            LirInstKind::Value { args, .. } => {
                for value in args {
                    *uses.entry(*value).or_insert(0) += 1;
                }
            }
            LirInstKind::StoreSlot { src, .. } => {
                *uses.entry(*src).or_insert(0) += 1;
            }
            LirInstKind::LoadSlot { .. } => {}
            LirInstKind::Boundary(LirBoundaryOp::MemoryGrow { .. })
            | LirInstKind::Boundary(LirBoundaryOp::TableGrow { .. })
            | LirInstKind::Boundary(LirBoundaryOp::MemoryInit { .. })
            | LirInstKind::Boundary(LirBoundaryOp::DataDrop { .. })
            | LirInstKind::Boundary(LirBoundaryOp::TableInit { .. })
            | LirInstKind::Boundary(LirBoundaryOp::ElemDrop { .. })
            | LirInstKind::Boundary(LirBoundaryOp::CallExternal { .. })
            | LirInstKind::Boundary(LirBoundaryOp::CallInternal { .. })
            | LirInstKind::Boundary(LirBoundaryOp::CallIndirect { .. }) => {}
        }
    }

    match &block.terminator {
        LirTerminator::Goto(edge) => count_edge_uses(edge, &mut uses),
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            *uses.entry(*cond).or_insert(0) += 1;
            count_edge_uses(then_edge, &mut uses);
            count_edge_uses(else_edge, &mut uses);
        }
        LirTerminator::BrTable { index, entries } => {
            *uses.entry(*index).or_insert(0) += 1;
            for edge in entries {
                count_edge_uses(edge, &mut uses);
            }
        }
        LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
    }

    uses
}

fn count_edge_uses(
    edge: &LirEdge,
    uses: &mut alloc::collections::BTreeMap<LirValue, u32>,
) {
    for binding in &edge.bindings {
        *uses.entry(binding.value).or_insert(0) += 1;
    }
}

fn span_region_with_slots(
    span: FrameSpan,
    slots: u16,
    context: &'static str,
) -> Result<HelperFrameRegion, WasmError> {
    if span.count != slots {
        return Err(WasmError::internal(alloc::format!(
            "{context} expected {} slots but got {}",
            slots,
            span.count,
        )));
    }
    Ok(span.into())
}

fn single_result(results: &[LirValue]) -> Result<LirValue, WasmError> {
    match results {
        [value] => Ok(*value),
        _ => Err(WasmError::internal("machine lowering expected exactly one result".into())),
    }
}

fn single_arg(args: &[LirValue]) -> Result<LirValue, WasmError> {
    match args {
        [value] => Ok(*value),
        _ => Err(WasmError::internal("machine lowering expected exactly one argument".into())),
    }
}

fn two_args(args: &[LirValue]) -> Result<(LirValue, LirValue), WasmError> {
    match args {
        [lhs, rhs] => Ok((*lhs, *rhs)),
        _ => Err(WasmError::internal("machine lowering expected exactly two arguments".into())),
    }
}
