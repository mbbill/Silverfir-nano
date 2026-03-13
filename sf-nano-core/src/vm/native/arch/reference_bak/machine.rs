//! Reference machine for finalized native IR.
//!
//! This module is the validation backend for placed `NativeProgram`.
//! It consumes the same target-independent machine contract that `arch/arm64`
//! will consume later:
//! - concrete `Hot/Tos/Tmp/fp[...]` locations
//! - explicit edges
//! - explicit calls and returns

use alloc::{rc::Rc, vec::Vec};

use crate::{
    constants::{MAX_CALL_STACK_DEPTH, WASM_PAGE_SIZE},
    error::WasmError,
    module::type_defs::FunctionType,
    vm::{
        entities::{Caller, FunctionInst},
        native::{
            abi::{NativeLocation, NativePlace, NativeSpillSlot, NativeValue},
            code::NativeCode,
            ir::{
                NativeBlockId, NativeBranchCond, NativeCompareKind, NativeEdge, NativeHelperInst,
                NativeInst, NativeInstKind, NativeProgram, NativeTerminator,
            },
            runtime::{context::NativeContext, layout},
        },
        store::Store,
        value::{RefHandle, Value},
    },
};
#[cfg(feature = "function-trace")]
use crate::vm::debug::function_trace;

use crate::vm::lir::leaf::LirLeafOp;

const MAX_REFERENCE_CALL_DEPTH: u64 = if MAX_CALL_STACK_DEPTH < 300 {
    MAX_CALL_STACK_DEPTH as u64
} else {
    300
};

unsafe extern "C" {
    fn ceilf(x: f32) -> f32;
    fn floorf(x: f32) -> f32;
    fn truncf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
    fn ceil(x: f64) -> f64;
    fn floor(x: f64) -> f64;
    fn trunc(x: f64) -> f64;
    fn sqrt(x: f64) -> f64;
}

enum MachineStep {
    Jump(NativeBlockId),
    Return,
}

enum ResolvedCallee {
    Local {
        code: *const NativeCode,
        program: *const NativeProgram,
        params_len: usize,
        results_len: usize,
    },
    External {
        func_type: Rc<FunctionType>,
        callback: crate::vm::entities::ExternalFn,
    },
}

/// Reference machine for the finalized native machine IR.
#[derive(Debug)]
pub struct ReferenceMachine<'a> {
    ctx: &'a mut NativeContext,
    program: &'a NativeProgram,
    fp: *mut u64,
    hot: Vec<u64>,
    tos: Vec<u64>,
    tmp: Vec<u64>,
}

impl<'a> ReferenceMachine<'a> {
    #[inline]
    pub fn new(ctx: &'a mut NativeContext, program: &'a NativeProgram, fp: *mut u64) -> Self {
        Self {
            ctx,
            program,
            fp,
            hot: alloc::vec![0; program.abi.hot_local_count as usize],
            tos: alloc::vec![0; program.abi.tos_register_count as usize],
            tmp: alloc::vec![0; program.abi.tmp_register_count as usize],
        }
    }

    pub fn run(mut self) -> Result<(), WasmError> {
        let mut block_id = self.program.entry;
        loop {
            let block = self
                .program
                .blocks
                .get(block_id.as_usize())
                .ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "native block {} is out of range",
                        block_id.as_usize()
                    ))
                })?;

            for inst in &block.ops {
                self.execute_inst(inst)?;
            }

            match self.execute_terminator(&block.terminator)? {
                MachineStep::Jump(next) => block_id = next,
                MachineStep::Return => return Ok(()),
            }
        }
    }

    fn execute_inst(&mut self, inst: &NativeInst) -> Result<(), WasmError> {
        match &inst.kind {
            NativeInstKind::Move(mov) => {
                let value = self.read_value(mov.src)?;
                self.write_place(mov.dst, value)
            }
            NativeInstKind::Leaf { op, args, results } => self.execute_leaf(op, args, results),
            NativeInstKind::Helper(helper) => self.execute_helper(helper),
            NativeInstKind::CallExternal {
                func_idx,
                args,
                results,
            } => self.execute_call_external(*func_idx, args, results),
            NativeInstKind::CallLocal {
                callee,
                callee_params_count,
                callee_frame_size,
                args,
                results,
            } => self.execute_call_local(
                *callee,
                *callee_params_count,
                *callee_frame_size,
                args,
                results,
            ),
            NativeInstKind::CallIndirect {
                type_idx,
                table_idx,
                index,
                args,
                results,
            } => self.execute_call_indirect(*type_idx, *table_idx, *index, args, results),
        }
    }

    fn execute_terminator(&mut self, term: &NativeTerminator) -> Result<MachineStep, WasmError> {
        match term {
            NativeTerminator::Goto(edge) => {
                self.apply_edge(edge)?;
                Ok(MachineStep::Jump(edge.target))
            }
            NativeTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                let edge = if self.eval_branch_cond(cond)? {
                    then_edge
                } else {
                    else_edge
                };
                self.apply_edge(edge)?;
                Ok(MachineStep::Jump(edge.target))
            }
            NativeTerminator::BrTable { index, entries } => {
                let selected = entries
                    .get(self.read_value(*index)? as usize)
                    .or_else(|| entries.last())
                    .ok_or_else(|| WasmError::internal("br_table has no entries".into()))?;
                self.apply_edge(selected)?;
                Ok(MachineStep::Jump(selected.target))
            }
            NativeTerminator::Return { values } => {
                let raws = self.read_values(values)?;
                for (index, raw) in raws.into_iter().enumerate() {
                    unsafe {
                        *self.fp.add(index) = raw;
                    }
                }
                #[cfg(feature = "function-trace")]
                if self.ctx.call_depth != 0 {
                    self.ctx.call_depth = self.ctx.call_depth.saturating_sub(1);
                    unsafe {
                        function_trace::native_function_trace_exit(
                            self.ctx as *mut NativeContext,
                            self.fp,
                            values.len() as u64,
                        );
                    }
                }
                Ok(MachineStep::Return)
            }
            NativeTerminator::TrapUnreachable => {
                Err(WasmError::invalid("native reached unreachable".into()))
            }
        }
    }

    fn execute_helper(&mut self, helper: &NativeHelperInst) -> Result<(), WasmError> {
        match helper {
            NativeHelperInst::Leaf {
                op, args, results, ..
            } => self.execute_leaf(op, args, results),
        }
    }

    fn eval_branch_cond(&self, cond: &NativeBranchCond) -> Result<bool, WasmError> {
        match cond {
            NativeBranchCond::Value(value) => Ok(self.read_value(*value)? != 0),
            NativeBranchCond::I32Eqz(value) => Ok((self.read_value(*value)? as u32) == 0),
            NativeBranchCond::I64Eqz(value) => Ok(self.read_value(*value)? == 0),
            NativeBranchCond::I32Compare { kind, lhs, rhs } => {
                let lhs = self.read_value(*lhs)? as u32;
                let rhs = self.read_value(*rhs)? as u32;
                Ok(eval_i32_compare(*kind, lhs, rhs))
            }
            NativeBranchCond::I64Compare { kind, lhs, rhs } => {
                let lhs = self.read_value(*lhs)?;
                let rhs = self.read_value(*rhs)?;
                Ok(eval_i64_compare(*kind, lhs, rhs))
            }
            NativeBranchCond::F32Compare { kind, lhs, rhs } => {
                let lhs = self.read_value(*lhs)? as u32;
                let rhs = self.read_value(*rhs)? as u32;
                Ok(eval_f32_compare(*kind, lhs, rhs))
            }
            NativeBranchCond::F64Compare { kind, lhs, rhs } => {
                let lhs = self.read_value(*lhs)?;
                let rhs = self.read_value(*rhs)?;
                Ok(eval_f64_compare(*kind, lhs, rhs))
            }
        }
    }

    fn apply_edge(&mut self, edge: &NativeEdge) -> Result<(), WasmError> {
        let snapshot = edge
            .copies
            .iter()
            .map(|mov| self.read_value(mov.src))
            .collect::<Result<Vec<_>, _>>()?;

        for (mov, value) in edge.copies.iter().zip(snapshot.into_iter()) {
            self.write_place(mov.dst, value)?;
        }
        Ok(())
    }

    fn read_values(&self, values: &[NativeValue]) -> Result<Vec<u64>, WasmError> {
        values
            .iter()
            .copied()
            .map(|value| self.read_value(value))
            .collect()
    }

    fn read_value(&self, value: NativeValue) -> Result<u64, WasmError> {
        match value {
            NativeValue::Imm64(value) => Ok(value),
            NativeValue::Place(place) => self.read_place(place),
        }
    }

    fn read_place(&self, place: NativePlace) -> Result<u64, WasmError> {
        match place {
            NativePlace::Location(NativeLocation::Ctx) => {
                Ok(self.ctx as *const NativeContext as u64)
            }
            NativePlace::Location(NativeLocation::Fp) => Ok(self.fp as u64),
            NativePlace::Location(NativeLocation::Hot(reg)) => self
                .hot
                .get(reg as usize)
                .copied()
                .ok_or_else(|| WasmError::internal("hot register read out of range".into())),
            NativePlace::Location(NativeLocation::Tos(lane)) => self
                .tos
                .get(lane as usize)
                .copied()
                .ok_or_else(|| WasmError::internal("TOS register read out of range".into())),
            NativePlace::Location(NativeLocation::Tmp(reg)) => self
                .tmp
                .get(reg as usize)
                .copied()
                .ok_or_else(|| WasmError::internal("tmp register read out of range".into())),
            NativePlace::Frame(slot) => Ok(unsafe { *self.fp.add(slot.0 as usize) }),
            NativePlace::Spill(slot) => Ok(unsafe { *self.spill_ptr(slot) }),
        }
    }

    fn write_place(&mut self, place: NativePlace, value: u64) -> Result<(), WasmError> {
        match place {
            NativePlace::Location(NativeLocation::Ctx)
            | NativePlace::Location(NativeLocation::Fp) => Err(WasmError::internal(
                "reference machine cannot assign to reserved ctx/fp locations".into(),
            )),
            NativePlace::Location(NativeLocation::Hot(reg)) => {
                let slot = self
                    .hot
                    .get_mut(reg as usize)
                    .ok_or_else(|| WasmError::internal("hot register write out of range".into()))?;
                *slot = value;
                Ok(())
            }
            NativePlace::Location(NativeLocation::Tos(lane)) => {
                let slot = self
                    .tos
                    .get_mut(lane as usize)
                    .ok_or_else(|| WasmError::internal("TOS register write out of range".into()))?;
                *slot = value;
                Ok(())
            }
            NativePlace::Location(NativeLocation::Tmp(reg)) => {
                let slot = self
                    .tmp
                    .get_mut(reg as usize)
                    .ok_or_else(|| WasmError::internal("tmp register write out of range".into()))?;
                *slot = value;
                Ok(())
            }
            NativePlace::Frame(slot) => {
                unsafe {
                    *self.fp.add(slot.0 as usize) = value;
                }
                Ok(())
            }
            NativePlace::Spill(slot) => {
                unsafe {
                    *self.spill_ptr(slot) = value;
                }
                Ok(())
            }
        }
    }

    #[inline]
    unsafe fn spill_ptr(&self, slot: NativeSpillSlot) -> *mut u64 {
        self.fp
            .add(self.program.frame.operands.end().0 as usize + slot.0 as usize)
    }

    fn execute_leaf(
        &mut self,
        op: &LirLeafOp,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let args = self.read_values(args)?;
        let output = self.eval_leaf(op, &args)?;
        if output.len() != results.len() {
            return Err(WasmError::internal(alloc::format!(
                "leaf {:?} produced {} results but instruction expects {}",
                op,
                output.len(),
                results.len(),
            )));
        }
        for (place, value) in results.iter().copied().zip(output.into_iter()) {
            self.write_place(place, value)?;
        }
        Ok(())
    }

    fn eval_leaf(&mut self, op: &LirLeafOp, args: &[u64]) -> Result<Vec<u64>, WasmError> {
        use LirLeafOp::*;

        let out = match op {
            I32Const { value } => alloc::vec![u64::from(*value)],
            I64Const { value } => alloc::vec![*value],
            F32Const { value } => alloc::vec![u64::from(*value)],
            F64Const { value } => alloc::vec![*value],
            I32Add => alloc::vec![u64::from(bin_i32(args, i32::wrapping_add)? as u32)],
            I32Sub => alloc::vec![u64::from(bin_i32(args, i32::wrapping_sub)? as u32)],
            I32Mul => alloc::vec![u64::from(bin_i32(args, i32::wrapping_mul)? as u32)],
            I32DivS => alloc::vec![i32_div_s(args)?],
            I32DivU => alloc::vec![i32_div_u(args)?],
            I32RemS => alloc::vec![i32_rem_s(args)?],
            I32RemU => alloc::vec![i32_rem_u(args)?],
            I32And => alloc::vec![u64::from((arg_u32(args, 0)? & arg_u32(args, 1)?) as u32)],
            I32Or => alloc::vec![u64::from((arg_u32(args, 0)? | arg_u32(args, 1)?) as u32)],
            I32Xor => alloc::vec![u64::from((arg_u32(args, 0)? ^ arg_u32(args, 1)?) as u32)],
            I32Shl => alloc::vec![u64::from(
                arg_u32(args, 0)?.wrapping_shl(arg_u32(args, 1)? & 31)
            )],
            I32ShrS => alloc::vec![u64::from(
                (arg_i32(args, 0)? >> (arg_u32(args, 1)? & 31)) as u32
            )],
            I32ShrU => alloc::vec![u64::from(arg_u32(args, 0)? >> (arg_u32(args, 1)? & 31))],
            I32Rotl => alloc::vec![u64::from(
                arg_u32(args, 0)?.rotate_left(arg_u32(args, 1)? & 31)
            )],
            I32Rotr => alloc::vec![u64::from(
                arg_u32(args, 0)?.rotate_right(arg_u32(args, 1)? & 31)
            )],
            I32Eqz => alloc::vec![u64::from((arg_u32(args, 0)? == 0) as u32)],
            I32Eq => alloc::vec![u64::from((arg_u32(args, 0)? == arg_u32(args, 1)?) as u32)],
            I32Ne => alloc::vec![u64::from((arg_u32(args, 0)? != arg_u32(args, 1)?) as u32)],
            I32LtS => alloc::vec![u64::from((arg_i32(args, 0)? < arg_i32(args, 1)?) as u32)],
            I32LtU => alloc::vec![u64::from((arg_u32(args, 0)? < arg_u32(args, 1)?) as u32)],
            I32GtS => alloc::vec![u64::from((arg_i32(args, 0)? > arg_i32(args, 1)?) as u32)],
            I32GtU => alloc::vec![u64::from((arg_u32(args, 0)? > arg_u32(args, 1)?) as u32)],
            I32LeS => alloc::vec![u64::from((arg_i32(args, 0)? <= arg_i32(args, 1)?) as u32)],
            I32LeU => alloc::vec![u64::from((arg_u32(args, 0)? <= arg_u32(args, 1)?) as u32)],
            I32GeS => alloc::vec![u64::from((arg_i32(args, 0)? >= arg_i32(args, 1)?) as u32)],
            I32GeU => alloc::vec![u64::from((arg_u32(args, 0)? >= arg_u32(args, 1)?) as u32)],
            I32Clz => alloc::vec![u64::from(arg_u32(args, 0)?.leading_zeros())],
            I32Ctz => alloc::vec![u64::from(arg_u32(args, 0)?.trailing_zeros())],
            I32Popcnt => alloc::vec![u64::from(arg_u32(args, 0)?.count_ones())],
            I64Add => alloc::vec![arg_u64(args, 0)?.wrapping_add(arg_u64(args, 1)?)],
            I64Sub => alloc::vec![arg_u64(args, 0)?.wrapping_sub(arg_u64(args, 1)?)],
            I64Mul => alloc::vec![arg_u64(args, 0)?.wrapping_mul(arg_u64(args, 1)?)],
            I64DivS => alloc::vec![i64_div_s(args)?],
            I64DivU => alloc::vec![i64_div_u(args)?],
            I64RemS => alloc::vec![i64_rem_s(args)?],
            I64RemU => alloc::vec![i64_rem_u(args)?],
            I64And => alloc::vec![arg_u64(args, 0)? & arg_u64(args, 1)?],
            I64Or => alloc::vec![arg_u64(args, 0)? | arg_u64(args, 1)?],
            I64Xor => alloc::vec![arg_u64(args, 0)? ^ arg_u64(args, 1)?],
            I64Shl => alloc::vec![arg_u64(args, 0)?.wrapping_shl((arg_u64(args, 1)? & 63) as u32)],
            I64ShrS => {
                alloc::vec![((arg_i64(args, 0)? >> ((arg_u64(args, 1)? & 63) as u32)) as u64)]
            }
            I64ShrU => alloc::vec![arg_u64(args, 0)? >> ((arg_u64(args, 1)? & 63) as u32)],
            I64Rotl => alloc::vec![arg_u64(args, 0)?.rotate_left((arg_u64(args, 1)? & 63) as u32)],
            I64Rotr => alloc::vec![arg_u64(args, 0)?.rotate_right((arg_u64(args, 1)? & 63) as u32)],
            I64Eqz => alloc::vec![u64::from((arg_u64(args, 0)? == 0) as u32)],
            I64Eq => alloc::vec![u64::from((arg_u64(args, 0)? == arg_u64(args, 1)?) as u32)],
            I64Ne => alloc::vec![u64::from((arg_u64(args, 0)? != arg_u64(args, 1)?) as u32)],
            I64LtS => alloc::vec![u64::from((arg_i64(args, 0)? < arg_i64(args, 1)?) as u32)],
            I64LtU => alloc::vec![u64::from((arg_u64(args, 0)? < arg_u64(args, 1)?) as u32)],
            I64GtS => alloc::vec![u64::from((arg_i64(args, 0)? > arg_i64(args, 1)?) as u32)],
            I64GtU => alloc::vec![u64::from((arg_u64(args, 0)? > arg_u64(args, 1)?) as u32)],
            I64LeS => alloc::vec![u64::from((arg_i64(args, 0)? <= arg_i64(args, 1)?) as u32)],
            I64LeU => alloc::vec![u64::from((arg_u64(args, 0)? <= arg_u64(args, 1)?) as u32)],
            I64GeS => alloc::vec![u64::from((arg_i64(args, 0)? >= arg_i64(args, 1)?) as u32)],
            I64GeU => alloc::vec![u64::from((arg_u64(args, 0)? >= arg_u64(args, 1)?) as u32)],
            I64Clz => alloc::vec![u64::from(arg_u64(args, 0)?.leading_zeros())],
            I64Ctz => alloc::vec![u64::from(arg_u64(args, 0)?.trailing_zeros())],
            I64Popcnt => alloc::vec![u64::from(arg_u64(args, 0)?.count_ones())],
            F32Add => alloc::vec![f32_bits(
                as_f32(arg_u32(args, 0)?) + as_f32(arg_u32(args, 1)?)
            )],
            F32Sub => alloc::vec![f32_bits(
                as_f32(arg_u32(args, 0)?) - as_f32(arg_u32(args, 1)?)
            )],
            F32Mul => alloc::vec![f32_bits(
                as_f32(arg_u32(args, 0)?) * as_f32(arg_u32(args, 1)?)
            )],
            F32Div => alloc::vec![f32_bits(
                as_f32(arg_u32(args, 0)?) / as_f32(arg_u32(args, 1)?)
            )],
            F32Min => alloc::vec![wasm_f32_min_bits(arg_u32(args, 0)?, arg_u32(args, 1)?)],
            F32Max => alloc::vec![wasm_f32_max_bits(arg_u32(args, 0)?, arg_u32(args, 1)?)],
            F32Copysign => alloc::vec![f32_bits(
                as_f32(arg_u32(args, 0)?).copysign(as_f32(arg_u32(args, 1)?))
            )],
            F64Add => alloc::vec![f64_bits(
                as_f64(arg_u64(args, 0)?) + as_f64(arg_u64(args, 1)?)
            )],
            F64Sub => alloc::vec![f64_bits(
                as_f64(arg_u64(args, 0)?) - as_f64(arg_u64(args, 1)?)
            )],
            F64Mul => alloc::vec![f64_bits(
                as_f64(arg_u64(args, 0)?) * as_f64(arg_u64(args, 1)?)
            )],
            F64Div => alloc::vec![f64_bits(
                as_f64(arg_u64(args, 0)?) / as_f64(arg_u64(args, 1)?)
            )],
            F64Min => alloc::vec![wasm_f64_min_bits(arg_u64(args, 0)?, arg_u64(args, 1)?)],
            F64Max => alloc::vec![wasm_f64_max_bits(arg_u64(args, 0)?, arg_u64(args, 1)?)],
            F64Copysign => alloc::vec![f64_bits(
                as_f64(arg_u64(args, 0)?).copysign(as_f64(arg_u64(args, 1)?))
            )],
            F32Eq => alloc::vec![u64::from(
                (as_f32(arg_u32(args, 0)?) == as_f32(arg_u32(args, 1)?)) as u32
            )],
            F32Ne => alloc::vec![u64::from(
                (as_f32(arg_u32(args, 0)?) != as_f32(arg_u32(args, 1)?)) as u32
            )],
            F32Lt => alloc::vec![u64::from(
                (as_f32(arg_u32(args, 0)?) < as_f32(arg_u32(args, 1)?)) as u32
            )],
            F32Gt => alloc::vec![u64::from(
                (as_f32(arg_u32(args, 0)?) > as_f32(arg_u32(args, 1)?)) as u32
            )],
            F32Le => alloc::vec![u64::from(
                (as_f32(arg_u32(args, 0)?) <= as_f32(arg_u32(args, 1)?)) as u32
            )],
            F32Ge => alloc::vec![u64::from(
                (as_f32(arg_u32(args, 0)?) >= as_f32(arg_u32(args, 1)?)) as u32
            )],
            F64Eq => alloc::vec![u64::from(
                (as_f64(arg_u64(args, 0)?) == as_f64(arg_u64(args, 1)?)) as u32
            )],
            F64Ne => alloc::vec![u64::from(
                (as_f64(arg_u64(args, 0)?) != as_f64(arg_u64(args, 1)?)) as u32
            )],
            F64Lt => alloc::vec![u64::from(
                (as_f64(arg_u64(args, 0)?) < as_f64(arg_u64(args, 1)?)) as u32
            )],
            F64Gt => alloc::vec![u64::from(
                (as_f64(arg_u64(args, 0)?) > as_f64(arg_u64(args, 1)?)) as u32
            )],
            F64Le => alloc::vec![u64::from(
                (as_f64(arg_u64(args, 0)?) <= as_f64(arg_u64(args, 1)?)) as u32
            )],
            F64Ge => alloc::vec![u64::from(
                (as_f64(arg_u64(args, 0)?) >= as_f64(arg_u64(args, 1)?)) as u32
            )],
            F32Abs => alloc::vec![f32_bits(as_f32(arg_u32(args, 0)?).abs())],
            F32Neg => alloc::vec![f32_bits(-as_f32(arg_u32(args, 0)?))],
            F32Ceil => alloc::vec![f32_bits(ceil_f32(as_f32(arg_u32(args, 0)?)))],
            F32Floor => alloc::vec![f32_bits(floor_f32(as_f32(arg_u32(args, 0)?)))],
            F32Trunc => alloc::vec![f32_bits(trunc_f32(as_f32(arg_u32(args, 0)?)))],
            F32Nearest => alloc::vec![wasm_f32_nearest_bits(arg_u32(args, 0)?)],
            F32Sqrt => alloc::vec![f32_bits(sqrt_f32(as_f32(arg_u32(args, 0)?)))],
            F64Abs => alloc::vec![f64_bits(as_f64(arg_u64(args, 0)?).abs())],
            F64Neg => alloc::vec![f64_bits(-as_f64(arg_u64(args, 0)?))],
            F64Ceil => alloc::vec![f64_bits(ceil_f64(as_f64(arg_u64(args, 0)?)))],
            F64Floor => alloc::vec![f64_bits(floor_f64(as_f64(arg_u64(args, 0)?)))],
            F64Trunc => alloc::vec![f64_bits(trunc_f64(as_f64(arg_u64(args, 0)?)))],
            F64Nearest => alloc::vec![wasm_f64_nearest_bits(arg_u64(args, 0)?)],
            F64Sqrt => alloc::vec![f64_bits(sqrt_f64(as_f64(arg_u64(args, 0)?)))],
            I32WrapI64 => alloc::vec![u64::from(arg_u64(args, 0)? as u32)],
            I64ExtendI32S => alloc::vec![(arg_i32(args, 0)? as i64) as u64],
            I64ExtendI32U => alloc::vec![u64::from(arg_u32(args, 0)?)],
            F32ConvertI32S => alloc::vec![f32_bits(arg_i32(args, 0)? as f32)],
            F32ConvertI32U => alloc::vec![f32_bits(arg_u32(args, 0)? as f32)],
            F32ConvertI64S => alloc::vec![f32_bits(arg_i64(args, 0)? as f32)],
            F32ConvertI64U => alloc::vec![f32_bits(arg_u64(args, 0)? as f32)],
            F32DemoteF64 => alloc::vec![f32_bits(as_f64(arg_u64(args, 0)?) as f32)],
            F64ConvertI32S => alloc::vec![f64_bits(arg_i32(args, 0)? as f64)],
            F64ConvertI32U => alloc::vec![f64_bits(arg_u32(args, 0)? as f64)],
            F64ConvertI64S => alloc::vec![f64_bits(arg_i64(args, 0)? as f64)],
            F64ConvertI64U => alloc::vec![f64_bits(arg_u64(args, 0)? as f64)],
            F64PromoteF32 => alloc::vec![f64_bits(as_f32(arg_u32(args, 0)?) as f64)],
            I32ReinterpretF32 => alloc::vec![u64::from(arg_u32(args, 0)?)],
            I64ReinterpretF64 => alloc::vec![arg_u64(args, 0)?],
            F32ReinterpretI32 => alloc::vec![u64::from(arg_u32(args, 0)?)],
            F64ReinterpretI64 => alloc::vec![arg_u64(args, 0)?],
            I32Extend8S => alloc::vec![u64::from((arg_u32(args, 0)? as i8 as i32) as u32)],
            I32Extend16S => alloc::vec![u64::from((arg_u32(args, 0)? as i16 as i32) as u32)],
            I64Extend8S => alloc::vec![(arg_u64(args, 0)? as i8 as i64) as u64],
            I64Extend16S => alloc::vec![(arg_u64(args, 0)? as i16 as i64) as u64],
            I64Extend32S => alloc::vec![(arg_u64(args, 0)? as i32 as i64) as u64],
            I32TruncF32S => alloc::vec![trunc_f32_to_i32_s(arg_u32(args, 0)?)?],
            I32TruncF32U => alloc::vec![trunc_f32_to_i32_u(arg_u32(args, 0)?)?],
            I32TruncF64S => alloc::vec![trunc_f64_to_i32_s(arg_u64(args, 0)?)?],
            I32TruncF64U => alloc::vec![trunc_f64_to_i32_u(arg_u64(args, 0)?)?],
            I64TruncF32S => alloc::vec![trunc_f32_to_i64_s(arg_u32(args, 0)?)?],
            I64TruncF32U => alloc::vec![trunc_f32_to_i64_u(arg_u32(args, 0)?)?],
            I64TruncF64S => alloc::vec![trunc_f64_to_i64_s(arg_u64(args, 0)?)?],
            I64TruncF64U => alloc::vec![trunc_f64_to_i64_u(arg_u64(args, 0)?)?],
            I32TruncSatF32S => alloc::vec![trunc_sat_f32_to_i32_s(arg_u32(args, 0)?)],
            I32TruncSatF32U => alloc::vec![trunc_sat_f32_to_i32_u(arg_u32(args, 0)?)],
            I32TruncSatF64S => alloc::vec![trunc_sat_f64_to_i32_s(arg_u64(args, 0)?)],
            I32TruncSatF64U => alloc::vec![trunc_sat_f64_to_i32_u(arg_u64(args, 0)?)],
            I64TruncSatF32S => alloc::vec![trunc_sat_f32_to_i64_s(arg_u32(args, 0)?)],
            I64TruncSatF32U => alloc::vec![trunc_sat_f32_to_i64_u(arg_u32(args, 0)?)],
            I64TruncSatF64S => alloc::vec![trunc_sat_f64_to_i64_s(arg_u64(args, 0)?)],
            I64TruncSatF64U => alloc::vec![trunc_sat_f64_to_i64_u(arg_u64(args, 0)?)],
            Select => alloc::vec![if arg_u64(args, 2)? != 0 {
                arg_u64(args, 0)?
            } else {
                arg_u64(args, 1)?
            }],
            Drop | Nop => Vec::new(),
            Unreachable => {
                return Err(WasmError::invalid(
                    "native leaf unreachable executed".into(),
                ))
            }
            GlobalGet { idx } => alloc::vec![self.read_global(*idx)?],
            GlobalSet { idx } => {
                self.write_global(*idx, arg_u64(args, 0)?)?;
                Vec::new()
            }
            MemorySize { mem_idx } => alloc::vec![self.memory_size(*mem_idx)? as u64],
            MemoryGrow { mem_idx } => {
                alloc::vec![self.memory_grow(*mem_idx, arg_u32(args, 0)? as usize)?]
            }
            MemoryFill { imm0, .. } => {
                self.memory_fill(
                    *imm0,
                    arg_u32(args, 0)? as usize,
                    arg_u32(args, 1)? as u8,
                    arg_u32(args, 2)? as usize,
                )?;
                Vec::new()
            }
            MemoryCopy { imm0, imm1 } => {
                self.memory_copy(
                    *imm0,
                    *imm1,
                    arg_u32(args, 0)? as usize,
                    arg_u32(args, 1)? as usize,
                    arg_u32(args, 2)? as usize,
                )?;
                Vec::new()
            }
            MemoryInit { imm0, imm1 } => {
                self.memory_init(
                    *imm0,
                    *imm1,
                    arg_u32(args, 0)? as usize,
                    arg_u32(args, 1)? as usize,
                    arg_u32(args, 2)? as usize,
                )?;
                Vec::new()
            }
            DataDrop { data_idx } => {
                self.data_drop(*data_idx)?;
                Vec::new()
            }
            I32Load { offset, memidx } => {
                alloc::vec![u64::from(
                    self.load_i32(*memidx, arg_u32(args, 0)?, *offset)? as u32
                )]
            }
            I64Load { offset, memidx } => {
                alloc::vec![self.load_i64(*memidx, arg_u32(args, 0)?, *offset)? as u64]
            }
            F32Load { offset, memidx } => alloc::vec![u64::from(self.load_u32(
                *memidx,
                arg_u32(args, 0)?,
                *offset
            )?)],
            F64Load { offset, memidx } => {
                alloc::vec![self.load_u64(*memidx, arg_u32(args, 0)?, *offset)?]
            }
            I32Load8S { offset, memidx } => {
                alloc::vec![u64::from(
                    self.load_i8(*memidx, arg_u32(args, 0)?, *offset)? as i32 as u32
                )]
            }
            I32Load8U { offset, memidx } => {
                alloc::vec![u64::from(
                    self.load_u8(*memidx, arg_u32(args, 0)?, *offset)? as u32
                )]
            }
            I32Load16S { offset, memidx } => {
                alloc::vec![u64::from(
                    self.load_i16(*memidx, arg_u32(args, 0)?, *offset)? as i32 as u32
                )]
            }
            I32Load16U { offset, memidx } => {
                alloc::vec![u64::from(
                    self.load_u16(*memidx, arg_u32(args, 0)?, *offset)? as u32
                )]
            }
            I64Load8S { offset, memidx } => {
                alloc::vec![(self.load_i8(*memidx, arg_u32(args, 0)?, *offset)? as i64) as u64]
            }
            I64Load8U { offset, memidx } => {
                alloc::vec![self.load_u8(*memidx, arg_u32(args, 0)?, *offset)? as u64]
            }
            I64Load16S { offset, memidx } => {
                alloc::vec![(self.load_i16(*memidx, arg_u32(args, 0)?, *offset)? as i64) as u64]
            }
            I64Load16U { offset, memidx } => {
                alloc::vec![self.load_u16(*memidx, arg_u32(args, 0)?, *offset)? as u64]
            }
            I64Load32S { offset, memidx } => {
                alloc::vec![(self.load_i32(*memidx, arg_u32(args, 0)?, *offset)? as i64) as u64]
            }
            I64Load32U { offset, memidx } => {
                alloc::vec![self.load_u32(*memidx, arg_u32(args, 0)?, *offset)? as u64]
            }
            I32Store { offset, memidx } => {
                self.store_u32(*memidx, arg_u32(args, 0)?, *offset, arg_u32(args, 1)?)?;
                Vec::new()
            }
            I64Store { offset, memidx } => {
                self.store_u64(*memidx, arg_u32(args, 0)?, *offset, arg_u64(args, 1)?)?;
                Vec::new()
            }
            F32Store { offset, memidx } => {
                self.store_u32(*memidx, arg_u32(args, 0)?, *offset, arg_u32(args, 1)?)?;
                Vec::new()
            }
            F64Store { offset, memidx } => {
                self.store_u64(*memidx, arg_u32(args, 0)?, *offset, arg_u64(args, 1)?)?;
                Vec::new()
            }
            I32Store8 { offset, memidx } => {
                self.store_u8(*memidx, arg_u32(args, 0)?, *offset, arg_u32(args, 1)? as u8)?;
                Vec::new()
            }
            I32Store16 { offset, memidx } => {
                self.store_u16(
                    *memidx,
                    arg_u32(args, 0)?,
                    *offset,
                    arg_u32(args, 1)? as u16,
                )?;
                Vec::new()
            }
            I64Store8 { offset, memidx } => {
                self.store_u8(*memidx, arg_u32(args, 0)?, *offset, arg_u64(args, 1)? as u8)?;
                Vec::new()
            }
            I64Store16 { offset, memidx } => {
                self.store_u16(
                    *memidx,
                    arg_u32(args, 0)?,
                    *offset,
                    arg_u64(args, 1)? as u16,
                )?;
                Vec::new()
            }
            I64Store32 { offset, memidx } => {
                self.store_u32(
                    *memidx,
                    arg_u32(args, 0)?,
                    *offset,
                    arg_u64(args, 1)? as u32,
                )?;
                Vec::new()
            }
            TableGet { table_idx } => {
                alloc::vec![self.table_get(*table_idx, arg_u32(args, 0)? as usize)?]
            }
            TableSet { table_idx } => {
                self.table_set(*table_idx, arg_u32(args, 0)? as usize, arg_u64(args, 1)?)?;
                Vec::new()
            }
            TableSize { table_idx } => alloc::vec![self.table_size(*table_idx)? as u64],
            TableGrow { table_idx } => alloc::vec![self.table_grow(
                *table_idx,
                arg_u64(args, 0)?,
                arg_u32(args, 1)? as usize
            )?],
            TableFill { imm0, .. } => {
                self.table_fill(
                    *imm0,
                    arg_u32(args, 0)? as usize,
                    arg_u64(args, 1)?,
                    arg_u32(args, 2)? as usize,
                )?;
                Vec::new()
            }
            TableCopy { imm0, imm1 } => {
                self.table_copy(
                    *imm0,
                    *imm1,
                    arg_u32(args, 0)? as usize,
                    arg_u32(args, 1)? as usize,
                    arg_u32(args, 2)? as usize,
                )?;
                Vec::new()
            }
            TableInit { imm0, imm1 } => {
                self.table_init(
                    *imm0,
                    *imm1,
                    arg_u32(args, 0)? as usize,
                    arg_u32(args, 1)? as usize,
                    arg_u32(args, 2)? as usize,
                )?;
                Vec::new()
            }
            ElemDrop { elem_idx } => {
                self.elem_drop(*elem_idx)?;
                Vec::new()
            }
            RefNull => alloc::vec![usize::MAX as u64],
            RefIsNull => alloc::vec![u64::from((arg_u64(args, 0)? as usize == usize::MAX) as u32)],
            RefFunc { func_idx } => alloc::vec![u64::from(*func_idx)],
            _ => {
                return Err(WasmError::internal(alloc::format!(
                    "reference machine does not implement leaf op {:?} yet",
                    op
                )));
            }
        };
        Ok(out)
    }

    fn execute_call_external(
        &mut self,
        func_idx: u32,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let arg_values = self.read_values(args)?;
        match self.resolve_callee(func_idx)? {
            ResolvedCallee::External {
                func_type,
                callback,
            } => {
                if func_type.params().len() != arg_values.len()
                    || func_type.results().len() != results.len()
                {
                    return Err(WasmError::internal(
                        "native call_external arity does not match function type".into(),
                    ));
                }

                let wasm_args = arg_values
                    .iter()
                    .zip(func_type.params().iter().copied())
                    .map(|(raw, ty)| Value::from_raw(*raw, ty))
                    .collect::<Vec<_>>();
                let mut returns = func_type
                    .results()
                    .iter()
                    .copied()
                    .map(Value::default_for_type)
                    .collect::<Vec<_>>();

                let store = self.store_mut()?;
                let mem_slice = if store.module().memories.is_empty() {
                    None
                } else {
                    Some(store.module_mut().memories[0].data.as_mut_slice())
                };
                let mut caller = Caller::new(mem_slice);
                callback(&mut caller, &wasm_args, &mut returns)?;

                for (place, value) in results.iter().copied().zip(returns.into_iter()) {
                    self.write_place(place, value.to_raw())?;
                }
                Ok(())
            }
            ResolvedCallee::Local { .. } => Err(WasmError::internal(
                "CallExternal resolved to a local function".into(),
            )),
        }
    }

    fn execute_call_local(
        &mut self,
        callee: u32,
        callee_params_count: u16,
        callee_frame_size: u16,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let arg_values = self.read_values(args)?;
        match self.resolve_callee(callee)? {
            ResolvedCallee::Local {
                code,
                program,
                params_len,
                results_len,
            } => self.invoke_local(
                callee,
                code,
                unsafe { &*program },
                callee_params_count,
                params_len,
                results_len,
                callee_frame_size,
                &arg_values,
                results,
            ),
            ResolvedCallee::External { .. } => Err(WasmError::internal(
                "CallLocal resolved to an external function".into(),
            )),
        }
    }

    fn execute_call_indirect(
        &mut self,
        type_idx: u32,
        table_idx: u32,
        index: NativeValue,
        args: &[NativeValue],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        let elem_index = self.read_value(index)? as usize;
        let func_idx = {
            let store = self.store()?;
            let table = store.table(table_idx as usize);
            let handle = table.elements.get(elem_index).ok_or_else(|| {
                WasmError::invalid("call_indirect: table index out of range".into())
            })?;
            if handle.is_null() {
                return Err(WasmError::invalid(
                    "call_indirect: null function reference".into(),
                ));
            }
            handle.raw_value() as u32
        };

        {
            let store = self.store()?;
            let expected = store.module().get_type(type_idx).ok_or_else(|| {
                WasmError::invalid("call_indirect: type index out of range".into())
            })?;
            let actual = store.function(func_idx as usize).func_type();
            if expected.as_ref() != actual {
                return Err(WasmError::invalid(
                    "call_indirect: signature mismatch".into(),
                ));
            }
        }

        let arg_values = self.read_values(args)?;
        match self.resolve_callee(func_idx)? {
            ResolvedCallee::Local {
                code,
                program,
                params_len,
                results_len,
            } => self.invoke_local(
                func_idx,
                code,
                unsafe { &*program },
                params_len as u16,
                params_len,
                results_len,
                unsafe { (&*program).frame.frame_prefix_size },
                &arg_values,
                results,
            ),
            ResolvedCallee::External {
                func_type,
                callback,
            } => {
                if func_type.params().len() != arg_values.len()
                    || func_type.results().len() != results.len()
                {
                    return Err(WasmError::internal(
                        "native call_indirect arity does not match function type".into(),
                    ));
                }

                let wasm_args = arg_values
                    .iter()
                    .zip(func_type.params().iter().copied())
                    .map(|(raw, ty)| Value::from_raw(*raw, ty))
                    .collect::<Vec<_>>();
                let mut returns = func_type
                    .results()
                    .iter()
                    .copied()
                    .map(Value::default_for_type)
                    .collect::<Vec<_>>();

                let store = self.store_mut()?;
                let mem_slice = if store.module().memories.is_empty() {
                    None
                } else {
                    Some(store.module_mut().memories[0].data.as_mut_slice())
                };
                let mut caller = Caller::new(mem_slice);
                callback(&mut caller, &wasm_args, &mut returns)?;

                for (place, value) in results.iter().copied().zip(returns.into_iter()) {
                    self.write_place(place, value.to_raw())?;
                }
                Ok(())
            }
        }
    }

    fn resolve_callee(&self, func_idx: u32) -> Result<ResolvedCallee, WasmError> {
        let store = self.store()?;
        let func = store.function(func_idx as usize);
        Ok(match func {
            FunctionInst::Local { spec, .. } => {
                let code = spec.get_native_code().ok_or_else(|| {
                    WasmError::internal("local callee is missing native code".into())
                })?;
                let program = code.program().ok_or_else(|| {
                    WasmError::internal("local callee is missing finalized native program".into())
                })?;
                ResolvedCallee::Local {
                    code: code as *const NativeCode,
                    program: program as *const NativeProgram,
                    params_len: func.func_type().params().len(),
                    results_len: func.func_type().results().len(),
                }
            }
            FunctionInst::External {
                func_type,
                callback,
            } => ResolvedCallee::External {
                func_type: func_type.clone(),
                callback: *callback,
            },
        })
    }

    fn invoke_local(
        &mut self,
        callee_func_idx: u32,
        code: *const NativeCode,
        callee: &NativeProgram,
        callee_params_count: u16,
        params_len: usize,
        results_len: usize,
        callee_frame_size: u16,
        args: &[u64],
        results: &[NativePlace],
    ) -> Result<(), WasmError> {
        if params_len != args.len() || results_len != results.len() {
            return Err(WasmError::internal(
                "native call_local arity does not match callee function type".into(),
            ));
        }
        if callee.frame.frame_prefix_size != callee_frame_size {
            return Err(WasmError::internal(
                "native call_local frame contract does not match callee".into(),
            ));
        }
        if params_len as u16 != callee_params_count {
            return Err(WasmError::internal(
                "native call_local param contract does not match callee".into(),
            ));
        }

        let caller_slots = layout::frame_slots_used(self.program);
        let callee_slots = layout::frame_slots_used(callee);
        let callee_fp = unsafe { self.fp.add(caller_slots) };
        let callee_end = unsafe { callee_fp.add(callee_slots) };
        if callee_end > self.ctx.stack_end {
            return Err(WasmError::exhaustion("stack overflow".into()));
        }
        if self.ctx.call_depth >= MAX_REFERENCE_CALL_DEPTH {
            return Err(WasmError::exhaustion("call stack exhausted".into()));
        }

        unsafe {
            for (index, value) in args.iter().copied().enumerate() {
                *callee_fp.add(index) = value;
            }
            let frame_prefix_slots = callee_frame_size as usize;
            if frame_prefix_slots > params_len {
                core::ptr::write_bytes(
                    callee_fp.add(params_len),
                    0,
                    frame_prefix_slots - params_len,
                );
            }
        }

        let saved_code = self.ctx.current_code;
        self.ctx.current_code = code;
        self.ctx.call_depth = self.ctx.call_depth.saturating_add(1);
        #[cfg(feature = "function-trace")]
        function_trace::native_function_trace_enter_func_idx(self.ctx, callee_func_idx);
        let result = if unsafe { (&*code).internal_entry().is_some() } {
            self.invoke_compiled_local(unsafe { &*code }, callee_fp)
        } else {
            execute_program(self.ctx, callee, callee_fp)
        };
        self.ctx.current_code = saved_code;
        result?;

        for (index, place) in results.iter().copied().enumerate() {
            let raw = unsafe { *callee_fp.add(index) };
            self.write_place(place, raw)?;
        }
        Ok(())
    }

    fn invoke_compiled_local(
        &mut self,
        code: &NativeCode,
        callee_fp: *mut u64,
    ) -> Result<(), WasmError> {
        let entry = code
            .entry
            .ok_or_else(|| WasmError::internal("direct local callee is missing entry".into()))?;
        unsafe {
            entry(self.ctx, callee_fp, 0, 0, 0, 0, 0, 0, 0);
        }
        self.ctx.refresh_cached_views();
        if let Some(error) = self.ctx.error.take() {
            return Err(error);
        }
        Ok(())
    }

    fn read_global(&self, idx: u32) -> Result<u64, WasmError> {
        let store = self.store()?;
        Ok(store.global(idx as usize).raw())
    }

    fn write_global(&mut self, idx: u32, raw: u64) -> Result<(), WasmError> {
        let (value_type, mutable) = {
            let store = self.store()?;
            let global = store.global(idx as usize);
            (global.value_type, global.mutable)
        };
        if !mutable {
            return Err(WasmError::invalid("global.set: immutable global".into()));
        }
        let store = self.store_mut()?;
        debug_assert_eq!(store.global(idx as usize).value_type, value_type);
        store.global_mut(idx as usize).set_raw(raw);
        Ok(())
    }

    fn memory_size(&self, mem_idx: u32) -> Result<usize, WasmError> {
        let store = self.store()?;
        Ok(store.memory(mem_idx as usize).current_pages())
    }

    fn memory_grow(&mut self, mem_idx: u32, delta: usize) -> Result<u64, WasmError> {
        let store = self.store_mut()?;
        let memory = store.memory_mut(mem_idx as usize);
        let old_pages = memory.current_pages();
        let new_pages = old_pages.saturating_add(delta);
        if new_pages > memory.limits.get_max() {
            return Ok(u32::MAX as u64);
        }
        memory
            .data
            .resize(new_pages.saturating_mul(WASM_PAGE_SIZE), 0);
        Ok(old_pages as u64)
    }

    fn memory_fill(
        &mut self,
        mem_idx: u32,
        dst: usize,
        value: u8,
        size: usize,
    ) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        let memory = store.memory_mut(mem_idx as usize);
        if dst.saturating_add(size) > memory.data.len() {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        memory.data[dst..dst + size].fill(value);
        Ok(())
    }

    fn memory_copy(
        &mut self,
        dst_idx: u32,
        src_idx: u32,
        dst: usize,
        src: usize,
        size: usize,
    ) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        if dst_idx == src_idx {
            let memory = store.memory_mut(dst_idx as usize);
            if src.saturating_add(size) > memory.data.len()
                || dst.saturating_add(size) > memory.data.len()
            {
                return Err(WasmError::trap("out of bounds memory access".into()));
            }
            memory.data.copy_within(src..src + size, dst);
            return Ok(());
        }

        let module = store.module_mut();
        let (src_mem, dst_mem) = if (src_idx as usize) < (dst_idx as usize) {
            let (left, right) = module.memories.split_at_mut(dst_idx as usize);
            (&left[src_idx as usize], &mut right[0])
        } else {
            let (left, right) = module.memories.split_at_mut(src_idx as usize);
            (&right[0], &mut left[dst_idx as usize])
        };
        if src.saturating_add(size) > src_mem.data.len()
            || dst.saturating_add(size) > dst_mem.data.len()
        {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        dst_mem.data[dst..dst + size].copy_from_slice(&src_mem.data[src..src + size]);
        Ok(())
    }

    fn memory_init(
        &mut self,
        mem_idx: u32,
        data_idx: u32,
        dst: usize,
        src: usize,
        size: usize,
    ) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        let module = store.module_mut();
        let Some(data_inst) = module.data.get(data_idx as usize) else {
            return Err(WasmError::trap("out of bounds memory access".into()));
        };

        if size == 0 {
            if src > data_inst.bytes.len() || dst > module.memories[mem_idx as usize].data.len() {
                return Err(WasmError::trap("out of bounds memory access".into()));
            }
            return Ok(());
        }
        if data_inst.is_dropped() || src.saturating_add(size) > data_inst.bytes.len() {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }

        let src_ptr = data_inst.bytes[src..].as_ptr();
        let mem_data = &mut module.memories[mem_idx as usize].data;
        if dst.saturating_add(size) > mem_data.len() {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        unsafe {
            core::ptr::copy_nonoverlapping(src_ptr, mem_data[dst..].as_mut_ptr(), size);
        }
        Ok(())
    }

    fn data_drop(&mut self, data_idx: u32) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        if let Some(data) = store.module_mut().data.get_mut(data_idx as usize) {
            data.drop_segment();
        }
        Ok(())
    }

    fn memory_range(
        &self,
        mem_idx: u32,
        address: u32,
        offset: u32,
        size: usize,
    ) -> Result<usize, WasmError> {
        let start = (address as usize).saturating_add(offset as usize);
        let store = self.store()?;
        let memory = store.memory(mem_idx as usize);
        if start.saturating_add(size) > memory.data.len() {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        Ok(start)
    }

    fn load_u8(&self, mem_idx: u32, address: u32, offset: u32) -> Result<u8, WasmError> {
        let start = self.memory_range(mem_idx, address, offset, 1)?;
        Ok(self.store()?.memory(mem_idx as usize).data[start])
    }

    fn load_i8(&self, mem_idx: u32, address: u32, offset: u32) -> Result<i8, WasmError> {
        Ok(self.load_u8(mem_idx, address, offset)? as i8)
    }

    fn load_u16(&self, mem_idx: u32, address: u32, offset: u32) -> Result<u16, WasmError> {
        let start = self.memory_range(mem_idx, address, offset, 2)?;
        let bytes = &self.store()?.memory(mem_idx as usize).data[start..start + 2];
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn load_i16(&self, mem_idx: u32, address: u32, offset: u32) -> Result<i16, WasmError> {
        Ok(self.load_u16(mem_idx, address, offset)? as i16)
    }

    fn load_u32(&self, mem_idx: u32, address: u32, offset: u32) -> Result<u32, WasmError> {
        let start = self.memory_range(mem_idx, address, offset, 4)?;
        let bytes = &self.store()?.memory(mem_idx as usize).data[start..start + 4];
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn load_i32(&self, mem_idx: u32, address: u32, offset: u32) -> Result<i32, WasmError> {
        Ok(self.load_u32(mem_idx, address, offset)? as i32)
    }

    fn load_u64(&self, mem_idx: u32, address: u32, offset: u32) -> Result<u64, WasmError> {
        let start = self.memory_range(mem_idx, address, offset, 8)?;
        let bytes = &self.store()?.memory(mem_idx as usize).data[start..start + 8];
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn load_i64(&self, mem_idx: u32, address: u32, offset: u32) -> Result<i64, WasmError> {
        Ok(self.load_u64(mem_idx, address, offset)? as i64)
    }

    fn store_bytes(
        &mut self,
        mem_idx: u32,
        address: u32,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), WasmError> {
        let start = self.memory_range(mem_idx, address, offset, bytes.len())?;
        let store = self.store_mut()?;
        let memory = store.memory_mut(mem_idx as usize);
        memory.data[start..start + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn store_u8(
        &mut self,
        mem_idx: u32,
        address: u32,
        offset: u32,
        value: u8,
    ) -> Result<(), WasmError> {
        self.store_bytes(mem_idx, address, offset, &[value])
    }

    fn store_u16(
        &mut self,
        mem_idx: u32,
        address: u32,
        offset: u32,
        value: u16,
    ) -> Result<(), WasmError> {
        self.store_bytes(mem_idx, address, offset, &value.to_le_bytes())
    }

    fn store_u32(
        &mut self,
        mem_idx: u32,
        address: u32,
        offset: u32,
        value: u32,
    ) -> Result<(), WasmError> {
        self.store_bytes(mem_idx, address, offset, &value.to_le_bytes())
    }

    fn store_u64(
        &mut self,
        mem_idx: u32,
        address: u32,
        offset: u32,
        value: u64,
    ) -> Result<(), WasmError> {
        self.store_bytes(mem_idx, address, offset, &value.to_le_bytes())
    }

    fn table_get(&self, table_idx: u32, elem_index: usize) -> Result<u64, WasmError> {
        let store = self.store()?;
        let table = store.table(table_idx as usize);
        let Some(handle) = table.elements.get(elem_index) else {
            return Err(WasmError::trap("out of bounds table access".into()));
        };
        Ok(usize::from(*handle) as u64)
    }

    fn table_set(
        &mut self,
        table_idx: u32,
        elem_index: usize,
        value: u64,
    ) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        let table = store.table_mut(table_idx as usize);
        let Some(slot) = table.elements.get_mut(elem_index) else {
            return Err(WasmError::trap("out of bounds table access".into()));
        };
        *slot = RefHandle::new(value as usize);
        Ok(())
    }

    fn table_size(&self, table_idx: u32) -> Result<usize, WasmError> {
        Ok(self.store()?.table(table_idx as usize).size())
    }

    fn table_grow(&mut self, table_idx: u32, value: u64, delta: usize) -> Result<u64, WasmError> {
        let store = self.store_mut()?;
        let table = store.table_mut(table_idx as usize);
        let Some(new_size) = table.elements.len().checked_add(delta) else {
            return Ok(u32::MAX as u64);
        };
        if new_size > table.limits.get_max() {
            return Ok(u32::MAX as u64);
        }
        let old_size = table.elements.len();
        let fill = RefHandle::new(value as usize);
        table.elements.resize_with(new_size, || fill);
        Ok(old_size as u64)
    }

    fn table_fill(
        &mut self,
        table_idx: u32,
        start: usize,
        value: u64,
        size: usize,
    ) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        let table = store.table_mut(table_idx as usize);
        if start.saturating_add(size) > table.elements.len() {
            return Err(WasmError::trap("out of bounds table access".into()));
        }
        table.elements[start..start + size].fill(RefHandle::new(value as usize));
        Ok(())
    }

    fn table_copy(
        &mut self,
        dst_idx: u32,
        src_idx: u32,
        dst: usize,
        src: usize,
        size: usize,
    ) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        if dst_idx == src_idx {
            let table = store.table_mut(dst_idx as usize);
            if src.saturating_add(size) > table.elements.len()
                || dst.saturating_add(size) > table.elements.len()
            {
                return Err(WasmError::trap("out of bounds table access".into()));
            }
            table.elements.copy_within(src..src + size, dst);
            return Ok(());
        }

        let module = store.module_mut();
        let (src_table, dst_table) = if (src_idx as usize) < (dst_idx as usize) {
            let (left, right) = module.tables.split_at_mut(dst_idx as usize);
            (&left[src_idx as usize], &mut right[0])
        } else {
            let (left, right) = module.tables.split_at_mut(src_idx as usize);
            (&right[0], &mut left[dst_idx as usize])
        };
        if src.saturating_add(size) > src_table.elements.len()
            || dst.saturating_add(size) > dst_table.elements.len()
        {
            return Err(WasmError::trap("out of bounds table access".into()));
        }
        dst_table.elements[dst..dst + size].copy_from_slice(&src_table.elements[src..src + size]);
        Ok(())
    }

    fn table_init(
        &mut self,
        table_idx: u32,
        elem_idx: u32,
        dst: usize,
        src: usize,
        size: usize,
    ) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        let module = store.module_mut();
        let Some(elem_inst) = module.elements.get(elem_idx as usize) else {
            return Err(WasmError::trap("out of bounds table access".into()));
        };

        if size == 0 {
            if src > elem_inst.refs.len() || dst > module.tables[table_idx as usize].elements.len()
            {
                return Err(WasmError::trap("out of bounds table access".into()));
            }
            return Ok(());
        }
        if src.saturating_add(size) > elem_inst.refs.len()
            || dst.saturating_add(size) > module.tables[table_idx as usize].elements.len()
        {
            return Err(WasmError::trap("out of bounds table access".into()));
        }
        if elem_inst.is_dropped() {
            return Err(WasmError::trap("out of bounds table access".into()));
        }

        for index in 0..size {
            module.tables[table_idx as usize].elements[dst + index] =
                module.elements[elem_idx as usize].refs[src + index];
        }
        Ok(())
    }

    fn elem_drop(&mut self, elem_idx: u32) -> Result<(), WasmError> {
        let store = self.store_mut()?;
        if let Some(elem) = store.module_mut().elements.get_mut(elem_idx as usize) {
            elem.drop_segment();
        }
        Ok(())
    }

    fn store(&self) -> Result<&Store, WasmError> {
        let Some(store) = (unsafe { self.ctx.store.as_ref() }) else {
            return Err(WasmError::internal(
                "native reference machine is missing store context".into(),
            ));
        };
        Ok(store)
    }

    fn store_mut(&mut self) -> Result<&mut Store, WasmError> {
        let Some(store) = (unsafe { self.ctx.store.as_mut() }) else {
            return Err(WasmError::internal(
                "native reference machine is missing mutable store context".into(),
            ));
        };
        Ok(store)
    }
}

fn eval_i32_compare(kind: NativeCompareKind, lhs: u32, rhs: u32) -> bool {
    match kind {
        NativeCompareKind::Eq => lhs == rhs,
        NativeCompareKind::Ne => lhs != rhs,
        NativeCompareKind::LtS => (lhs as i32) < (rhs as i32),
        NativeCompareKind::LtU => lhs < rhs,
        NativeCompareKind::GtS => (lhs as i32) > (rhs as i32),
        NativeCompareKind::GtU => lhs > rhs,
        NativeCompareKind::LeS => (lhs as i32) <= (rhs as i32),
        NativeCompareKind::LeU => lhs <= rhs,
        NativeCompareKind::GeS => (lhs as i32) >= (rhs as i32),
        NativeCompareKind::GeU => lhs >= rhs,
        NativeCompareKind::Lt | NativeCompareKind::Gt | NativeCompareKind::Le | NativeCompareKind::Ge => {
            false
        }
    }
}

fn eval_i64_compare(kind: NativeCompareKind, lhs: u64, rhs: u64) -> bool {
    match kind {
        NativeCompareKind::Eq => lhs == rhs,
        NativeCompareKind::Ne => lhs != rhs,
        NativeCompareKind::LtS => (lhs as i64) < (rhs as i64),
        NativeCompareKind::LtU => lhs < rhs,
        NativeCompareKind::GtS => (lhs as i64) > (rhs as i64),
        NativeCompareKind::GtU => lhs > rhs,
        NativeCompareKind::LeS => (lhs as i64) <= (rhs as i64),
        NativeCompareKind::LeU => lhs <= rhs,
        NativeCompareKind::GeS => (lhs as i64) >= (rhs as i64),
        NativeCompareKind::GeU => lhs >= rhs,
        NativeCompareKind::Lt | NativeCompareKind::Gt | NativeCompareKind::Le | NativeCompareKind::Ge => {
            false
        }
    }
}

fn eval_f32_compare(kind: NativeCompareKind, lhs: u32, rhs: u32) -> bool {
    let lhs = f32::from_bits(lhs);
    let rhs = f32::from_bits(rhs);
    match kind {
        NativeCompareKind::Eq => lhs == rhs,
        NativeCompareKind::Ne => lhs != rhs,
        NativeCompareKind::Lt => lhs < rhs,
        NativeCompareKind::Gt => lhs > rhs,
        NativeCompareKind::Le => lhs <= rhs,
        NativeCompareKind::Ge => lhs >= rhs,
        NativeCompareKind::LtS
        | NativeCompareKind::LtU
        | NativeCompareKind::GtS
        | NativeCompareKind::GtU
        | NativeCompareKind::LeS
        | NativeCompareKind::LeU
        | NativeCompareKind::GeS
        | NativeCompareKind::GeU => false,
    }
}

fn eval_f64_compare(kind: NativeCompareKind, lhs: u64, rhs: u64) -> bool {
    let lhs = f64::from_bits(lhs);
    let rhs = f64::from_bits(rhs);
    match kind {
        NativeCompareKind::Eq => lhs == rhs,
        NativeCompareKind::Ne => lhs != rhs,
        NativeCompareKind::Lt => lhs < rhs,
        NativeCompareKind::Gt => lhs > rhs,
        NativeCompareKind::Le => lhs <= rhs,
        NativeCompareKind::Ge => lhs >= rhs,
        NativeCompareKind::LtS
        | NativeCompareKind::LtU
        | NativeCompareKind::GtS
        | NativeCompareKind::GtU
        | NativeCompareKind::LeS
        | NativeCompareKind::LeU
        | NativeCompareKind::GeS
        | NativeCompareKind::GeU => false,
    }
}

#[inline]
fn arg_u64(args: &[u64], index: usize) -> Result<u64, WasmError> {
    args.get(index)
        .copied()
        .ok_or_else(|| WasmError::internal("native leaf argument out of range".into()))
}

#[inline]
fn arg_u32(args: &[u64], index: usize) -> Result<u32, WasmError> {
    Ok(arg_u64(args, index)? as u32)
}

#[inline]
fn arg_i32(args: &[u64], index: usize) -> Result<i32, WasmError> {
    Ok(arg_u64(args, index)? as i32)
}

#[inline]
fn arg_i64(args: &[u64], index: usize) -> Result<i64, WasmError> {
    Ok(arg_u64(args, index)? as i64)
}

#[inline]
fn bin_i32(args: &[u64], op: impl FnOnce(i32, i32) -> i32) -> Result<i32, WasmError> {
    Ok(op(arg_i32(args, 0)?, arg_i32(args, 1)?))
}

#[inline]
fn as_f32(bits: u32) -> f32 {
    f32::from_bits(bits)
}

#[inline]
fn as_f64(bits: u64) -> f64 {
    f64::from_bits(bits)
}

#[inline]
fn f32_bits(value: f32) -> u64 {
    u64::from(value.to_bits())
}

#[inline]
fn f64_bits(value: f64) -> u64 {
    value.to_bits()
}

#[inline]
fn ceil_f32(value: f32) -> f32 {
    unsafe { ceilf(value) }
}

#[inline]
fn floor_f32(value: f32) -> f32 {
    unsafe { floorf(value) }
}

#[inline]
fn trunc_f32(value: f32) -> f32 {
    unsafe { truncf(value) }
}

#[inline]
fn sqrt_f32(value: f32) -> f32 {
    unsafe { sqrtf(value) }
}

#[inline]
fn ceil_f64(value: f64) -> f64 {
    unsafe { ceil(value) }
}

#[inline]
fn floor_f64(value: f64) -> f64 {
    unsafe { floor(value) }
}

#[inline]
fn trunc_f64(value: f64) -> f64 {
    unsafe { trunc(value) }
}

#[inline]
fn sqrt_f64(value: f64) -> f64 {
    unsafe { sqrt(value) }
}

#[inline]
fn trap_div_by_zero() -> WasmError {
    WasmError::trap("integer divide by zero".into())
}

#[inline]
fn trap_integer_overflow() -> WasmError {
    WasmError::trap("integer overflow".into())
}

#[inline]
fn trap_invalid_conversion() -> WasmError {
    WasmError::trap("invalid conversion to integer".into())
}

fn i32_div_s(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_i32(args, 0)?;
    let rhs = arg_i32(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    if lhs == i32::MIN && rhs == -1 {
        return Err(trap_integer_overflow());
    }
    Ok(u64::from(lhs.wrapping_div(rhs) as u32))
}

fn i32_div_u(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_u32(args, 0)?;
    let rhs = arg_u32(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    Ok(u64::from(lhs / rhs))
}

fn i32_rem_s(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_i32(args, 0)?;
    let rhs = arg_i32(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    Ok(u64::from(lhs.wrapping_rem(rhs) as u32))
}

fn i32_rem_u(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_u32(args, 0)?;
    let rhs = arg_u32(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    Ok(u64::from(lhs % rhs))
}

fn i64_div_s(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_i64(args, 0)?;
    let rhs = arg_i64(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    if lhs == i64::MIN && rhs == -1 {
        return Err(trap_integer_overflow());
    }
    Ok(lhs.wrapping_div(rhs) as u64)
}

fn i64_div_u(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_u64(args, 0)?;
    let rhs = arg_u64(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    Ok(lhs / rhs)
}

fn i64_rem_s(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_i64(args, 0)?;
    let rhs = arg_i64(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    Ok(lhs.wrapping_rem(rhs) as u64)
}

fn i64_rem_u(args: &[u64]) -> Result<u64, WasmError> {
    let lhs = arg_u64(args, 0)?;
    let rhs = arg_u64(args, 1)?;
    if rhs == 0 {
        return Err(trap_div_by_zero());
    }
    Ok(lhs % rhs)
}

fn wasm_f32_min_bits(left_bits: u32, right_bits: u32) -> u64 {
    const NEG_ZERO: u32 = 0x8000_0000;
    let left = as_f32(left_bits);
    let right = as_f32(right_bits);
    if left.is_nan() || right.is_nan() {
        return f32_bits(f32::NAN);
    }
    if left == right {
        return u64::from(if left_bits == NEG_ZERO || right_bits == NEG_ZERO {
            NEG_ZERO
        } else {
            left_bits
        });
    }
    u64::from(if left < right { left_bits } else { right_bits })
}

fn wasm_f32_max_bits(left_bits: u32, right_bits: u32) -> u64 {
    let left = as_f32(left_bits);
    let right = as_f32(right_bits);
    if left.is_nan() || right.is_nan() {
        return f32_bits(f32::NAN);
    }
    if left == right {
        return u64::from(if left_bits == 0 || right_bits == 0 {
            0
        } else {
            left_bits
        });
    }
    u64::from(if left > right { left_bits } else { right_bits })
}

fn wasm_f64_min_bits(left_bits: u64, right_bits: u64) -> u64 {
    const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
    let left = as_f64(left_bits);
    let right = as_f64(right_bits);
    if left.is_nan() || right.is_nan() {
        return f64_bits(f64::NAN);
    }
    if left == right {
        return if left_bits == NEG_ZERO || right_bits == NEG_ZERO {
            NEG_ZERO
        } else {
            left_bits
        };
    }
    if left < right {
        left_bits
    } else {
        right_bits
    }
}

fn wasm_f64_max_bits(left_bits: u64, right_bits: u64) -> u64 {
    let left = as_f64(left_bits);
    let right = as_f64(right_bits);
    if left.is_nan() || right.is_nan() {
        return f64_bits(f64::NAN);
    }
    if left == right {
        return if left_bits == 0 || right_bits == 0 {
            0
        } else {
            left_bits
        };
    }
    if left > right {
        left_bits
    } else {
        right_bits
    }
}

fn wasm_f32_nearest_bits(bits: u32) -> u64 {
    let value = as_f32(bits);
    if !value.is_finite() {
        return u64::from(bits);
    }
    let floor = floor_f32(value);
    let diff = value - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    f32_bits(rounded)
}

fn wasm_f64_nearest_bits(bits: u64) -> u64 {
    let value = as_f64(bits);
    if !value.is_finite() {
        return bits;
    }
    let floor = floor_f64(value);
    let diff = value - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    f64_bits(rounded)
}

fn trunc_f32_to_i32_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits) as f64;
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -2147483649.0 || value >= 2147483648.0 {
        return Err(trap_integer_overflow());
    }
    Ok(u64::from((value as i32) as u32))
}

fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits) as f64;
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -1.0 || value >= 4294967296.0 {
        return Err(trap_integer_overflow());
    }
    Ok(u64::from(value as u32))
}

fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -2147483649.0 || value >= 2147483648.0 {
        return Err(trap_integer_overflow());
    }
    Ok(u64::from((value as i32) as u32))
}

fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -1.0 || value >= 4294967296.0 {
        return Err(trap_integer_overflow());
    }
    Ok(u64::from(value as u32))
}

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits) as f64;
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(trap_integer_overflow());
    }
    Ok((value as i64) as u64)
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits) as f64;
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(trap_integer_overflow());
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(trap_integer_overflow());
    }
    Ok((value as i64) as u64)
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(trap_invalid_conversion());
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(trap_integer_overflow());
    }
    Ok(value as u64)
}

fn trunc_sat_f32_to_i32_s(bits: u32) -> u64 {
    let value = as_f32(bits) as f64;
    if value.is_nan() {
        0
    } else if value <= i32::MIN as f64 {
        u64::from(i32::MIN as u32)
    } else if value >= i32::MAX as f64 {
        u64::from(i32::MAX as u32)
    } else {
        u64::from((value as i32) as u32)
    }
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let value = as_f32(bits) as f64;
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f64 {
        u64::from(u32::MAX)
    } else {
        u64::from(value as u32)
    }
}

fn trunc_sat_f64_to_i32_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        0
    } else if value <= i32::MIN as f64 {
        u64::from(i32::MIN as u32)
    } else if value >= i32::MAX as f64 {
        u64::from(i32::MAX as u32)
    } else {
        u64::from((value as i32) as u32)
    }
}

fn trunc_sat_f64_to_i32_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f64 {
        u64::from(u32::MAX)
    } else {
        u64::from(value as u32)
    }
}

fn trunc_sat_f32_to_i64_s(bits: u32) -> u64 {
    let value = as_f32(bits) as f64;
    if value.is_nan() {
        0
    } else if value <= i64::MIN as f64 {
        i64::MIN as u64
    } else if value >= i64::MAX as f64 {
        i64::MAX as u64
    } else {
        (value as i64) as u64
    }
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let value = as_f32(bits) as f64;
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

fn trunc_sat_f64_to_i64_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        0
    } else if value <= i64::MIN as f64 {
        i64::MIN as u64
    } else if value >= i64::MAX as f64 {
        i64::MAX as u64
    } else {
        (value as i64) as u64
    }
}

fn trunc_sat_f64_to_i64_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

#[inline]
pub fn execute_program(
    ctx: &mut NativeContext,
    program: &NativeProgram,
    fp: *mut u64,
) -> Result<(), WasmError> {
    let result = ReferenceMachine::new(ctx, program, fp).run();
    #[cfg(feature = "function-trace")]
    if let Err(ref error) = result {
        function_trace::native_trap_current(ctx, error);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::execute_program;
    use crate::vm::lir::leaf::LirLeafOp;
    use crate::vm::{
        backend::BackendConfig,
        native::{
            abi::{BlockAbi, NativeAbi, NativeLocation, NativePlace, NativeValue},
            ir::{
                NativeBlock, NativeBlockId, NativeInst, NativeInstKind, NativeProgram,
                NativeTerminator,
            },
            runtime::context::NativeContext,
        },
        plan::frame::FramePlanner,
    };

    fn test_abi() -> NativeAbi {
        NativeAbi::from(BackendConfig {
            ctx_register_count: 1,
            fp_register_count: 1,
            tmp_register_count: 1,
            hot_local_count: 0,
            tos_register_count: 0,
        })
    }

    #[test]
    fn executes_simple_add_program() {
        let frame = {
            let planner = FramePlanner::new(2);
            let (planner, _) = planner.reserve_operands(0);
            planner.finish()
        };
        let program = NativeProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![NativeBlock {
                id: NativeBlockId(0),
                entry_abi: BlockAbi::default(),
                ops: alloc::vec![NativeInst {
                    kind: NativeInstKind::Leaf {
                        op: LirLeafOp::I32Add,
                        args: alloc::vec![
                            NativeValue::Place(NativePlace::Frame(frame.local_slot(0))),
                            NativeValue::Place(NativePlace::Frame(frame.local_slot(1))),
                        ],
                        results: alloc::vec![NativePlace::Location(NativeLocation::Tmp(0))],
                    },
                }],
                terminator: NativeTerminator::Return {
                    values: alloc::vec![NativeValue::Place(NativePlace::Location(
                        NativeLocation::Tmp(0),
                    ))],
                },
            }],
            abi: test_abi(),
            frame,
            spill_slots: 0,
            hot_locals: None,
        };

        let mut frame_mem = [0u64; 8];
        frame_mem[0] = 3;
        frame_mem[1] = 4;
        let mut ctx = NativeContext::new(
            core::ptr::null_mut(),
            unsafe { frame_mem.as_mut_ptr().add(frame_mem.len()) },
            core::ptr::null(),
        );

        execute_program(&mut ctx, &program, frame_mem.as_mut_ptr()).expect("program executes");
        assert_eq!(frame_mem[0], 7);
    }
}
