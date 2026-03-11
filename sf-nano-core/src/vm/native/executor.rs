//! Native-owned executor for finalized native IR.
//!
//! This is the semantic bring-up path for the native backend:
//! - the shared frontend still produces `NativeProgram`
//! - native finalization still produces a native code object
//! - runtime enters through the native entry ABI
//! - the shared native entry executes finalized native IR directly
//!
//! This keeps execution within the native backend boundary while direct ARM64
//! block lowering is still being filled in.

use alloc::{format, vec, vec::Vec};

use crate::{
    constants::WASM_PAGE_SIZE,
    error::WasmError,
    module::entities::FunctionSpec,
    value_type::ValueType,
    vm::{
        entities::{Caller, FunctionInst, MemInst},
        value::{RefHandle, Value},
    },
};

use super::{
    code::NativeCode,
    context::NativeContext,
    ir::{
        NativeBlock, NativeBlockId, NativeEdge, NativeInst, NativeInstKind, NativeProgram,
        NativeReg, NativeTerminator,
    },
    precompile,
};

const MAX_CALL_DEPTH: u64 = 300;

/// Shared runtime entry for all compiled native functions during bring-up.
pub unsafe extern "C" fn shared_native_entry(
    ctx: *mut NativeContext,
    fp: *mut u64,
    _l0: u64,
    _l1: u64,
    _l2: u64,
    _t0: u64,
    _t1: u64,
    _t2: u64,
    _t3: u64,
) {
    let ctx = unsafe { &mut *ctx };
    let Some(code) = (unsafe { ctx.current_code.as_ref() }) else {
        ctx.error = Some(WasmError::internal(
            "native entry called without current code".into(),
        ));
        return;
    };

    match execute_code(ctx, code, fp) {
        Ok(results) => {
            for (index, value) in results.into_iter().enumerate() {
                unsafe {
                    *fp.add(index) = value;
                }
            }
        }
        Err(error) => {
            ctx.error = Some(error);
        }
    }
}

pub fn execute_code(
    ctx: &mut NativeContext,
    code: &NativeCode,
    fp: *mut u64,
) -> Result<Vec<u64>, WasmError> {
    let program = code
        .program()
        .ok_or_else(|| WasmError::internal("native code is missing finalized program".into()))?;
    let mut executor = NativeExecutor {
        ctx,
        code,
        program,
        fp,
        regs: vec![0; code.vreg_count()],
        hot_locals: vec![0; program.abi.hot_local_count as usize],
    };
    executor.execute()
}

pub fn frame_slots_used(program: &NativeProgram) -> usize {
    program.frame.operands.end().0 as usize
}

struct NativeExecutor<'a> {
    ctx: &'a mut NativeContext,
    code: &'a NativeCode,
    program: &'a NativeProgram,
    fp: *mut u64,
    regs: Vec<u64>,
    hot_locals: Vec<u64>,
}

impl<'a> NativeExecutor<'a> {
    fn execute(&mut self) -> Result<Vec<u64>, WasmError> {
        let mut current = self.program.entry;

        loop {
            let block = self.block(current)?.clone();
            for op in &block.ops {
                self.execute_inst(op)?;
            }

            match &block.terminator {
                NativeTerminator::Goto(edge) => {
                    self.apply_edge(edge);
                    current = edge.target;
                }
                NativeTerminator::Branch {
                    cond,
                    then_edge,
                    else_edge,
                } => {
                    let edge = if (self.read_reg(*cond) as u32) != 0 {
                        then_edge
                    } else {
                        else_edge
                    };
                    self.apply_edge(edge);
                    current = edge.target;
                }
                NativeTerminator::BrTable { index, entries } => {
                    let table_index = self.read_reg(*index) as u32 as usize;
                    let edge = entries
                        .get(table_index)
                        .or_else(|| entries.last())
                        .ok_or_else(|| WasmError::internal("native br_table has no entries".into()))?;
                    self.apply_edge(edge);
                    current = edge.target;
                }
                NativeTerminator::Return { values } => {
                    return Ok(values.iter().map(|value| self.read_reg(*value)).collect());
                }
                NativeTerminator::TrapUnreachable => {
                    return Err(WasmError::trap("unreachable".into()));
                }
            }
        }
    }

    fn execute_inst(&mut self, inst: &NativeInst) -> Result<(), WasmError> {
        match &inst.kind {
            NativeInstKind::Leaf { op, args, results } => {
                let args = args.iter().map(|arg| self.read_reg(*arg)).collect::<Vec<_>>();
                match execute_leaf(self.ctx, op, &args)? {
                    LeafResult::None => {
                        if !results.is_empty() {
                            return Err(WasmError::internal(
                                "native leaf returned no value for non-empty result list".into(),
                            ));
                        }
                    }
                    LeafResult::One(value) => {
                        if results.len() != 1 {
                            return Err(WasmError::internal(
                                "native leaf returned one value for non-unary result list".into(),
                            ));
                        }
                        self.write_reg(results[0], value);
                    }
                }
            }
            NativeInstKind::ReadOperandSlot { slot, dst }
            | NativeInstKind::ReadFrameLocal {
                frame_slot: slot,
                dst,
            } => {
                self.write_reg(*dst, self.read_frame(*slot));
            }
            NativeInstKind::WriteOperandSlot { slot, src }
            | NativeInstKind::WriteFrameLocal {
                frame_slot: slot,
                src,
            } => {
                self.write_frame(*slot, self.read_reg(*src));
            }
            NativeInstKind::ReadHotLocal { reg, dst } => {
                let value = *self
                    .hot_locals
                    .get(*reg as usize)
                    .ok_or_else(|| WasmError::internal("native hot-local read out of range".into()))?;
                self.write_reg(*dst, value);
            }
            NativeInstKind::WriteHotLocal { reg, src } => {
                let value = self.read_reg(*src);
                let slot = self
                    .hot_locals
                    .get_mut(*reg as usize)
                    .ok_or_else(|| WasmError::internal("native hot-local write out of range".into()))?;
                *slot = value;
            }
            NativeInstKind::CallExternal {
                func_idx,
                args,
                results,
            } => {
                let raw_args = args.iter().map(|arg| self.read_reg(*arg)).collect::<Vec<_>>();
                let returned = self.call_function(*func_idx, &raw_args)?;
                self.write_results(results, &returned)?;
            }
            NativeInstKind::CallInternal {
                callee,
                args,
                results,
            } => {
                let raw_args = args.iter().map(|arg| self.read_reg(*arg)).collect::<Vec<_>>();
                let returned = self.call_function(*callee, &raw_args)?;
                self.write_results(results, &returned)?;
            }
            NativeInstKind::CallIndirect {
                type_idx,
                table_idx,
                index,
                args,
                results,
            } => {
                let raw_args = args.iter().map(|arg| self.read_reg(*arg)).collect::<Vec<_>>();
                let returned = self.call_indirect(*type_idx, *table_idx, self.read_reg(*index), &raw_args)?;
                self.write_results(results, &returned)?;
            }
        }
        Ok(())
    }

    fn write_results(&mut self, results: &[NativeReg], returned: &[u64]) -> Result<(), WasmError> {
        if results.len() != returned.len() {
            return Err(WasmError::internal(format!(
                "native call result arity mismatch: got {}, expected {}",
                returned.len(),
                results.len()
            )));
        }
        for (reg, value) in results.iter().zip(returned.iter().copied()) {
            self.write_reg(*reg, value);
        }
        Ok(())
    }

    fn call_indirect(
        &mut self,
        type_idx: u32,
        table_idx: u32,
        raw_index: u64,
        args: &[u64],
    ) -> Result<Vec<u64>, WasmError> {
        let elem_index = raw_index as u32 as usize;
        let store = self.store();
        let expected_ty = store
            .module()
            .get_type(type_idx)
            .cloned()
            .ok_or_else(|| WasmError::trap("indirect call type error".into()))?;
        let table = store.table(table_idx as usize);
        if elem_index >= table.elements.len() {
            return Err(WasmError::trap("undefined element".into()));
        }
        let func_ref = table.elements[elem_index];
        if func_ref.is_null() {
            return Err(WasmError::trap("uninitialized element".into()));
        }

        let func_index = func_ref.raw_value();
        if func_index >= store.module().functions.len() {
            return Err(WasmError::trap("uninitialized element".into()));
        }
        let callee = store.function(func_index);
        let actual_type = callee.func_type();
        if *actual_type != *expected_ty {
            let type_context = &store.module().types;
            let mut equivalent = false;
            for (index, func_type) in type_context.as_slice().iter().enumerate() {
                if **func_type == *actual_type {
                    equivalent = type_context.types_equivalent(index as u32, type_idx);
                    break;
                }
            }
            if !equivalent {
                return Err(WasmError::trap("indirect call type mismatch".into()));
            }
        }

        self.call_function(func_index as u32, args)
    }

    fn call_function(&mut self, func_idx: u32, args: &[u64]) -> Result<Vec<u64>, WasmError> {
        let func_ptr = self.store().function(func_idx as usize) as *const FunctionInst;
        let func = unsafe { &*func_ptr };
        match func {
            FunctionInst::External { .. } => invoke_external_function(self.store_mut(), func, args),
            FunctionInst::Local { spec, .. } => self.call_local(spec, args),
        }
    }

    fn call_local(&mut self, spec: &FunctionSpec, args: &[u64]) -> Result<Vec<u64>, WasmError> {
        if !spec.has_native_code() {
            precompile::precompile_module(self.store())?;
        }
        let code = spec
            .get_native_code()
            .ok_or_else(|| WasmError::internal("native callee is not compiled".into()))?;
        let program = code
            .program()
            .ok_or_else(|| WasmError::internal("native callee is missing finalized program".into()))?;
        let params_len = spec.func_type().params().len();
        if args.len() != params_len {
            return Err(WasmError::internal(format!(
                "native call arg mismatch: got {}, expected {}",
                args.len(),
                params_len
            )));
        }

        if self.ctx.call_depth >= MAX_CALL_DEPTH {
            return Err(WasmError::exhaustion("call stack exhausted".into()));
        }
        let current_slots = frame_slots_used(self.program);
        let callee_slots = frame_slots_used(program);
        let callee_fp = unsafe { self.fp.add(current_slots) };
        let callee_end = unsafe { callee_fp.add(callee_slots) };
        if callee_end > self.ctx.stack_end {
            return Err(WasmError::exhaustion("stack overflow".into()));
        }

        unsafe {
            core::ptr::write_bytes(callee_fp, 0, callee_slots);
            for (index, value) in args.iter().copied().enumerate() {
                *callee_fp.add(index) = value;
            }
        }

        self.ctx.call_depth += 1;
        let result = execute_code(self.ctx, code, callee_fp);
        self.ctx.call_depth -= 1;
        result
    }

    fn apply_edge(&mut self, edge: &NativeEdge) {
        let snapshot = edge
            .copies
            .iter()
            .map(|copy| self.read_reg(copy.src))
            .collect::<Vec<_>>();
        for (copy, value) in edge.copies.iter().zip(snapshot) {
            self.write_reg(copy.dst, value);
        }
    }

    fn block(&self, id: NativeBlockId) -> Result<&NativeBlock, WasmError> {
        self.program
            .blocks
            .get(id.as_usize())
            .ok_or_else(|| WasmError::internal(format!("native block {} is out of range", id.0)))
    }

    fn read_reg(&self, reg: NativeReg) -> u64 {
        self.regs[reg.0 as usize]
    }

    fn write_reg(&mut self, reg: NativeReg, value: u64) {
        self.regs[reg.0 as usize] = value;
    }

    fn read_frame(&self, slot: crate::vm::plan::frame::FrameSlot) -> u64 {
        unsafe { *self.fp.add(slot.0 as usize) }
    }

    fn write_frame(&mut self, slot: crate::vm::plan::frame::FrameSlot, value: u64) {
        unsafe {
            *self.fp.add(slot.0 as usize) = value;
        }
    }

    fn store(&self) -> &crate::vm::store::Store {
        unsafe { &*self.ctx.store }
    }

    fn store_mut(&mut self) -> &mut crate::vm::store::Store {
        unsafe { &mut *self.ctx.store }
    }
}

enum LeafResult {
    None,
    One(u64),
}

fn execute_leaf(
    ctx: &mut NativeContext,
    op: &crate::vm::lir::leaf::LirLeafOp,
    args: &[u64],
) -> Result<LeafResult, WasmError> {
    use crate::vm::lir::leaf::LirLeafOp as Op;

    match op {
        Op::I32Const { value } => Ok(LeafResult::One(*value as u64)),
        Op::I64Const { value } => Ok(LeafResult::One(*value)),
        Op::F32Const { value } => Ok(LeafResult::One(*value as u64)),
        Op::F64Const { value } => Ok(LeafResult::One(*value)),

        Op::I32Add => Ok(LeafResult::One((as_u32(args[0]).wrapping_add(as_u32(args[1]))) as u64)),
        Op::I32Sub => Ok(LeafResult::One((as_u32(args[0]).wrapping_sub(as_u32(args[1]))) as u64)),
        Op::I32Mul => Ok(LeafResult::One((as_u32(args[0]).wrapping_mul(as_u32(args[1]))) as u64)),
        Op::I32And => Ok(LeafResult::One((as_u32(args[0]) & as_u32(args[1])) as u64)),
        Op::I32Or => Ok(LeafResult::One((as_u32(args[0]) | as_u32(args[1])) as u64)),
        Op::I32Xor => Ok(LeafResult::One((as_u32(args[0]) ^ as_u32(args[1])) as u64)),
        Op::I32Shl => Ok(LeafResult::One(
            as_u32(args[0]).wrapping_shl(as_u32(args[1]) & 31) as u64,
        )),
        Op::I32ShrU => Ok(LeafResult::One(
            as_u32(args[0]).wrapping_shr(as_u32(args[1]) & 31) as u64,
        )),
        Op::I32ShrS => Ok(LeafResult::One(
            (as_i32(args[0]) >> (as_u32(args[1]) & 31)) as u32 as u64,
        )),
        Op::I32Rotl => Ok(LeafResult::One(
            as_u32(args[0]).rotate_left(as_u32(args[1]) & 31) as u64,
        )),
        Op::I32Rotr => Ok(LeafResult::One(
            as_u32(args[0]).rotate_right(as_u32(args[1]) & 31) as u64,
        )),
        Op::I32DivS => {
            let rhs = as_i32(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            let lhs = as_i32(args[0]);
            if lhs == i32::MIN && rhs == -1 {
                return Err(WasmError::trap("integer overflow".into()));
            }
            Ok(LeafResult::One((lhs / rhs) as u32 as u64))
        }
        Op::I32DivU => {
            let rhs = as_u32(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            Ok(LeafResult::One((as_u32(args[0]) / rhs) as u64))
        }
        Op::I32RemS => {
            let rhs = as_i32(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            let lhs = as_i32(args[0]);
            if lhs == i32::MIN && rhs == -1 {
                return Ok(LeafResult::One(0));
            }
            Ok(LeafResult::One((lhs % rhs) as u32 as u64))
        }
        Op::I32RemU => {
            let rhs = as_u32(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            Ok(LeafResult::One((as_u32(args[0]) % rhs) as u64))
        }

        Op::I64Add => Ok(LeafResult::One(as_u64(args[0]).wrapping_add(as_u64(args[1])))),
        Op::I64Sub => Ok(LeafResult::One(as_u64(args[0]).wrapping_sub(as_u64(args[1])))),
        Op::I64Mul => Ok(LeafResult::One(as_u64(args[0]).wrapping_mul(as_u64(args[1])))),
        Op::I64And => Ok(LeafResult::One(as_u64(args[0]) & as_u64(args[1]))),
        Op::I64Or => Ok(LeafResult::One(as_u64(args[0]) | as_u64(args[1]))),
        Op::I64Xor => Ok(LeafResult::One(as_u64(args[0]) ^ as_u64(args[1]))),
        Op::I64Shl => Ok(LeafResult::One(
            as_u64(args[0]).wrapping_shl((as_u64(args[1]) & 63) as u32),
        )),
        Op::I64ShrU => Ok(LeafResult::One(
            as_u64(args[0]).wrapping_shr((as_u64(args[1]) & 63) as u32),
        )),
        Op::I64ShrS => Ok(LeafResult::One(
            (as_i64(args[0]) >> ((as_u64(args[1]) & 63) as u32)) as u64,
        )),
        Op::I64Rotl => Ok(LeafResult::One(
            as_u64(args[0]).rotate_left((as_u64(args[1]) & 63) as u32),
        )),
        Op::I64Rotr => Ok(LeafResult::One(
            as_u64(args[0]).rotate_right((as_u64(args[1]) & 63) as u32),
        )),
        Op::I64DivS => {
            let rhs = as_i64(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            let lhs = as_i64(args[0]);
            if lhs == i64::MIN && rhs == -1 {
                return Err(WasmError::trap("integer overflow".into()));
            }
            Ok(LeafResult::One((lhs / rhs) as u64))
        }
        Op::I64DivU => {
            let rhs = as_u64(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            Ok(LeafResult::One(as_u64(args[0]) / rhs))
        }
        Op::I64RemS => {
            let rhs = as_i64(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            let lhs = as_i64(args[0]);
            if lhs == i64::MIN && rhs == -1 {
                return Ok(LeafResult::One(0));
            }
            Ok(LeafResult::One((lhs % rhs) as u64))
        }
        Op::I64RemU => {
            let rhs = as_u64(args[1]);
            if rhs == 0 {
                return Err(WasmError::trap("integer divide by zero".into()));
            }
            Ok(LeafResult::One(as_u64(args[0]) % rhs))
        }

        Op::F32Add => Ok(LeafResult::One((as_f32(args[0]) + as_f32(args[1])).to_bits() as u64)),
        Op::F32Sub => Ok(LeafResult::One((as_f32(args[0]) - as_f32(args[1])).to_bits() as u64)),
        Op::F32Mul => Ok(LeafResult::One((as_f32(args[0]) * as_f32(args[1])).to_bits() as u64)),
        Op::F32Div => Ok(LeafResult::One((as_f32(args[0]) / as_f32(args[1])).to_bits() as u64)),
        Op::F32Min => Ok(LeafResult::One(f32_min_bits(as_u32(args[0]), as_u32(args[1])) as u64)),
        Op::F32Max => Ok(LeafResult::One(f32_max_bits(as_u32(args[0]), as_u32(args[1])) as u64)),
        Op::F32Copysign => Ok(LeafResult::One(
            f32_copysign_bits(as_u32(args[0]), as_u32(args[1])) as u64,
        )),
        Op::F64Add => Ok(LeafResult::One((as_f64(args[0]) + as_f64(args[1])).to_bits())),
        Op::F64Sub => Ok(LeafResult::One((as_f64(args[0]) - as_f64(args[1])).to_bits())),
        Op::F64Mul => Ok(LeafResult::One((as_f64(args[0]) * as_f64(args[1])).to_bits())),
        Op::F64Div => Ok(LeafResult::One((as_f64(args[0]) / as_f64(args[1])).to_bits())),
        Op::F64Min => Ok(LeafResult::One(f64_min_bits(as_u64(args[0]), as_u64(args[1])))),
        Op::F64Max => Ok(LeafResult::One(f64_max_bits(as_u64(args[0]), as_u64(args[1])))),
        Op::F64Copysign => Ok(LeafResult::One(
            f64_copysign_bits(as_u64(args[0]), as_u64(args[1])),
        )),

        Op::I32Eq => Ok(LeafResult::One(bool32(as_u32(args[0]) == as_u32(args[1])))),
        Op::I32Ne => Ok(LeafResult::One(bool32(as_u32(args[0]) != as_u32(args[1])))),
        Op::I32LtS => Ok(LeafResult::One(bool32(as_i32(args[0]) < as_i32(args[1])))),
        Op::I32LtU => Ok(LeafResult::One(bool32(as_u32(args[0]) < as_u32(args[1])))),
        Op::I32GtS => Ok(LeafResult::One(bool32(as_i32(args[0]) > as_i32(args[1])))),
        Op::I32GtU => Ok(LeafResult::One(bool32(as_u32(args[0]) > as_u32(args[1])))),
        Op::I32LeS => Ok(LeafResult::One(bool32(as_i32(args[0]) <= as_i32(args[1])))),
        Op::I32LeU => Ok(LeafResult::One(bool32(as_u32(args[0]) <= as_u32(args[1])))),
        Op::I32GeS => Ok(LeafResult::One(bool32(as_i32(args[0]) >= as_i32(args[1])))),
        Op::I32GeU => Ok(LeafResult::One(bool32(as_u32(args[0]) >= as_u32(args[1])))),

        Op::I64Eq => Ok(LeafResult::One(bool32(as_u64(args[0]) == as_u64(args[1])))),
        Op::I64Ne => Ok(LeafResult::One(bool32(as_u64(args[0]) != as_u64(args[1])))),
        Op::I64LtS => Ok(LeafResult::One(bool32(as_i64(args[0]) < as_i64(args[1])))),
        Op::I64LtU => Ok(LeafResult::One(bool32(as_u64(args[0]) < as_u64(args[1])))),
        Op::I64GtS => Ok(LeafResult::One(bool32(as_i64(args[0]) > as_i64(args[1])))),
        Op::I64GtU => Ok(LeafResult::One(bool32(as_u64(args[0]) > as_u64(args[1])))),
        Op::I64LeS => Ok(LeafResult::One(bool32(as_i64(args[0]) <= as_i64(args[1])))),
        Op::I64LeU => Ok(LeafResult::One(bool32(as_u64(args[0]) <= as_u64(args[1])))),
        Op::I64GeS => Ok(LeafResult::One(bool32(as_i64(args[0]) >= as_i64(args[1])))),
        Op::I64GeU => Ok(LeafResult::One(bool32(as_u64(args[0]) >= as_u64(args[1])))),

        Op::F32Eq => Ok(LeafResult::One(bool32(as_f32(args[0]) == as_f32(args[1])))),
        Op::F32Ne => Ok(LeafResult::One(bool32(as_f32(args[0]) != as_f32(args[1])))),
        Op::F32Lt => Ok(LeafResult::One(bool32(as_f32(args[0]) < as_f32(args[1])))),
        Op::F32Gt => Ok(LeafResult::One(bool32(as_f32(args[0]) > as_f32(args[1])))),
        Op::F32Le => Ok(LeafResult::One(bool32(as_f32(args[0]) <= as_f32(args[1])))),
        Op::F32Ge => Ok(LeafResult::One(bool32(as_f32(args[0]) >= as_f32(args[1])))),
        Op::F64Eq => Ok(LeafResult::One(bool32(as_f64(args[0]) == as_f64(args[1])))),
        Op::F64Ne => Ok(LeafResult::One(bool32(as_f64(args[0]) != as_f64(args[1])))),
        Op::F64Lt => Ok(LeafResult::One(bool32(as_f64(args[0]) < as_f64(args[1])))),
        Op::F64Gt => Ok(LeafResult::One(bool32(as_f64(args[0]) > as_f64(args[1])))),
        Op::F64Le => Ok(LeafResult::One(bool32(as_f64(args[0]) <= as_f64(args[1])))),
        Op::F64Ge => Ok(LeafResult::One(bool32(as_f64(args[0]) >= as_f64(args[1])))),

        Op::I32Eqz => Ok(LeafResult::One(bool32(as_u32(args[0]) == 0))),
        Op::I32Clz => Ok(LeafResult::One(as_u32(args[0]).leading_zeros() as u64)),
        Op::I32Ctz => Ok(LeafResult::One(as_u32(args[0]).trailing_zeros() as u64)),
        Op::I32Popcnt => Ok(LeafResult::One(as_u32(args[0]).count_ones() as u64)),
        Op::I64Eqz => Ok(LeafResult::One(bool32(as_u64(args[0]) == 0))),
        Op::I64Clz => Ok(LeafResult::One(as_u64(args[0]).leading_zeros() as u64)),
        Op::I64Ctz => Ok(LeafResult::One(as_u64(args[0]).trailing_zeros() as u64)),
        Op::I64Popcnt => Ok(LeafResult::One(as_u64(args[0]).count_ones() as u64)),

        Op::F32Abs => Ok(LeafResult::One(f32_abs_bits(as_u32(args[0])) as u64)),
        Op::F32Neg => Ok(LeafResult::One(f32_neg_bits(as_u32(args[0])) as u64)),
        Op::F32Ceil => Ok(LeafResult::One(f32_ceil_bits(as_u32(args[0])) as u64)),
        Op::F32Floor => Ok(LeafResult::One(f32_floor_bits(as_u32(args[0])) as u64)),
        Op::F32Trunc => Ok(LeafResult::One(f32_trunc_bits(as_u32(args[0])) as u64)),
        Op::F32Nearest => Ok(LeafResult::One(f32_nearest_bits(as_u32(args[0])) as u64)),
        Op::F32Sqrt => Ok(LeafResult::One(f32_sqrt_bits(as_u32(args[0])) as u64)),
        Op::F64Abs => Ok(LeafResult::One(f64_abs_bits(as_u64(args[0])))),
        Op::F64Neg => Ok(LeafResult::One(f64_neg_bits(as_u64(args[0])))),
        Op::F64Ceil => Ok(LeafResult::One(f64_ceil_bits(as_u64(args[0])))),
        Op::F64Floor => Ok(LeafResult::One(f64_floor_bits(as_u64(args[0])))),
        Op::F64Trunc => Ok(LeafResult::One(f64_trunc_bits(as_u64(args[0])))),
        Op::F64Nearest => Ok(LeafResult::One(f64_nearest_bits(as_u64(args[0])))),
        Op::F64Sqrt => Ok(LeafResult::One(f64_sqrt_bits(as_u64(args[0])))),

        Op::I32WrapI64 => Ok(LeafResult::One((as_u64(args[0]) as u32) as u64)),
        Op::I64ExtendI32S => Ok(LeafResult::One((as_i32(args[0]) as i64) as u64)),
        Op::I64ExtendI32U => Ok(LeafResult::One(as_u32(args[0]) as u64)),
        Op::F32ConvertI32S => Ok(LeafResult::One((as_i32(args[0]) as f32).to_bits() as u64)),
        Op::F32ConvertI32U => Ok(LeafResult::One((as_u32(args[0]) as f32).to_bits() as u64)),
        Op::F32ConvertI64S => Ok(LeafResult::One((as_i64(args[0]) as f32).to_bits() as u64)),
        Op::F32ConvertI64U => Ok(LeafResult::One((as_u64(args[0]) as f32).to_bits() as u64)),
        Op::F32DemoteF64 => Ok(LeafResult::One((as_f64(args[0]) as f32).to_bits() as u64)),
        Op::F64ConvertI32S => Ok(LeafResult::One((as_i32(args[0]) as f64).to_bits())),
        Op::F64ConvertI32U => Ok(LeafResult::One((as_u32(args[0]) as f64).to_bits())),
        Op::F64ConvertI64S => Ok(LeafResult::One((as_i64(args[0]) as f64).to_bits())),
        Op::F64ConvertI64U => Ok(LeafResult::One((as_u64(args[0]) as f64).to_bits())),
        Op::F64PromoteF32 => Ok(LeafResult::One((as_f32(args[0]) as f64).to_bits())),
        Op::I32ReinterpretF32 => Ok(LeafResult::One(as_u32(args[0]) as u64)),
        Op::I64ReinterpretF64 => Ok(LeafResult::One(as_u64(args[0]))),
        Op::F32ReinterpretI32 => Ok(LeafResult::One(as_u32(args[0]) as u64)),
        Op::F64ReinterpretI64 => Ok(LeafResult::One(as_u64(args[0]))),
        Op::I32Extend8S => Ok(LeafResult::One((as_i32(args[0]) as i8 as i32) as u32 as u64)),
        Op::I32Extend16S => Ok(LeafResult::One((as_i32(args[0]) as i16 as i32) as u32 as u64)),
        Op::I64Extend8S => Ok(LeafResult::One((as_i64(args[0]) as i8 as i64) as u64)),
        Op::I64Extend16S => Ok(LeafResult::One((as_i64(args[0]) as i16 as i64) as u64)),
        Op::I64Extend32S => Ok(LeafResult::One((as_i64(args[0]) as i32 as i64) as u64)),

        Op::I32TruncF32S => Ok(LeafResult::One(trunc_f32_to_i32(as_f32(args[0]))? as u32 as u64)),
        Op::I32TruncF32U => Ok(LeafResult::One(trunc_f32_to_u32(as_f32(args[0]))? as u64)),
        Op::I32TruncF64S => Ok(LeafResult::One(trunc_f64_to_i32(as_f64(args[0]))? as u32 as u64)),
        Op::I32TruncF64U => Ok(LeafResult::One(trunc_f64_to_u32(as_f64(args[0]))? as u64)),
        Op::I64TruncF32S => Ok(LeafResult::One(trunc_f32_to_i64(as_f32(args[0]))? as u64)),
        Op::I64TruncF32U => Ok(LeafResult::One(trunc_f32_to_u64(as_f32(args[0]))?)),
        Op::I64TruncF64S => Ok(LeafResult::One(trunc_f64_to_i64(as_f64(args[0]))? as u64)),
        Op::I64TruncF64U => Ok(LeafResult::One(trunc_f64_to_u64(as_f64(args[0]))?)),

        Op::I32TruncSatF32S => Ok(LeafResult::One(trunc_sat_f32_to_i32(as_f32(args[0])) as u32 as u64)),
        Op::I32TruncSatF32U => Ok(LeafResult::One(trunc_sat_f32_to_u32(as_f32(args[0])) as u64)),
        Op::I32TruncSatF64S => Ok(LeafResult::One(trunc_sat_f64_to_i32(as_f64(args[0])) as u32 as u64)),
        Op::I32TruncSatF64U => Ok(LeafResult::One(trunc_sat_f64_to_u32(as_f64(args[0])) as u64)),
        Op::I64TruncSatF32S => Ok(LeafResult::One(trunc_sat_f32_to_i64(as_f32(args[0])) as u64)),
        Op::I64TruncSatF32U => Ok(LeafResult::One(trunc_sat_f32_to_u64(as_f32(args[0])))),
        Op::I64TruncSatF64S => Ok(LeafResult::One(trunc_sat_f64_to_i64(as_f64(args[0])) as u64)),
        Op::I64TruncSatF64U => Ok(LeafResult::One(trunc_sat_f64_to_u64(as_f64(args[0])))),

        Op::I32Load { offset, memidx } => Ok(LeafResult::One(load_u32(ctx, *memidx, args[0], *offset)? as u64)),
        Op::I64Load { offset, memidx } => Ok(LeafResult::One(load_u64(ctx, *memidx, args[0], *offset)?)),
        Op::F32Load { offset, memidx } => Ok(LeafResult::One(load_u32(ctx, *memidx, args[0], *offset)? as u64)),
        Op::F64Load { offset, memidx } => Ok(LeafResult::One(load_u64(ctx, *memidx, args[0], *offset)?)),
        Op::I32Load8S { offset, memidx } => Ok(LeafResult::One((load_u8(ctx, *memidx, args[0], *offset)? as i8 as i32) as u32 as u64)),
        Op::I32Load8U { offset, memidx } => Ok(LeafResult::One(load_u8(ctx, *memidx, args[0], *offset)? as u64)),
        Op::I32Load16S { offset, memidx } => Ok(LeafResult::One((load_u16(ctx, *memidx, args[0], *offset)? as i16 as i32) as u32 as u64)),
        Op::I32Load16U { offset, memidx } => Ok(LeafResult::One(load_u16(ctx, *memidx, args[0], *offset)? as u64)),
        Op::I64Load8S { offset, memidx } => Ok(LeafResult::One((load_u8(ctx, *memidx, args[0], *offset)? as i8 as i64) as u64)),
        Op::I64Load8U { offset, memidx } => Ok(LeafResult::One(load_u8(ctx, *memidx, args[0], *offset)? as u64)),
        Op::I64Load16S { offset, memidx } => Ok(LeafResult::One((load_u16(ctx, *memidx, args[0], *offset)? as i16 as i64) as u64)),
        Op::I64Load16U { offset, memidx } => Ok(LeafResult::One(load_u16(ctx, *memidx, args[0], *offset)? as u64)),
        Op::I64Load32S { offset, memidx } => Ok(LeafResult::One((load_u32(ctx, *memidx, args[0], *offset)? as i32 as i64) as u64)),
        Op::I64Load32U { offset, memidx } => Ok(LeafResult::One(load_u32(ctx, *memidx, args[0], *offset)? as u64)),

        Op::I32Store { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &as_u32(args[1]).to_le_bytes())?;
            Ok(LeafResult::None)
        }
        Op::I64Store { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &as_u64(args[1]).to_le_bytes())?;
            Ok(LeafResult::None)
        }
        Op::F32Store { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &as_u32(args[1]).to_le_bytes())?;
            Ok(LeafResult::None)
        }
        Op::F64Store { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &as_u64(args[1]).to_le_bytes())?;
            Ok(LeafResult::None)
        }
        Op::I32Store8 { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &[as_u32(args[1]) as u8])?;
            Ok(LeafResult::None)
        }
        Op::I32Store16 { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &(as_u32(args[1]) as u16).to_le_bytes())?;
            Ok(LeafResult::None)
        }
        Op::I64Store8 { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &[as_u64(args[1]) as u8])?;
            Ok(LeafResult::None)
        }
        Op::I64Store16 { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &(as_u64(args[1]) as u16).to_le_bytes())?;
            Ok(LeafResult::None)
        }
        Op::I64Store32 { offset, memidx } => {
            store_bytes(ctx, *memidx, args[0], *offset, &(as_u64(args[1]) as u32).to_le_bytes())?;
            Ok(LeafResult::None)
        }

        Op::GlobalGet { idx } => {
            let global = unsafe { &*ctx.store }.global(*idx as usize);
            Ok(LeafResult::One(global.value.to_raw()))
        }
        Op::GlobalSet { idx } => {
            let store = unsafe { &mut *ctx.store };
            let global = store.global_mut(*idx as usize);
            global.value = Value::from_raw(args[0], global.value_type);
            Ok(LeafResult::None)
        }

        Op::MemorySize { mem_idx } => {
            let store = unsafe { &*ctx.store };
            Ok(LeafResult::One(store.memory(*mem_idx as usize).current_pages() as u64))
        }
        Op::MemoryGrow { mem_idx } => Ok(LeafResult::One(memory_grow(ctx, *mem_idx, args[0])?)),
        Op::MemoryFill { imm0, .. } => {
            memory_fill(ctx, *imm0, args[0], args[1], args[2])?;
            Ok(LeafResult::None)
        }
        Op::MemoryCopy { imm0, imm1 } => {
            memory_copy(ctx, *imm0, *imm1, args[0], args[1], args[2])?;
            Ok(LeafResult::None)
        }
        Op::MemoryInit { imm0, imm1 } => {
            memory_init(ctx, *imm0, *imm1, args[0], args[1], args[2])?;
            Ok(LeafResult::None)
        }
        Op::DataDrop { data_idx } => {
            unsafe { &mut *ctx.store }
                .module_mut()
                .data[*data_idx as usize]
                .drop_segment();
            Ok(LeafResult::None)
        }

        Op::TableGet { table_idx } => Ok(LeafResult::One(table_get(ctx, *table_idx, args[0])?)),
        Op::TableSet { table_idx } => {
            table_set(ctx, *table_idx, args[0], args[1])?;
            Ok(LeafResult::None)
        }
        Op::TableSize { table_idx } => {
            let table = unsafe { &*ctx.store }.table(*table_idx as usize);
            Ok(LeafResult::One(table.size() as u64))
        }
        Op::TableGrow { table_idx } => Ok(LeafResult::One(table_grow(ctx, *table_idx, args[0], args[1])?)),
        Op::TableFill { imm0, .. } => {
            table_fill(ctx, *imm0, args[0], args[1], args[2])?;
            Ok(LeafResult::None)
        }
        Op::TableCopy { imm0, imm1 } => {
            table_copy(ctx, *imm0, *imm1, args[0], args[1], args[2])?;
            Ok(LeafResult::None)
        }
        Op::TableInit { imm0, imm1 } => {
            table_init(ctx, *imm0, *imm1, args[0], args[1], args[2])?;
            Ok(LeafResult::None)
        }
        Op::ElemDrop { elem_idx } => {
            unsafe { &mut *ctx.store }
                .module_mut()
                .elements[*elem_idx as usize]
                .drop_segment();
            Ok(LeafResult::None)
        }

        Op::RefNull => Ok(LeafResult::One(RefHandle::null().0 as u64)),
        Op::RefIsNull => Ok(LeafResult::One(bool32(args[0] as usize == usize::MAX))),
        Op::RefFunc { func_idx } => Ok(LeafResult::One(*func_idx as u64)),

        Op::Drop | Op::Nop => Ok(LeafResult::None),
        Op::Select => Ok(LeafResult::One(if (args[2] as u32) != 0 { args[0] } else { args[1] })),
        Op::Unreachable => Err(WasmError::trap("unreachable".into())),
    }
}

fn invoke_external_function(
    store: &mut crate::vm::store::Store,
    func: &FunctionInst,
    args: &[u64],
) -> Result<Vec<u64>, WasmError> {
    let FunctionInst::External {
        func_type,
        callback,
    } = func
    else {
        return Err(WasmError::internal(
            "native external call expected an external function".into(),
        ));
    };

    let params = func_type.params();
    let results = func_type.results();
    if args.len() != params.len() {
        return Err(WasmError::invalid(format!(
            "invalid argument count: got {}, expected {}",
            args.len(),
            params.len()
        )));
    }

    let ext_args = args
        .iter()
        .zip(params.iter())
        .map(|(raw, ty)| Value::from_raw(*raw, *ty))
        .collect::<Vec<_>>();
    let mut ret_vals = vec![Value::default(); results.len()];
    let mem_slice = if !store.module().memories.is_empty() {
        let mem = &store.module().memories[0] as *const MemInst as *mut MemInst;
        unsafe { Some((*mem).data.as_mut_slice()) }
    } else {
        None
    };
    let mut caller = Caller::new(mem_slice);
    callback(&mut caller, &ext_args, &mut ret_vals)?;
    Ok(ret_vals.into_iter().map(|value| value.to_raw()).collect())
}

fn bool32(value: bool) -> u64 {
    if value { 1 } else { 0 }
}

fn as_u32(raw: u64) -> u32 {
    raw as u32
}

fn as_i32(raw: u64) -> i32 {
    raw as u32 as i32
}

fn as_u64(raw: u64) -> u64 {
    raw
}

fn as_i64(raw: u64) -> i64 {
    raw as i64
}

fn as_f32(raw: u64) -> f32 {
    f32::from_bits(raw as u32)
}

fn as_f64(raw: u64) -> f64 {
    f64::from_bits(raw)
}

unsafe extern "C" {
    fn ceilf(x: f32) -> f32;
    fn floorf(x: f32) -> f32;
    fn truncf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
    fn copysignf(x: f32, y: f32) -> f32;

    fn ceil(x: f64) -> f64;
    fn floor(x: f64) -> f64;
    fn trunc(x: f64) -> f64;
    fn sqrt(x: f64) -> f64;
    fn copysign(x: f64, y: f64) -> f64;
}

fn f32_abs_bits(bits: u32) -> u32 {
    bits & 0x7fff_ffff
}

fn f32_neg_bits(bits: u32) -> u32 {
    bits ^ 0x8000_0000
}

fn f32_ceil_bits(bits: u32) -> u32 {
    unsafe { ceilf(f32::from_bits(bits)).to_bits() }
}

fn f32_floor_bits(bits: u32) -> u32 {
    unsafe { floorf(f32::from_bits(bits)).to_bits() }
}

fn f32_trunc_bits(bits: u32) -> u32 {
    unsafe { truncf(f32::from_bits(bits)).to_bits() }
}

fn f32_sqrt_bits(bits: u32) -> u32 {
    unsafe { sqrtf(f32::from_bits(bits)).to_bits() }
}

fn f32_copysign_bits(lhs: u32, rhs: u32) -> u32 {
    unsafe { copysignf(f32::from_bits(lhs), f32::from_bits(rhs)).to_bits() }
}

fn f32_min_bits(lhs: u32, rhs: u32) -> u32 {
    let left = f32::from_bits(lhs);
    let right = f32::from_bits(rhs);
    if left.is_nan() || right.is_nan() {
        return f32::NAN.to_bits();
    }
    if left == right {
        return if lhs == 0x8000_0000 || rhs == 0x8000_0000 {
            0x8000_0000
        } else {
            lhs
        };
    }
    if left < right { lhs } else { rhs }
}

fn f32_max_bits(lhs: u32, rhs: u32) -> u32 {
    let left = f32::from_bits(lhs);
    let right = f32::from_bits(rhs);
    if left.is_nan() || right.is_nan() {
        return f32::NAN.to_bits();
    }
    if left == right {
        return if lhs == 0 || rhs == 0 { 0 } else { lhs };
    }
    if left > right { lhs } else { rhs }
}

fn f32_nearest_bits(bits: u32) -> u32 {
    let value = f32::from_bits(bits);
    if !value.is_finite() {
        return bits;
    }
    let floor = unsafe { floorf(value) };
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
    rounded.to_bits()
}

fn f64_abs_bits(bits: u64) -> u64 {
    bits & 0x7fff_ffff_ffff_ffff
}

fn f64_neg_bits(bits: u64) -> u64 {
    bits ^ 0x8000_0000_0000_0000
}

fn f64_ceil_bits(bits: u64) -> u64 {
    unsafe { ceil(f64::from_bits(bits)).to_bits() }
}

fn f64_floor_bits(bits: u64) -> u64 {
    unsafe { floor(f64::from_bits(bits)).to_bits() }
}

fn f64_trunc_bits(bits: u64) -> u64 {
    unsafe { trunc(f64::from_bits(bits)).to_bits() }
}

fn f64_sqrt_bits(bits: u64) -> u64 {
    unsafe { sqrt(f64::from_bits(bits)).to_bits() }
}

fn f64_copysign_bits(lhs: u64, rhs: u64) -> u64 {
    unsafe { copysign(f64::from_bits(lhs), f64::from_bits(rhs)).to_bits() }
}

fn f64_min_bits(lhs: u64, rhs: u64) -> u64 {
    let left = f64::from_bits(lhs);
    let right = f64::from_bits(rhs);
    if left.is_nan() || right.is_nan() {
        return f64::NAN.to_bits();
    }
    if left == right {
        return if lhs == 0x8000_0000_0000_0000 || rhs == 0x8000_0000_0000_0000 {
            0x8000_0000_0000_0000
        } else {
            lhs
        };
    }
    if left < right { lhs } else { rhs }
}

fn f64_max_bits(lhs: u64, rhs: u64) -> u64 {
    let left = f64::from_bits(lhs);
    let right = f64::from_bits(rhs);
    if left.is_nan() || right.is_nan() {
        return f64::NAN.to_bits();
    }
    if left == right {
        return if lhs == 0 || rhs == 0 { 0 } else { lhs };
    }
    if left > right { lhs } else { rhs }
}

fn f64_nearest_bits(bits: u64) -> u64 {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return bits;
    }
    let floor = unsafe { floor(value) };
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
    rounded.to_bits()
}

fn load_u8(ctx: &NativeContext, mem_idx: u32, addr: u64, offset: u32) -> Result<u8, WasmError> {
    let bytes = load_range(ctx, mem_idx, addr, offset, 1)?;
    Ok(bytes[0])
}

fn load_u16(ctx: &NativeContext, mem_idx: u32, addr: u64, offset: u32) -> Result<u16, WasmError> {
    let bytes = load_range(ctx, mem_idx, addr, offset, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn load_u32(ctx: &NativeContext, mem_idx: u32, addr: u64, offset: u32) -> Result<u32, WasmError> {
    let bytes = load_range(ctx, mem_idx, addr, offset, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn load_u64(ctx: &NativeContext, mem_idx: u32, addr: u64, offset: u32) -> Result<u64, WasmError> {
    let bytes = load_range(ctx, mem_idx, addr, offset, 8)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn load_range<'a>(
    ctx: &'a NativeContext,
    mem_idx: u32,
    addr: u64,
    offset: u32,
    size: usize,
) -> Result<&'a [u8], WasmError> {
    let start = effective_addr(addr, offset)?;
    let store = unsafe { &*ctx.store };
    let mem = store.memory(mem_idx as usize);
    let end = start
        .checked_add(size)
        .ok_or_else(|| WasmError::trap("out of bounds memory access".into()))?;
    if end > mem.data.len() {
        return Err(WasmError::trap("out of bounds memory access".into()));
    }
    Ok(&mem.data[start..end])
}

fn store_bytes(
    ctx: &mut NativeContext,
    mem_idx: u32,
    addr: u64,
    offset: u32,
    bytes: &[u8],
) -> Result<(), WasmError> {
    let start = effective_addr(addr, offset)?;
    let store = unsafe { &mut *ctx.store };
    let mem = store.memory_mut(mem_idx as usize);
    let end = start
        .checked_add(bytes.len())
        .ok_or_else(|| WasmError::trap("out of bounds memory access".into()))?;
    if end > mem.data.len() {
        return Err(WasmError::trap("out of bounds memory access".into()));
    }
    mem.data[start..end].copy_from_slice(bytes);
    Ok(())
}

fn effective_addr(addr: u64, offset: u32) -> Result<usize, WasmError> {
    (addr as u32 as usize)
        .checked_add(offset as usize)
        .ok_or_else(|| WasmError::trap("out of bounds memory access".into()))
}

fn memory_grow(ctx: &mut NativeContext, mem_idx: u32, delta_raw: u64) -> Result<u64, WasmError> {
    let store = unsafe { &mut *ctx.store };
    let mem = store.memory_mut(mem_idx as usize);
    let is_64 = mem.limits.is64;
    let error_value = if is_64 { u64::MAX } else { u32::MAX as u64 };

    let delta_pages = if is_64 {
        delta_raw as i64 as usize
    } else {
        let signed = delta_raw as i32;
        if signed < 0 {
            return Ok(error_value);
        }
        signed as usize
    };

    let old_pages = mem.current_pages();
    let Some(new_pages) = old_pages.checked_add(delta_pages) else {
        return Ok(error_value);
    };
    if new_pages > mem.limits.get_max() {
        return Ok(error_value);
    }
    mem.data.resize(new_pages * WASM_PAGE_SIZE, 0);
    Ok(old_pages as u64)
}

fn memory_init(
    ctx: &mut NativeContext,
    mem_idx: u32,
    data_idx: u32,
    dst_raw: u64,
    src_raw: u64,
    size_raw: u64,
) -> Result<(), WasmError> {
    let dst = dst_raw as usize;
    let src = src_raw as usize;
    let size = size_raw as usize;
    let store = unsafe { &mut *ctx.store };
    let module = store.module_mut();
    if data_idx as usize >= module.data.len() {
        return Err(WasmError::trap("out of bounds memory access".into()));
    }
    let data = &module.data[data_idx as usize];
    if size == 0 {
        if src > data.bytes.len() || dst > module.memories[mem_idx as usize].data.len() {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        return Ok(());
    }
    if data.is_dropped() || src.saturating_add(size) > data.bytes.len() {
        return Err(WasmError::trap("out of bounds memory access".into()));
    }
    let src_ptr = data.bytes[src..].as_ptr();
    let mem_data = &mut module.memories[mem_idx as usize].data;
    if dst.saturating_add(size) > mem_data.len() {
        return Err(WasmError::trap("out of bounds memory access".into()));
    }
    unsafe {
        core::ptr::copy_nonoverlapping(src_ptr, mem_data[dst..].as_mut_ptr(), size);
    }
    Ok(())
}

fn memory_copy(
    ctx: &mut NativeContext,
    dst_idx: u32,
    src_idx: u32,
    dst_raw: u64,
    src_raw: u64,
    size_raw: u64,
) -> Result<(), WasmError> {
    let dst = dst_raw as usize;
    let src = src_raw as usize;
    let size = size_raw as usize;
    let store = unsafe { &mut *ctx.store };

    if dst_idx == src_idx {
        let mem = &mut store.memory_mut(dst_idx as usize).data;
        if src.saturating_add(size) > mem.len() || dst.saturating_add(size) > mem.len() {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        mem.copy_within(src..src + size, dst);
        return Ok(());
    }

    let module = store.module_mut();
    let (src_mem, dst_mem) = if (src_idx as usize) < (dst_idx as usize) {
        let (left, right) = module.memories.split_at_mut(dst_idx as usize);
        (&left[src_idx as usize], &mut right[0])
    } else {
        let (left, right) = module.memories.split_at_mut(src_idx as usize);
        (&right[0] as &MemInst, &mut left[dst_idx as usize])
    };
    if src.saturating_add(size) > src_mem.data.len() || dst.saturating_add(size) > dst_mem.data.len() {
        return Err(WasmError::trap("out of bounds memory access".into()));
    }
    dst_mem.data[dst..dst + size].copy_from_slice(&src_mem.data[src..src + size]);
    Ok(())
}

fn memory_fill(
    ctx: &mut NativeContext,
    mem_idx: u32,
    dst_raw: u64,
    value_raw: u64,
    size_raw: u64,
) -> Result<(), WasmError> {
    let dst = dst_raw as usize;
    let value = value_raw as u8;
    let size = size_raw as usize;
    let mem = &mut unsafe { &mut *ctx.store }.memory_mut(mem_idx as usize).data;
    if dst.saturating_add(size) > mem.len() {
        return Err(WasmError::trap("out of bounds memory access".into()));
    }
    mem[dst..dst + size].fill(value);
    Ok(())
}

fn table_get(ctx: &NativeContext, table_idx: u32, index_raw: u64) -> Result<u64, WasmError> {
    let index = index_raw as usize;
    let table = unsafe { &*ctx.store }.table(table_idx as usize);
    if index >= table.elements.len() {
        return Err(WasmError::trap("out of bounds table access".into()));
    }
    Ok(usize::from(table.elements[index]) as u64)
}

fn table_set(
    ctx: &mut NativeContext,
    table_idx: u32,
    index_raw: u64,
    value_raw: u64,
) -> Result<(), WasmError> {
    let index = index_raw as usize;
    let table = unsafe { &mut *ctx.store }.table_mut(table_idx as usize);
    if index >= table.elements.len() {
        return Err(WasmError::trap("out of bounds table access".into()));
    }
    table.elements[index] = RefHandle::new(value_raw as usize);
    Ok(())
}

fn table_grow(
    ctx: &mut NativeContext,
    table_idx: u32,
    init_raw: u64,
    delta_raw: u64,
) -> Result<u64, WasmError> {
    let delta = delta_raw as usize;
    let init = RefHandle::new(init_raw as usize);
    let table = unsafe { &mut *ctx.store }.table_mut(table_idx as usize);
    let new_size = table.elements.len().checked_add(delta).unwrap_or(usize::MAX);
    if new_size > table.limits.get_max() {
        return Ok(u32::MAX as u64);
    }
    let old = table.elements.len();
    table.elements.resize(old + delta, init);
    Ok(old as u64)
}

fn table_fill(
    ctx: &mut NativeContext,
    table_idx: u32,
    start_raw: u64,
    value_raw: u64,
    size_raw: u64,
) -> Result<(), WasmError> {
    let start = start_raw as usize;
    let value = RefHandle::new(value_raw as usize);
    let size = size_raw as usize;
    let table = unsafe { &mut *ctx.store }.table_mut(table_idx as usize);
    if start.saturating_add(size) > table.elements.len() {
        return Err(WasmError::trap("out of bounds table access".into()));
    }
    table.elements[start..start + size].fill(value);
    Ok(())
}

fn table_init(
    ctx: &mut NativeContext,
    table_idx: u32,
    elem_idx: u32,
    dst_raw: u64,
    src_raw: u64,
    size_raw: u64,
) -> Result<(), WasmError> {
    let dst = dst_raw as usize;
    let src = src_raw as usize;
    let size = size_raw as usize;
    let store = unsafe { &mut *ctx.store };
    let module = store.module_mut();
    if elem_idx as usize >= module.elements.len() {
        return Err(WasmError::trap("out of bounds table access".into()));
    }
    let elem = &module.elements[elem_idx as usize];
    if size == 0 {
        if src > elem.refs.len() || dst > module.tables[table_idx as usize].elements.len() {
            return Err(WasmError::trap("out of bounds table access".into()));
        }
        return Ok(());
    }
    if elem.is_dropped()
        || src.saturating_add(size) > elem.refs.len()
        || dst.saturating_add(size) > module.tables[table_idx as usize].elements.len()
    {
        return Err(WasmError::trap("out of bounds table access".into()));
    }
    for offset in 0..size {
        module.tables[table_idx as usize].elements[dst + offset] =
            module.elements[elem_idx as usize].refs[src + offset];
    }
    Ok(())
}

fn table_copy(
    ctx: &mut NativeContext,
    dst_idx: u32,
    src_idx: u32,
    dst_raw: u64,
    src_raw: u64,
    size_raw: u64,
) -> Result<(), WasmError> {
    let dst = dst_raw as usize;
    let src = src_raw as usize;
    let size = size_raw as usize;
    let store = unsafe { &mut *ctx.store };
    let module = store.module_mut();
    if dst_idx == src_idx {
        let table = &mut module.tables[dst_idx as usize].elements;
        if src.saturating_add(size) > table.len() || dst.saturating_add(size) > table.len() {
            return Err(WasmError::trap("out of bounds table access".into()));
        }
        table.copy_within(src..src + size, dst);
        return Ok(());
    }
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

fn trunc_f32_to_i32(value: f32) -> Result<i32, WasmError> {
    let numeric = value as f64;
    if !numeric.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(numeric) };
    if truncated < i32::MIN as f64 || truncated > i32::MAX as f64 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as i32)
}

fn trunc_f32_to_u32(value: f32) -> Result<u32, WasmError> {
    let numeric = value as f64;
    if !numeric.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(numeric) };
    if truncated < 0.0 || truncated >= 4_294_967_296.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as u32)
}

fn trunc_f64_to_i32(value: f64) -> Result<i32, WasmError> {
    if !value.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(value) };
    if truncated < i32::MIN as f64 || truncated > i32::MAX as f64 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as i32)
}

fn trunc_f64_to_u32(value: f64) -> Result<u32, WasmError> {
    if !value.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(value) };
    if truncated < 0.0 || truncated >= 4_294_967_296.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as u32)
}

fn trunc_f32_to_i64(value: f32) -> Result<i64, WasmError> {
    let numeric = value as f64;
    if !numeric.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(numeric) };
    if truncated < -9_223_372_036_854_775_808.0 || truncated >= 9_223_372_036_854_775_808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as i64)
}

fn trunc_f32_to_u64(value: f32) -> Result<u64, WasmError> {
    let numeric = value as f64;
    if !numeric.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(numeric) };
    if truncated < 0.0 || truncated >= 18_446_744_073_709_551_616.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as u64)
}

fn trunc_f64_to_i64(value: f64) -> Result<i64, WasmError> {
    if !value.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(value) };
    if truncated < -9_223_372_036_854_775_808.0 || truncated >= 9_223_372_036_854_775_808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as i64)
}

fn trunc_f64_to_u64(value: f64) -> Result<u64, WasmError> {
    if !value.is_finite() {
        return Err(WasmError::trap("integer overflow".into()));
    }
    let truncated = unsafe { trunc(value) };
    if truncated < 0.0 || truncated >= 18_446_744_073_709_551_616.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(truncated as u64)
}

fn trunc_sat_f32_to_i32(value: f32) -> i32 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { i32::MIN } else { i32::MAX }
    } else {
        let truncated = unsafe { truncf(value) };
        if truncated < i32::MIN as f32 {
            i32::MIN
        } else if truncated > i32::MAX as f32 {
            i32::MAX
        } else {
            truncated as i32
        }
    }
}

fn trunc_sat_f32_to_u32(value: f32) -> u32 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { 0 } else { u32::MAX }
    } else {
        let truncated = unsafe { truncf(value) };
        if truncated < 0.0 {
            0
        } else if truncated > u32::MAX as f32 {
            u32::MAX
        } else {
            truncated as u32
        }
    }
}

fn trunc_sat_f64_to_i32(value: f64) -> i32 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { i32::MIN } else { i32::MAX }
    } else {
        let truncated = unsafe { trunc(value) };
        if truncated < i32::MIN as f64 {
            i32::MIN
        } else if truncated > i32::MAX as f64 {
            i32::MAX
        } else {
            truncated as i32
        }
    }
}

fn trunc_sat_f64_to_u32(value: f64) -> u32 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { 0 } else { u32::MAX }
    } else {
        let truncated = unsafe { trunc(value) };
        if truncated < 0.0 {
            0
        } else if truncated > u32::MAX as f64 {
            u32::MAX
        } else {
            truncated as u32
        }
    }
}

fn trunc_sat_f32_to_i64(value: f32) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { i64::MIN } else { i64::MAX }
    } else {
        let truncated = unsafe { truncf(value) };
        if truncated < i64::MIN as f32 {
            i64::MIN
        } else if truncated > i64::MAX as f32 {
            i64::MAX
        } else {
            truncated as i64
        }
    }
}

fn trunc_sat_f32_to_u64(value: f32) -> u64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { 0 } else { u64::MAX }
    } else {
        let truncated = unsafe { truncf(value) };
        if truncated < 0.0 {
            0
        } else if truncated > u64::MAX as f32 {
            u64::MAX
        } else {
            truncated as u64
        }
    }
}

fn trunc_sat_f64_to_i64(value: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { i64::MIN } else { i64::MAX }
    } else {
        let truncated = unsafe { trunc(value) };
        if truncated < i64::MIN as f64 {
            i64::MIN
        } else if truncated > i64::MAX as f64 {
            i64::MAX
        } else {
            truncated as i64
        }
    }
}

fn trunc_sat_f64_to_u64(value: f64) -> u64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() { 0 } else { u64::MAX }
    } else {
        let truncated = unsafe { trunc(value) };
        if truncated < 0.0 {
            0
        } else if truncated > u64::MAX as f64 {
            u64::MAX
        } else {
            truncated as u64
        }
    }
}
