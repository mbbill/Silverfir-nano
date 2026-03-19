//! MachineIR emulator backend.
//!
//! This consumes the current machine module/runtime contract directly. It is a
//! backend under `arch/` rather than part of `native/runtime`, because its job
//! is to execute finalized MachineIR the same way a real ISA backend will.

mod address_space;
pub mod config;

use self::address_space::EmulatorAddressSpace;
use crate::{
    constants::{MAX_CALL_STACK_DEPTH, MAX_STACK_SIZE},
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        native::{
            code::{CompiledNativeModule, NativeCode},
            ir::{
                machine::{
                    MachineAddr, MachineBlockId, MachineBranchCond, MachineCompareKind,
                    MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth,
                    MachineHelperCall, MachineInst, MachineInstKind, MachineIntBinaryOp,
                    MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
                    MachineReg, MachineSign, MachineTerminator, MachineTrapKind, MachineValue,
                    MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
                    MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
                },
                runtime::MachineFrameRegion,
            },
            runtime::{
                context::NativeContext,
                helpers::{resolve_helper_entry, NativeHelperStatus},
            },
        },
        raw_value::{
            as_f32, as_f64, as_i32, as_i64, as_u32, as_u64, from_f32, from_f64, from_i32, from_i64,
        },
        stack::InterpreterStack,
        store::Store,
        value::Value,
    },
};
use alloc::{vec, vec::Vec};

#[cfg(feature = "function-trace")]
use crate::vm::debug::function_trace;

const MAX_STACK_SLOTS: usize = MAX_STACK_SIZE / core::mem::size_of::<u64>();

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

#[derive(Debug)]
struct SavedCaller {
    func_id: crate::vm::native::ir::machine::MachineFuncId,
    regs: Vec<u64>,
    addr_kinds: Vec<RegAddrKind>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RegAddrKind {
    #[default]
    Unknown,
    Mem0,
}

#[derive(Debug)]
struct Emulator<'a> {
    ctx: &'a mut NativeContext,
    compiled: &'a CompiledNativeModule,
    root_frame: *mut u64,
    func_id: crate::vm::native::ir::machine::MachineFuncId,
    block_id: MachineBlockId,
    fp: *mut u64,
    regs: Vec<u64>,
    addr_kinds: Vec<RegAddrKind>,
    call_stack: Vec<SavedCaller>,
    address_space: EmulatorAddressSpace,
}

pub(crate) fn eval_root_with_context(
    compiled: &CompiledNativeModule,
    func_id: crate::vm::native::ir::machine::MachineFuncId,
    ctx: &mut NativeContext,
    fp: *mut u64,
) -> Result<(), WasmError> {
    let program = compiled.function(func_id).ok_or_else(|| {
        WasmError::internal("native entry function is missing machine code".into())
    })?;
    let address_space = EmulatorAddressSpace::new(compiled, fp, ctx.stack_end);
    let runtime_base = address_space.runtime_base_value(ctx);
    let fp_base = address_space.frame_base_value(fp)?;
    let mem0_base = address_space.mem0_base_value(ctx);
    let mem0_size = ctx.mem0_size;
    Emulator {
        ctx,
        compiled,
        root_frame: fp,
        func_id,
        block_id: program.program.entry,
        fp,
        regs: init_entry_regs(
            compiled,
            program.program.reg_count,
            runtime_base,
            fp_base,
            mem0_base,
            mem0_size,
        ),
        addr_kinds: init_entry_addr_kinds(program.program.reg_count),
        call_stack: Vec::new(),
        address_space,
    }
    .run()
}

pub fn eval(
    spec: &FunctionSpec,
    code: &NativeCode,
    store: &mut Store,
    args: &[Value],
    backend: &'static str,
) -> Result<InterpreterStack, WasmError> {
    let func_type = spec.func_type();
    if args.len() != func_type.params().len() {
        return Err(WasmError::invalid(alloc::format!(
            "invalid argument count: got {}, expected {}",
            args.len(),
            func_type.params().len()
        )));
    }

    let compiled = code.compiled();
    let func_id = code.func_id();
    let runtime = compiled
        .runtime()
        .functions
        .get(func_id.0 as usize)
        .ok_or_else(|| {
            WasmError::internal("native entry function is missing runtime metadata".into())
        })?;

    let mut stack = vec![0u64; MAX_STACK_SLOTS];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(MAX_STACK_SLOTS) };

    unsafe {
        for (index, arg) in args.iter().enumerate() {
            *stack_base.add(index) = arg.to_raw();
        }
        if runtime.frame_prefix_slots as usize > args.len() {
            core::ptr::write_bytes(
                stack_base.add(args.len()),
                0,
                runtime.frame_prefix_slots as usize - args.len(),
            );
        }
    }
    ensure_stack_capacity(stack_base, stack_end, runtime.total_frame_slots)?;

    let mut ctx = NativeContext::new(store as *mut Store, stack_end);
    #[cfg(feature = "function-trace")]
    {
        function_trace::init_from_env();
        function_trace::native_root_entry(&mut ctx, spec, backend);
    }

    let result = eval_root_with_context(compiled, func_id, &mut ctx, stack_base);

    if let Err(ref error) = result {
        #[cfg(feature = "function-trace")]
        function_trace::native_trap_current(&mut ctx, error);
        return Err(error.clone());
    }

    let results_len = func_type.results().len();
    let out = unsafe {
        crate::vm::native::runtime::collect_native_results_from_stack(
            stack_base,
            func_type.results(),
            compiled.backend().gp_unit_bytes,
        )
    };
    #[cfg(feature = "function-trace")]
    {
        let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
        function_trace::native_root_exit(&mut ctx, spec, results);
    }
    Ok(out)
}

impl<'a> Emulator<'a> {
    fn run(mut self) -> Result<(), WasmError> {
        loop {
            let block = self.current_block()?.clone();
            for (inst_idx, inst) in block.ops.iter().enumerate() {
                if let Err(error) = self.execute_inst(inst) {
                    self.log_execution_error(block.id, Some(inst_idx), Some(inst), None, &error);
                    return Err(error);
                }
            }
            let terminator = block.terminator.clone();
            let terminator_result = match &terminator {
                MachineTerminator::Jump(edge) => self.jump_to_edge(edge),
                MachineTerminator::Branch {
                    cond,
                    then_edge,
                    else_edge,
                } => {
                    let edge = if self.eval_branch_cond(*cond)? {
                        then_edge
                    } else {
                        else_edge
                    };
                    self.jump_to_edge(edge)
                }
                MachineTerminator::JumpTable { index, entries } => {
                    let index = self.read_value(*index)? as usize;
                    let edge = entries
                        .get(index)
                        .or_else(|| entries.last())
                        .ok_or_else(|| {
                            WasmError::internal("machine jump table has no entries".into())
                        })?;
                    self.jump_to_edge(edge)
                }
                MachineTerminator::CallDirect {
                    callee,
                    callee_frame_base,
                    ..
                } => self.enter_direct_call(*callee, *callee_frame_base),
                MachineTerminator::CallIndirect {
                    callee_target,
                    callee_frame_base,
                    arg_slots,
                    caller_result_base,
                    continuation,
                } => self.enter_indirect_call(
                    *callee_target,
                    *callee_frame_base,
                    *arg_slots,
                    *caller_result_base,
                    *continuation,
                ),
                MachineTerminator::Return => {
                    if self.handle_return()? {
                        return Ok(());
                    }
                    Ok(())
                }
                MachineTerminator::Trap { kind } => Err(trap_from_kind(*kind)),
            };
            if let Err(error) = terminator_result {
                self.log_execution_error(block.id, None, None, Some(&terminator), &error);
                return Err(error);
            }
        }
    }

    fn log_execution_error(
        &self,
        block_id: MachineBlockId,
        inst_idx: Option<usize>,
        inst: Option<&MachineInst>,
        terminator: Option<&MachineTerminator>,
        error: &WasmError,
    ) {
        #[cfg(any(feature = "std", feature = "wasi", test))]
        {
            if std::env::var_os("SF_EMU_TRAP_TRACE").is_none() {
                return;
            }
            match (inst_idx, inst, terminator) {
                (Some(inst_idx), Some(inst), _) => std::eprintln!(
                    "[emu-trap] func={} depth={} block=b{} inst={} {:?}: {}",
                    self.func_id.0,
                    self.call_stack.len(),
                    block_id.0,
                    inst_idx,
                    inst.kind,
                    error
                ),
                (_, _, Some(terminator)) => std::eprintln!(
                    "[emu-trap] func={} depth={} block=b{} term {:?}: {}",
                    self.func_id.0,
                    self.call_stack.len(),
                    block_id.0,
                    terminator,
                    error
                ),
                _ => {}
            }
        }
    }

    fn execute_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src, .. } => {
                let value = self.read_value(*src)?;
                self.write_reg_with_kind(*dst, value, self.value_addr_kind(*src))?;
            }
            MachineInstKind::FloatConst { width, dst, bits } => {
                let value = match width {
                    MachineFloatWidth::F32 => u64::from(*bits as u32),
                    MachineFloatWidth::F64 => *bits,
                };
                self.write_reg_with_kind(*dst, value, fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::Lea { dst, addr } => {
                self.write_reg_with_kind(
                    *dst,
                    self.addr_value(*addr),
                    self.reg_addr_kind(addr.base),
                )?;
            }
            MachineInstKind::Load {
                ty: _,
                dst,
                addr,
                width,
                extension,
            } => {
                let value = self.load(*addr, *width, *extension)?;
                self.write_reg_with_kind(*dst, value, fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::Store {
                ty: _,
                addr,
                width,
                src,
            } => {
                let value = self.read_value(*src)?;
                self.store(*addr, *width, value)?;
            }
            MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src,
            } => {
                self.write_reg_with_kind(
                    *dst,
                    eval_int_unary(*width, *op, self.read_value(*src)?)?,
                    fixed_reg_addr_kind(*dst),
                )?;
            }
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => {
                self.write_reg_with_kind(
                    *dst,
                    eval_int_binary(*width, *op, self.read_value(*lhs)?, self.read_value(*rhs)?)?,
                    int_binary_addr_kind(
                        *op,
                        self.value_addr_kind(*lhs),
                        self.value_addr_kind(*rhs),
                    ),
                )?;
            }
            MachineInstKind::IntMulWide {
                sign,
                dst_lo,
                dst_hi,
                lhs,
                rhs,
            } => {
                let (lo, hi) =
                    eval_int_mul_wide(*sign, self.read_value(*lhs)?, self.read_value(*rhs)?);
                self.write_reg_with_kind(*dst_lo, lo, fixed_reg_addr_kind(*dst_lo))?;
                self.write_reg_with_kind(*dst_hi, hi, fixed_reg_addr_kind(*dst_hi))?;
            }
            MachineInstKind::Int64PairDivRem {
                sign,
                rem,
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
            } => {
                let (lo, hi) = eval_i64_pair_div_rem(
                    *sign,
                    *rem,
                    self.read_value(*lhs_lo)?,
                    self.read_value(*lhs_hi)?,
                    self.read_value(*rhs_lo)?,
                    self.read_value(*rhs_hi)?,
                )?;
                self.write_reg_with_kind(*dst_lo, lo, fixed_reg_addr_kind(*dst_lo))?;
                self.write_reg_with_kind(*dst_hi, hi, fixed_reg_addr_kind(*dst_hi))?;
            }
            MachineInstKind::Int64PairUnary {
                op,
                dst_lo,
                dst_hi,
                src_lo,
                src_hi,
            } => {
                let (lo, hi) =
                    eval_i64_pair_unary(*op, self.read_value(*src_lo)?, self.read_value(*src_hi)?)?;
                self.write_reg_with_kind(*dst_lo, lo, fixed_reg_addr_kind(*dst_lo))?;
                self.write_reg_with_kind(*dst_hi, hi, fixed_reg_addr_kind(*dst_hi))?;
            }
            MachineInstKind::Int64PairShift {
                op,
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs,
            } => {
                let (lo, hi) = eval_i64_pair_shift(
                    *op,
                    self.read_value(*lhs_lo)?,
                    self.read_value(*lhs_hi)?,
                    self.read_value(*rhs)?,
                )?;
                self.write_reg_with_kind(*dst_lo, lo, fixed_reg_addr_kind(*dst_lo))?;
                self.write_reg_with_kind(*dst_hi, hi, fixed_reg_addr_kind(*dst_hi))?;
            }
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => {
                let value = eval_int_compare(
                    *width,
                    *kind,
                    *sign,
                    self.read_value(*lhs)?,
                    self.read_value(*rhs)?,
                );
                self.write_reg_with_kind(*dst, value, fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::FloatUnary {
                width,
                op,
                dst,
                src,
            } => {
                self.write_reg_with_kind(
                    *dst,
                    eval_float_unary(*width, *op, self.read_value(*src)?)?,
                    fixed_reg_addr_kind(*dst),
                )?;
            }
            MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => {
                self.write_reg_with_kind(
                    *dst,
                    eval_float_binary(*width, *op, self.read_value(*lhs)?, self.read_value(*rhs)?)?,
                    fixed_reg_addr_kind(*dst),
                )?;
            }
            MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs,
                rhs,
            } => {
                self.write_reg_with_kind(
                    *dst,
                    eval_float_compare(
                        *width,
                        *kind,
                        self.read_value(*lhs)?,
                        self.read_value(*rhs)?,
                    ),
                    fixed_reg_addr_kind(*dst),
                )?;
            }
            MachineInstKind::Convert { op, dst, src } => {
                self.write_reg_with_kind(
                    *dst,
                    eval_convert(*op, self.read_value(*src)?)?,
                    convert_addr_kind(*op, self.value_addr_kind(*src)),
                )?;
            }
            MachineInstKind::ConvertI64PairToFloat {
                width,
                sign,
                dst,
                src_lo,
                src_hi,
            } => {
                self.write_reg_with_kind(
                    *dst,
                    eval_i64_pair_to_float(
                        *width,
                        *sign,
                        self.read_value(*src_lo)?,
                        self.read_value(*src_hi)?,
                    ),
                    fixed_reg_addr_kind(*dst),
                )?;
            }
            MachineInstKind::ConvertFloatToI64Pair {
                op,
                dst_lo,
                dst_hi,
                src,
            } => {
                let value = eval_convert(*op, self.read_value(*src)?)?;
                self.write_reg_with_kind(
                    *dst_lo,
                    u64::from(value as u32),
                    fixed_reg_addr_kind(*dst_lo),
                )?;
                self.write_reg_with_kind(
                    *dst_hi,
                    u64::from((value >> 32) as u32),
                    fixed_reg_addr_kind(*dst_hi),
                )?;
            }
            MachineInstKind::ReinterpretF64ToI64Pair {
                dst_lo,
                dst_hi,
                src,
            } => {
                let bits = self.read_value(*src)?;
                self.write_reg_with_kind(
                    *dst_lo,
                    u64::from(bits as u32),
                    fixed_reg_addr_kind(*dst_lo),
                )?;
                self.write_reg_with_kind(
                    *dst_hi,
                    u64::from((bits >> 32) as u32),
                    fixed_reg_addr_kind(*dst_hi),
                )?;
            }
            MachineInstKind::ReinterpretI64PairToF64 {
                dst,
                src_lo,
                src_hi,
            } => {
                let bits = u64::from(self.read_value(*src_lo)? as u32)
                    | (u64::from(self.read_value(*src_hi)? as u32) << 32);
                self.write_reg_with_kind(*dst, bits, fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::Select {
                dst,
                on_true,
                on_false,
                cond,
                ..
            } => {
                let cond = self.read_value(*cond)?;
                let value = if cond != 0 {
                    self.read_value(*on_true)?
                } else {
                    self.read_value(*on_false)?
                };
                let kind = if cond != 0 {
                    self.value_addr_kind(*on_true)
                } else {
                    self.value_addr_kind(*on_false)
                };
                self.write_reg_with_kind(*dst, value, kind)?;
            }
            MachineInstKind::TrapIf { kind, cond } => {
                if self.eval_branch_cond(*cond)? {
                    return Err(trap_from_kind(*kind));
                }
            }
            MachineInstKind::CallHelper(call) => self.execute_helper(call)?,
        }
        Ok(())
    }

    fn execute_helper(&mut self, call: &MachineHelperCall) -> Result<(), WasmError> {
        let binding = self
            .compiled
            .module()
            .externs
            .get(call.target.0 as usize)
            .ok_or_else(|| WasmError::internal("machine helper target is out of range".into()))?;
        let metadata = self
            .compiled
            .const_ptr(call.metadata)
            .ok_or_else(|| WasmError::internal("machine helper metadata is out of range".into()))?;
        let entry = resolve_helper_entry(binding.symbol);
        let status = unsafe { entry(self.ctx as *mut NativeContext, self.fp, metadata) };
        if status == NativeHelperStatus::Ok as u32 {
            return Ok(());
        }
        Err(self
            .ctx
            .error
            .take()
            .unwrap_or_else(|| trap_from_kind(MachineTrapKind::HelperFailure)))
    }

    fn jump_to_edge(
        &mut self,
        edge: &crate::vm::native::ir::machine::MachineEdge,
    ) -> Result<(), WasmError> {
        let target_params = self
            .current_program()?
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| WasmError::internal("machine edge target is out of range".into()))?
            .params
            .clone();
        let mut values = Vec::with_capacity(edge.args.len());
        for value in &edge.args {
            values.push(self.read_value(*value)?);
        }
        for (param, value) in target_params.into_iter().zip(values.into_iter()) {
            self.write_reg(param.reg, value)?;
        }
        self.block_id = edge.target;
        Ok(())
    }

    fn enter_direct_call(
        &mut self,
        callee: crate::vm::native::ir::machine::MachineFuncId,
        callee_frame_base: MachineReg,
    ) -> Result<(), WasmError> {
        let callee_fp = self
            .address_space
            .host_stack_ptr(self.read_reg(callee_frame_base)?)?;
        self.enter_callee(callee, callee_fp)
    }

    fn enter_indirect_call(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee =
            crate::vm::native::ir::machine::MachineFuncId(self.read_value(callee_target)? as u32);
        let callee_runtime = self.runtime_for(callee)?;
        if arg_slots > callee_runtime.frame_prefix_slots {
            return Err(WasmError::internal(
                "indirect local call arg span exceeds callee frame prefix".into(),
            ));
        }
        let callee_fp = self
            .address_space
            .host_stack_ptr(self.read_reg(callee_frame_base)?)?;
        for slot in arg_slots..callee_runtime.frame_prefix_slots {
            unsafe {
                *callee_fp.add(slot as usize) = 0;
            }
        }
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("indirect local call requires callee call scratch".into())
        })?;
        self.write_call_link(callee_fp, call_scratch, continuation, caller_result_base)?;
        self.enter_callee(callee, callee_fp)
    }

    fn enter_callee(
        &mut self,
        callee: crate::vm::native::ir::machine::MachineFuncId,
        callee_fp: *mut u64,
    ) -> Result<(), WasmError> {
        let callee_function = self
            .compiled
            .function(callee)
            .ok_or_else(|| WasmError::internal("machine local callee is out of range".into()))?;
        let callee_runtime = self.runtime_for(callee)?;
        ensure_stack_capacity(
            callee_fp,
            self.ctx.stack_end,
            callee_runtime.total_frame_slots,
        )?;
        ensure_call_depth_capacity(self.call_stack.len())?;
        self.call_stack.push(SavedCaller {
            func_id: self.func_id,
            regs: core::mem::take(&mut self.regs),
            addr_kinds: core::mem::take(&mut self.addr_kinds),
        });
        self.func_id = callee;
        self.fp = callee_fp;
        self.block_id = callee_function.program.entry;
        self.regs = init_entry_regs(
            self.compiled,
            callee_function.program.reg_count,
            self.address_space.runtime_base_value(self.ctx),
            self.address_space.frame_base_value(callee_fp)?,
            self.address_space.mem0_base_value(self.ctx),
            self.ctx.mem0_size,
        );
        self.addr_kinds = init_entry_addr_kinds(callee_function.program.reg_count);
        #[cfg(feature = "function-trace")]
        function_trace::native_function_trace_enter_func_idx(self.ctx, callee.0);
        Ok(())
    }

    fn handle_return(&mut self) -> Result<bool, WasmError> {
        let current_runtime = *self.runtime_for(self.func_id)?;
        let results = current_runtime.return_results;
        if let Some(saved) = self.call_stack.pop() {
            let call_scratch = current_runtime.call_scratch.ok_or_else(|| {
                WasmError::internal("machine local return requires call scratch".into())
            })?;
            let continuation = MachineBlockId(self.read_call_link_word(
                self.fp,
                call_scratch,
                self.compiled.runtime().call_link.continuation_offset,
            )? as u32);
            let caller_fp = self.address_space.host_stack_ptr(self.read_call_link_word(
                self.fp,
                call_scratch,
                self.compiled.runtime().call_link.caller_frame_offset,
            )?)?;
            let caller_result_base_bytes = self.read_call_link_word(
                self.fp,
                call_scratch,
                self.compiled.runtime().call_link.caller_result_base_offset,
            )? as usize;
            self.copy_results(
                results,
                self.fp,
                caller_fp
                    .cast::<u8>()
                    .wrapping_add(caller_result_base_bytes)
                    .cast::<u64>(),
            )?;
            #[cfg(feature = "function-trace")]
            {
                let arity = results.map(|region| region.slots).unwrap_or(0) as u64;
                let result_fp = results
                    .map(|region| unsafe { self.fp.add(region.base_slot as usize) })
                    .unwrap_or(self.fp);
                unsafe {
                    function_trace::native_function_trace_exit(self.ctx, result_fp, arity);
                }
            }
            self.func_id = saved.func_id;
            self.fp = caller_fp;
            self.regs = saved.regs;
            self.addr_kinds = saved.addr_kinds;
            self.block_id = continuation;
            self.init_reserved_regs()?;
            return Ok(false);
        }

        self.copy_results(results, self.fp, self.root_frame)?;
        Ok(true)
    }

    fn copy_results(
        &self,
        results: Option<MachineFrameRegion>,
        source_fp: *mut u64,
        dest_fp: *mut u64,
    ) -> Result<(), WasmError> {
        let Some(results) = results else {
            return Ok(());
        };
        unsafe {
            for index in 0..results.slots as usize {
                *dest_fp.add(index) = *source_fp.add(results.base_slot as usize + index);
            }
        }
        Ok(())
    }

    fn eval_branch_cond(&self, cond: MachineBranchCond) -> Result<bool, WasmError> {
        match cond {
            MachineBranchCond::Value(value) => Ok(self.read_value(value)? != 0),
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => Ok(eval_int_compare(
                width,
                kind,
                sign,
                self.read_value(lhs)?,
                self.read_value(rhs)?,
            ) != 0),
            MachineBranchCond::FloatCompare {
                width,
                kind,
                lhs,
                rhs,
            } => Ok(
                eval_float_compare(width, kind, self.read_value(lhs)?, self.read_value(rhs)?) != 0,
            ),
        }
    }

    fn current_program(
        &self,
    ) -> Result<&crate::vm::native::ir::machine::MachineProgram, WasmError> {
        Ok(&self
            .compiled
            .function(self.func_id)
            .ok_or_else(|| WasmError::internal("machine current function is out of range".into()))?
            .program)
    }

    fn current_block(&self) -> Result<&crate::vm::native::ir::machine::MachineBlock, WasmError> {
        self.current_program()?
            .blocks
            .get(self.block_id.as_usize())
            .ok_or_else(|| WasmError::internal("machine current block is out of range".into()))
    }

    fn runtime_for(
        &self,
        func_id: crate::vm::native::ir::machine::MachineFuncId,
    ) -> Result<&crate::vm::native::ir::runtime::MachineFunctionRuntime, WasmError> {
        self.compiled
            .runtime()
            .functions
            .get(func_id.0 as usize)
            .ok_or_else(|| WasmError::internal("machine runtime record is out of range".into()))
    }

    fn init_reserved_regs(&mut self) -> Result<(), WasmError> {
        if self.regs.len() < MACHINE_FIXED_REG_COUNT as usize {
            return Err(WasmError::internal(
                "machine register file is smaller than reserved native ABI registers".into(),
            ));
        }
        self.write_reg_with_kind(
            MACHINE_CTX_REG,
            self.address_space.runtime_base_value(self.ctx),
            fixed_reg_addr_kind(MACHINE_CTX_REG),
        )?;
        self.write_reg_with_kind(
            MACHINE_FP_REG,
            self.address_space.frame_base_value(self.fp)?,
            fixed_reg_addr_kind(MACHINE_FP_REG),
        )?;
        self.write_reg_with_kind(
            MACHINE_MEM0_BASE_REG,
            self.address_space.mem0_base_value(self.ctx),
            fixed_reg_addr_kind(MACHINE_MEM0_BASE_REG),
        )?;
        self.write_reg_with_kind(
            MACHINE_MEM0_SIZE_REG,
            self.ctx.mem0_size,
            fixed_reg_addr_kind(MACHINE_MEM0_SIZE_REG),
        )?;
        Ok(())
    }

    fn read_reg(&self, reg: MachineReg) -> Result<u64, WasmError> {
        self.regs
            .get(reg.0 as usize)
            .copied()
            .ok_or_else(|| WasmError::internal("machine register is out of range".into()))
    }

    fn write_reg(&mut self, reg: MachineReg, value: u64) -> Result<(), WasmError> {
        let slot = self
            .regs
            .get_mut(reg.0 as usize)
            .ok_or_else(|| WasmError::internal("machine register is out of range".into()))?;
        *slot = value;
        Ok(())
    }

    fn write_reg_with_kind(
        &mut self,
        reg: MachineReg,
        value: u64,
        kind: RegAddrKind,
    ) -> Result<(), WasmError> {
        self.write_reg(reg, value)?;
        self.set_reg_addr_kind(reg, kind)
    }

    fn reg_addr_kind(&self, reg: MachineReg) -> RegAddrKind {
        self.addr_kinds
            .get(reg.0 as usize)
            .copied()
            .unwrap_or(RegAddrKind::Unknown)
    }

    fn value_addr_kind(&self, value: MachineValue) -> RegAddrKind {
        match value {
            MachineValue::Reg(reg) => self.reg_addr_kind(reg),
            MachineValue::Imm64(_) => RegAddrKind::Unknown,
        }
    }

    fn set_reg_addr_kind(&mut self, reg: MachineReg, kind: RegAddrKind) -> Result<(), WasmError> {
        let slot = self
            .addr_kinds
            .get_mut(reg.0 as usize)
            .ok_or_else(|| WasmError::internal("machine register is out of range".into()))?;
        *slot = kind;
        Ok(())
    }

    fn read_value(&self, value: MachineValue) -> Result<u64, WasmError> {
        match value {
            MachineValue::Reg(reg) => self.read_reg(reg),
            MachineValue::Imm64(value) => Ok(value),
        }
    }

    fn addr_value(&self, addr: MachineAddr) -> u64 {
        self.read_reg(addr.base)
            .expect("validated base register must exist")
            .wrapping_add_signed(i64::from(addr.offset))
    }

    /// Check that a pointer dereference targets valid memory.  When guard-page
    /// backing is active the MIR omits explicit bounds-check TrapIf
    /// instructions, relying on a signal handler that the emulator does not
    /// install.  Catch out-of-bounds wasm memory accesses here instead.
    ///
    /// We identify wasm-memory pointers by checking whether they fall inside
    /// the guard-page virtual reservation (8 GB from mem0_base).  Frame/stack
    /// pointers live in a heap Vec and never land in that range.
    fn check_access(&self, ptr: usize, size: usize) -> Result<(), WasmError> {
        let mem_base = self.ctx.mem0_base as usize;
        let mem_size = self.ctx.mem0_size as usize;
        if mem_base == 0 {
            return Ok(());
        }
        // On 64-bit with guard pages, the 8 GB virtual reservation reliably
        // separates wasm-memory pointers from frame/stack pointers.  On 32-bit
        // (no guard pages) we just check the actual committed memory range.
        #[cfg(target_pointer_width = "64")]
        const GUARD_WINDOW: usize = 8 * 1024 * 1024 * 1024 + 64 * 1024;
        #[cfg(target_pointer_width = "64")]
        let in_wasm_region = ptr >= mem_base && ptr < mem_base.saturating_add(GUARD_WINDOW);
        #[cfg(target_pointer_width = "32")]
        let in_wasm_region = ptr >= mem_base && ptr < mem_base + mem_size;
        if in_wasm_region {
            if ptr.saturating_add(size) > mem_base + mem_size {
                return Err(WasmError::trap("out of bounds memory access".into()));
            }
        }
        Ok(())
    }

    fn check_mem0_access(&self, addr: u64, size: usize) -> Result<(), WasmError> {
        let mem_base = self.ctx.mem0_base as u64;
        let mem_size = self.ctx.mem0_size;
        if mem_base == 0 || mem_size == 0 {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        let end = addr
            .checked_add(size as u64)
            .ok_or_else(|| WasmError::trap("out of bounds memory access".into()))?;
        let mem_end = mem_base
            .checked_add(mem_size)
            .ok_or_else(|| WasmError::trap("out of bounds memory access".into()))?;
        if addr < mem_base || end > mem_end {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        Ok(())
    }

    fn load(
        &self,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<u64, WasmError> {
        let addr_value = self.addr_value(addr);
        if let Some(result) = self.address_space.load(self.ctx, addr_value, width) {
            return result.map(|raw| match (width, extension) {
                (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                    (raw as u8 as i8 as i64) as u64
                }
                (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                    (raw as u16 as i16 as i64) as u64
                }
                (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                    (raw as u32 as i32 as i64) as u64
                }
                _ => raw,
            });
        }
        if matches!(self.address_space, EmulatorAddressSpace::Target32(_)) {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit load uses unmapped address 0x{addr_value:08x}"
            )));
        }
        let ptr = addr_value as *const u8;
        if self.reg_addr_kind(addr.base) == RegAddrKind::Mem0 {
            self.check_mem0_access(addr_value, mem_width_bytes(width))?;
        } else {
            self.check_access(ptr as usize, mem_width_bytes(width))?;
        }
        let raw = unsafe {
            match width {
                MachineMemWidth::U8 => core::ptr::read_unaligned(ptr.cast::<u8>()) as u64,
                MachineMemWidth::U16 => core::ptr::read_unaligned(ptr.cast::<u16>()) as u64,
                MachineMemWidth::U32 => core::ptr::read_unaligned(ptr.cast::<u32>()) as u64,
                MachineMemWidth::U64 => core::ptr::read_unaligned(ptr.cast::<u64>()),
            }
        };
        Ok(match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                (raw as u8 as i8 as i64) as u64
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                (raw as u16 as i16 as i64) as u64
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                (raw as u32 as i32 as i64) as u64
            }
            (MachineMemWidth::U8, _)
            | (MachineMemWidth::U16, _)
            | (MachineMemWidth::U32, _)
            | (MachineMemWidth::U64, _) => raw,
        })
    }

    fn store(
        &self,
        addr: MachineAddr,
        width: MachineMemWidth,
        value: u64,
    ) -> Result<(), WasmError> {
        let addr_value = self.addr_value(addr);
        if let Some(result) = self.address_space.store(self.ctx, addr_value, width, value) {
            return result;
        }
        if matches!(self.address_space, EmulatorAddressSpace::Target32(_)) {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit store uses unmapped address 0x{addr_value:08x}"
            )));
        }
        let ptr = addr_value as *mut u8;
        if self.reg_addr_kind(addr.base) == RegAddrKind::Mem0 {
            self.check_mem0_access(addr_value, mem_width_bytes(width))?;
        } else {
            self.check_access(ptr as usize, mem_width_bytes(width))?;
        }
        unsafe {
            match width {
                MachineMemWidth::U8 => core::ptr::write_unaligned(ptr.cast::<u8>(), value as u8),
                MachineMemWidth::U16 => core::ptr::write_unaligned(ptr.cast::<u16>(), value as u16),
                MachineMemWidth::U32 => core::ptr::write_unaligned(ptr.cast::<u32>(), value as u32),
                MachineMemWidth::U64 => core::ptr::write_unaligned(ptr.cast::<u64>(), value),
            }
        }
        Ok(())
    }

    fn write_call_link(
        &self,
        callee_fp: *mut u64,
        call_scratch: MachineFrameRegion,
        continuation: MachineBlockId,
        caller_result_base: u16,
    ) -> Result<(), WasmError> {
        let layout = self.compiled.runtime().call_link;
        self.write_call_link_word_slot(
            callee_fp,
            call_scratch,
            layout.continuation_offset,
            continuation.0 as u64,
        )?;
        self.write_call_link_word_slot(
            callee_fp,
            call_scratch,
            layout.caller_frame_offset,
            self.address_space.frame_base_value(self.fp)?,
        )?;
        self.write_call_link_word_slot(
            callee_fp,
            call_scratch,
            layout.caller_result_base_offset,
            u64::from(caller_result_base) * 8,
        )
    }

    fn write_call_link_word_slot(
        &self,
        callee_fp: *mut u64,
        call_scratch: MachineFrameRegion,
        offset: i32,
        value: u64,
    ) -> Result<(), WasmError> {
        let addr = frame_region_addr(callee_fp, call_scratch, offset)?;
        unsafe {
            match self.compiled.backend().gp_unit_bytes {
                4 => core::ptr::write_unaligned(addr.cast::<u32>(), value as u32),
                8 => core::ptr::write_unaligned(addr.cast::<u64>(), value),
                _ => {
                    return Err(WasmError::internal(
                        "unsupported GP unit size in emulator call-link write".into(),
                    ))
                }
            }
        }
        Ok(())
    }

    fn read_call_link_word(
        &self,
        callee_fp: *mut u64,
        call_scratch: MachineFrameRegion,
        offset: i32,
    ) -> Result<u64, WasmError> {
        let addr = frame_region_addr(callee_fp, call_scratch, offset)?;
        Ok(unsafe {
            match self.compiled.backend().gp_unit_bytes {
                4 => u64::from(core::ptr::read_unaligned(addr.cast::<u32>())),
                8 => core::ptr::read_unaligned(addr.cast::<u64>()),
                _ => {
                    return Err(WasmError::internal(
                        "unsupported GP unit size in emulator call-link read".into(),
                    ))
                }
            }
        })
    }
}

fn init_entry_regs(
    _compiled: &CompiledNativeModule,
    reg_count: u16,
    ctx_ptr: u64,
    fp: u64,
    mem0_base: u64,
    mem0_size: u64,
) -> Vec<u64> {
    let mut regs = vec![0; reg_count as usize];
    if !regs.is_empty() {
        regs[MACHINE_CTX_REG.0 as usize] = ctx_ptr;
    }
    let frame_base = MACHINE_FP_REG.0 as usize;
    if frame_base < regs.len() {
        regs[frame_base] = fp;
    }
    let mem0_base_slot = MACHINE_MEM0_BASE_REG.0 as usize;
    if mem0_base_slot < regs.len() {
        regs[mem0_base_slot] = mem0_base;
    }
    let mem0_size_slot = MACHINE_MEM0_SIZE_REG.0 as usize;
    if mem0_size_slot < regs.len() {
        regs[mem0_size_slot] = mem0_size;
    }
    regs
}

fn init_entry_addr_kinds(reg_count: u16) -> Vec<RegAddrKind> {
    let mut kinds = vec![RegAddrKind::Unknown; reg_count as usize];
    if (MACHINE_MEM0_BASE_REG.0 as usize) < kinds.len() {
        kinds[MACHINE_MEM0_BASE_REG.0 as usize] = RegAddrKind::Mem0;
    }
    kinds
}

fn fixed_reg_addr_kind(reg: MachineReg) -> RegAddrKind {
    if reg == MACHINE_MEM0_BASE_REG {
        RegAddrKind::Mem0
    } else {
        RegAddrKind::Unknown
    }
}

fn int_binary_addr_kind(op: MachineIntBinaryOp, lhs: RegAddrKind, rhs: RegAddrKind) -> RegAddrKind {
    match op {
        MachineIntBinaryOp::Add => match (lhs, rhs) {
            (RegAddrKind::Mem0, RegAddrKind::Unknown)
            | (RegAddrKind::Unknown, RegAddrKind::Mem0) => RegAddrKind::Mem0,
            _ => RegAddrKind::Unknown,
        },
        MachineIntBinaryOp::Sub => match (lhs, rhs) {
            (RegAddrKind::Mem0, RegAddrKind::Unknown) => RegAddrKind::Mem0,
            _ => RegAddrKind::Unknown,
        },
        _ => RegAddrKind::Unknown,
    }
}

fn convert_addr_kind(op: MachineConvertOp, src: RegAddrKind) -> RegAddrKind {
    if src == RegAddrKind::Unknown {
        return RegAddrKind::Unknown;
    }
    match op {
        MachineConvertOp::I64ExtendI32S
        | MachineConvertOp::I64ExtendI32U
        | MachineConvertOp::I32WrapI64 => src,
        _ => RegAddrKind::Unknown,
    }
}

pub(crate) fn ensure_stack_capacity(
    fp: *mut u64,
    stack_end: *mut u64,
    total_frame_slots: u16,
) -> Result<(), WasmError> {
    let end =
        (fp as usize).saturating_add(total_frame_slots as usize * core::mem::size_of::<u64>());
    if end > stack_end as usize {
        return Err(WasmError::exhaustion("stack overflow".into()));
    }
    Ok(())
}

pub(crate) fn ensure_call_depth_capacity(current_depth: usize) -> Result<(), WasmError> {
    if current_depth >= MAX_CALL_STACK_DEPTH {
        return Err(trap_from_kind(MachineTrapKind::CallStackExhausted));
    }
    Ok(())
}

fn frame_region_addr(
    fp: *mut u64,
    region: MachineFrameRegion,
    offset: i32,
) -> Result<*mut u8, WasmError> {
    let base = (region.base_slot as usize)
        .checked_mul(core::mem::size_of::<u64>())
        .ok_or_else(|| WasmError::internal("frame region offset overflow".into()))?;
    Ok((fp as *mut u8)
        .wrapping_add(base)
        .wrapping_offset(offset as isize))
}

fn trap_from_kind(kind: MachineTrapKind) -> WasmError {
    match kind {
        MachineTrapKind::Unreachable => WasmError::trap("unreachable executed".into()),
        MachineTrapKind::MemoryOutOfBounds => WasmError::trap("out of bounds memory access".into()),
        MachineTrapKind::TableOutOfBounds => WasmError::trap("out of bounds table access".into()),
        MachineTrapKind::InvalidFunctionReference => {
            WasmError::trap("invalid function reference".into())
        }
        MachineTrapKind::IndirectCallTypeMismatch => {
            WasmError::trap("indirect call type mismatch".into())
        }
        MachineTrapKind::IntegerDivideByZero => WasmError::trap("integer divide by zero".into()),
        MachineTrapKind::IntegerOverflow => WasmError::trap("integer overflow".into()),
        MachineTrapKind::CallStackExhausted => WasmError::exhaustion("call stack exhausted".into()),
        MachineTrapKind::StackOverflow => WasmError::exhaustion("stack overflow".into()),
        MachineTrapKind::HelperFailure => WasmError::trap("native helper failed".into()),
    }
}

fn mem_width_bytes(width: MachineMemWidth) -> usize {
    match width {
        MachineMemWidth::U8 => 1,
        MachineMemWidth::U16 => 2,
        MachineMemWidth::U32 => 4,
        MachineMemWidth::U64 => 8,
    }
}

fn eval_int_unary(
    width: MachineIntWidth,
    op: MachineIntUnaryOp,
    src: u64,
) -> Result<u64, WasmError> {
    Ok(match (width, op) {
        (MachineIntWidth::I32, MachineIntUnaryOp::Eqz) => u64::from((src as u32 == 0) as u32),
        (MachineIntWidth::I64, MachineIntUnaryOp::Eqz) => u64::from((src == 0) as u32),
        (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => u64::from((src as u32).leading_zeros()),
        (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => u64::from(src.leading_zeros()),
        (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => u64::from((src as u32).trailing_zeros()),
        (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => u64::from(src.trailing_zeros()),
        (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => u64::from((src as u32).count_ones()),
        (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => u64::from(src.count_ones()),
        (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
            u64::from((src as u8 as i8 as i32) as u32)
        }
        (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
            u64::from((src as u16 as i16 as i32) as u32)
        }
        (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => (src as u8 as i8 as i64) as u64,
        (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => (src as u16 as i16 as i64) as u64,
        (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => (src as u32 as i32 as i64) as u64,
        _ => {
            return Err(WasmError::internal(
                "machine integer unary op is invalid for its width".into(),
            ))
        }
    })
}

fn eval_int_binary(
    width: MachineIntWidth,
    op: MachineIntBinaryOp,
    lhs: u64,
    rhs: u64,
) -> Result<u64, WasmError> {
    Ok(match width {
        MachineIntWidth::I32 => {
            let lhs_u = lhs as u32;
            let rhs_u = rhs as u32;
            let lhs_s = lhs_u as i32;
            let rhs_s = rhs_u as i32;
            let value = match op {
                MachineIntBinaryOp::Add => lhs_u.wrapping_add(rhs_u),
                MachineIntBinaryOp::Sub => lhs_u.wrapping_sub(rhs_u),
                MachineIntBinaryOp::Mul => lhs_u.wrapping_mul(rhs_u),
                MachineIntBinaryOp::DivS => {
                    if rhs_s == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    if lhs_s == i32::MIN && rhs_s == -1 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerOverflow));
                    }
                    lhs_s.wrapping_div(rhs_s) as u32
                }
                MachineIntBinaryOp::DivU => {
                    if rhs_u == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    lhs_u / rhs_u
                }
                MachineIntBinaryOp::RemS => {
                    if rhs_s == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    lhs_s.wrapping_rem(rhs_s) as u32
                }
                MachineIntBinaryOp::RemU => {
                    if rhs_u == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    lhs_u % rhs_u
                }
                MachineIntBinaryOp::And => lhs_u & rhs_u,
                MachineIntBinaryOp::Or => lhs_u | rhs_u,
                MachineIntBinaryOp::Xor => lhs_u ^ rhs_u,
                MachineIntBinaryOp::Shl => lhs_u.wrapping_shl(rhs_u & 31),
                MachineIntBinaryOp::ShrS => (lhs_s >> (rhs_u & 31)) as u32,
                MachineIntBinaryOp::ShrU => lhs_u >> (rhs_u & 31),
                MachineIntBinaryOp::Rotl => lhs_u.rotate_left(rhs_u & 31),
                MachineIntBinaryOp::Rotr => lhs_u.rotate_right(rhs_u & 31),
            };
            u64::from(value)
        }
        MachineIntWidth::I64 => {
            let lhs_s = lhs as i64;
            let rhs_s = rhs as i64;
            match op {
                MachineIntBinaryOp::Add => lhs.wrapping_add(rhs),
                MachineIntBinaryOp::Sub => lhs.wrapping_sub(rhs),
                MachineIntBinaryOp::Mul => lhs.wrapping_mul(rhs),
                MachineIntBinaryOp::DivS => {
                    if rhs_s == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    if lhs_s == i64::MIN && rhs_s == -1 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerOverflow));
                    }
                    lhs_s.wrapping_div(rhs_s) as u64
                }
                MachineIntBinaryOp::DivU => {
                    if rhs == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    lhs / rhs
                }
                MachineIntBinaryOp::RemS => {
                    if rhs_s == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    lhs_s.wrapping_rem(rhs_s) as u64
                }
                MachineIntBinaryOp::RemU => {
                    if rhs == 0 {
                        return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
                    }
                    lhs % rhs
                }
                MachineIntBinaryOp::And => lhs & rhs,
                MachineIntBinaryOp::Or => lhs | rhs,
                MachineIntBinaryOp::Xor => lhs ^ rhs,
                MachineIntBinaryOp::Shl => lhs.wrapping_shl((rhs & 63) as u32),
                MachineIntBinaryOp::ShrS => (lhs_s >> ((rhs & 63) as u32)) as u64,
                MachineIntBinaryOp::ShrU => lhs >> ((rhs & 63) as u32),
                MachineIntBinaryOp::Rotl => lhs.rotate_left((rhs & 63) as u32),
                MachineIntBinaryOp::Rotr => lhs.rotate_right((rhs & 63) as u32),
            }
        }
    })
}

fn eval_int_mul_wide(sign: MachineSign, lhs: u64, rhs: u64) -> (u64, u64) {
    let product = match sign {
        MachineSign::Signed => {
            let lhs = i64::from(lhs as u32 as i32);
            let rhs = i64::from(rhs as u32 as i32);
            lhs.wrapping_mul(rhs) as u64
        }
        MachineSign::Unsigned => u64::from(lhs as u32).wrapping_mul(u64::from(rhs as u32)),
    };
    (u64::from(product as u32), u64::from((product >> 32) as u32))
}

fn eval_i64_pair_div_rem(
    sign: MachineSign,
    rem: bool,
    lhs_lo: u64,
    lhs_hi: u64,
    rhs_lo: u64,
    rhs_hi: u64,
) -> Result<(u64, u64), WasmError> {
    let lhs = u64::from(lhs_lo as u32) | (u64::from(lhs_hi as u32) << 32);
    let rhs = u64::from(rhs_lo as u32) | (u64::from(rhs_hi as u32) << 32);
    let value = match sign {
        MachineSign::Unsigned => {
            if rhs == 0 {
                return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
            }
            if rem {
                lhs % rhs
            } else {
                lhs / rhs
            }
        }
        MachineSign::Signed => {
            let lhs = lhs as i64;
            let rhs = rhs as i64;
            if rhs == 0 {
                return Err(trap_from_kind(MachineTrapKind::IntegerDivideByZero));
            }
            if !rem && lhs == i64::MIN && rhs == -1 {
                return Err(trap_from_kind(MachineTrapKind::IntegerOverflow));
            }
            if rem {
                lhs.wrapping_rem(rhs) as u64
            } else {
                lhs.wrapping_div(rhs) as u64
            }
        }
    };
    Ok((u64::from(value as u32), u64::from((value >> 32) as u32)))
}

fn eval_i64_pair_unary(
    op: MachineIntUnaryOp,
    src_lo: u64,
    src_hi: u64,
) -> Result<(u64, u64), WasmError> {
    let src = u64::from(src_lo as u32) | (u64::from(src_hi as u32) << 32);
    let value = match op {
        MachineIntUnaryOp::Clz => u64::from(src.leading_zeros()),
        MachineIntUnaryOp::Ctz => u64::from(src.trailing_zeros()),
        MachineIntUnaryOp::Popcnt => u64::from(src.count_ones()),
        _ => {
            return Err(WasmError::internal(
                "machine Int64PairUnary requires a supported i64 unary op".into(),
            ))
        }
    };
    Ok((u64::from(value as u32), u64::from((value >> 32) as u32)))
}

fn eval_i64_pair_shift(
    op: MachineIntBinaryOp,
    lhs_lo: u64,
    lhs_hi: u64,
    rhs: u64,
) -> Result<(u64, u64), WasmError> {
    let lhs = u64::from(lhs_lo as u32) | (u64::from(lhs_hi as u32) << 32);
    let shift = (rhs as u32) & 63;
    let value = match op {
        MachineIntBinaryOp::Shl => lhs.wrapping_shl(shift),
        MachineIntBinaryOp::ShrS => ((lhs as i64) >> shift) as u64,
        MachineIntBinaryOp::ShrU => lhs >> shift,
        MachineIntBinaryOp::Rotl => lhs.rotate_left(shift),
        MachineIntBinaryOp::Rotr => lhs.rotate_right(shift),
        _ => {
            return Err(WasmError::internal(
                "machine Int64PairShift requires a shift/rotate op".into(),
            ))
        }
    };
    Ok((u64::from(value as u32), u64::from((value >> 32) as u32)))
}

fn eval_i64_pair_to_float(
    width: MachineFloatWidth,
    sign: MachineSign,
    src_lo: u64,
    src_hi: u64,
) -> u64 {
    let src = u64::from(src_lo as u32) | (u64::from(src_hi as u32) << 32);
    match (width, sign) {
        (MachineFloatWidth::F32, MachineSign::Signed) => from_f32((src as i64) as f32),
        (MachineFloatWidth::F32, MachineSign::Unsigned) => from_f32(src as f32),
        (MachineFloatWidth::F64, MachineSign::Signed) => from_f64((src as i64) as f64),
        (MachineFloatWidth::F64, MachineSign::Unsigned) => from_f64(src as f64),
    }
}

fn eval_int_compare(
    width: MachineIntWidth,
    kind: MachineCompareKind,
    sign: MachineSign,
    lhs: u64,
    rhs: u64,
) -> u64 {
    let result = match (width, sign) {
        (MachineIntWidth::I32, MachineSign::Signed) => {
            compare_i64(kind, lhs as u32 as i32 as i64, rhs as u32 as i32 as i64)
        }
        (MachineIntWidth::I32, MachineSign::Unsigned) => {
            compare_u64(kind, u64::from(lhs as u32), u64::from(rhs as u32))
        }
        (MachineIntWidth::I64, MachineSign::Signed) => compare_i64(kind, lhs as i64, rhs as i64),
        (MachineIntWidth::I64, MachineSign::Unsigned) => compare_u64(kind, lhs, rhs),
    };
    u64::from(result as u32)
}

fn eval_float_unary(
    width: MachineFloatWidth,
    op: MachineFloatUnaryOp,
    src: u64,
) -> Result<u64, WasmError> {
    Ok(match width {
        MachineFloatWidth::F32 => {
            let value = as_f32(src);
            match op {
                MachineFloatUnaryOp::Abs => from_f32(value.abs()),
                MachineFloatUnaryOp::Neg => from_f32(-value),
                MachineFloatUnaryOp::Ceil => from_f32(ceil_f32(value)),
                MachineFloatUnaryOp::Floor => from_f32(floor_f32(value)),
                MachineFloatUnaryOp::Trunc => from_f32(trunc_f32(value)),
                MachineFloatUnaryOp::Nearest => wasm_f32_nearest_bits(src as u32),
                MachineFloatUnaryOp::Sqrt => from_f32(sqrt_f32(value)),
            }
        }
        MachineFloatWidth::F64 => {
            let value = as_f64(src);
            match op {
                MachineFloatUnaryOp::Abs => from_f64(value.abs()),
                MachineFloatUnaryOp::Neg => from_f64(-value),
                MachineFloatUnaryOp::Ceil => from_f64(ceil_f64(value)),
                MachineFloatUnaryOp::Floor => from_f64(floor_f64(value)),
                MachineFloatUnaryOp::Trunc => from_f64(trunc_f64(value)),
                MachineFloatUnaryOp::Nearest => wasm_f64_nearest_bits(src),
                MachineFloatUnaryOp::Sqrt => from_f64(sqrt_f64(value)),
            }
        }
    })
}

fn eval_float_binary(
    width: MachineFloatWidth,
    op: MachineFloatBinaryOp,
    lhs: u64,
    rhs: u64,
) -> Result<u64, WasmError> {
    Ok(match width {
        MachineFloatWidth::F32 => match op {
            MachineFloatBinaryOp::Add => from_f32(as_f32(lhs) + as_f32(rhs)),
            MachineFloatBinaryOp::Sub => from_f32(as_f32(lhs) - as_f32(rhs)),
            MachineFloatBinaryOp::Mul => from_f32(as_f32(lhs) * as_f32(rhs)),
            MachineFloatBinaryOp::Div => from_f32(as_f32(lhs) / as_f32(rhs)),
            MachineFloatBinaryOp::Min => wasm_f32_min_bits(lhs as u32, rhs as u32),
            MachineFloatBinaryOp::Max => wasm_f32_max_bits(lhs as u32, rhs as u32),
            MachineFloatBinaryOp::Copysign => from_f32(as_f32(lhs).copysign(as_f32(rhs))),
        },
        MachineFloatWidth::F64 => match op {
            MachineFloatBinaryOp::Add => from_f64(as_f64(lhs) + as_f64(rhs)),
            MachineFloatBinaryOp::Sub => from_f64(as_f64(lhs) - as_f64(rhs)),
            MachineFloatBinaryOp::Mul => from_f64(as_f64(lhs) * as_f64(rhs)),
            MachineFloatBinaryOp::Div => from_f64(as_f64(lhs) / as_f64(rhs)),
            MachineFloatBinaryOp::Min => wasm_f64_min_bits(lhs, rhs),
            MachineFloatBinaryOp::Max => wasm_f64_max_bits(lhs, rhs),
            MachineFloatBinaryOp::Copysign => from_f64(as_f64(lhs).copysign(as_f64(rhs))),
        },
    })
}

fn eval_float_compare(
    width: MachineFloatWidth,
    kind: MachineCompareKind,
    lhs: u64,
    rhs: u64,
) -> u64 {
    let result = match width {
        MachineFloatWidth::F32 => compare_f64(kind, as_f32(lhs) as f64, as_f32(rhs) as f64),
        MachineFloatWidth::F64 => compare_f64(kind, as_f64(lhs), as_f64(rhs)),
    };
    u64::from(result as u32)
}

fn eval_convert(op: MachineConvertOp, src: u64) -> Result<u64, WasmError> {
    match op {
        MachineConvertOp::I32WrapI64 => Ok(u64::from(src as u32)),
        MachineConvertOp::I64ExtendI32S => Ok((src as u32 as i32 as i64) as u64),
        MachineConvertOp::I64ExtendI32U => Ok(u64::from(src as u32)),
        MachineConvertOp::I32TruncF32S => trunc_f32_to_i32_s(src as u32),
        MachineConvertOp::I32TruncF32U => trunc_f32_to_i32_u(src as u32),
        MachineConvertOp::I32TruncF64S => trunc_f64_to_i32_s(src),
        MachineConvertOp::I32TruncF64U => trunc_f64_to_i32_u(src),
        MachineConvertOp::I64TruncF32S => trunc_f32_to_i64_s(src as u32),
        MachineConvertOp::I64TruncF32U => trunc_f32_to_i64_u(src as u32),
        MachineConvertOp::I64TruncF64S => trunc_f64_to_i64_s(src),
        MachineConvertOp::I64TruncF64U => trunc_f64_to_i64_u(src),
        MachineConvertOp::I32TruncSatF32S => Ok(trunc_sat_f32_to_i32_s(src as u32)),
        MachineConvertOp::I32TruncSatF32U => Ok(trunc_sat_f32_to_i32_u(src as u32)),
        MachineConvertOp::I32TruncSatF64S => Ok(trunc_sat_f64_to_i32_s(src)),
        MachineConvertOp::I32TruncSatF64U => Ok(trunc_sat_f64_to_i32_u(src)),
        MachineConvertOp::I64TruncSatF32S => Ok(trunc_sat_f32_to_i64_s(src as u32)),
        MachineConvertOp::I64TruncSatF32U => Ok(trunc_sat_f32_to_i64_u(src as u32)),
        MachineConvertOp::I64TruncSatF64S => Ok(trunc_sat_f64_to_i64_s(src)),
        MachineConvertOp::I64TruncSatF64U => Ok(trunc_sat_f64_to_i64_u(src)),
        MachineConvertOp::F32ConvertI32S => Ok(from_f32(as_i32(src) as f32)),
        MachineConvertOp::F32ConvertI32U => Ok(from_f32(as_u32(src) as f32)),
        MachineConvertOp::F32ConvertI64S => Ok(from_f32(as_i64(src) as f32)),
        MachineConvertOp::F32ConvertI64U => Ok(from_f32(as_u64(src) as f32)),
        MachineConvertOp::F64ConvertI32S => Ok(from_f64(as_i32(src) as f64)),
        MachineConvertOp::F64ConvertI32U => Ok(from_f64(as_u32(src) as f64)),
        MachineConvertOp::F64ConvertI64S => Ok(from_f64(as_i64(src) as f64)),
        MachineConvertOp::F64ConvertI64U => Ok(from_f64(as_u64(src) as f64)),
        MachineConvertOp::F32DemoteF64 => Ok(from_f32(as_f64(src) as f32)),
        MachineConvertOp::F64PromoteF32 => Ok(from_f64(as_f32(src) as f64)),
        MachineConvertOp::I32ReinterpretF32 => Ok(u64::from(src as u32)),
        MachineConvertOp::I64ReinterpretF64 => Ok(src),
        MachineConvertOp::F32ReinterpretI32 => Ok(u64::from(src as u32)),
        MachineConvertOp::F64ReinterpretI64 => Ok(src),
    }
}

fn compare_u64(kind: MachineCompareKind, lhs: u64, rhs: u64) -> bool {
    match kind {
        MachineCompareKind::Eq => lhs == rhs,
        MachineCompareKind::Ne => lhs != rhs,
        MachineCompareKind::Lt => lhs < rhs,
        MachineCompareKind::Gt => lhs > rhs,
        MachineCompareKind::Le => lhs <= rhs,
        MachineCompareKind::Ge => lhs >= rhs,
    }
}

fn compare_i64(kind: MachineCompareKind, lhs: i64, rhs: i64) -> bool {
    match kind {
        MachineCompareKind::Eq => lhs == rhs,
        MachineCompareKind::Ne => lhs != rhs,
        MachineCompareKind::Lt => lhs < rhs,
        MachineCompareKind::Gt => lhs > rhs,
        MachineCompareKind::Le => lhs <= rhs,
        MachineCompareKind::Ge => lhs >= rhs,
    }
}

fn compare_f64(kind: MachineCompareKind, lhs: f64, rhs: f64) -> bool {
    match kind {
        MachineCompareKind::Eq => lhs == rhs,
        MachineCompareKind::Ne => lhs != rhs,
        MachineCompareKind::Lt => lhs < rhs,
        MachineCompareKind::Gt => lhs > rhs,
        MachineCompareKind::Le => lhs <= rhs,
        MachineCompareKind::Ge => lhs >= rhs,
    }
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

fn wasm_f32_min_bits(left_bits: u32, right_bits: u32) -> u64 {
    const NEG_ZERO: u32 = 0x8000_0000;
    let left = as_f32(left_bits as u64);
    let right = as_f32(right_bits as u64);
    if left.is_nan() || right.is_nan() {
        return from_f32(f32::NAN);
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
    let left = as_f32(left_bits as u64);
    let right = as_f32(right_bits as u64);
    if left.is_nan() || right.is_nan() {
        return from_f32(f32::NAN);
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
        return from_f64(f64::NAN);
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
        return from_f64(f64::NAN);
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
    let value = as_f32(bits as u64);
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
    from_f32(rounded)
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
    from_f64(rounded)
}

fn trunc_f32_to_i32_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -2147483649.0 || value >= 2147483648.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -1.0 || value >= 4294967296.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -2147483649.0 || value >= 2147483648.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -1.0 || value >= 4294967296.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

fn trunc_sat_f32_to_i32_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        0
    } else if value <= i32::MIN as f64 {
        from_i32(i32::MIN)
    } else if value >= i32::MAX as f64 {
        from_i32(i32::MAX)
    } else {
        from_i32(value as i32)
    }
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64) as f64;
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
        from_i32(i32::MIN)
    } else if value >= i32::MAX as f64 {
        from_i32(i32::MAX)
    } else {
        from_i32(value as i32)
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
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        0
    } else if value <= i64::MIN as f64 {
        from_i64(i64::MIN)
    } else if value >= i64::MAX as f64 {
        from_i64(i64::MAX)
    } else {
        from_i64(value as i64)
    }
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64) as f64;
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
        from_i64(i64::MIN)
    } else if value >= i64::MAX as f64 {
        from_i64(i64::MAX)
    } else {
        from_i64(value as i64)
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

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::eval;
    use crate::{
        error::WasmError,
        module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType},
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            entities::{Caller, FunctionInst, MemInst, ModuleInst},
            native::{
                arch::{
                    backend_mode_test_lock, set_reference_backend, set_reference_backend_mode,
                    ReferenceBackendMode,
                },
                build::ensure_module_compiled,
                ir::machine::{
                    MachineBranchCond, MachineCompareKind, MachineInstKind, MachineIntWidth,
                    MachineSign, MachineTrapKind, MachineValue,
                },
            },
            runtime,
            store::Store,
            value::Value,
        },
    };

    fn host_double(
        _caller: &mut Caller<'_>,
        args: &[Value],
        results: &mut [Value],
    ) -> Result<(), WasmError> {
        results[0] = Value::I32(i32::from(args[0]) * 2);
        Ok(())
    }

    fn native_test_memory(limits: Limits) -> MemInst {
        #[cfg(has_guard_pages)]
        {
            MemInst::new_guarded(limits).expect("guarded memory")
        }
        #[cfg(not(has_guard_pages))]
        {
            MemInst::new(limits)
        }
    }

    struct ReferenceBackendGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for ReferenceBackendGuard {
        fn drop(&mut self) {
            set_reference_backend(false).expect("reset reference backend");
        }
    }

    fn enable_reference_backend() -> ReferenceBackendGuard {
        enable_reference_backend_mode(ReferenceBackendMode::Emu64)
    }

    fn enable_reference_backend_mode(mode: ReferenceBackendMode) -> ReferenceBackendGuard {
        let lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        set_reference_backend_mode(mode).expect("reference backend");
        ReferenceBackendGuard { _lock: lock }
    }

    #[test]
    fn evaluates_native_machine_module_with_helper_external_call() {
        let _guard = enable_reference_backend();

        let ty = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
        ));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::External {
            func_type: Rc::clone(&ty),
            callback: host_double,
        });
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_locals(vec![]);
        spec.set_code((&[0x20, 0x00, 0x10, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("native compile");
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let FunctionInst::Local { spec, .. } = (unsafe { &*func_ptr }) else {
            panic!("expected local function");
        };
        let code = spec.get_native_code().expect("native code");
        let results =
            eval(spec, code, &mut store, &[Value::I32(21)], "reference").expect("native eval");
        assert_eq!(results.peek_at_index(0), Value::I32(42).to_raw());
    }

    #[test]
    fn runtime_eval_uses_emulator_backend() {
        let _guard = enable_reference_backend();

        let ty = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
        ));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x41, 0x01, 0x6a, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[Value::I32(4)]).expect("runtime eval");
        assert_eq!(results.peek_at_index(0), Value::I32(5).to_raw());
    }

    #[test]
    fn runtime_eval_preserves_first_local_call_result_across_second_local_call() {
        let _guard = enable_reference_backend();

        let malloc_ty = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
        ));
        let diff_ty = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![Rc::clone(&malloc_ty), Rc::clone(&diff_ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut malloc_spec = FunctionSpec::new(Rc::clone(&malloc_ty), 0);
        malloc_spec.set_code((&[0x41, 0x10, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: malloc_spec,
            type_index: 0,
        });

        let mut diff_spec = FunctionSpec::new(Rc::clone(&diff_ty), 1);
        diff_spec.set_locals(vec![ValueType::I32, ValueType::I32]);
        diff_spec.set_code(
            (&[
                0x41, 0x04, // i32.const 4
                0x10, 0x00, // call 0
                0x21, 0x00, // local.set 0
                0x41, 0x04, // i32.const 4
                0x10, 0x00, // call 0
                0x21, 0x01, // local.set 1
                0x20, 0x01, // local.get 1
                0x20, 0x00, // local.get 0
                0x6b, // i32.sub
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec: diff_spec,
            type_index: 1,
        });

        let mut store = Store::new(module);
        ensure_module_compiled(&store).expect("native compile");
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[]).expect("runtime eval");
        assert_eq!(results.peek_at_index(0), Value::I32(0).to_raw());
    }

    #[test]
    fn runtime_eval_emu32_traps_on_recursive_call_exhaustion() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let ty = Rc::new(FunctionType::new(vec![], vec![]));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x10, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let error =
            runtime::eval(func_ref, &mut store, &[]).expect_err("recursive call should exhaust");
        assert_eq!(error.message(), "call stack exhausted");
    }

    #[test]
    fn runtime_eval_emu32_traps_on_wrapping_memory_address() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let ty = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
        ));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x2d, 0x00, 0x01, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut mem = MemInst::new(Limits::new(1, Some(1)).unwrap());
        mem.data[..26].copy_from_slice(b"abcdefghijklmnopqrstuvwxyz");
        module.memories.push(mem);
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let ok = runtime::eval(func_ref, &mut store, &[Value::I32(0)]).expect("load at 0");
        assert_eq!(ok.peek_at_index(0), Value::I32(98).to_raw());

        let error = runtime::eval(func_ref, &mut store, &[Value::I32(-1)])
            .expect_err("wrapping effective address should trap");
        assert_eq!(error.message(), "out of bounds memory access");
    }

    #[test]
    fn compiled_emu32_keeps_access_wrap_trap_for_max_offset_memory_load() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let ty = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
        ));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x2d, 0x00, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        let store = Store::new(module);

        ensure_module_compiled(&store).expect("native compile");

        let code = store.module().functions[0]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("compiled native code");
        let ops = &code.compiled().module().functions[0].program.blocks[0].ops;
        assert!(
            ops.iter().any(|inst| matches!(
                &inst.kind,
                MachineInstKind::TrapIf {
                    kind: MachineTrapKind::MemoryOutOfBounds,
                    cond: MachineBranchCond::IntCompare {
                        width: MachineIntWidth::I32,
                        kind: MachineCompareKind::Lt,
                        sign: MachineSign::Unsigned,
                        rhs: MachineValue::Imm64(1),
                        ..
                    },
                }
            )),
            "compiled emu32 module must keep the access-size wrap trap for max-offset loads"
        );
    }

    #[test]
    fn runtime_eval_emu32_traps_on_max_offset_memory_address() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let ty = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
        ));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x2d, 0x00, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut mem = MemInst::new(Limits::new(1, Some(1)).unwrap());
        mem.data[..26].copy_from_slice(b"abcdefghijklmnopqrstuvwxyz");
        module.memories.push(mem);
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let error = runtime::eval(func_ref, &mut store, &[Value::I32(0)])
            .expect_err("max-offset memory access should trap");
        assert_eq!(error.message(), "out of bounds memory access");
    }

    #[test]
    fn runtime_eval_emu32_loads_i64_without_clobbering_address_base() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let ty = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I64],
        ));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x29, 0x00, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut mem = MemInst::new(Limits::new(1, Some(1)).unwrap());
        mem.data[..8].copy_from_slice(b"abcdefgh");
        module.memories.push(mem);
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[Value::I32(0)]).expect("i64 load");
        assert_eq!(
            results.peek_at_index(0),
            Value::I64(0x6867_6665_6463_6261).to_raw()
        );
    }

    #[test]
    fn runtime_eval_emu64_refreshes_mem0_regs_after_zero_page_memory_grow() {
        let _guard = enable_reference_backend();

        let ty = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code(
            (&[
                0x41, 0x01, // i32.const 1
                0x40, 0x00, // memory.grow 0
                0x1a, // drop
                0x41, 0x00, // i32.const 0
                0x41, 0x2a, // i32.const 42
                0x36, 0x02, 0x00, // i32.store align=2 offset=0
                0x41, 0x00, // i32.const 0
                0x28, 0x02, 0x00, // i32.load align=2 offset=0
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        module
            .memories
            .push(native_test_memory(Limits::new(0, Some(2)).unwrap()));
        let mut store = Store::new(module);

        ensure_module_compiled(&store).expect("native compile");
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let FunctionInst::Local { spec, .. } = (unsafe { &*func_ptr }) else {
            panic!("expected local function");
        };
        let code = spec.get_native_code().expect("native code");
        let results =
            eval(spec, code, &mut store, &[], "reference").expect("memory.grow + store/load");
        assert_eq!(results.peek_at_index(0), Value::I32(42).to_raw());
    }

    #[test]
    fn runtime_eval_emu64_traps_on_zero_page_memory_access() {
        let _guard = enable_reference_backend();

        let ty = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code(
            (&[
                0x41, 0x00, // i32.const 0
                0x28, 0x02, 0x00, // i32.load align=2 offset=0
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        module
            .memories
            .push(native_test_memory(Limits::new(0, Some(1)).unwrap()));
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let error = runtime::eval(func_ref, &mut store, &[])
            .expect_err("zero-page memory access should trap");
        assert_eq!(error.message(), "out of bounds memory access");
    }
}
