//! MachineIR emulator backend.
//!
//! This consumes the current machine module/runtime contract directly. It is a
//! backend under `arch/` rather than part of `native/runtime`, because its job
//! is to execute finalized MachineIR the same way a real ISA backend will.

mod address_space;
pub(crate) mod config;

use crate::collections;

use self::address_space::EmulatorAddressSpace;
use crate::{
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        jit::machine::machine_ir::{
            MachineAddr, MachineArgSrc, MachineBlock, MachineBlockId, MachineBranchCond,
            MachineCallArgs, MachineCallResults, MachineCallRuntime, MachineCallTarget,
            MachineCompareKind, MachineConvertOp, MachineEdge, MachineFloatBinaryOp,
            MachineFloatUnaryOp, MachineFloatWidth, MachineFrameRegion, MachineFuncId,
            MachineFunctionAbi, MachineIndexExtend, MachineInst, MachineInstKind,
            MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension,
            MachineMemWidth, MachineParamLoc, MachineProgram, MachineReg, MachineShiftOp,
            MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
            MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG,
            MACHINE_MEM0_SIZE_REG,
        },
        jit::runtime::{
            code::{CompiledNativeModule, NativeCode},
            collect_native_results_from_stack,
            common::NativeCallStatus,
            context::NativeContext,
            preserved::{self, io as preserved_io, op as preserved_op},
            runtime_call::call_runtime_entry_ptr,
        },
        result_buffer::ResultBuffer,
        store::Store,
        value::Value,
        value_encoding::{
            as_f32, as_f64, as_i32, as_i64, as_u32, as_u64, from_f32, from_f64, from_i32, from_i64,
            value_to_machine_raw_in_store,
        },
    },
};

#[cfg(sf_call_trace)]
use crate::vm::jit::debug::function_trace;

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

fn caller_results_base_delta(results: &MachineCallResults) -> u32 {
    match results {
        MachineCallResults::FrameFallback { caller_results, .. } => {
            u32::from(caller_results.base_slot) * 8
        }
        MachineCallResults::None
        | MachineCallResults::ScalarGp { .. }
        | MachineCallResults::ScalarGpPair { .. }
        | MachineCallResults::ScalarFp { .. } => 0,
    }
}

#[derive(Debug)]
struct SavedCaller {
    func_id: MachineFuncId,
    regs: collections::Vec<u64>,
    addr_kinds: collections::Vec<RegAddrKind>,
    /// CFG block to resume on return. The new local-call ABI carries the
    /// continuation as an explicit MIR field rather than as a memory slot in
    /// the callee's frame, so the emulator stashes it on its logical call
    /// stack instead of reading it back from frame memory.
    continuation: MachineBlockId,
    /// Caller frame pointer to restore on return.
    caller_fp: *mut u64,
    /// Absolute pointer to the caller's result-receive region. The callee's
    /// `Return` will copy its `return_results` slots here.
    caller_result_base: *mut u64,
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
    func_id: MachineFuncId,
    block_id: MachineBlockId,
    fp: *mut u64,
    regs: collections::Vec<u64>,
    addr_kinds: collections::Vec<RegAddrKind>,
    call_stack: collections::Vec<SavedCaller>,
    address_space: EmulatorAddressSpace,
}

#[derive(Clone, Copy, Debug)]
struct CapturedArgLane {
    dst: MachineReg,
    value: u64,
    kind: RegAddrKind,
}

pub(crate) fn eval_root_with_context(
    compiled: &CompiledNativeModule,
    func_id: MachineFuncId,
    ctx: &mut NativeContext,
    fp: *mut u64,
) -> Result<(), WasmError> {
    let program = compiled
        .function(func_id)
        .ok_or_else(|| WasmError::internal("native entry function is missing machine code"))?;
    let address_space = EmulatorAddressSpace::new(compiled, fp, ctx.stack_end);
    address_space.validate_runtime_shape(ctx)?;
    let runtime_base = address_space.runtime_base_value(ctx);
    let fp_base = address_space.frame_base_value(fp)?;
    let mem0_base = address_space.mem0_base_value(ctx);
    let mem0_size = ctx.mem0_size;
    let mut emulator = Emulator {
        ctx,
        compiled,
        root_frame: fp,
        func_id,
        block_id: program.program.entry,
        fp,
        regs: init_entry_regs(
            compiled,
            compiled.backend().total_reg_count(),
            runtime_base,
            fp_base,
            mem0_base,
            mem0_size,
        ),
        addr_kinds: init_entry_addr_kinds(compiled.backend().total_reg_count()),
        call_stack: collections::Vec::new(),
        address_space,
    };
    emulator.load_entry_param_lanes_from_frame(fp)?;
    emulator.run()
}

pub(crate) fn eval(
    spec: &FunctionSpec,
    code: &NativeCode,
    store: &mut Store,
    args: &[Value],
    backend: &'static str,
) -> Result<ResultBuffer, WasmError> {
    let _ = backend; // consumed by function-trace feature
    let func_type = spec.func_type();
    if args.len() != func_type.params().len() {
        return Err(WasmError::invalid("invalid argument count"));
    }

    let compiled = code.compiled();
    let func_id = code.func_id();
    let runtime = compiled
        .abi()
        .functions
        .get(func_id.0 as usize)
        .ok_or_else(|| WasmError::internal("native entry function is missing runtime metadata"))?;

    // At least one slot is guaranteed: `Engine::new` refuses a budget that
    // rounds down to none.
    let stack_slots = store.config().get_wasm_stack_bytes() / core::mem::size_of::<u64>();
    let mut stack = collections::vec![0u64; stack_slots];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(stack_slots) };

    unsafe {
        for (index, arg) in args.iter().enumerate() {
            *stack_base.add(index) =
                value_to_machine_raw_in_store(*arg, compiled.backend().gp_unit_bytes, store);
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

    let n_globals = store.module().globals.len();
    let mut ctx = NativeContext::new(store as *mut Store, stack_end, n_globals);
    ctx.seed_local_call_infos(compiled);
    #[cfg(sf_call_trace)]
    {
        function_trace::init_from_env();
        function_trace::native_root_entry(&mut *ctx, spec, backend);
    }

    let result = eval_root_with_context(compiled, func_id, &mut *ctx, stack_base);

    if let Err(ref error) = result {
        #[cfg(sf_call_trace)]
        function_trace::native_trap_current(&mut *ctx, error);
        return Err(error.clone());
    }

    let out = unsafe {
        collect_native_results_from_stack(
            stack_base,
            func_type.results(),
            compiled.backend().gp_unit_bytes,
            store,
        )
    }?;
    #[cfg(sf_call_trace)]
    {
        let results_len = func_type.results().len();
        let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
        function_trace::native_root_exit(&mut *ctx, spec, results);
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
                        .ok_or_else(|| WasmError::internal("machine jump table has no entries"))?;
                    self.jump_to_edge(edge)
                }
                MachineTerminator::Call {
                    target,
                    frame_delta,
                    args,
                    results,
                    success,
                } => self.enter_call(target, *frame_delta, args, results, success),
                MachineTerminator::TailCall { target, args } => self.enter_tail_call(target, args),
                MachineTerminator::Return | MachineTerminator::ReturnScalar { .. } => {
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

    #[cfg_attr(not(any(sf_has_std, test)), allow(unused_variables))]
    fn log_execution_error(
        &self,
        block_id: MachineBlockId,
        inst_idx: Option<usize>,
        inst: Option<&MachineInst>,
        terminator: Option<&MachineTerminator>,
        error: &WasmError,
    ) {
        #[cfg(any(sf_has_std, test))]
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
            MachineInstKind::Load {
                dst,
                addr,
                width,
                extension,
                ..
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
            MachineInstKind::Int64PairBinary {
                op,
                dst_lo,
                dst_hi,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
            } => {
                let (lo, hi) = eval_i64_pair_binary(
                    *op,
                    self.read_value(*lhs_lo)?,
                    self.read_value(*lhs_hi)?,
                    self.read_value(*rhs_lo)?,
                    self.read_value(*rhs_hi)?,
                )?;
                self.write_reg_with_kind(*dst_lo, lo, fixed_reg_addr_kind(*dst_lo))?;
                self.write_reg_with_kind(*dst_hi, hi, fixed_reg_addr_kind(*dst_hi))?;
            }
            MachineInstKind::Int64MulFromSignExt32 {
                dst_lo,
                dst_hi,
                lhs,
                rhs,
            } => {
                let lhs_lo = self.read_value(*lhs)? as i32;
                let rhs_lo = self.read_value(*rhs)? as i32;
                let product = (lhs_lo as i64).wrapping_mul(rhs_lo as i64) as u64;
                let lo = (product & 0xFFFF_FFFF) as u64;
                let hi = (product >> 32) & 0xFFFF_FFFF;
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
            MachineInstKind::Int64PairCompare {
                kind,
                sign,
                dst,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
            } => {
                let value = eval_i64_pair_compare(
                    *kind,
                    *sign,
                    self.read_value(*lhs_lo)?,
                    self.read_value(*lhs_hi)?,
                    self.read_value(*rhs_lo)?,
                    self.read_value(*rhs_hi)?,
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
            MachineInstKind::IndexedLoad {
                dst,
                base,
                index,
                index_extend,
                offset,
                width,
                extension,
            } => {
                let (addr_value, base_kind) =
                    self.indexed_addr_value(*base, *index, *index_extend, *offset)?;
                let value = self.load_at(addr_value, base_kind, *width, *extension)?;
                self.write_reg_with_kind(*dst, value, fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::IndexedStore {
                base,
                index,
                index_extend,
                offset,
                width,
                src,
            } => {
                let (addr_value, base_kind) =
                    self.indexed_addr_value(*base, *index, *index_extend, *offset)?;
                let value = self.read_value(*src)?;
                self.store_at(addr_value, base_kind, *width, value)?;
            }
            MachineInstKind::BitfieldExtractU {
                width,
                dst,
                src,
                lsb,
                bits,
            } => {
                let src_val = self.read_reg(*src)?;
                let result = match width {
                    MachineIntWidth::I32 => {
                        let v = src_val as u32;
                        let extracted = (v >> *lsb) & ((1u32 << *bits) - 1);
                        u64::from(extracted)
                    }
                    MachineIntWidth::I64 => (src_val >> *lsb) & ((1u64 << *bits) - 1),
                };
                self.write_reg_with_kind(*dst, result, fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::IntBinaryShifted {
                width,
                op,
                dst,
                lhs,
                rhs,
                shift,
                amount,
            } => {
                let lhs_val = self.read_reg(*lhs)?;
                let rhs_val = self.read_reg(*rhs)?;
                let shifted = apply_shift(*width, *shift, rhs_val, *amount);
                self.write_reg_with_kind(
                    *dst,
                    eval_int_binary(*width, *op, lhs_val, shifted)?,
                    fixed_reg_addr_kind(*dst),
                )?;
            }
            MachineInstKind::TestBits {
                width,
                kind,
                dst,
                src,
                mask,
            } => {
                let src_val = self.read_reg(*src)?;
                let mask_val = self.read_value(*mask)?;
                let anded = match width {
                    MachineIntWidth::I32 => u64::from((src_val as u32) & (mask_val as u32)),
                    MachineIntWidth::I64 => src_val & mask_val,
                };
                let result = match kind {
                    MachineCompareKind::Eq => u64::from(anded == 0),
                    MachineCompareKind::Ne => u64::from(anded != 0),
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits only supports Eq/Ne compare kinds".into(),
                        ));
                    }
                };
                self.write_reg_with_kind(*dst, result, fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::CallRuntime(call) => self.execute_call_runtime(call)?,
            MachineInstKind::EhThrow { tag_idx, args } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*tag_idx);
                io[preserved_io::ARG0] = self.fp as usize as u64;
                io[preserved_io::ARG1] = u64::from(args.start.0);
                io[preserved_io::ARG2] = u64::from(args.count);
                self.execute_preserved_helper(preserved_op::EH_THROW, &mut io)?;
            }
            MachineInstKind::EhThrowRef { exnref_slot } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::ARG0] = self.fp as usize as u64;
                io[preserved_io::ARG1] = u64::from(exnref_slot.0);
                self.execute_preserved_helper(preserved_op::EH_THROW_REF, &mut io)?;
            }
            MachineInstKind::EhAllocExnRef { tag_idx, dst } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*tag_idx);
                io[preserved_io::ARG0] = self.fp as usize as u64;
                self.execute_preserved_helper(preserved_op::EH_ALLOC_EXN_REF, &mut io)?;
                self.write_reg_with_kind(*dst, io[preserved_io::RET0], fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::MemoryGrow {
                mem_idx,
                dst,
                delta,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*mem_idx);
                io[preserved_io::ARG0] = self.read_value(*delta)?;
                self.execute_preserved_helper(preserved_op::MEMORY_GROW, &mut io)?;
                self.write_reg_with_kind(*dst, io[preserved_io::RET0], fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::MemoryFill {
                mem_idx,
                dest,
                val,
                len,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*mem_idx);
                io[preserved_io::ARG0] = self.read_value(*dest)?;
                io[preserved_io::ARG1] = self.read_value(*val)?;
                io[preserved_io::ARG2] = self.read_value(*len)?;
                self.execute_preserved_helper(preserved_op::MEMORY_FILL, &mut io)?;
            }
            MachineInstKind::MemoryCopy {
                dst_mem,
                src_mem,
                dest,
                src,
                len,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*dst_mem);
                io[preserved_io::IMM1] = u64::from(*src_mem);
                io[preserved_io::ARG0] = self.read_value(*dest)?;
                io[preserved_io::ARG1] = self.read_value(*src)?;
                io[preserved_io::ARG2] = self.read_value(*len)?;
                self.execute_preserved_helper(preserved_op::MEMORY_COPY, &mut io)?;
            }
            MachineInstKind::MemoryInit {
                mem_idx,
                data_idx,
                dest,
                src,
                len,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*mem_idx);
                io[preserved_io::IMM1] = u64::from(*data_idx);
                io[preserved_io::ARG0] = self.read_value(*dest)?;
                io[preserved_io::ARG1] = self.read_value(*src)?;
                io[preserved_io::ARG2] = self.read_value(*len)?;
                self.execute_preserved_helper(preserved_op::MEMORY_INIT, &mut io)?;
            }
            MachineInstKind::DataDrop { data_idx } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*data_idx);
                self.execute_preserved_helper(preserved_op::DATA_DROP, &mut io)?;
            }
            MachineInstKind::TableGrow {
                table_idx,
                dst,
                init_val,
                delta,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*table_idx);
                io[preserved_io::ARG0] = self.read_value(*init_val)?;
                io[preserved_io::ARG1] = self.read_value(*delta)?;
                self.execute_preserved_helper(preserved_op::TABLE_GROW, &mut io)?;
                self.write_reg_with_kind(*dst, io[preserved_io::RET0], fixed_reg_addr_kind(*dst))?;
            }
            MachineInstKind::TableFill {
                table_idx,
                start,
                val,
                len,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*table_idx);
                io[preserved_io::ARG0] = self.read_value(*start)?;
                io[preserved_io::ARG1] = self.read_value(*val)?;
                io[preserved_io::ARG2] = self.read_value(*len)?;
                self.execute_preserved_helper(preserved_op::TABLE_FILL, &mut io)?;
            }
            MachineInstKind::TableCopy {
                dst_tbl,
                src_tbl,
                dest,
                src,
                len,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*dst_tbl);
                io[preserved_io::IMM1] = u64::from(*src_tbl);
                io[preserved_io::ARG0] = self.read_value(*dest)?;
                io[preserved_io::ARG1] = self.read_value(*src)?;
                io[preserved_io::ARG2] = self.read_value(*len)?;
                self.execute_preserved_helper(preserved_op::TABLE_COPY, &mut io)?;
            }
            MachineInstKind::TableInit {
                table_idx,
                elem_idx,
                dest,
                src,
                len,
            } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*table_idx);
                io[preserved_io::IMM1] = u64::from(*elem_idx);
                io[preserved_io::ARG0] = self.read_value(*dest)?;
                io[preserved_io::ARG1] = self.read_value(*src)?;
                io[preserved_io::ARG2] = self.read_value(*len)?;
                self.execute_preserved_helper(preserved_op::TABLE_INIT, &mut io)?;
            }
            MachineInstKind::ElemDrop { elem_idx } => {
                let mut io = [0u64; preserved_io::SLOT_COUNT];
                io[preserved_io::IMM0] = u64::from(*elem_idx);
                self.execute_preserved_helper(preserved_op::ELEM_DROP, &mut io)?;
            }
            MachineInstKind::RefFunc { func_idx, dst } => {
                self.execute_preserved_result(
                    preserved_op::REF_FUNC,
                    *func_idx,
                    0,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::RefAsNonNull { src, dst } => {
                self.execute_preserved_result(
                    preserved_op::REF_AS_NON_NULL,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::RefEq { lhs, rhs, dst } => {
                self.execute_preserved_result(
                    preserved_op::REF_EQ,
                    0,
                    0,
                    *lhs,
                    *rhs,
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::RefI31 { src, dst } => {
                self.execute_preserved_result(
                    preserved_op::REF_I31,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::I31GetS { src, dst } => {
                self.execute_preserved_result(
                    preserved_op::I31_GET_S,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::I31GetU { src, dst } => {
                self.execute_preserved_result(
                    preserved_op::I31_GET_U,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::AnyConvertExtern { src, dst } => {
                self.execute_preserved_result(
                    preserved_op::ANY_CONVERT_EXTERN,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::ExternConvertAny { src, dst } => {
                self.execute_preserved_result(
                    preserved_op::EXTERN_CONVERT_ANY,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::RefTest { ref_type, src, dst } => {
                let encoded = ref_type.encode_to_u64();
                self.execute_preserved_result(
                    preserved_op::REF_TEST,
                    encoded as u32,
                    (encoded >> 32) as u32,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::RefCast { ref_type, src, dst } => {
                let encoded = ref_type.encode_to_u64();
                self.execute_preserved_result(
                    preserved_op::REF_CAST,
                    encoded as u32,
                    (encoded >> 32) as u32,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::StructNew {
                type_idx,
                fields,
                dst,
            } => {
                self.execute_struct_new(*type_idx, fields, *dst)?;
            }
            MachineInstKind::StructNewDefault { type_idx, dst } => {
                self.execute_preserved_result(
                    preserved_op::STRUCT_NEW_DEFAULT,
                    *type_idx,
                    0,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::StructGet {
                type_idx,
                field_idx,
                signed,
                ty,
                src,
                dst,
                dst_hi,
            } => {
                let op_code = match signed {
                    None => preserved_op::STRUCT_GET,
                    Some(true) => preserved_op::STRUCT_GET_S,
                    Some(false) => preserved_op::STRUCT_GET_U,
                };
                self.execute_preserved_result(
                    op_code,
                    *type_idx,
                    *field_idx,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    *ty,
                    *dst,
                    *dst_hi,
                )?;
            }
            MachineInstKind::StructSet {
                type_idx,
                field_idx,
                ref_src,
                value_lo,
                value_hi,
            } => {
                self.execute_struct_set(*type_idx, *field_idx, *ref_src, *value_lo, *value_hi)?;
            }
            MachineInstKind::ArrayNew {
                type_idx,
                init_lo,
                init_hi,
                length,
                dst,
            } => {
                self.execute_array_new(*type_idx, *init_lo, *init_hi, *length, *dst)?;
            }
            MachineInstKind::ArrayNewDefault {
                type_idx,
                length,
                dst,
            } => {
                self.execute_preserved_result(
                    preserved_op::ARRAY_NEW_DEFAULT,
                    *type_idx,
                    0,
                    *length,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
            MachineInstKind::ArrayNewFixed {
                type_idx,
                elements,
                dst,
            } => {
                self.execute_array_new_fixed(*type_idx, elements, *dst)?;
            }
            MachineInstKind::ArrayNewData {
                type_idx,
                data_idx,
                src,
                len,
                dst,
            } => {
                self.execute_array_new_data(*type_idx, *data_idx, *src, *len, *dst)?;
            }
            MachineInstKind::ArrayNewElem {
                type_idx,
                elem_idx,
                src,
                len,
                dst,
            } => {
                self.execute_array_new_elem(*type_idx, *elem_idx, *src, *len, *dst)?;
            }
            MachineInstKind::ArrayGet {
                type_idx,
                signed,
                ty,
                ref_src,
                index,
                dst,
                dst_hi,
            } => {
                let op_code = match signed {
                    None => preserved_op::ARRAY_GET,
                    Some(true) => preserved_op::ARRAY_GET_S,
                    Some(false) => preserved_op::ARRAY_GET_U,
                };
                self.execute_preserved_result(
                    op_code,
                    *type_idx,
                    0,
                    *ref_src,
                    *index,
                    MachineValue::Imm64(0),
                    *ty,
                    *dst,
                    *dst_hi,
                )?;
            }
            MachineInstKind::ArraySet {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
            } => {
                self.execute_array_set(*type_idx, *ref_src, *index, *value_lo, *value_hi)?;
            }
            MachineInstKind::ArrayFill {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
                len,
            } => {
                self.execute_array_fill(*type_idx, *ref_src, *index, *value_lo, *value_hi, *len)?;
            }
            MachineInstKind::ArrayCopy {
                dst_type_idx,
                src_type_idx,
                dst_ref,
                dst_index,
                src_ref,
                src_index,
                len,
            } => {
                self.execute_array_copy(
                    *dst_type_idx,
                    *src_type_idx,
                    *dst_ref,
                    *dst_index,
                    *src_ref,
                    *src_index,
                    *len,
                )?;
            }
            MachineInstKind::ArrayInitData {
                type_idx,
                data_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            } => {
                self.execute_array_init_data(
                    *type_idx, *data_idx, *ref_src, *dst_index, *src_index, *len,
                )?;
            }
            MachineInstKind::ArrayInitElem {
                type_idx,
                elem_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            } => {
                self.execute_array_init_elem(
                    *type_idx, *elem_idx, *ref_src, *dst_index, *src_index, *len,
                )?;
            }
            MachineInstKind::ArrayLen { src, dst } => {
                self.execute_preserved_result(
                    preserved_op::ARRAY_LEN,
                    0,
                    0,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                    None,
                )?;
            }
        }
        Ok(())
    }

    fn execute_call_runtime(&mut self, call: &MachineCallRuntime) -> Result<(), WasmError> {
        let metadata = self
            .compiled
            .const_ptr(call.metadata)
            .ok_or_else(|| WasmError::internal("machine runtime-call metadata is out of range"))?;
        let entry = call_runtime_entry_ptr();
        let status = unsafe { entry(self.ctx as *mut NativeContext, self.fp, metadata) };
        if status == NativeCallStatus::Ok as u32 {
            self.address_space.validate_runtime_shape(self.ctx)?;
            return Ok(());
        }
        Err(self
            .ctx
            .error
            .take()
            .or_else(|| core::mem::take(&mut self.ctx.pending_escape).into_error())
            .unwrap_or_else(|| trap_from_kind(MachineTrapKind::HelperFailure)))
    }

    fn execute_preserved_result(
        &mut self,
        op_code: u32,
        imm0: u32,
        imm1: u32,
        arg0: MachineValue,
        arg1: MachineValue,
        arg2: MachineValue,
        _ty: MachineStorageType,
        dst: MachineReg,
        dst_hi: Option<MachineReg>,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(imm0);
        io[preserved_io::IMM1] = u64::from(imm1);
        io[preserved_io::ARG0] = self.read_value(arg0)?;
        io[preserved_io::ARG1] = self.read_value(arg1)?;
        io[preserved_io::ARG2] = self.read_value(arg2)?;
        self.execute_preserved_helper(op_code, &mut io)?;
        if let Some(dst_hi) = dst_hi {
            self.write_reg_with_kind(
                dst,
                u64::from(io[preserved_io::RET0] as u32),
                fixed_reg_addr_kind(dst),
            )?;
            self.write_reg_with_kind(
                dst_hi,
                u64::from((io[preserved_io::RET0] >> 32) as u32),
                fixed_reg_addr_kind(dst_hi),
            )?;
        } else {
            self.write_reg_with_kind(dst, io[preserved_io::RET0], fixed_reg_addr_kind(dst))?;
        }
        Ok(())
    }

    fn pack_preserved_io_value(
        &mut self,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
    ) -> Result<u64, WasmError> {
        Ok(if let Some(value_hi) = value_hi {
            u64::from(self.read_value(value_lo)? as u32)
                | (u64::from(self.read_value(value_hi)? as u32) << 32)
        } else {
            self.read_value(value_lo)?
        })
    }

    fn execute_struct_new(
        &mut self,
        type_idx: u32,
        fields: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        // Keep the packed payload alive across the helper call; the helper only
        // borrows this buffer for the duration of `execute_preserved_helper`.
        let mut payload = collections::Vec::with_capacity(fields.len());
        for (value_lo, value_hi) in fields {
            payload.push(self.pack_preserved_io_value(*value_lo, *value_hi)?);
        }
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::IMM1] = fields.len() as u64;
        io[preserved_io::ARG0] = if payload.is_empty() {
            0
        } else {
            payload.as_ptr() as usize as u64
        };
        self.execute_preserved_helper(preserved_op::STRUCT_NEW, &mut io)?;
        self.write_reg_with_kind(dst, io[preserved_io::RET0], fixed_reg_addr_kind(dst))?;
        Ok(())
    }

    fn execute_struct_set(
        &mut self,
        type_idx: u32,
        field_idx: u32,
        ref_src: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::IMM1] = u64::from(field_idx);
        io[preserved_io::ARG0] = self.read_value(ref_src)?;
        io[preserved_io::ARG1] = self.pack_preserved_io_value(value_lo, value_hi)?;
        self.execute_preserved_helper(preserved_op::STRUCT_SET, &mut io)
    }

    fn execute_array_new(
        &mut self,
        type_idx: u32,
        init_lo: MachineValue,
        init_hi: Option<MachineValue>,
        length: MachineValue,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::ARG0] = self.pack_preserved_io_value(init_lo, init_hi)?;
        io[preserved_io::ARG1] = self.read_value(length)?;
        self.execute_preserved_helper(preserved_op::ARRAY_NEW, &mut io)?;
        self.write_reg_with_kind(dst, io[preserved_io::RET0], fixed_reg_addr_kind(dst))?;
        Ok(())
    }

    fn execute_array_new_fixed(
        &mut self,
        type_idx: u32,
        elements: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        // Keep the packed payload alive across the helper call; the helper only
        // borrows this buffer for the duration of `execute_preserved_helper`.
        let mut payload = collections::Vec::with_capacity(elements.len());
        for (value_lo, value_hi) in elements {
            payload.push(self.pack_preserved_io_value(*value_lo, *value_hi)?);
        }
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::IMM1] = elements.len() as u64;
        io[preserved_io::ARG0] = if payload.is_empty() {
            0
        } else {
            payload.as_ptr() as usize as u64
        };
        self.execute_preserved_helper(preserved_op::ARRAY_NEW_FIXED, &mut io)?;
        self.write_reg_with_kind(dst, io[preserved_io::RET0], fixed_reg_addr_kind(dst))?;
        Ok(())
    }

    fn execute_array_new_data(
        &mut self,
        type_idx: u32,
        data_idx: u32,
        src: MachineValue,
        len: MachineValue,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::IMM1] = u64::from(data_idx);
        io[preserved_io::ARG0] = self.read_value(src)?;
        io[preserved_io::ARG1] = self.read_value(len)?;
        self.execute_preserved_helper(preserved_op::ARRAY_NEW_DATA, &mut io)?;
        self.write_reg_with_kind(dst, io[preserved_io::RET0], fixed_reg_addr_kind(dst))?;
        Ok(())
    }

    fn execute_array_new_elem(
        &mut self,
        type_idx: u32,
        elem_idx: u32,
        src: MachineValue,
        len: MachineValue,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::IMM1] = u64::from(elem_idx);
        io[preserved_io::ARG0] = self.read_value(src)?;
        io[preserved_io::ARG1] = self.read_value(len)?;
        self.execute_preserved_helper(preserved_op::ARRAY_NEW_ELEM, &mut io)?;
        self.write_reg_with_kind(dst, io[preserved_io::RET0], fixed_reg_addr_kind(dst))?;
        Ok(())
    }

    fn execute_array_set(
        &mut self,
        type_idx: u32,
        ref_src: MachineValue,
        index: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::ARG0] = self.read_value(ref_src)?;
        io[preserved_io::ARG1] = self.read_value(index)?;
        io[preserved_io::ARG2] = self.pack_preserved_io_value(value_lo, value_hi)?;
        self.execute_preserved_helper(preserved_op::ARRAY_SET, &mut io)
    }

    fn execute_array_fill(
        &mut self,
        type_idx: u32,
        ref_src: MachineValue,
        index: MachineValue,
        value_lo: MachineValue,
        value_hi: Option<MachineValue>,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::ARG0] = self.read_value(ref_src)?;
        io[preserved_io::ARG1] = self.read_value(index)?;
        io[preserved_io::ARG2] = self.pack_preserved_io_value(value_lo, value_hi)?;
        io[preserved_io::ARG3] = self.read_value(len)?;
        self.execute_preserved_helper(preserved_op::ARRAY_FILL, &mut io)
    }

    fn execute_array_copy(
        &mut self,
        dst_type_idx: u32,
        src_type_idx: u32,
        dst_ref: MachineValue,
        dst_index: MachineValue,
        src_ref: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(dst_type_idx);
        io[preserved_io::IMM1] = u64::from(src_type_idx);
        io[preserved_io::ARG0] = self.read_value(dst_ref)?;
        io[preserved_io::ARG1] = self.read_value(dst_index)?;
        io[preserved_io::ARG2] = self.read_value(src_ref)?;
        io[preserved_io::ARG3] = self.read_value(src_index)?;
        io[preserved_io::ARG4] = self.read_value(len)?;
        self.execute_preserved_helper(preserved_op::ARRAY_COPY, &mut io)
    }

    fn execute_array_init_data(
        &mut self,
        type_idx: u32,
        data_idx: u32,
        ref_src: MachineValue,
        dst_index: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::IMM1] = u64::from(data_idx);
        io[preserved_io::ARG0] = self.read_value(ref_src)?;
        io[preserved_io::ARG1] = self.read_value(dst_index)?;
        io[preserved_io::ARG2] = self.read_value(src_index)?;
        io[preserved_io::ARG3] = self.read_value(len)?;
        self.execute_preserved_helper(preserved_op::ARRAY_INIT_DATA, &mut io)
    }

    fn execute_array_init_elem(
        &mut self,
        type_idx: u32,
        elem_idx: u32,
        ref_src: MachineValue,
        dst_index: MachineValue,
        src_index: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        let mut io = [0u64; preserved_io::SLOT_COUNT];
        io[preserved_io::IMM0] = u64::from(type_idx);
        io[preserved_io::IMM1] = u64::from(elem_idx);
        io[preserved_io::ARG0] = self.read_value(ref_src)?;
        io[preserved_io::ARG1] = self.read_value(dst_index)?;
        io[preserved_io::ARG2] = self.read_value(src_index)?;
        io[preserved_io::ARG3] = self.read_value(len)?;
        self.execute_preserved_helper(preserved_op::ARRAY_INIT_ELEM, &mut io)
    }

    fn execute_preserved_helper(
        &mut self,
        op_code: u32,
        io: &mut [u64; preserved_io::SLOT_COUNT],
    ) -> Result<(), WasmError> {
        let status = unsafe {
            preserved::preserved_entry(self.ctx as *mut NativeContext, op_code, io.as_mut_ptr())
        };
        if status == NativeCallStatus::Ok as u32 {
            self.address_space.validate_runtime_shape(self.ctx)?;
            return Ok(());
        }
        Err(self
            .ctx
            .error
            .take()
            .or_else(|| core::mem::take(&mut self.ctx.pending_escape).into_error())
            .unwrap_or_else(|| trap_from_kind(MachineTrapKind::HelperFailure)))
    }

    fn jump_to_edge(&mut self, edge: &MachineEdge) -> Result<(), WasmError> {
        let target_params = self
            .current_program()?
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| WasmError::internal("machine edge target is out of range"))?
            .params
            .clone();
        // Read every non-reserved arg up front so sequential writes below
        // cannot clobber a source register before it is consumed. Reserved
        // cache edges are identity-only: the target register already holds
        // the cached-local value, so no move happens across the edge — we
        // only verify the identity invariant the native backends rely on.
        let mut pending: collections::Vec<(MachineReg, u64, RegAddrKind)> =
            collections::Vec::with_capacity(edge.args.len());
        for (param, arg) in target_params.iter().zip(edge.args.iter()) {
            match *arg {
                MachineValue::ReservedReg(reg) => {
                    if reg != param.reg {
                        return Err(WasmError::internal(
                            "emulator received non-identity reserved cache edge move into from",
                        ));
                    }
                }
                other => {
                    let value = self.read_value(other)?;
                    let kind = self.value_addr_kind(other);
                    pending.push((param.reg, value, kind));
                }
            }
        }
        for (reg, value, kind) in pending.into_iter() {
            let kind = match fixed_reg_addr_kind(reg) {
                RegAddrKind::Unknown => kind,
                fixed => fixed,
            };
            self.write_reg_with_kind(reg, value, kind)?;
        }
        self.block_id = edge.target;
        Ok(())
    }

    fn enter_call(
        &mut self,
        target: &MachineCallTarget,
        frame_delta: u32,
        args: &MachineCallArgs,
        results: &MachineCallResults,
        success: &MachineEdge,
    ) -> Result<(), WasmError> {
        let (callee, check_stack_capacity) = match target {
            MachineCallTarget::Direct(callee) => (*callee, true),
            MachineCallTarget::Indirect { callee_target, .. } => {
                (MachineFuncId(self.read_reg(*callee_target)? as u32), false)
            }
        };
        let lane_args = self.capture_call_arg_lanes(args)?;
        let callee_fp = unsafe { self.fp.add((frame_delta / 8) as usize) };
        let result_base_ptr = unsafe {
            self.fp
                .add((caller_results_base_delta(results) / 8) as usize)
        };
        self.enter_callee(
            callee,
            callee_fp,
            result_base_ptr,
            success.target,
            check_stack_capacity,
            lane_args,
        )
    }

    fn enter_tail_call(
        &mut self,
        target: &MachineCallTarget,
        args: &MachineCallArgs,
    ) -> Result<(), WasmError> {
        let (callee, check_stack_capacity) = match target {
            MachineCallTarget::Direct(callee) => (*callee, true),
            MachineCallTarget::Indirect { callee_target, .. } => {
                (MachineFuncId(self.read_reg(*callee_target)? as u32), false)
            }
        };
        let lane_args = self.capture_call_arg_lanes(args)?;
        self.enter_tail_callee(callee, self.fp, check_stack_capacity, lane_args)
    }

    fn enter_callee(
        &mut self,
        callee: MachineFuncId,
        callee_fp: *mut u64,
        caller_result_base: *mut u64,
        continuation: MachineBlockId,
        check_stack_capacity: bool,
        lane_args: collections::Vec<CapturedArgLane>,
    ) -> Result<(), WasmError> {
        let callee_function = self
            .compiled
            .function(callee)
            .ok_or_else(|| WasmError::internal("machine local callee is out of range"))?;
        let callee_runtime = self.runtime_for(callee)?;
        if check_stack_capacity {
            ensure_stack_capacity(
                callee_fp,
                self.ctx.stack_end,
                callee_runtime.total_frame_slots,
            )?;
        }
        // Save the caller state on the emulator's logical call stack. The
        // continuation block, caller frame pointer, and caller_result_base
        // travel via this stack rather than via memory slots in the callee's
        // frame, mirroring how native backends keep them in a backend-private
        // host-stack call record.
        self.call_stack.push(SavedCaller {
            func_id: self.func_id,
            regs: core::mem::take(&mut self.regs),
            addr_kinds: core::mem::take(&mut self.addr_kinds),
            continuation,
            caller_fp: self.fp,
            caller_result_base,
        });
        self.func_id = callee;
        self.fp = callee_fp;
        self.block_id = callee_function.program.entry;
        self.regs = init_entry_regs(
            self.compiled,
            self.compiled.backend().total_reg_count(),
            self.address_space.runtime_base_value(self.ctx),
            self.address_space.frame_base_value(callee_fp)?,
            self.address_space.mem0_base_value(self.ctx),
            self.ctx.mem0_size,
        );
        self.addr_kinds = init_entry_addr_kinds(self.compiled.backend().total_reg_count());
        self.install_captured_arg_lanes(&lane_args)?;
        #[cfg(sf_call_trace)]
        function_trace::native_function_trace_enter_func_idx(self.ctx, callee.0);
        Ok(())
    }

    fn enter_tail_callee(
        &mut self,
        callee: MachineFuncId,
        callee_fp: *mut u64,
        check_stack_capacity: bool,
        lane_args: collections::Vec<CapturedArgLane>,
    ) -> Result<(), WasmError> {
        let callee_function = self
            .compiled
            .function(callee)
            .ok_or_else(|| WasmError::internal("machine local callee is out of range"))?;
        let callee_runtime = self.runtime_for(callee)?;
        if check_stack_capacity {
            ensure_stack_capacity(
                callee_fp,
                self.ctx.stack_end,
                callee_runtime.total_frame_slots,
            )?;
        }
        self.func_id = callee;
        self.fp = callee_fp;
        self.block_id = callee_function.program.entry;
        self.regs = init_entry_regs(
            self.compiled,
            self.compiled.backend().total_reg_count(),
            self.address_space.runtime_base_value(self.ctx),
            self.address_space.frame_base_value(callee_fp)?,
            self.address_space.mem0_base_value(self.ctx),
            self.ctx.mem0_size,
        );
        self.addr_kinds = init_entry_addr_kinds(self.compiled.backend().total_reg_count());
        self.install_captured_arg_lanes(&lane_args)?;
        #[cfg(sf_call_trace)]
        function_trace::native_function_trace_tail_call_enter_func_idx(self.ctx, callee.0);
        Ok(())
    }

    fn load_entry_param_lanes_from_frame(&mut self, source_fp: *mut u64) -> Result<(), WasmError> {
        let param_locs = self.runtime_for(self.func_id)?.param_locs.clone();
        for loc in param_locs {
            match loc {
                MachineParamLoc::Frame { .. } => {}
                MachineParamLoc::GpArg {
                    param_index, lane, ..
                } => {
                    let reg = self.gp_arg_lane_reg(lane)?;
                    let value = unsafe { *source_fp.add(param_index as usize) };
                    self.write_reg_with_kind(reg, value, fixed_reg_addr_kind(reg))?;
                }
                MachineParamLoc::GpArgPair {
                    param_index,
                    lo_lane,
                    hi_lane,
                } => {
                    let raw = unsafe { *source_fp.add(param_index as usize) };
                    let lo = self.gp_arg_lane_reg(lo_lane)?;
                    let hi = self.gp_arg_lane_reg(hi_lane)?;
                    self.write_reg_with_kind(lo, u64::from(raw as u32), fixed_reg_addr_kind(lo))?;
                    self.write_reg_with_kind(hi, raw >> 32, fixed_reg_addr_kind(hi))?;
                }
                MachineParamLoc::FpArg {
                    param_index, lane, ..
                } => {
                    let reg = self.fp_arg_lane_reg(lane)?;
                    let value = unsafe { *source_fp.add(param_index as usize) };
                    self.write_reg_with_kind(reg, value, fixed_reg_addr_kind(reg))?;
                }
            }
        }
        Ok(())
    }

    fn capture_call_arg_lanes(
        &self,
        args: &MachineCallArgs,
    ) -> Result<collections::Vec<CapturedArgLane>, WasmError> {
        let mut out = collections::Vec::with_capacity(args.lane_args.len().saturating_mul(2));
        for arg in &args.lane_args {
            match arg {
                crate::vm::jit::machine::machine_ir::MachineCallLaneArg::Gp {
                    lane, src, ..
                } => {
                    let dst = self.gp_arg_lane_reg(*lane)?;
                    let (value, kind) = self.read_arg_src(*src)?;
                    out.push(CapturedArgLane { dst, value, kind });
                }
                crate::vm::jit::machine::machine_ir::MachineCallLaneArg::GpPair {
                    lo_lane,
                    hi_lane,
                    src,
                    ..
                } => {
                    let lo = self.gp_arg_lane_reg(*lo_lane)?;
                    let hi = self.gp_arg_lane_reg(*hi_lane)?;
                    let (lo_value, lo_kind) = self.read_arg_src(src.lo)?;
                    let (hi_value, hi_kind) = self.read_arg_src(src.hi)?;
                    out.push(CapturedArgLane {
                        dst: lo,
                        value: lo_value,
                        kind: lo_kind,
                    });
                    out.push(CapturedArgLane {
                        dst: hi,
                        value: hi_value,
                        kind: hi_kind,
                    });
                }
                crate::vm::jit::machine::machine_ir::MachineCallLaneArg::Fp {
                    lane, src, ..
                } => {
                    let dst = self.fp_arg_lane_reg(*lane)?;
                    let (value, kind) = self.read_arg_src(*src)?;
                    out.push(CapturedArgLane { dst, value, kind });
                }
            }
        }
        Ok(out)
    }

    fn install_captured_arg_lanes(&mut self, lanes: &[CapturedArgLane]) -> Result<(), WasmError> {
        for lane in lanes {
            let kind = match fixed_reg_addr_kind(lane.dst) {
                RegAddrKind::Unknown => lane.kind,
                fixed => fixed,
            };
            self.write_reg_with_kind(lane.dst, lane.value, kind)?;
        }
        Ok(())
    }

    fn read_arg_src(&self, src: MachineArgSrc) -> Result<(u64, RegAddrKind), WasmError> {
        match src {
            MachineArgSrc::Reg(reg) => Ok((self.read_reg(reg)?, self.reg_addr_kind(reg))),
            MachineArgSrc::FrameSlot(slot) => {
                let value = unsafe { *self.fp.add(slot.0 as usize) };
                Ok((value, RegAddrKind::Unknown))
            }
            MachineArgSrc::FrameSlotOffset { slot, byte_offset } => {
                let base = unsafe { self.fp.add(slot.0 as usize).cast::<u8>() };
                let value = match byte_offset {
                    0 => unsafe { *base.cast::<u64>() },
                    4 => unsafe { u64::from(*base.add(4).cast::<u32>()) },
                    offset => {
                        let ptr = unsafe { base.offset(isize::from(offset)) };
                        unsafe { *ptr.cast::<u64>() }
                    }
                };
                Ok((value, RegAddrKind::Unknown))
            }
        }
    }

    fn gp_arg_lane_reg(&self, lane: u8) -> Result<MachineReg, WasmError> {
        let config = self.compiled.backend();
        if lane >= config.allocatable_gp_dynamic_budget() {
            return Err(WasmError::internal(
                "internal GP argument lane is out of range",
            ));
        }
        Ok(MachineReg(MACHINE_FIXED_REG_COUNT + u16::from(lane)))
    }

    fn fp_arg_lane_reg(&self, lane: u8) -> Result<MachineReg, WasmError> {
        let config = self.compiled.backend();
        if lane >= config.fp_dynamic_budget {
            return Err(WasmError::internal(
                "internal FP argument lane is out of range",
            ));
        }
        Ok(MachineReg(config.first_fp_reg() + u16::from(lane)))
    }

    fn handle_return(&mut self) -> Result<bool, WasmError> {
        let current_runtime = self.runtime_for(self.func_id)?.clone();
        let results = current_runtime.return_results;
        if let Some(saved) = self.call_stack.pop() {
            // The unified Return mechanism: copy results into the caller's
            // result-receive region (an absolute pointer the caller pushed
            // onto the logical call stack), restore caller state, and resume
            // at the continuation block.
            self.copy_results(results, self.fp, saved.caller_result_base)?;
            #[cfg(sf_call_trace)]
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
            self.fp = saved.caller_fp;
            self.regs = saved.regs;
            self.addr_kinds = saved.addr_kinds;
            self.block_id = saved.continuation;
            self.init_reserved_regs()?;
            return Ok(false);
        }

        // Root return: copy results into the root frame, where the host
        // entry path reads them from (`collect_native_results_from_stack`).
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
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                let src_val = self.read_value(src)?;
                let mask_val = self.read_value(mask)?;
                let anded = match width {
                    MachineIntWidth::I32 => u64::from((src_val as u32) & (mask_val as u32)),
                    MachineIntWidth::I64 => src_val & mask_val,
                };
                Ok(match kind {
                    MachineCompareKind::Eq => anded == 0,
                    MachineCompareKind::Ne => anded != 0,
                    _ => {
                        return Err(WasmError::internal(
                            "TestBits branch only supports Eq/Ne compare kinds".into(),
                        ));
                    }
                })
            }
        }
    }

    fn current_program(&self) -> Result<&MachineProgram, WasmError> {
        Ok(&self
            .compiled
            .function(self.func_id)
            .ok_or_else(|| WasmError::internal("machine current function is out of range"))?
            .program)
    }

    fn current_block(&self) -> Result<&MachineBlock, WasmError> {
        self.current_program()?
            .blocks
            .get(self.block_id.as_usize())
            .ok_or_else(|| WasmError::internal("machine current block is out of range"))
    }

    fn runtime_for(&self, func_id: MachineFuncId) -> Result<&MachineFunctionAbi, WasmError> {
        self.compiled
            .abi()
            .functions
            .get(func_id.0 as usize)
            .ok_or_else(|| WasmError::internal("machine runtime record is out of range"))
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
            .ok_or_else(|| WasmError::internal("machine register is out of range"))
    }

    fn write_reg(&mut self, reg: MachineReg, value: u64) -> Result<(), WasmError> {
        let slot = self
            .regs
            .get_mut(reg.0 as usize)
            .ok_or_else(|| WasmError::internal("machine register is out of range"))?;
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
            MachineValue::ReservedReg(_) => RegAddrKind::Unknown,
            MachineValue::Imm64(_) => RegAddrKind::Unknown,
        }
    }

    fn set_reg_addr_kind(&mut self, reg: MachineReg, kind: RegAddrKind) -> Result<(), WasmError> {
        let slot = self
            .addr_kinds
            .get_mut(reg.0 as usize)
            .ok_or_else(|| WasmError::internal("machine register is out of range"))?;
        *slot = kind;
        Ok(())
    }

    fn read_value(&self, value: MachineValue) -> Result<u64, WasmError> {
        match value {
            MachineValue::Reg(reg) => self.read_reg(reg),
            MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
                "emulator attempted to read reserved cache register as a real value",
            )),
            MachineValue::Imm64(value) => Ok(value),
        }
    }

    fn addr_value(&self, addr: MachineAddr) -> u64 {
        self.read_reg(addr.base)
            .expect("validated base register must exist")
            .wrapping_add_signed(i64::from(addr.offset))
    }

    fn indexed_addr_value(
        &self,
        base: MachineReg,
        index: MachineReg,
        index_extend: MachineIndexExtend,
        offset: i32,
    ) -> Result<(u64, RegAddrKind), WasmError> {
        let base_value = self.read_reg(base)?;
        let index = self.read_reg(index)?;
        let index = match index_extend {
            MachineIndexExtend::None => index,
            MachineIndexExtend::ZeroExtend32 => u64::from(index as u32),
        };
        Ok((
            base_value
                .wrapping_add(index)
                .wrapping_add_signed(i64::from(offset)),
            self.reg_addr_kind(base),
        ))
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
        // Only guard-page targets reserve the large virtual window that lets
        // us distinguish wasm-memory addresses from unrelated host pointers.
        // Without guard pages, frame/stack pointers can legitimately sit near
        // the committed memory allocation, so we must check only the committed
        // range.
        #[cfg(all(target_pointer_width = "64", sf_has_guard_pages))]
        const GUARD_WINDOW: usize = 8 * 1024 * 1024 * 1024 + 64 * 1024;
        #[cfg(all(target_pointer_width = "64", sf_has_guard_pages))]
        let in_wasm_region = ptr >= mem_base && ptr < mem_base.saturating_add(GUARD_WINDOW);
        #[cfg(all(target_pointer_width = "64", not(sf_has_guard_pages)))]
        let in_wasm_region = ptr >= mem_base && ptr < mem_base + mem_size;
        #[cfg(target_pointer_width = "32")]
        let in_wasm_region = ptr >= mem_base && ptr < mem_base + mem_size;
        if in_wasm_region {
            if ptr.saturating_add(size) > mem_base + mem_size {
                return Err(WasmError::trap("out of bounds memory access"));
            }
        }
        Ok(())
    }

    fn check_mem0_access(&self, addr: u64, size: usize) -> Result<(), WasmError> {
        let mem_base = self.ctx.mem0_base as u64;
        let mem_size = self.ctx.mem0_size;
        if mem_base == 0 || mem_size == 0 {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        let end = addr
            .checked_add(size as u64)
            .ok_or_else(|| WasmError::trap("out of bounds memory access"))?;
        let mem_end = mem_base
            .checked_add(mem_size)
            .ok_or_else(|| WasmError::trap("out of bounds memory access"))?;
        if addr < mem_base || end > mem_end {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        Ok(())
    }

    fn load(
        &self,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<u64, WasmError> {
        self.load_at(
            self.addr_value(addr),
            self.reg_addr_kind(addr.base),
            width,
            extension,
        )
    }

    fn load_at(
        &self,
        addr_value: u64,
        base_kind: RegAddrKind,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<u64, WasmError> {
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
            return Err(WasmError::internal(
                "synthetic 32-bit load uses unmapped address 0x",
            ));
        }
        let ptr = addr_value as *const u8;
        if base_kind == RegAddrKind::Mem0 {
            self.check_mem0_access(addr_value, width.bytes() as usize)?;
        } else {
            self.check_access(ptr as usize, width.bytes() as usize)?;
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
        self.store_at(
            self.addr_value(addr),
            self.reg_addr_kind(addr.base),
            width,
            value,
        )
    }

    fn store_at(
        &self,
        addr_value: u64,
        base_kind: RegAddrKind,
        width: MachineMemWidth,
        value: u64,
    ) -> Result<(), WasmError> {
        if let Some(result) = self.address_space.store(self.ctx, addr_value, width, value) {
            return result;
        }
        if matches!(self.address_space, EmulatorAddressSpace::Target32(_)) {
            return Err(WasmError::internal(
                "synthetic 32-bit store uses unmapped address 0x",
            ));
        }
        let ptr = addr_value as *mut u8;
        if base_kind == RegAddrKind::Mem0 {
            self.check_mem0_access(addr_value, width.bytes() as usize)?;
        } else {
            self.check_access(ptr as usize, width.bytes() as usize)?;
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
}

#[inline]
fn init_entry_regs(
    _compiled: &CompiledNativeModule,
    reg_count: u16,
    ctx_ptr: u64,
    fp: u64,
    mem0_base: u64,
    mem0_size: u64,
) -> collections::Vec<u64> {
    let mut regs = collections::vec![0; reg_count as usize];
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

fn init_entry_addr_kinds(reg_count: u16) -> collections::Vec<RegAddrKind> {
    let mut kinds = collections::vec![RegAddrKind::Unknown; reg_count as usize];
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
        return Err(WasmError::exhaustion("stack overflow"));
    }
    Ok(())
}

fn trap_from_kind(kind: MachineTrapKind) -> WasmError {
    match kind {
        MachineTrapKind::Unreachable => WasmError::trap("unreachable executed"),
        MachineTrapKind::MemoryOutOfBounds => WasmError::trap("out of bounds memory access"),
        MachineTrapKind::TableOutOfBounds => WasmError::trap("out of bounds table access"),
        MachineTrapKind::InvalidFunctionReference => WasmError::trap("invalid function reference"),
        MachineTrapKind::IndirectCallTypeMismatch => WasmError::trap("indirect call type mismatch"),
        MachineTrapKind::IntegerDivideByZero => WasmError::trap("integer divide by zero"),
        MachineTrapKind::IntegerOverflow => WasmError::trap("integer overflow"),
        MachineTrapKind::InvalidConversion => WasmError::trap("invalid conversion to integer"),
        MachineTrapKind::StackOverflow => WasmError::exhaustion("stack overflow"),
        MachineTrapKind::HelperFailure => WasmError::trap("native helper failed"),
    }
}

fn eval_int_unary(
    width: MachineIntWidth,
    op: MachineIntUnaryOp,
    src: u64,
) -> Result<u64, WasmError> {
    Ok(match (width, op) {
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

fn apply_shift(width: MachineIntWidth, shift: MachineShiftOp, value: u64, amount: u8) -> u64 {
    match width {
        MachineIntWidth::I32 => {
            let v = value as u32;
            let r = match shift {
                MachineShiftOp::Lsl => v.wrapping_shl(amount as u32),
                MachineShiftOp::Lsr => v.wrapping_shr(amount as u32),
                MachineShiftOp::Asr => (v as i32).wrapping_shr(amount as u32) as u32,
                MachineShiftOp::Ror => v.rotate_right(amount as u32),
            };
            u64::from(r)
        }
        MachineIntWidth::I64 => match shift {
            MachineShiftOp::Lsl => value.wrapping_shl(amount as u32),
            MachineShiftOp::Lsr => value.wrapping_shr(amount as u32),
            MachineShiftOp::Asr => (value as i64).wrapping_shr(amount as u32) as u64,
            MachineShiftOp::Ror => value.rotate_right(amount as u32),
        },
    }
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

fn eval_i64_pair_binary(
    op: MachineIntBinaryOp,
    lhs_lo: u64,
    lhs_hi: u64,
    rhs_lo: u64,
    rhs_hi: u64,
) -> Result<(u64, u64), WasmError> {
    let lhs = u64::from(lhs_lo as u32) | (u64::from(lhs_hi as u32) << 32);
    let rhs = u64::from(rhs_lo as u32) | (u64::from(rhs_hi as u32) << 32);
    let value = match op {
        MachineIntBinaryOp::Add => lhs.wrapping_add(rhs),
        MachineIntBinaryOp::Sub => lhs.wrapping_sub(rhs),
        MachineIntBinaryOp::Mul => lhs.wrapping_mul(rhs),
        MachineIntBinaryOp::And => lhs & rhs,
        MachineIntBinaryOp::Or => lhs | rhs,
        MachineIntBinaryOp::Xor => lhs ^ rhs,
        _ => {
            return Err(WasmError::internal(
                "machine Int64PairBinary requires a supported i64 binary op".into(),
            ))
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
        MachineIntUnaryOp::Extend8S => (src as u8 as i8 as i64) as u64,
        MachineIntUnaryOp::Extend16S => (src as u16 as i16 as i64) as u64,
        MachineIntUnaryOp::Extend32S => (src as u32 as i32 as i64) as u64,
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

fn eval_i64_pair_compare(
    kind: MachineCompareKind,
    sign: MachineSign,
    lhs_lo: u64,
    lhs_hi: u64,
    rhs_lo: u64,
    rhs_hi: u64,
) -> u64 {
    let lhs = u64::from(lhs_lo as u32) | (u64::from(lhs_hi as u32) << 32);
    let rhs = u64::from(rhs_lo as u32) | (u64::from(rhs_hi as u32) << 32);
    let result = match sign {
        MachineSign::Signed => compare_i64(kind, lhs as i64, rhs as i64),
        MachineSign::Unsigned => compare_u64(kind, lhs, rhs),
    };
    u64::from(result as u32)
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
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -2147483649.0 || value >= 2147483648.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -1.0 || value >= 4294967296.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -2147483649.0 || value >= 2147483648.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -1.0 || value >= 4294967296.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow"));
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
    use tracked_alloc::{boxed::Box, rc::Rc, string::String};

    use super::{eval, Emulator, EmulatorAddressSpace, RegAddrKind};
    use crate::collections;
    use crate::{
        error::WasmError,
        module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType},
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            entities::{Caller, FunctionInst, MemInst, ModuleInst, TableInst},
            jit::arch::{backend_mode_test_lock, NativeBackend},
            jit::build::ensure_module_compiled,
            jit::machine::machine_ir::{
                MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
                MachineCompareKind, MachineEdge, MachineFuncId, MachineFunction,
                MachineIndexExtend, MachineInst, MachineInstKind, MachineIntWidth, MachineMemWidth,
                MachineModule, MachineModuleAbi, MachineProgram, MachineReg, MachineRegOwner,
                MachineSign, MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
                MACHINE_FIXED_REG_COUNT, MACHINE_MEM0_BASE_REG,
            },
            jit::runtime::{self, code::CompiledNativeModule},
            store::Store,
            value::{RefHandle, Value},
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
        #[cfg(sf_has_guard_pages)]
        {
            MemInst::new_guarded(limits).expect("guarded memory")
        }
        #[cfg(not(sf_has_guard_pages))]
        {
            MemInst::new(limits).expect("test memory within runtime limits")
        }
    }

    struct CompiledBackendGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    fn enable_compiled_backend() -> CompiledBackendGuard {
        let lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        CompiledBackendGuard { _lock: lock }
    }

    const fn compiled_native_backend() -> NativeBackend {
        #[cfg(sf_backend_emu64)]
        {
            return NativeBackend::Emu64;
        }

        #[cfg(sf_backend_emu32)]
        {
            return NativeBackend::Emu32;
        }

        unreachable!("emulator tests require an emulator backend build")
    }

    const fn compiled_backend_config() -> crate::vm::jit::backend::BackendConfig {
        super::config::compile_backend_config()
    }

    #[test]
    fn evaluates_native_machine_module_with_helper_runtime_call() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::Host {
            func_type: Rc::clone(&ty),
            callback: crate::vm::entities::HostCallback::new(host_double),
        });
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_locals(collections::vec![]);
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
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
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

    #[cfg(sf_backend_emu64)]
    #[test]
    fn runtime_eval_emu64_call_indirect_accepts_block_result_as_first_argument() {
        let _guard = enable_compiled_backend();

        let check_ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let caller_ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&check_ty),
            Rc::clone(&caller_ty)
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        // Callee: `(param i32 i32) -> i32 { local.get 0 }`
        let mut callee_spec = FunctionSpec::new(Rc::clone(&check_ty), 0);
        callee_spec.set_code((&[0x20, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: callee_spec,
            type_index: 0,
        });

        // Caller:
        //   (block (result i32)
        //     (call_indirect (type 0)
        //       (block (result i32) (i32.const 1))
        //       (i32.const 2)
        //       (i32.const 0)))
        let mut caller_spec = FunctionSpec::new(Rc::clone(&caller_ty), 1);
        caller_spec.set_code(
            (&[
                0x02, 0x7f, // block (result i32)
                0x02, 0x7f, //   block (result i32)
                0x41, 0x01, //     i32.const 1
                0x0b, //   end
                0x41, 0x02, //   i32.const 2
                0x41, 0x00, //   i32.const 0
                0x11, 0x00, 0x00, //   call_indirect (type 0) (table 0)
                0x0b, // end
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec: caller_spec,
            type_index: 1,
        });

        let mut table = TableInst::new(Limits::new(1, Some(1)).unwrap(), ValueType::funcref());
        table.elements[0] = RefHandle::new(0);
        module.tables.push(table);

        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[])
            .expect("call_indirect with block-produced first arg should run");
        assert_eq!(results.peek_at_index(0), Value::I32(1).to_raw());
    }

    #[cfg(sf_backend_emu64)]
    #[test]
    fn runtime_eval_emu64_simple_block_result_survives_with_memory_present() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);

        // (func (result i32)
        //   (block (nop))
        //   (block (result i32) (i32.const 7)))
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code(
            (&[
                0x02, 0x40, // block
                0x01, //   nop
                0x0b, // end
                0x02, 0x7f, // block (result i32)
                0x41, 0x07, //   i32.const 7
                0x0b, // end
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
            .push(native_test_memory(Limits::new(1, Some(1)).unwrap()));

        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results =
            runtime::eval(func_ref, &mut store, &[]).expect("block result should not trap");
        assert_eq!(results.peek_at_index(0), Value::I32(7).to_raw());
    }
    #[cfg(sf_backend_emu64)]
    #[test]
    fn runtime_eval_emu64_call_indirect_accepts_simple_local_target() {
        let _guard = enable_compiled_backend();

        let check_ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let caller_ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&check_ty),
            Rc::clone(&caller_ty)
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut callee_spec = FunctionSpec::new(Rc::clone(&check_ty), 0);
        callee_spec.set_code((&[0x20, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: callee_spec,
            type_index: 0,
        });

        let mut caller_spec = FunctionSpec::new(Rc::clone(&caller_ty), 1);
        caller_spec.set_code(
            (&[
                0x41, 0x01, // i32.const 1
                0x41, 0x02, // i32.const 2
                0x41, 0x00, // i32.const 0
                0x11, 0x00, 0x00, // call_indirect (type 0) (table 0)
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec: caller_spec,
            type_index: 1,
        });

        let mut table = TableInst::new(Limits::new(1, Some(1)).unwrap(), ValueType::funcref());
        table.elements[0] = RefHandle::new(0);
        module.tables.push(table);

        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[])
            .expect("simple local call_indirect should run");
        assert_eq!(results.peek_at_index(0), Value::I32(1).to_raw());
    }

    #[test]
    fn runtime_eval_preserves_first_local_call_result_across_second_local_call() {
        let _guard = enable_compiled_backend();

        let malloc_ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let diff_ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&malloc_ty),
            Rc::clone(&diff_ty)
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut malloc_spec = FunctionSpec::new(Rc::clone(&malloc_ty), 0);
        malloc_spec.set_code((&[0x41, 0x10, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: malloc_spec,
            type_index: 0,
        });

        let mut diff_spec = FunctionSpec::new(Rc::clone(&diff_ty), 1);
        diff_spec.set_locals(collections::vec![ValueType::I32, ValueType::I32]);
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

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_preserves_first_local_call_result_across_second_local_call() {
        let _guard = enable_compiled_backend();

        let malloc_ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let diff_ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&malloc_ty),
            Rc::clone(&diff_ty)
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut malloc_spec = FunctionSpec::new(Rc::clone(&malloc_ty), 0);
        malloc_spec.set_code((&[0x41, 0x10, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: malloc_spec,
            type_index: 0,
        });

        let mut diff_spec = FunctionSpec::new(Rc::clone(&diff_ty), 1);
        diff_spec.set_locals(collections::vec![ValueType::I32, ValueType::I32]);
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

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_recursive_i32_fib_matches_expected_result() {
        let _guard = enable_compiled_backend();

        let fib_ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&fib_ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut fib_spec = FunctionSpec::new(Rc::clone(&fib_ty), 0);
        fib_spec.set_code(
            (&[
                0x20, 0x00, // local.get 0
                0x41, 0x01, // i32.const 1
                0x4c, // i32.le_s
                0x04, 0x7f, // if (result i32)
                0x20, 0x00, //   local.get 0
                0x05, // else
                0x20, 0x00, //   local.get 0
                0x41, 0x02, //   i32.const 2
                0x6b, //   i32.sub
                0x10, 0x00, //   call 0
                0x20, 0x00, //   local.get 0
                0x41, 0x01, //   i32.const 1
                0x6b, //   i32.sub
                0x10, 0x00, //   call 0
                0x6a, //   i32.add
                0x0b, // end if
                0x0b, // end func
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec: fib_spec,
            type_index: 0,
        });

        let mut store = Store::new(module);
        ensure_module_compiled(&store).expect("native compile");
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[Value::I32(10)]).expect("runtime eval");
        assert_eq!(results.peek_at_index(0), Value::I32(55).to_raw());
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_preserves_positive_i64_across_identity_call() {
        let _guard = enable_compiled_backend();

        let id_ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I64],
            collections::vec![ValueType::I64],
        ));
        let caller_ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I64],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&id_ty), Rc::clone(&caller_ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut id_spec = FunctionSpec::new(Rc::clone(&id_ty), 0);
        id_spec.set_code((&[0x20, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: id_spec,
            type_index: 0,
        });

        let mut caller_spec = FunctionSpec::new(Rc::clone(&caller_ty), 1);
        caller_spec.set_code(
            (&[
                0x42, 0x26, // i64.const 38
                0x10, 0x00, // call 0
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec: caller_spec,
            type_index: 1,
        });

        let mut store = Store::new(module);
        ensure_module_compiled(&store).expect("native compile");
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[]).expect("runtime eval");
        assert_eq!(results.peek_at_index(0), Value::I64(38).to_raw());
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_preserves_mixed_i32_i64_i32_call_arguments() {
        let _guard = enable_compiled_backend();

        let callee_ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I64, ValueType::I32],
            collections::vec![ValueType::I64],
        ));
        let caller_ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I64],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&callee_ty),
            Rc::clone(&caller_ty)
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut callee_spec = FunctionSpec::new(Rc::clone(&callee_ty), 0);
        callee_spec.set_code((&[0x20, 0x01, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: callee_spec,
            type_index: 0,
        });

        let mut caller_spec = FunctionSpec::new(Rc::clone(&caller_ty), 1);
        caller_spec.set_code(
            (&[
                0x41, 0x01, // i32.const 1
                0x42, 0x26, // i64.const 38
                0x41, 0x02, // i32.const 2
                0x10, 0x00, // call 0
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec: caller_spec,
            type_index: 1,
        });

        let mut store = Store::new(module);
        ensure_module_compiled(&store).expect("native compile");
        let func_ptr = &store.module().functions[1] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let results = runtime::eval(func_ref, &mut store, &[]).expect("runtime eval");
        assert_eq!(results.peek_at_index(0), Value::I64(38).to_raw());
    }

    #[test]
    fn jump_to_edge_preserves_mem0_provenance_for_block_params() {
        let compiled = Rc::new(
            CompiledNativeModule::new(
                compiled_native_backend(),
                compiled_backend_config(),
                MachineModule {
                    config: compiled_backend_config(),
                    functions: collections::vec![MachineFunction {
                        id: MachineFuncId(0),
                        program: MachineProgram {
                            entry: MachineBlockId(0),
                            fp_reg_init_widths: collections::Vec::new(),
                            blocks: collections::vec![
                                MachineBlock {
                                    id: MachineBlockId(0),
                                    params: collections::Vec::new(),
                                    ops: collections::Vec::new(),
                                    terminator: MachineTerminator::Jump(MachineEdge {
                                        target: MachineBlockId(1),
                                        args: collections::vec![MachineValue::Reg(
                                            MACHINE_MEM0_BASE_REG
                                        )],
                                    }),
                                },
                                MachineBlock {
                                    id: MachineBlockId(1),
                                    params: collections::vec![MachineBlockParam::gp_word(
                                        MachineReg(MACHINE_FIXED_REG_COUNT,)
                                    )],
                                    ops: collections::Vec::new(),
                                    terminator: MachineTerminator::Return,
                                },
                            ],
                        },
                        preserved_clobbers: collections::Vec::new(),
                    }],
                    consts: collections::Vec::new(),
                },
                MachineModuleAbi::default(),
            )
            .expect("compiled machine module"),
        );

        let mut store = Store::new(ModuleInst::new(
            String::from("m"),
            TypeContext::new(collections::vec![]),
        ));
        let n_globals = store.module().globals.len();
        let mut ctx = crate::vm::jit::runtime::context::NativeContext::new(
            (&mut store) as *mut Store,
            core::ptr::null_mut(),
            n_globals,
        );
        let mut emulator = Emulator {
            ctx: &mut *ctx,
            compiled: &compiled,
            root_frame: core::ptr::null_mut(),
            func_id: MachineFuncId(0),
            block_id: MachineBlockId(0),
            fp: core::ptr::null_mut(),
            regs: super::init_entry_regs(
                &compiled,
                MACHINE_FIXED_REG_COUNT + 1,
                0,
                0,
                0x4000_0000,
                64,
            ),
            addr_kinds: super::init_entry_addr_kinds(MACHINE_FIXED_REG_COUNT + 1),
            call_stack: collections::Vec::new(),
            address_space: EmulatorAddressSpace::Host,
        };

        emulator
            .jump_to_edge(&MachineEdge {
                target: MachineBlockId(1),
                args: collections::vec![MachineValue::Reg(MACHINE_MEM0_BASE_REG)],
            })
            .expect("jump to edge");

        assert_eq!(
            emulator.reg_addr_kind(MachineReg(MACHINE_FIXED_REG_COUNT)),
            RegAddrKind::Mem0
        );
    }

    #[test]
    fn indexed_addr_zero_extends_i32_index_and_preserves_mem0_base() {
        let compiled = Rc::new(
            CompiledNativeModule::new(
                compiled_native_backend(),
                compiled_backend_config(),
                MachineModule {
                    config: compiled_backend_config(),
                    functions: collections::vec![MachineFunction {
                        id: MachineFuncId(0),
                        program: MachineProgram {
                            entry: MachineBlockId(0),
                            fp_reg_init_widths: collections::Vec::new(),
                            blocks: collections::vec![MachineBlock {
                                id: MachineBlockId(0),
                                params: collections::Vec::new(),
                                ops: collections::Vec::new(),
                                terminator: MachineTerminator::Return,
                            }],
                        },
                        preserved_clobbers: collections::Vec::new(),
                    }],
                    consts: collections::Vec::new(),
                },
                MachineModuleAbi::default(),
            )
            .expect("compiled machine module"),
        );

        let mut store = Store::new(ModuleInst::new(
            String::from("m"),
            TypeContext::new(collections::vec![]),
        ));
        let n_globals = store.module().globals.len();
        let mut ctx = crate::vm::jit::runtime::context::NativeContext::new(
            (&mut store) as *mut Store,
            core::ptr::null_mut(),
            n_globals,
        );
        let mut emulator = Emulator {
            ctx: &mut *ctx,
            compiled: &compiled,
            root_frame: core::ptr::null_mut(),
            func_id: MachineFuncId(0),
            block_id: MachineBlockId(0),
            fp: core::ptr::null_mut(),
            regs: super::init_entry_regs(
                &compiled,
                MACHINE_FIXED_REG_COUNT + 2,
                0,
                0,
                0x4000_0000,
                64,
            ),
            addr_kinds: super::init_entry_addr_kinds(MACHINE_FIXED_REG_COUNT + 2),
            call_stack: collections::Vec::new(),
            address_space: EmulatorAddressSpace::Host,
        };
        emulator
            .write_reg(MachineReg(MACHINE_FIXED_REG_COUNT), u64::MAX)
            .expect("index reg");

        let (addr_value, kind) = emulator
            .indexed_addr_value(
                MACHINE_MEM0_BASE_REG,
                MachineReg(MACHINE_FIXED_REG_COUNT),
                MachineIndexExtend::ZeroExtend32,
                4,
            )
            .expect("indexed addr");
        let base = emulator.read_reg(MACHINE_MEM0_BASE_REG).expect("mem0 base");
        assert_eq!(
            addr_value,
            base.wrapping_add(u64::from(u32::MAX)).wrapping_add(4)
        );
        assert_eq!(kind, RegAddrKind::Mem0);
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn compiled_emu32_rejects_unfinalized_gpi64_machine_ir() {
        let backend = compiled_backend_config();
        let err = CompiledNativeModule::new(
            compiled_native_backend(),
            backend,
            MachineModule {
                config: backend,
                functions: collections::vec![MachineFunction {
                    id: MachineFuncId(0),
                    program: MachineProgram {
                        entry: MachineBlockId(0),
                        fp_reg_init_widths: collections::Vec::new(),
                        blocks: collections::vec![MachineBlock {
                            id: MachineBlockId(0),
                            params: collections::Vec::new(),
                            ops: collections::vec![MachineInst {
                                kind: MachineInstKind::Move {
                                    owner: MachineRegOwner::LinearValue,
                                    ty: MachineStorageType::GpI64,
                                    dst: MachineReg(MACHINE_FIXED_REG_COUNT),
                                    src: MachineValue::Imm64(7),
                                },
                            }],
                            terminator: MachineTerminator::Return,
                        }],
                    },
                    preserved_clobbers: collections::Vec::new(),
                }],
                consts: collections::Vec::new(),
            },
            MachineModuleAbi::default(),
        )
        .expect_err("emu32 should reject scalar gpi64 IR on a 32-bit GP target");

        assert!(err
            .message()
            .contains("not valid 32-bit GP-target MachineIR"));
        assert!(err.message().contains("GpI64"));
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn compiled_emu32_rejects_wrong_gp_fp_boundary() {
        let backend = compiled_backend_config();
        // Create a module config with a mismatched GP/FP boundary by reducing
        // the GP dynamic budget by 1, so module.config.first_fp_reg() differs
        // from backend.first_fp_reg().
        let mut wrong_config = backend;
        wrong_config.gp_dynamic_budget = wrong_config.gp_dynamic_budget.saturating_sub(1);
        let err = CompiledNativeModule::new(
            compiled_native_backend(),
            backend,
            MachineModule {
                config: wrong_config,
                functions: collections::vec![MachineFunction {
                    id: MachineFuncId(0),
                    program: MachineProgram {
                        entry: MachineBlockId(0),
                        fp_reg_init_widths: collections::Vec::new(),
                        blocks: collections::vec![MachineBlock {
                            id: MachineBlockId(0),
                            params: collections::Vec::new(),
                            ops: collections::vec![MachineInst {
                                kind: MachineInstKind::Store {
                                    ty: MachineStorageType::GpWord,
                                    addr: MachineAddr {
                                        base: MachineReg(1),
                                        offset: 0,
                                    },
                                    width: MachineMemWidth::U32,
                                    src: MachineValue::Imm64(0),
                                },
                            }],
                            terminator: MachineTerminator::Return,
                        }],
                    },
                    preserved_clobbers: collections::Vec::new(),
                }],
                consts: collections::Vec::new(),
            },
            MachineModuleAbi::default(),
        )
        .expect_err("emu32 should reject machine IR with a wrong 32-bit GP/FP bank boundary");

        assert!(err.message().contains("mismatched first_fp_reg boundary"));
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_rejects_more_than_thirty_two_memories() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x41, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        for _ in 0..33 {
            module
                .memories
                .push(native_test_memory(Limits::new(0, Some(1)).unwrap()));
        }
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let error = runtime::eval(func_ref, &mut store, &[])
            .expect_err("emu32 should reject synthetic address-space memory overlap");
        assert!(error
            .message()
            .contains("emu32 synthetic address space supports at most 32 memories"));
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_rejects_more_than_sixteen_tables() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x41, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        for _ in 0..17 {
            module.tables.push(TableInst::new(
                Limits::new(0, Some(1)).unwrap(),
                ValueType::funcref(),
            ));
        }
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let error = runtime::eval(func_ref, &mut store, &[])
            .expect_err("emu32 should reject synthetic address-space table overlap");
        assert!(error
            .message()
            .contains("emu32 synthetic address space supports at most 16 tables"));
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_traps_on_wrapping_memory_address() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x2d, 0x00, 0x01, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut mem = MemInst::new(Limits::new(1, Some(1)).unwrap())
            .expect("test memory within runtime limits");
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

    #[cfg(sf_backend_emu32)]
    #[test]
    fn compiled_emu32_keeps_access_wrap_trap_for_max_offset_memory_load() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x2d, 0x00, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        module.memories.push(
            MemInst::new(Limits::new(1, Some(1)).unwrap())
                .expect("test memory within runtime limits"),
        );
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

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_traps_on_max_offset_memory_address() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x2d, 0x00, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut mem = MemInst::new(Limits::new(1, Some(1)).unwrap())
            .expect("test memory within runtime limits");
        mem.data[..26].copy_from_slice(b"abcdefghijklmnopqrstuvwxyz");
        module.memories.push(mem);
        let mut store = Store::new(module);
        let func_ptr = &store.module().functions[0] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let error = runtime::eval(func_ref, &mut store, &[Value::I32(0)])
            .expect_err("max-offset memory access should trap");
        assert_eq!(error.message(), "out of bounds memory access");
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn runtime_eval_emu32_loads_i64_without_clobbering_address_base() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I64],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_code((&[0x20, 0x00, 0x29, 0x00, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let mut mem = MemInst::new(Limits::new(1, Some(1)).unwrap())
            .expect("test memory within runtime limits");
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

    #[cfg(sf_backend_emu64)]
    #[test]
    fn runtime_eval_emu64_refreshes_mem0_regs_after_zero_page_memory_grow() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
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

    #[cfg(sf_backend_emu64)]
    #[test]
    fn runtime_eval_emu64_traps_on_zero_page_memory_access() {
        let _guard = enable_compiled_backend();

        let ty = Rc::new(FunctionType::new(
            collections::vec![],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
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
