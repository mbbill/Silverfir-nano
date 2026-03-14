use alloc::{vec, vec::Vec};

use crate::{
    constants::MAX_CALL_STACK_DEPTH,
    error::WasmError,
    vm::{
        entities::ModuleInst,
        native::{
            code::{Arm64CodePtr, Arm64RootEntry, CompiledNativeModule},
            ir::{
                machine::{
                    MachineAddr, MachineBlock, MachineBlockId, MachineBranchCond,
                    MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp,
                    MachineFloatUnaryOp, MachineFloatWidth,
                    MachineFunction, MachineInst, MachineInstKind, MachineIntBinaryOp,
                    MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
                    MachineReg, MachineSign, MachineTerminator, MachineTrapKind, MachineValue,
                    MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
                    MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
                },
                runtime::MachineHelperSymbol,
            },
            runtime::context::ctx_offset,
            runtime::helpers::resolve_helper_entry,
        },
    },
};

use super::{
    arm64_raise_trap, arm64_raise_unsupported,
    emit::Arm64TextEmitter,
    enc::{self, Cond},
    reg::Arm64Reg,
};

const SCRATCH0: Arm64Reg = Arm64Reg::X16;
const SCRATCH1: Arm64Reg = Arm64Reg::X17;
/// FP scratch registers (caller-saved, not used for parameter passing in our ABI)
const FP_SCRATCH0: u32 = 0; // D0/S0
const FP_SCRATCH1: u32 = 1; // D1/S1
const FP_SCRATCH2: u32 = 2; // D2/S2
const CALLEE_SAVED_FRAME_SIZE: u32 = 96;
const ARM64_FUNCTION_INFO_SIZE: usize = 32;
const MAX_ARM64_CALL_DEPTH: u64 = if MAX_CALL_STACK_DEPTH < 300 {
    MAX_CALL_STACK_DEPTH as u64
} else {
    300
};
const DYNAMIC_REGS: [Arm64Reg; 13] = [
    Arm64Reg::X9,
    Arm64Reg::X10,
    Arm64Reg::X11,
    Arm64Reg::X12,
    Arm64Reg::X13,
    Arm64Reg::X14,
    Arm64Reg::X15,
    Arm64Reg::X23,
    Arm64Reg::X24,
    Arm64Reg::X25,
    Arm64Reg::X26,
    Arm64Reg::X27,
    Arm64Reg::X28,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LabelKind {
    Block,
    Edge,
    StackOverflow,
    CallDepthExhausted,
    ReturnOk,
    ReturnError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BranchFixupKind {
    B,
    BCond(Cond),
    Cbz(Arm64Reg),
    Cbnz(Arm64Reg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BranchFixup {
    inst_offset: usize,
    label: usize,
    kind: BranchFixupKind,
}

#[derive(Clone, Debug)]
struct EdgeStub {
    label: usize,
    target: MachineBlockId,
    params: Vec<MachineReg>,
    args: Vec<MachineValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LocalPtrPatch {
    literal_offset: usize,
    target_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingLocalPtrPatch {
    literal_offset: usize,
    target_label: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectCallPatch {
    literal_offset: usize,
    callee: crate::vm::native::ir::machine::MachineFuncId,
}

/// One debug region within a compiled function (for profiler symbols).
#[derive(Clone, Debug)]
pub struct DebugRegion {
    /// Byte offset within the function text.
    pub offset: usize,
    /// Byte length of this region.
    pub len: usize,
    /// Human-readable label (e.g. "b0", "edge_3", "prologue", "return_ok").
    pub label: alloc::string::String,
}

#[derive(Debug)]
struct FunctionArtifact {
    text: Arm64TextEmitter,
    local_ptr_patches: Vec<LocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    root_return_offset: usize,
    /// Offset of the internal entry point (after prologue), for local calls.
    internal_entry_offset: usize,
    /// Per-block/region debug map for profiler symbols.
    debug_regions: Vec<DebugRegion>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Arm64FunctionInfo {
    entry: u64,
    total_frame_bytes: u64,
    frame_prefix_slots: u64,
    call_scratch_base_slot: u64,
}

#[derive(Clone, Debug)]
pub struct CompiledArm64Entry {
    pub entry: Arm64RootEntry,
    pub text_len: usize,
    pub debug_regions: Vec<DebugRegion>,
    pub root_return: Arm64CodePtr,
}

#[derive(Debug)]
struct FunctionCompiler<'a> {
    compiled: &'a CompiledNativeModule,
    function: &'a MachineFunction,
    text: Arm64TextEmitter,
    labels: Vec<Option<usize>>,
    fixups: Vec<BranchFixup>,
    block_labels: Vec<usize>,
    edge_stubs: Vec<EdgeStub>,
    resolved_ptr_patches: Vec<LocalPtrPatch>,
    local_ptr_patches: Vec<PendingLocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    deferred_traps: Vec<(usize, MachineTrapKind)>,
    stack_overflow_label: usize,
    call_depth_label: usize,
    return_ok_label: usize,
    return_error_label: usize,
}

pub fn compile_module(
    module: &ModuleInst,
    compiled: &CompiledNativeModule,
) -> Result<Vec<Option<CompiledArm64Entry>>, WasmError> {
    let mut artifacts = Vec::with_capacity(compiled.module().functions.len());
    for function in &compiled.module().functions {
        match compile_function(compiled, function) {
            Ok(artifact) => artifacts.push(artifact),
            // No unsupported stubs - all functions must compile successfully.
            // Any unsupported operation should be implemented in the backend.
            Err(err) => return Err(err),
        }
    }

    let mut base_offsets = Vec::with_capacity(artifacts.len());
    let mut running_offset = 0usize;
    for artifact in &artifacts {
        base_offsets.push(running_offset);
        running_offset = running_offset.saturating_add(artifact.text.len());
    }
    let function_info_table_offset = running_offset;

    let mut entry_addrs = Vec::with_capacity(artifacts.len());
    let mut internal_entry_addrs = Vec::with_capacity(artifacts.len());
    let base_ptr = {
        let executable = module
            .native_code_buffer()
            .map_err(|err| WasmError::internal(err.into()))?;
        executable.as_ptr()
    };
    for (i, base_offset) in base_offsets.iter().enumerate() {
        entry_addrs.push(unsafe { base_ptr.add(*base_offset) } as usize);
        internal_entry_addrs
            .push(unsafe { base_ptr.add(*base_offset + artifacts[i].internal_entry_offset) }
                as usize);
    }

    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let function_base = base_offsets[index];
        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as u64;
            artifact.text.patch_u64(patch.literal_offset, target_addr);
        }
        for patch in &artifact.direct_call_patches {
            // Direct local calls use internal entry (skip prologue)
            let callee_addr =
                *internal_entry_addrs.get(patch.callee.0 as usize).ok_or_else(|| {
                    WasmError::internal("arm64 direct callee address is out of range".into())
                })? as u64;
            artifact.text.patch_u64(patch.literal_offset, callee_addr);
        }
        for &literal_offset in &artifact.function_table_patches {
            artifact.text.patch_u64(literal_offset, unsafe {
                base_ptr.add(function_info_table_offset)
            } as u64);
        }
    }

    let mut function_info_bytes = Vec::with_capacity(artifacts.len() * ARM64_FUNCTION_INFO_SIZE);
    for (func_idx, runtime) in compiled.runtime().functions.iter().enumerate() {
        let info = Arm64FunctionInfo {
            // Function info table uses internal entry (for indirect calls)
            entry: *internal_entry_addrs
                .get(func_idx)
                .ok_or_else(|| WasmError::internal("arm64 function entry is out of range".into()))?
                as u64,
            total_frame_bytes: u64::from(runtime.total_frame_slots) * 8,
            frame_prefix_slots: u64::from(runtime.frame_prefix_slots),
            call_scratch_base_slot: u64::from(
                runtime
                    .call_scratch
                    .map(|region| region.base_slot)
                    .unwrap_or(0),
            ),
        };
        function_info_bytes.extend_from_slice(&info.entry.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.call_scratch_base_slot.to_le_bytes());
    }

    let mut executable = module
        .native_code_buffer()
        .map_err(|err| WasmError::internal(err.into()))?;
    executable.begin_write();
    executable.reset();

    let written_start = executable.len();
    let mut entries = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        let text_bytes = artifact.text.finish();
        let text_len = text_bytes.len();
        let debug_regions = artifact.debug_regions;
        let offset = executable.emit_bytes(&text_bytes);
        let entry = unsafe { executable.fn_ptr::<Arm64RootEntry>(offset) };
        let root_return = unsafe { executable.ptr(offset + artifact.root_return_offset) };
        entries.push(Some(CompiledArm64Entry { entry, root_return, text_len, debug_regions }));
    }
    executable.emit_bytes(&function_info_bytes);
    let written_len = executable.len().saturating_sub(written_start);
    executable.finish_write(written_start, written_len);

    // Record per-block JIT symbols for profiling tools (samply-for-ai, perf)
    let module_name = &module.name;
    for (func_idx, entry) in entries.iter().enumerate() {
        if let Some(entry) = entry {
            let func_base = entry.entry as *const u8;
            for region in &entry.debug_regions {
                if region.len > 0 {
                    let region_start = unsafe { func_base.add(region.offset) };
                    let code_bytes =
                        unsafe { core::slice::from_raw_parts(region_start, region.len) };
                    let symbol = alloc::format!(
                        "jit::{}::func{}::{}",
                        module_name, func_idx, region.label
                    );
                    crate::vm::native::profiler::record_function(
                        region_start,
                        code_bytes,
                        &symbol,
                    );
                }
            }
        }
    }

    Ok(entries)
}

fn compile_function(
    compiled: &CompiledNativeModule,
    function: &MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    let max_reg = MACHINE_FIXED_REG_COUNT as usize + DYNAMIC_REGS.len();
    if function.program.reg_count as usize > max_reg {
        return Err(WasmError::invalid(alloc::format!(
            "arm64 MachineIR backend supports at most {} machine regs, got {} in function {}",
            max_reg,
            function.program.reg_count,
            function.id.0
        )));
    }
    if function
        .program
        .blocks
        .iter()
        .find(|block| block.id == function.program.entry)
        .map(|block| !block.params.is_empty())
        .unwrap_or(false)
    {
        return Err(WasmError::invalid(
            "arm64 MachineIR backend does not support entry block params yet".into(),
        ));
    }

    let mut compiler = FunctionCompiler::new(compiled, function);
    let mut debug_regions = Vec::new();

    let prologue_start = compiler.text.len();
    compiler.emit_prologue();
    let internal_entry_offset = compiler.text.len();
    debug_regions.push(DebugRegion {
        offset: prologue_start,
        len: internal_entry_offset - prologue_start,
        label: alloc::format!("prologue"),
    });

    for block in &function.program.blocks {
        let label = compiler.block_label(block.id)?;
        compiler.bind_label(label);
        let block_start = compiler.text.len();
        compiler.emit_block(block)?;
        let block_end = compiler.text.len();
        debug_regions.push(DebugRegion {
            offset: block_start,
            len: block_end - block_start,
            label: alloc::format!("b{}", block.id.0),
        });
    }

    let edge_start = compiler.text.len();
    for edge in compiler.edge_stubs.clone() {
        compiler.bind_label(edge.label);
        compiler.emit_parallel_moves(&edge.params, &edge.args)?;
        compiler.emit_branch_to_block(edge.target)?;
    }
    let edge_end = compiler.text.len();
    if edge_end > edge_start {
        debug_regions.push(DebugRegion {
            offset: edge_start,
            len: edge_end - edge_start,
            label: alloc::format!("edges"),
        });
    }

    let tail_start = compiler.text.len();
    compiler.bind_label(compiler.return_ok_label);
    compiler.materialize_u64(Arm64Reg::X0, 0);
    compiler.emit_epilogue();

    compiler.bind_label(compiler.stack_overflow_label);
    compiler.emit_trap(MachineTrapKind::StackOverflow);

    compiler.bind_label(compiler.call_depth_label);
    compiler.emit_trap(MachineTrapKind::CallStackExhausted);

    compiler.bind_label(compiler.return_error_label);
    compiler.emit_epilogue();

    // Emit deferred trap stubs
    let deferred = core::mem::take(&mut compiler.deferred_traps);
    for (label, kind) in deferred {
        compiler.bind_label(label);
        compiler.emit_trap(kind);
    }
    let tail_end = compiler.text.len();
    if tail_end > tail_start {
        debug_regions.push(DebugRegion {
            offset: tail_start,
            len: tail_end - tail_start,
            label: alloc::format!("tail"),
        });
    }

    compiler.patch_fixups()?;
    let root_return_offset = compiler
        .labels
        .get(compiler.return_ok_label)
        .and_then(|offset| *offset)
        .ok_or_else(|| WasmError::internal("arm64 root return label is unresolved".into()))?;
    let mut local_ptr_patches = compiler.resolved_ptr_patches;
    local_ptr_patches.reserve(compiler.local_ptr_patches.len());
    for patch in compiler.local_ptr_patches {
        let target_offset = compiler
            .labels
            .get(patch.target_label)
            .and_then(|offset| *offset)
            .ok_or_else(|| {
                WasmError::internal("arm64 local continuation label is unresolved".into())
            })?;
        local_ptr_patches.push(LocalPtrPatch {
            literal_offset: patch.literal_offset,
            target_offset,
        });
    }
    Ok(FunctionArtifact {
        text: compiler.text,
        local_ptr_patches,
        direct_call_patches: compiler.direct_call_patches,
        function_table_patches: compiler.function_table_patches,
        root_return_offset,
        internal_entry_offset,
        debug_regions,
    })
}

fn compile_unsupported_stub(
    func_id: crate::vm::native::ir::machine::MachineFuncId,
) -> FunctionArtifact {
    let mut text = Arm64TextEmitter::new();
    emit_shared_prologue(&mut text);
    text.emit_u32(enc::mov_reg_64(
        Arm64Reg::X0,
        map_fixed_reg(MACHINE_CTX_REG),
    ));
    text.emit_u32(enc::movz_64(Arm64Reg::X1, func_id.0 as u16, 0));
    if (func_id.0 >> 16) != 0 {
        text.emit_u32(enc::movk_64(
            Arm64Reg::X1,
            ((func_id.0 >> 16) & 0xffff) as u16,
            16,
        ));
    }
    materialize_u64_into(&mut text, SCRATCH0, arm64_raise_unsupported as usize as u64);
    text.emit_u32(enc::blr(SCRATCH0));
    emit_shared_epilogue(&mut text);
    FunctionArtifact {
        text,
        local_ptr_patches: Vec::new(),
        direct_call_patches: Vec::new(),
        function_table_patches: Vec::new(),
        root_return_offset: 0,
        internal_entry_offset: 0, // stub uses root entry
        debug_regions: Vec::new(),
    }
}

impl<'a> FunctionCompiler<'a> {
    fn new(compiled: &'a CompiledNativeModule, function: &'a MachineFunction) -> Self {
        let block_cap = function
            .program
            .blocks
            .iter()
            .map(|block| block.id.0 as usize)
            .max()
            .unwrap_or(0)
            + 1;
        let mut labels = Vec::new();
        let mut block_labels = vec![usize::MAX; block_cap];
        for block in &function.program.blocks {
            let label = labels.len();
            labels.push(None);
            block_labels[block.id.0 as usize] = label;
        }
        let stack_overflow_label = labels.len();
        labels.push(None);
        let call_depth_label = labels.len();
        labels.push(None);
        let return_ok_label = labels.len();
        labels.push(None);
        let return_error_label = labels.len();
        labels.push(None);
        Self {
            compiled,
            function,
            text: Arm64TextEmitter::new(),
            labels,
            fixups: Vec::new(),
            block_labels,
            edge_stubs: Vec::new(),
            resolved_ptr_patches: Vec::new(),
            local_ptr_patches: Vec::new(),
            direct_call_patches: Vec::new(),
            function_table_patches: Vec::new(),
            deferred_traps: Vec::new(),
            stack_overflow_label,
            call_depth_label,
            return_ok_label,
            return_error_label,
        }
    }

    fn emit_prologue(&mut self) {
        emit_shared_prologue(&mut self.text);
        self.text.emit_u32(enc::mov_reg_64(
            map_fixed_reg(MACHINE_CTX_REG),
            Arm64Reg::X0,
        ));
        self.text
            .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), Arm64Reg::X1));
        self.text.emit_u32(enc::ldr_64(
            map_fixed_reg(MACHINE_MEM0_BASE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::MEM0_BASE / 8) as u32,
        ));
        self.text.emit_u32(enc::ldr_64(
            map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::MEM0_SIZE / 8) as u32,
        ));
    }

    fn emit_epilogue(&mut self) {
        emit_shared_epilogue(&mut self.text);
    }

    fn emit_block(&mut self, block: &MachineBlock) -> Result<(), WasmError> {
        for inst in &block.ops {
            self.emit_inst(inst)?;
        }
        self.emit_terminator(&block.terminator)
    }

    fn emit_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src } => self.emit_move(*dst, *src),
            MachineInstKind::Lea { dst, addr } => self.emit_addr_into(map_reg(*dst)?, *addr),
            MachineInstKind::Load {
                dst,
                addr,
                width,
                extension,
            } => self.emit_load(*dst, *addr, *width, *extension),
            MachineInstKind::Store { addr, width, src } => self.emit_store(*addr, *width, *src),
            MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src,
            } => self.emit_int_unary(*width, *op, *dst, *src),
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => self.emit_int_binary(*width, *op, *dst, *lhs, *rhs),
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => self.emit_int_compare(*width, *kind, *sign, *dst, *lhs, *rhs),
            MachineInstKind::Select {
                dst,
                on_true,
                on_false,
                cond,
            } => self.emit_select(*dst, *on_true, *on_false, *cond),
            MachineInstKind::CallHelper(call) => {
                self.emit_call_helper(call.target.0 as usize, call.metadata.0 as usize)
            }
            MachineInstKind::FloatUnary {
                width,
                op,
                dst,
                src,
            } => self.emit_float_unary(*width, *op, *dst, *src),
            MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => self.emit_float_binary(*width, *op, *dst, *lhs, *rhs),
            MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs,
                rhs,
            } => self.emit_float_compare(*width, *kind, *dst, *lhs, *rhs),
            MachineInstKind::Convert { op, dst, src } => self.emit_convert(*op, *dst, *src),
        }
    }

    fn emit_terminator(&mut self, term: &MachineTerminator) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => {
                let label = self.emit_edge(edge.target, &edge.args)?;
                self.emit_b(label);
                Ok(())
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.emit_branch(cond, then_edge, else_edge),
            MachineTerminator::Return => self.emit_return_sequence(),
            MachineTerminator::Trap { kind } => {
                self.emit_trap(*kind);
                Ok(())
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.emit_jump_table(*index, entries)
            }
            MachineTerminator::CallDirect {
                callee,
                callee_frame_base,
                continuation,
            } => self.emit_call_direct(*callee, *callee_frame_base, *continuation),
            MachineTerminator::CallIndirect {
                callee_target,
                callee_frame_base,
                arg_slots,
                caller_result_base,
                continuation,
            } => self.emit_call_indirect(
                *callee_target,
                *callee_frame_base,
                *arg_slots,
                *caller_result_base,
                *continuation,
            ),
        }
    }

    fn emit_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &crate::vm::native::ir::machine::MachineEdge,
        else_edge: &crate::vm::native::ir::machine::MachineEdge,
    ) -> Result<(), WasmError> {
        let then_label = self.emit_edge(then_edge.target, &then_edge.args)?;
        let else_label = self.emit_edge(else_edge.target, &else_edge.args)?;
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => self.emit_b(else_label),
                MachineValue::Imm64(_) => self.emit_b(then_label),
                MachineValue::Reg(reg) => {
                    self.emit_cbnz(map_reg(reg)?, then_label);
                    self.emit_b(else_label);
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                self.emit_cmp_values(width, lhs, rhs)?;
                self.emit_b_cond(map_int_cond(kind, sign), then_label);
                self.emit_b(else_label);
            }
            MachineBranchCond::FloatCompare {
                width,
                kind,
                lhs,
                rhs,
            } => {
                return self.emit_float_branch(width, kind, lhs, rhs, then_label, else_label);
            }
        }
        Ok(())
    }

    fn emit_call_direct(
        &mut self,
        callee: crate::vm::native::ir::machine::MachineFuncId,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_runtime = self.runtime_for(callee)?;
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("arm64 direct local call requires callee call scratch".into())
        })?;
        let continuation_slot = call_scratch.base_slot
            + (self.compiled.runtime().call_link.continuation_offset / 8) as u16;
        let callee_fp = map_reg(callee_frame_base)?;

        self.emit_stack_overflow_check(callee_fp, callee_runtime.total_frame_slots)?;
        self.emit_call_depth_increment()?;

        let continuation_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        if continuation_slot < 4096 {
            self.text
                .emit_u32(enc::str_64(SCRATCH0, callee_fp, continuation_slot as u32));
        } else {
            self.materialize_u64(SCRATCH1, u64::from(continuation_slot) * 8);
            self.text
                .emit_u32(enc::add_reg_64(SCRATCH1, callee_fp, SCRATCH1));
            self.text
                .emit_u32(enc::str_reg_64(SCRATCH0, SCRATCH1, Arm64Reg::Xzr));
        }

        let callee_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        self.text
            .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
        self.text.emit_u32(enc::br(SCRATCH0));

        let continuation_literal = self.text.emit_u64(0);
        let callee_literal = self.text.emit_u64(0);

        let continuation_label = self.block_label(continuation)?;
        let continuation_delta =
            ((continuation_literal as isize - continuation_load as isize) / 4) as i32;
        self.text.patch_u32(
            continuation_load,
            enc::ldr_lit_64(SCRATCH0, continuation_delta),
        );
        let callee_delta = ((callee_literal as isize - callee_load as isize) / 4) as i32;
        self.text
            .patch_u32(callee_load, enc::ldr_lit_64(SCRATCH0, callee_delta));

        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: continuation_literal,
            target_label: continuation_label,
        });
        self.direct_call_patches.push(DirectCallPatch {
            literal_offset: callee_literal,
            callee,
        });
        Ok(())
    }

    fn emit_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[crate::vm::native::ir::machine::MachineEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal(
                "arm64 MachineIR jump table requires at least one entry".into(),
            ));
        }
        if entries.len() == 1 {
            let label = self.emit_edge(entries[0].target, &entries[0].args)?;
            self.emit_b(label);
            return Ok(());
        }

        let index_reg = self.materialize_value(SCRATCH1, index)?;
        self.materialize_u64(Arm64Reg::X0, (entries.len() - 1) as u64);
        self.text.emit_u32(enc::cmp_reg_64(index_reg, Arm64Reg::X0));
        self.text
            .emit_u32(enc::csel_64(SCRATCH1, index_reg, Arm64Reg::X0, Cond::Ls));

        let table_base_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        self.materialize_u64(Arm64Reg::X0, 3);
        self.text
            .emit_u32(enc::lslv_64(SCRATCH1, SCRATCH1, Arm64Reg::X0));
        self.text
            .emit_u32(enc::ldr_reg_64(SCRATCH0, SCRATCH0, SCRATCH1));
        self.text.emit_u32(enc::br(SCRATCH0));

        let table_base_literal = self.text.emit_u64(0);
        let table_offset = self.text.len();
        let table_base_delta =
            ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
        self.text
            .patch_u32(table_base_load, enc::ldr_lit_64(SCRATCH0, table_base_delta));
        self.resolved_ptr_patches.push(LocalPtrPatch {
            literal_offset: table_base_literal,
            target_offset: table_offset,
        });

        for entry in entries {
            let label = self.emit_edge(entry.target, &entry.args)?;
            let literal_offset = self.text.emit_u64(0);
            self.local_ptr_patches.push(PendingLocalPtrPatch {
                literal_offset,
                target_label: label,
            });
        }
        Ok(())
    }

    fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = *self.runtime_for(self.function.id)?;
        let call_scratch = runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("arm64 local return requires call scratch".into())
        })?;
        let call_link = self.compiled.runtime().call_link;
        let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;
        let caller_frame_slot = call_scratch.base_slot + (call_link.caller_frame_offset / 8) as u16;
        let caller_result_base_slot =
            call_scratch.base_slot + (call_link.caller_result_base_offset / 8) as u16;

        self.text.emit_u32(enc::ldr_64(
            SCRATCH0,
            map_fixed_reg(MACHINE_FP_REG),
            continuation_slot as u32,
        ));
        self.text.emit_u32(enc::ldr_64(
            SCRATCH1,
            map_fixed_reg(MACHINE_FP_REG),
            caller_frame_slot as u32,
        ));
        self.text.emit_u32(enc::ldr_64(
            Arm64Reg::X0,
            map_fixed_reg(MACHINE_FP_REG),
            caller_result_base_slot as u32,
        ));
        self.text
            .emit_u32(enc::add_reg_64(Arm64Reg::X0, SCRATCH1, Arm64Reg::X0));

        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as u32 {
                self.text.emit_u32(enc::ldr_64(
                    Arm64Reg::X1,
                    map_fixed_reg(MACHINE_FP_REG),
                    results.base_slot as u32 + index,
                ));
                self.text
                    .emit_u32(enc::str_64(Arm64Reg::X1, Arm64Reg::X0, index));
            }
        }

        self.text.emit_u32(enc::ldr_64(
            Arm64Reg::X1,
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::CALL_DEPTH / 8) as u32,
        ));
        self.text
            .emit_u32(enc::sub_imm_64(Arm64Reg::X1, Arm64Reg::X1, 1));
        self.text.emit_u32(enc::str_64(
            Arm64Reg::X1,
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::CALL_DEPTH / 8) as u32,
        ));
        self.text
            .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), SCRATCH1));
        self.text.emit_u32(enc::br(SCRATCH0));
        Ok(())
    }

    fn emit_call_indirect(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        self.emit_call_depth_increment()?;

        let callee_id_reg = self.materialize_value(SCRATCH1, callee_target)?;
        let table_base_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        let skip_table_literal = self.text.emit_u32(enc::b(0)); // skip over literal
        self.function_table_patches.push(self.text.emit_u64(0));
        let table_base_literal = self
            .function_table_patches
            .last()
            .copied()
            .expect("function table literal recorded");
        let after_table_literal = self.text.len();
        // Patch the skip branch
        let skip_delta = ((after_table_literal as isize - skip_table_literal as isize) / 4) as i32;
        self.text.patch_u32(skip_table_literal, enc::b(skip_delta));
        // Patch the ldr literal offset
        let table_base_delta =
            ((table_base_literal as isize - table_base_load as isize) / 4) as i32;
        self.text
            .patch_u32(table_base_load, enc::ldr_lit_64(SCRATCH0, table_base_delta));

        self.materialize_u64(Arm64Reg::X0, 5);
        self.text
            .emit_u32(enc::lslv_64(SCRATCH1, callee_id_reg, Arm64Reg::X0));
        self.text
            .emit_u32(enc::add_reg_64(SCRATCH0, SCRATCH0, SCRATCH1));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X0, SCRATCH0, 0));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X1, SCRATCH0, 1));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X2, SCRATCH0, 2));
        self.text.emit_u32(enc::ldr_64(Arm64Reg::X3, SCRATCH0, 3));

        let callee_fp = map_reg(callee_frame_base)?;
        self.text
            .emit_u32(enc::add_reg_64(SCRATCH0, callee_fp, Arm64Reg::X1));
        self.text.emit_u32(enc::ldr_64(
            SCRATCH1,
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::STACK_END / 8) as u32,
        ));
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, SCRATCH1));
        self.emit_b_cond(Cond::Hi, self.stack_overflow_label);

        self.emit_zero_dynamic_callee_prefix(callee_fp, arg_slots)?;

        let continuation_load = self.text.emit_u32(enc::ldr_lit_64(SCRATCH0, 0));
        let skip_cont_literal = self.text.emit_u32(enc::b(0)); // skip over literal
        let continuation_literal = self.text.emit_u64(0);
        let after_cont_literal = self.text.len();
        let skip_cont_delta =
            ((after_cont_literal as isize - skip_cont_literal as isize) / 4) as i32;
        self.text.patch_u32(skip_cont_literal, enc::b(skip_cont_delta));
        let continuation_label = self.block_label(continuation)?;
        let continuation_delta =
            ((continuation_literal as isize - continuation_load as isize) / 4) as i32;
        self.text.patch_u32(
            continuation_load,
            enc::ldr_lit_64(SCRATCH0, continuation_delta),
        );
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: continuation_literal,
            target_label: continuation_label,
        });

        self.materialize_u64(SCRATCH1, 3);
        self.text
            .emit_u32(enc::lslv_64(Arm64Reg::X3, Arm64Reg::X3, SCRATCH1));
        self.text
            .emit_u32(enc::add_reg_64(Arm64Reg::X3, callee_fp, Arm64Reg::X3));
        self.text
            .emit_u32(enc::str_reg_64(SCRATCH0, Arm64Reg::X3, Arm64Reg::Xzr));
        self.text
            .emit_u32(enc::str_64(map_fixed_reg(MACHINE_FP_REG), Arm64Reg::X3, 1));
        self.materialize_u64(SCRATCH1, u64::from(caller_result_base) * 8);
        self.text.emit_u32(enc::str_64(SCRATCH1, Arm64Reg::X3, 2));

        self.text
            .emit_u32(enc::mov_reg_64(map_fixed_reg(MACHINE_FP_REG), callee_fp));
        self.text.emit_u32(enc::br(Arm64Reg::X0));
        Ok(())
    }

    fn emit_zero_dynamic_callee_prefix(
        &mut self,
        callee_fp: Arm64Reg,
        arg_slots: u16,
    ) -> Result<(), WasmError> {
        self.materialize_u64(SCRATCH0, u64::from(arg_slots) * 8);
        self.text
            .emit_u32(enc::add_reg_64(SCRATCH0, callee_fp, SCRATCH0));
        self.materialize_u64(SCRATCH1, 3);
        self.text
            .emit_u32(enc::lslv_64(Arm64Reg::X2, Arm64Reg::X2, SCRATCH1));
        self.text
            .emit_u32(enc::add_reg_64(Arm64Reg::X2, callee_fp, Arm64Reg::X2));
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, Arm64Reg::X2));
        let done = self.new_label(LabelKind::Edge);
        let loop_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Hs, done);
        self.bind_label(loop_label);
        self.text
            .emit_u32(enc::str_reg_64(Arm64Reg::Xzr, SCRATCH0, Arm64Reg::Xzr));
        self.text.emit_u32(enc::add_imm_64(SCRATCH0, SCRATCH0, 8));
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, Arm64Reg::X2));
        self.emit_b_cond(Cond::Lo, loop_label);
        self.bind_label(done);
        Ok(())
    }

    fn emit_stack_overflow_check(
        &mut self,
        callee_fp: Arm64Reg,
        callee_total_frame_slots: u16,
    ) -> Result<(), WasmError> {
        let callee_end_bytes = u64::from(callee_total_frame_slots) * 8;
        if callee_end_bytes < 4096 {
            self.text.emit_u32(enc::add_imm_64(
                SCRATCH0,
                callee_fp,
                callee_end_bytes as u32,
            ));
        } else {
            self.materialize_u64(SCRATCH0, callee_end_bytes);
            self.text
                .emit_u32(enc::add_reg_64(SCRATCH0, callee_fp, SCRATCH0));
        }
        self.text.emit_u32(enc::ldr_64(
            SCRATCH1,
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::STACK_END / 8) as u32,
        ));
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, SCRATCH1));
        self.emit_b_cond(Cond::Hi, self.stack_overflow_label);
        Ok(())
    }

    fn emit_call_depth_increment(&mut self) -> Result<(), WasmError> {
        self.text.emit_u32(enc::ldr_64(
            SCRATCH0,
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::CALL_DEPTH / 8) as u32,
        ));
        self.materialize_u64(SCRATCH1, MAX_ARM64_CALL_DEPTH);
        self.text.emit_u32(enc::cmp_reg_64(SCRATCH0, SCRATCH1));
        self.emit_b_cond(Cond::Hs, self.call_depth_label);
        self.text.emit_u32(enc::add_imm_64(SCRATCH0, SCRATCH0, 1));
        self.text.emit_u32(enc::str_64(
            SCRATCH0,
            map_fixed_reg(MACHINE_CTX_REG),
            (ctx_offset::CALL_DEPTH / 8) as u32,
        ));
        Ok(())
    }

    fn runtime_for(
        &self,
        func_id: crate::vm::native::ir::machine::MachineFuncId,
    ) -> Result<&crate::vm::native::ir::runtime::MachineFunctionRuntime, WasmError> {
        self.compiled
            .runtime()
            .functions
            .get(func_id.0 as usize)
            .ok_or_else(|| {
                WasmError::internal(alloc::format!(
                    "arm64 runtime metadata missing for machine function {}",
                    func_id.0
                ))
            })
    }

    fn emit_move(&mut self, dst: MachineReg, src: MachineValue) -> Result<(), WasmError> {
        let dst = map_reg(dst)?;
        match src {
            MachineValue::Reg(src) => {
                let src = map_reg(src)?;
                if dst != src {
                    self.text.emit_u32(enc::mov_reg_64(dst, src));
                }
            }
            MachineValue::Imm64(value) => self.materialize_u64(dst, value),
        }
        Ok(())
    }

    fn emit_addr_into(&mut self, dst: Arm64Reg, addr: MachineAddr) -> Result<(), WasmError> {
        let base = map_reg(addr.base)?;
        let offset = addr.offset as i64;
        if offset == 0 {
            if dst != base {
                self.text.emit_u32(enc::mov_reg_64(dst, base));
            }
            return Ok(());
        }
        if offset > 0 && offset < 4096 {
            self.text
                .emit_u32(enc::add_imm_64(dst, base, offset as u32));
            return Ok(());
        }
        if offset < 0 && -offset < 4096 {
            self.text
                .emit_u32(enc::sub_imm_64(dst, base, (-offset) as u32));
            return Ok(());
        }
        self.materialize_u64(SCRATCH1, offset.unsigned_abs());
        if offset >= 0 {
            self.text.emit_u32(enc::add_reg_64(dst, base, SCRATCH1));
        } else {
            self.text.emit_u32(enc::sub_reg_64(dst, base, SCRATCH1));
        }
        Ok(())
    }

    fn emit_load(
        &mut self,
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let dst = map_reg(dst)?;
        // Fast path: U64 load with aligned immediate offset → single ldr_64
        if matches!(
            (width, extension),
            (MachineMemWidth::U64, MachineLoadExtension::None)
                | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend)
        ) {
            let base = map_reg(addr.base)?;
            let offset = addr.offset as i64;
            if offset >= 0 && (offset % 8) == 0 && (offset / 8) < 4096 {
                self.text
                    .emit_u32(enc::ldr_64(dst, base, (offset / 8) as u32));
                return Ok(());
            }
        }
        self.emit_addr_into(SCRATCH0, addr)?;
        let inst = match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::ldrb_reg(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::ldrsb_reg_64(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::ldrh_reg(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::ldrsh_reg_64(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::ldr_reg_32(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::ldrsw_reg(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::ldr_reg_64(dst, SCRATCH0, Arm64Reg::Xzr)
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "arm64 MachineIR backend does not support sign-extending U64 loads".into(),
                ))
            }
        };
        self.text.emit_u32(inst);
        Ok(())
    }

    fn emit_store(
        &mut self,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        // Fast path: U64 store with aligned immediate offset → single str_64
        if width == MachineMemWidth::U64 {
            let base = map_reg(addr.base)?;
            let offset = addr.offset as i64;
            if offset >= 0 && (offset % 8) == 0 && (offset / 8) < 4096 {
                let src_reg = self.materialize_value(SCRATCH1, src)?;
                self.text
                    .emit_u32(enc::str_64(src_reg, base, (offset / 8) as u32));
                return Ok(());
            }
        }
        self.emit_addr_into(SCRATCH0, addr)?;
        let src_reg = self.materialize_value(SCRATCH1, src)?;
        let inst = match width {
            MachineMemWidth::U8 => enc::strb_reg(src_reg, SCRATCH0, Arm64Reg::Xzr),
            MachineMemWidth::U16 => enc::strh_reg(src_reg, SCRATCH0, Arm64Reg::Xzr),
            MachineMemWidth::U32 => enc::str_reg_32(src_reg, SCRATCH0, Arm64Reg::Xzr),
            MachineMemWidth::U64 => enc::str_reg_64(src_reg, SCRATCH0, Arm64Reg::Xzr),
        };
        self.text.emit_u32(inst);
        Ok(())
    }

    fn emit_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = map_reg(dst)?;
        let src = self.materialize_value(SCRATCH0, src)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntUnaryOp::Eqz) => {
                self.text.emit_u32(enc::cmp_reg_32(src, Arm64Reg::Xzr));
                self.text.emit_u32(enc::cset_32(dst, Cond::Eq));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Eqz) => {
                self.text.emit_u32(enc::cmp_reg_64(src, Arm64Reg::Xzr));
                self.text.emit_u32(enc::cset_64(dst, Cond::Eq));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
                self.text.emit_u32(enc::clz_32(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
                self.text.emit_u32(enc::clz_64(dst, src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                self.text.emit_u32(enc::sxtb_32(dst, src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                self.text.emit_u32(enc::sxth_32(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                self.text.emit_u32(enc::sxtb_64(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                self.text.emit_u32(enc::sxth_64(dst, src));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
                self.text.emit_u32(enc::sxtw(dst, src));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
                self.text.emit_u32(enc::rbit_32(dst, src));
                self.text.emit_u32(enc::clz_32(dst, dst));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
                self.text.emit_u32(enc::rbit_64(dst, src));
                self.text.emit_u32(enc::clz_64(dst, dst));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
                // FMOV D0, X_src (move GP to FP); CNT V0.8B; ADDV B0; UMOV Wd, V0.B[0]
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, src));
                self.text.emit_u32(enc::cnt_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::addv_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::umov_b0(dst, FP_SCRATCH0));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, src));
                self.text.emit_u32(enc::cnt_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::addv_8b(FP_SCRATCH0, FP_SCRATCH0));
                self.text.emit_u32(enc::umov_b0(dst, FP_SCRATCH0));
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
                // i32.extend32_s is a nop (already 32-bit)
                if dst != src {
                    self.text.emit_u32(enc::mov_reg_64(dst, src));
                }
            }
        }
        Ok(())
    }

    fn emit_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = map_reg(dst)?;
        let lhs = self.materialize_value(SCRATCH0, lhs)?;
        let rhs = self.materialize_value(SCRATCH1, rhs)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                self.text.emit_u32(enc::add_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                self.text.emit_u32(enc::add_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                self.text.emit_u32(enc::sub_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                self.text.emit_u32(enc::sub_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Mul) => {
                self.text.emit_u32(enc::mul_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Mul) => {
                self.text.emit_u32(enc::mul_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                self.text.emit_u32(enc::and_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                self.text.emit_u32(enc::and_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                self.text.emit_u32(enc::orr_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                self.text.emit_u32(enc::orr_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                self.text.emit_u32(enc::eor_reg_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                self.text.emit_u32(enc::eor_reg_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                self.text.emit_u32(enc::lslv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                self.text.emit_u32(enc::lslv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                self.text.emit_u32(enc::asrv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                self.text.emit_u32(enc::asrv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                self.text.emit_u32(enc::lsrv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                self.text.emit_u32(enc::lsrv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                self.text.emit_u32(enc::rorv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                self.text.emit_u32(enc::rorv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                // rotl(x, n) = rotr(x, 32 - n)
                self.text.emit_u32(enc::neg_reg_32(SCRATCH0, rhs));
                self.text.emit_u32(enc::rorv_32(dst, lhs, SCRATCH0));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                self.text.emit_u32(enc::neg_reg_64(SCRATCH0, rhs));
                self.text.emit_u32(enc::rorv_64(dst, lhs, SCRATCH0));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::DivS) => {
                self.emit_div_s_32(dst, lhs, rhs);
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::DivS) => {
                self.emit_div_s_64(dst, lhs, rhs);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::DivU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I32);
                self.text.emit_u32(enc::udiv_32(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::DivU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I64);
                self.text.emit_u32(enc::udiv_64(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::RemS) => {
                self.emit_rem_s_32(dst, lhs, rhs);
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::RemS) => {
                self.emit_rem_s_64(dst, lhs, rhs);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::RemU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I32);
                // rem = lhs - (lhs / rhs) * rhs
                self.text.emit_u32(enc::udiv_32(SCRATCH0, lhs, rhs));
                self.text.emit_u32(enc::msub_32(dst, SCRATCH0, rhs, lhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::RemU) => {
                self.emit_div_u_check(lhs, rhs, MachineIntWidth::I64);
                self.text.emit_u32(enc::udiv_64(SCRATCH0, lhs, rhs));
                self.text.emit_u32(enc::msub_64(dst, SCRATCH0, rhs, lhs));
            }
        };
        Ok(())
    }

    fn emit_int_compare(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_cmp_values(width, lhs, rhs)?;
        let dst = map_reg(dst)?;
        let cond = map_int_cond(kind, sign);
        match width {
            MachineIntWidth::I32 => {
                self.text.emit_u32(enc::cset_32(dst, cond));
            }
            MachineIntWidth::I64 => {
                self.text.emit_u32(enc::cset_64(dst, cond));
            }
        };
        Ok(())
    }

    fn emit_select(
        &mut self,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = map_reg(dst)?;
        match cond {
            MachineValue::Imm64(value) => {
                let selected = if value != 0 { on_true } else { on_false };
                return self.emit_move(inv_map_reg(dst), selected);
            }
            MachineValue::Reg(reg) => {
                self.text.emit_u32(enc::cmp_imm_64(map_reg(reg)?, 0));
            }
        }
        let true_reg = self.materialize_value(SCRATCH0, on_true)?;
        let false_reg = self.materialize_value(SCRATCH1, on_false)?;
        self.text
            .emit_u32(enc::csel_64(dst, true_reg, false_reg, Cond::Ne));
        Ok(())
    }

    fn emit_call_helper(&mut self, extern_idx: usize, const_idx: usize) -> Result<(), WasmError> {
        let binding = self
            .compiled
            .module()
            .externs
            .get(extern_idx)
            .ok_or_else(|| WasmError::internal("arm64 helper target is out of range".into()))?;
        let metadata = self
            .compiled
            .const_ptr(crate::vm::native::ir::machine::MachineConstId(
                const_idx as u32,
            ))
            .ok_or_else(|| WasmError::internal("arm64 helper metadata is out of range".into()))?;
        self.text.emit_u32(enc::mov_reg_64(
            Arm64Reg::X0,
            map_fixed_reg(MACHINE_CTX_REG),
        ));
        self.text
            .emit_u32(enc::mov_reg_64(Arm64Reg::X1, map_fixed_reg(MACHINE_FP_REG)));
        self.materialize_u64(Arm64Reg::X2, metadata as u64);
        self.materialize_u64(
            SCRATCH0,
            resolve_helper_entry(binding.symbol) as usize as u64,
        );
        self.text.emit_u32(enc::blr(SCRATCH0));
        self.emit_cbnz(Arm64Reg::X0, self.return_error_label);
        Ok(())
    }

    fn emit_trap(&mut self, kind: MachineTrapKind) {
        self.text.emit_u32(enc::mov_reg_64(
            Arm64Reg::X0,
            map_fixed_reg(MACHINE_CTX_REG),
        ));
        self.materialize_u64(Arm64Reg::X1, trap_code(kind));
        self.materialize_u64(SCRATCH0, arm64_raise_trap as usize as u64);
        self.text.emit_u32(enc::blr(SCRATCH0));
        self.emit_b(self.return_error_label);
    }

    fn emit_cmp_values(
        &mut self,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let lhs = self.materialize_value(SCRATCH0, lhs)?;
        let rhs = self.materialize_value(SCRATCH1, rhs)?;
        match width {
            MachineIntWidth::I32 => {
                self.text.emit_u32(enc::cmp_reg_32(lhs, rhs));
            }
            MachineIntWidth::I64 => {
                self.text.emit_u32(enc::cmp_reg_64(lhs, rhs));
            }
        };
        Ok(())
    }

    // --- Division / remainder helpers with trap checks ---

    fn emit_div_by_zero_trap_label(&mut self) -> usize {
        let label = self.new_label(LabelKind::Edge);
        // We'll bind this later and emit a trap there.
        // Actually, we need to emit the trap inline. Let's use a pattern:
        // branch-to-trap on check, then fall through to the operation.
        label
    }

    fn emit_div_u_check(&mut self, _lhs: Arm64Reg, rhs: Arm64Reg, width: MachineIntWidth) {
        // rhs == 0 => trap IntegerDivideByZero
        match width {
            MachineIntWidth::I32 => self.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr)),
            MachineIntWidth::I64 => self.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr)),
        };
        // Branch to a trap stub
        let trap_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, trap_label);
        // Emit the trap stub at the end via deferred_traps
        self.deferred_traps.push((trap_label, MachineTrapKind::IntegerDivideByZero));
    }

    fn emit_div_s_32(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        // Check rhs == 0 => IntegerDivideByZero
        self.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps.push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        // Check lhs == i32::MIN && rhs == -1 => IntegerOverflow
        self.materialize_u64(SCRATCH0, i32::MIN as u32 as u64);
        self.text.emit_u32(enc::cmp_reg_32(lhs, SCRATCH0));
        let not_min = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Ne, not_min);
        // lhs is MIN, check rhs == -1
        self.materialize_u64(SCRATCH0, (-1i32) as u32 as u64);
        self.text.emit_u32(enc::cmp_reg_32(rhs, SCRATCH0));
        let overflow_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, overflow_label);
        self.deferred_traps.push((overflow_label, MachineTrapKind::IntegerOverflow));

        self.bind_label(not_min);
        self.text.emit_u32(enc::sdiv_32(dst, lhs, rhs));
    }

    fn emit_div_s_64(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        self.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps.push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        self.materialize_u64(SCRATCH0, i64::MIN as u64);
        self.text.emit_u32(enc::cmp_reg_64(lhs, SCRATCH0));
        let not_min = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Ne, not_min);
        self.materialize_u64(SCRATCH0, (-1i64) as u64);
        self.text.emit_u32(enc::cmp_reg_64(rhs, SCRATCH0));
        let overflow_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, overflow_label);
        self.deferred_traps.push((overflow_label, MachineTrapKind::IntegerOverflow));

        self.bind_label(not_min);
        self.text.emit_u32(enc::sdiv_64(dst, lhs, rhs));
    }

    fn emit_rem_s_32(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        // Check rhs == 0 => IntegerDivideByZero
        self.text.emit_u32(enc::cmp_reg_32(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps.push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        // rem = lhs - (lhs / rhs) * rhs  (wrapping, so MIN % -1 = 0, no trap)
        self.text.emit_u32(enc::sdiv_32(SCRATCH0, lhs, rhs));
        self.text.emit_u32(enc::msub_32(dst, SCRATCH0, rhs, lhs));
    }

    fn emit_rem_s_64(&mut self, dst: Arm64Reg, lhs: Arm64Reg, rhs: Arm64Reg) {
        self.text.emit_u32(enc::cmp_reg_64(rhs, Arm64Reg::Xzr));
        let div_zero_label = self.new_label(LabelKind::Edge);
        self.emit_b_cond(Cond::Eq, div_zero_label);
        self.deferred_traps.push((div_zero_label, MachineTrapKind::IntegerDivideByZero));

        self.text.emit_u32(enc::sdiv_64(SCRATCH0, lhs, rhs));
        self.text.emit_u32(enc::msub_64(dst, SCRATCH0, rhs, lhs));
    }

    // --- Float operations ---

    fn emit_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = map_reg(dst)?;
        let src_gp = self.materialize_value(SCRATCH0, src)?;
        // Move GP -> FP
        match width {
            MachineFloatWidth::F32 => self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH0, src_gp)),
            MachineFloatWidth::F64 => self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, src_gp)),
        };
        // Perform the FP operation
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => self.text.emit_u32(enc::fabs_s(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => self.text.emit_u32(enc::fabs_d(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => self.text.emit_u32(enc::fneg_s(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => self.text.emit_u32(enc::fneg_d(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => self.text.emit_u32(enc::fsqrt_s(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => self.text.emit_u32(enc::fsqrt_d(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Ceil) => self.text.emit_u32(enc::frintp_s(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Ceil) => self.text.emit_u32(enc::frintp_d(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Floor) => self.text.emit_u32(enc::frintm_s(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Floor) => self.text.emit_u32(enc::frintm_d(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Trunc) => self.text.emit_u32(enc::frintz_s(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Trunc) => self.text.emit_u32(enc::frintz_d(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Nearest) => self.text.emit_u32(enc::frintn_s(FP_SCRATCH1, FP_SCRATCH0)),
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Nearest) => self.text.emit_u32(enc::frintn_d(FP_SCRATCH1, FP_SCRATCH0)),
        };
        // Move FP -> GP
        match width {
            MachineFloatWidth::F32 => self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, FP_SCRATCH1)),
            MachineFloatWidth::F64 => self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, FP_SCRATCH1)),
        };
        Ok(())
    }

    fn emit_float_binary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = map_reg(dst)?;
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
        // Move both operands to FP
        match width {
            MachineFloatWidth::F32 => {
                self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH0, lhs_gp));
                self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH1, rhs_gp));
            }
            MachineFloatWidth::F64 => {
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, lhs_gp));
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH1, rhs_gp));
            }
        };
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => { self.text.emit_u32(enc::fadd_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => { self.text.emit_u32(enc::fadd_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => { self.text.emit_u32(enc::fsub_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => { self.text.emit_u32(enc::fsub_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => { self.text.emit_u32(enc::fmul_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => { self.text.emit_u32(enc::fmul_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => { self.text.emit_u32(enc::fdiv_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => { self.text.emit_u32(enc::fdiv_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1)); }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
                // Wasm fmin: NaN if either is NaN. ARM64 FMIN returns non-NaN operand.
                self.text.emit_u32(enc::fmin_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.text.emit_u32(enc::fcmp_s(FP_SCRATCH0, FP_SCRATCH1));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done); // no NaN => FMIN result is correct
                // NaN case: FADD produces NaN from NaN input
                self.text.emit_u32(enc::fadd_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.bind_label(done);
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
                self.text.emit_u32(enc::fmin_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.text.emit_u32(enc::fcmp_d(FP_SCRATCH0, FP_SCRATCH1));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done);
                self.text.emit_u32(enc::fadd_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.bind_label(done);
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
                self.text.emit_u32(enc::fmax_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.text.emit_u32(enc::fcmp_s(FP_SCRATCH0, FP_SCRATCH1));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done);
                self.text.emit_u32(enc::fadd_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.bind_label(done);
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
                self.text.emit_u32(enc::fmax_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.text.emit_u32(enc::fcmp_d(FP_SCRATCH0, FP_SCRATCH1));
                let done = self.new_label(LabelKind::Edge);
                self.emit_b_cond(Cond::Vc, done);
                self.text.emit_u32(enc::fadd_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH1));
                self.bind_label(done);
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
                // copysign: magnitude of lhs, sign of rhs
                // Extract sign bit from rhs, magnitude from lhs
                // F32: bit 31 is sign
                self.text.emit_u32(enc::fabs_s(FP_SCRATCH2, FP_SCRATCH0)); // |lhs|
                self.text.emit_u32(enc::fneg_s(FP_SCRATCH0, FP_SCRATCH2)); // -|lhs|
                // Test sign bit of rhs: if rhs_gp bit 31 is set, use -|lhs|, else |lhs|
                self.materialize_u64(SCRATCH0, 31);
                self.text.emit_u32(enc::lsrv_64(SCRATCH0, rhs_gp, SCRATCH0));
                self.text.emit_u32(enc::cmp_imm_64(SCRATCH0, 0));
                self.text.emit_u32(enc::fcsel_s(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH2, Cond::Ne));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
                self.text.emit_u32(enc::fabs_d(FP_SCRATCH2, FP_SCRATCH0));
                self.text.emit_u32(enc::fneg_d(FP_SCRATCH0, FP_SCRATCH2));
                self.materialize_u64(SCRATCH0, 63);
                self.text.emit_u32(enc::lsrv_64(SCRATCH0, rhs_gp, SCRATCH0));
                self.text.emit_u32(enc::cmp_imm_64(SCRATCH0, 0));
                self.text.emit_u32(enc::fcsel_d(FP_SCRATCH2, FP_SCRATCH0, FP_SCRATCH2, Cond::Ne));
            }
        };
        // Move result FP -> GP
        match width {
            MachineFloatWidth::F32 => self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, FP_SCRATCH2)),
            MachineFloatWidth::F64 => self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, FP_SCRATCH2)),
        };
        Ok(())
    }

    fn emit_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = map_reg(dst)?;
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
        match width {
            MachineFloatWidth::F32 => {
                self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH0, lhs_gp));
                self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH1, rhs_gp));
                self.text.emit_u32(enc::fcmp_s(FP_SCRATCH0, FP_SCRATCH1));
            }
            MachineFloatWidth::F64 => {
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, lhs_gp));
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH1, rhs_gp));
                self.text.emit_u32(enc::fcmp_d(FP_SCRATCH0, FP_SCRATCH1));
            }
        };
        // Wasm float comparisons: unordered (NaN) => false for all except Ne
        let cond = map_float_cond(kind);
        self.text.emit_u32(enc::cset_32(dst_gp, cond));
        Ok(())
    }

    fn emit_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = map_reg(dst)?;
        let src_gp = self.materialize_value(SCRATCH0, src)?;
        match op {
            // Integer wrapping / extension (no FP involved)
            MachineConvertOp::I32WrapI64 => {
                // Just mask to 32 bits
                self.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
            }
            MachineConvertOp::I64ExtendI32S => {
                self.text.emit_u32(enc::sxtw(dst_gp, src_gp));
            }
            MachineConvertOp::I64ExtendI32U => {
                self.text.emit_u32(enc::mov_reg_32(dst_gp, src_gp));
            }
            // Reinterpret (no-op, bits stay the same in GP reg)
            MachineConvertOp::I32ReinterpretF32
            | MachineConvertOp::I64ReinterpretF64
            | MachineConvertOp::F32ReinterpretI32
            | MachineConvertOp::F64ReinterpretI64 => {
                if dst_gp != src_gp {
                    self.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                }
            }
            // Float promotion / demotion
            MachineConvertOp::F64PromoteF32 => {
                self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fcvt_d_from_s(FP_SCRATCH1, FP_SCRATCH0));
                self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, FP_SCRATCH1));
            }
            MachineConvertOp::F32DemoteF64 => {
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fcvt_s_from_d(FP_SCRATCH1, FP_SCRATCH0));
                self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, FP_SCRATCH1));
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                self.text.emit_u32(enc::scvtf_s_32(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, FP_SCRATCH0));
            }
            MachineConvertOp::F32ConvertI32U => {
                // Zero-extend to 64-bit first to ensure unsigned interpretation
                self.text.emit_u32(enc::mov_reg_32(SCRATCH0, src_gp));
                self.text.emit_u32(enc::ucvtf_s_64(FP_SCRATCH0, SCRATCH0));
                self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, FP_SCRATCH0));
            }
            MachineConvertOp::F32ConvertI64S => {
                self.text.emit_u32(enc::scvtf_s_64(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, FP_SCRATCH0));
            }
            MachineConvertOp::F32ConvertI64U => {
                self.text.emit_u32(enc::ucvtf_s_64(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, FP_SCRATCH0));
            }
            MachineConvertOp::F64ConvertI32S => {
                self.text.emit_u32(enc::scvtf_d_32(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, FP_SCRATCH0));
            }
            MachineConvertOp::F64ConvertI32U => {
                self.text.emit_u32(enc::mov_reg_32(SCRATCH0, src_gp));
                self.text.emit_u32(enc::ucvtf_d_64(FP_SCRATCH0, SCRATCH0));
                self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, FP_SCRATCH0));
            }
            MachineConvertOp::F64ConvertI64S => {
                self.text.emit_u32(enc::scvtf_d_64(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, FP_SCRATCH0));
            }
            MachineConvertOp::F64ConvertI64U => {
                self.text.emit_u32(enc::ucvtf_d_64(FP_SCRATCH0, src_gp));
                self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, FP_SCRATCH0));
            }
            // Trapping truncations: call Rust helpers
            MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U => {
                self.emit_trapping_trunc(op, dst_gp, src_gp)?;
            }
            // Saturating truncations
            MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U => {
                self.emit_saturating_trunc(op, dst_gp, src_gp)?;
            }
        }
        Ok(())
    }

    fn emit_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: Arm64Reg,
        src: Arm64Reg,
    ) -> Result<(), WasmError> {
        // Call the helper: extern "C" fn(ctx, src_bits) -> status
        self.text.emit_u32(enc::mov_reg_64(Arm64Reg::X0, map_fixed_reg(MACHINE_CTX_REG)));
        self.text.emit_u32(enc::mov_reg_64(Arm64Reg::X1, src));
        self.materialize_u64(Arm64Reg::X2, convert_op_code(op));
        self.materialize_u64(SCRATCH0, arm64_trapping_trunc as usize as u64);
        self.text.emit_u32(enc::blr(SCRATCH0));
        // X0 = status (0 = ok), X1 = result value
        self.emit_cbnz(Arm64Reg::X0, self.return_error_label);
        self.text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::X1));
        Ok(())
    }

    fn emit_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: Arm64Reg,
        src: Arm64Reg,
    ) -> Result<(), WasmError> {
        self.text.emit_u32(enc::mov_reg_64(Arm64Reg::X0, src));
        self.materialize_u64(Arm64Reg::X1, convert_op_code(op));
        self.materialize_u64(SCRATCH0, arm64_saturating_trunc as usize as u64);
        self.text.emit_u32(enc::blr(SCRATCH0));
        // X0 = result value (no error possible for sat)
        self.text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::X0));
        Ok(())
    }

    fn emit_float_branch(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        lhs: MachineValue,
        rhs: MachineValue,
        then_label: usize,
        else_label: usize,
    ) -> Result<(), WasmError> {
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
        match width {
            MachineFloatWidth::F32 => {
                self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH0, lhs_gp));
                self.text.emit_u32(enc::fmov_s_from_gp(FP_SCRATCH1, rhs_gp));
                self.text.emit_u32(enc::fcmp_s(FP_SCRATCH0, FP_SCRATCH1));
            }
            MachineFloatWidth::F64 => {
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH0, lhs_gp));
                self.text.emit_u32(enc::fmov_d_from_gp(FP_SCRATCH1, rhs_gp));
                self.text.emit_u32(enc::fcmp_d(FP_SCRATCH0, FP_SCRATCH1));
            }
        };
        let cond = map_float_cond(kind);
        self.emit_b_cond(cond, then_label);
        self.emit_b(else_label);
        Ok(())
    }

    fn emit_parallel_moves(
        &mut self,
        params: &[MachineReg],
        args: &[MachineValue],
    ) -> Result<(), WasmError> {
        let mut pending = Vec::new();
        for (&dst_reg, &arg) in params.iter().zip(args.iter()) {
            let dst = map_reg(dst_reg)?;
            let src = match arg {
                MachineValue::Reg(reg) => ParallelSource::Reg(map_reg(reg)?),
                MachineValue::Imm64(value) => ParallelSource::Imm(value),
            };
            if matches!(src, ParallelSource::Reg(src_reg) if src_reg == dst) {
                continue;
            }
            pending.push((dst, src));
        }

        while !pending.is_empty() {
            let mut ready = None;
            for index in 0..pending.len() {
                let dst = pending[index].0;
                let blocked = pending.iter().enumerate().any(|(other_index, (_, src))| {
                    other_index != index
                        && matches!(src, ParallelSource::Reg(src_reg) if *src_reg == dst)
                });
                if !blocked {
                    ready = Some(index);
                    break;
                }
            }
            if let Some(index) = ready {
                let (dst, src) = pending.remove(index);
                self.emit_source_move(dst, src);
                continue;
            }

            let (dst, src) = pending.remove(0);
            let ParallelSource::Reg(src_reg) = src else {
                self.emit_source_move(dst, src);
                continue;
            };
            self.text.emit_u32(enc::mov_reg_64(SCRATCH1, dst));
            self.text.emit_u32(enc::mov_reg_64(dst, src_reg));
            for (_, source) in pending.iter_mut() {
                if matches!(*source, ParallelSource::Reg(reg) if reg == dst) {
                    *source = ParallelSource::Temp;
                }
            }
        }
        Ok(())
    }

    fn emit_source_move(&mut self, dst: Arm64Reg, src: ParallelSource) {
        match src {
            ParallelSource::Reg(src) => {
                self.text.emit_u32(enc::mov_reg_64(dst, src));
            }
            ParallelSource::Imm(value) => self.materialize_u64(dst, value),
            ParallelSource::Temp => {
                self.text.emit_u32(enc::mov_reg_64(dst, SCRATCH1));
            }
        }
    }

    fn materialize_value(
        &mut self,
        scratch: Arm64Reg,
        value: MachineValue,
    ) -> Result<Arm64Reg, WasmError> {
        match value {
            MachineValue::Reg(reg) => map_reg(reg),
            MachineValue::Imm64(value) => {
                self.materialize_u64(scratch, value);
                Ok(scratch)
            }
        }
    }

    fn materialize_u64(&mut self, dst: Arm64Reg, value: u64) {
        materialize_u64_into(&mut self.text, dst, value);
    }

    /// Check if an edge is a no-op (args match params exactly, no copies needed).
    fn is_identity_edge(
        &self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> bool {
        let Some(block) = self.function.program.blocks.get(target.as_usize()) else {
            return false;
        };
        if block.params.len() != args.len() {
            return false;
        }
        block.params.iter().zip(args.iter()).all(|(param, arg)| {
            matches!(arg, MachineValue::Reg(r) if *r == *param)
        })
    }

    /// For identity/empty edges, branch directly to the target block.
    /// For edges needing copies, create an edge stub.
    fn emit_edge(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        if self.is_identity_edge(target, args) {
            // No copies needed — branch directly to the target block.
            return self.block_label(target);
        }
        self.add_edge_stub(target, args)
    }

    fn add_edge_stub(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        let block = self
            .function
            .program
            .blocks
            .get(target.as_usize())
            .ok_or_else(|| WasmError::internal("arm64 edge target block is out of range".into()))?;
        let label = self.new_label(LabelKind::Edge);
        self.edge_stubs.push(EdgeStub {
            label,
            target,
            params: block.params.clone(),
            args: args.to_vec(),
        });
        Ok(label)
    }

    fn emit_branch_to_block(&mut self, target: MachineBlockId) -> Result<(), WasmError> {
        let label = self.block_label(target)?;
        self.emit_b(label);
        Ok(())
    }

    fn block_label(&self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.block_labels
            .get(target.0 as usize)
            .copied()
            .filter(|label| *label != usize::MAX)
            .ok_or_else(|| WasmError::internal("arm64 block label is out of range".into()))
    }

    fn new_label(&mut self, _kind: LabelKind) -> usize {
        let label = self.labels.len();
        self.labels.push(None);
        label
    }

    fn bind_label(&mut self, label: usize) {
        self.labels[label] = Some(self.text.len());
    }

    fn emit_b(&mut self, label: usize) {
        let inst_offset = self.text.emit_u32(enc::b(0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::B,
        });
    }

    fn emit_b_cond(&mut self, cond: Cond, label: usize) {
        let inst_offset = self.text.emit_u32(enc::b_cond(cond, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::BCond(cond),
        });
    }

    fn emit_cbnz(&mut self, reg: Arm64Reg, label: usize) {
        let inst_offset = self.text.emit_u32(enc::cbnz_64(reg, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::Cbnz(reg),
        });
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target = self
                .labels
                .get(fixup.label)
                .and_then(|value| *value)
                .ok_or_else(|| {
                    WasmError::internal("arm64 branch target label is unresolved".into())
                })?;
            let delta_words = ((target as isize) - (fixup.inst_offset as isize)) / 4;
            let patched = match fixup.kind {
                BranchFixupKind::B => enc::b(delta_words as i32),
                BranchFixupKind::BCond(cond) => enc::b_cond(cond, delta_words as i32),
                BranchFixupKind::Cbz(reg) => enc::cbz_64(reg, delta_words as i32),
                BranchFixupKind::Cbnz(reg) => enc::cbnz_64(reg, delta_words as i32),
            };
            self.text.patch_u32(fixup.inst_offset, patched);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ParallelSource {
    Reg(Arm64Reg),
    Imm(u64),
    Temp,
}

impl From<Arm64Reg> for ParallelSource {
    fn from(value: Arm64Reg) -> Self {
        Self::Reg(value)
    }
}

impl From<u64> for ParallelSource {
    fn from(value: u64) -> Self {
        Self::Imm(value)
    }
}

fn emit_shared_prologue(text: &mut Arm64TextEmitter) {
    text.emit_u32(enc::sub_imm_64(
        Arm64Reg::SP,
        Arm64Reg::SP,
        CALLEE_SAVED_FRAME_SIZE,
    ));
    text.emit_u32(enc::stp_64(Arm64Reg::X19, Arm64Reg::X20, Arm64Reg::SP, 0));
    text.emit_u32(enc::stp_64(Arm64Reg::X21, Arm64Reg::X22, Arm64Reg::SP, 2));
    text.emit_u32(enc::stp_64(Arm64Reg::X23, Arm64Reg::X24, Arm64Reg::SP, 4));
    text.emit_u32(enc::stp_64(Arm64Reg::X25, Arm64Reg::X26, Arm64Reg::SP, 6));
    text.emit_u32(enc::stp_64(Arm64Reg::X27, Arm64Reg::X28, Arm64Reg::SP, 8));
    text.emit_u32(enc::stp_64(Arm64Reg::X29, Arm64Reg::X30, Arm64Reg::SP, 10));
}

fn emit_shared_epilogue(text: &mut Arm64TextEmitter) {
    text.emit_u32(enc::ldp_64(Arm64Reg::X19, Arm64Reg::X20, Arm64Reg::SP, 0));
    text.emit_u32(enc::ldp_64(Arm64Reg::X21, Arm64Reg::X22, Arm64Reg::SP, 2));
    text.emit_u32(enc::ldp_64(Arm64Reg::X23, Arm64Reg::X24, Arm64Reg::SP, 4));
    text.emit_u32(enc::ldp_64(Arm64Reg::X25, Arm64Reg::X26, Arm64Reg::SP, 6));
    text.emit_u32(enc::ldp_64(Arm64Reg::X27, Arm64Reg::X28, Arm64Reg::SP, 8));
    text.emit_u32(enc::ldp_64(Arm64Reg::X29, Arm64Reg::X30, Arm64Reg::SP, 10));
    text.emit_u32(enc::add_imm_64(
        Arm64Reg::SP,
        Arm64Reg::SP,
        CALLEE_SAVED_FRAME_SIZE,
    ));
    text.emit_u32(enc::ret());
}

fn materialize_u64_into(text: &mut Arm64TextEmitter, dst: Arm64Reg, value: u64) {
    if value == 0 {
        text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::Xzr));
        return;
    }

    let chunks = [
        (value & 0xffff) as u16,
        ((value >> 16) & 0xffff) as u16,
        ((value >> 32) & 0xffff) as u16,
        ((value >> 48) & 0xffff) as u16,
    ];

    // Find first non-zero chunk for movz, then movk for remaining non-zero chunks
    let mut first = true;
    for (i, &chunk) in chunks.iter().enumerate() {
        if chunk != 0 || first && i == 3 {
            // Must emit at least one instruction
            if first {
                text.emit_u32(enc::movz_64(dst, chunk, (i as u32) * 16));
                first = false;
            } else {
                text.emit_u32(enc::movk_64(dst, chunk, (i as u32) * 16));
            }
        }
    }
    if first {
        // All chunks are zero but value != 0 (shouldn't happen, covered above)
        text.emit_u32(enc::movz_64(dst, 0, 0));
    }
}

fn map_fixed_reg(reg: MachineReg) -> Arm64Reg {
    match reg {
        MACHINE_CTX_REG => Arm64Reg::X19,
        MACHINE_FP_REG => Arm64Reg::X20,
        MACHINE_MEM0_BASE_REG => Arm64Reg::X21,
        MACHINE_MEM0_SIZE_REG => Arm64Reg::X22,
        _ => unreachable!("not a fixed machine reg"),
    }
}

fn map_reg(reg: MachineReg) -> Result<Arm64Reg, WasmError> {
    if reg.0 < MACHINE_FIXED_REG_COUNT {
        return Ok(map_fixed_reg(reg));
    }
    DYNAMIC_REGS
        .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
        .copied()
        .ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 MachineIR backend has no physical mapping for machine reg {}",
                reg.0
            ))
        })
}

fn inv_map_reg(reg: Arm64Reg) -> MachineReg {
    match reg {
        Arm64Reg::X19 => MACHINE_CTX_REG,
        Arm64Reg::X20 => MACHINE_FP_REG,
        Arm64Reg::X21 => MACHINE_MEM0_BASE_REG,
        Arm64Reg::X22 => MACHINE_MEM0_SIZE_REG,
        other => {
            let index = DYNAMIC_REGS
                .iter()
                .position(|candidate| *candidate == other)
                .expect("mapped reg must come from dynamic table");
            MachineReg(MACHINE_FIXED_REG_COUNT + index as u16)
        }
    }
}

fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cond {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cond::Eq,
        (MachineCompareKind::Ne, _) => Cond::Ne,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Lo,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
        (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Hs,
    }
}

fn trap_code(kind: MachineTrapKind) -> u64 {
    match kind {
        MachineTrapKind::Unreachable => 0,
        MachineTrapKind::MemoryOutOfBounds => 1,
        MachineTrapKind::TableOutOfBounds => 2,
        MachineTrapKind::InvalidFunctionReference => 3,
        MachineTrapKind::IndirectCallTypeMismatch => 4,
        MachineTrapKind::IntegerDivideByZero => 5,
        MachineTrapKind::IntegerOverflow => 6,
        MachineTrapKind::CallStackExhausted => 7,
        MachineTrapKind::StackOverflow => 8,
        MachineTrapKind::HelperFailure => 9,
    }
}

/// Map a Wasm float comparison kind to an ARM64 condition code.
///
/// Wasm float comparisons treat unordered (NaN) as false for all relations
/// except Ne (Ne returns true when either is NaN).
fn map_float_cond(kind: MachineCompareKind) -> Cond {
    match kind {
        // Eq: ordered & equal. FCMP: Z=1, V=0. Use EQ (but only if not unordered).
        // On ARM64 after FCMP, EQ is true only when ordered & equal (V=0 && Z=1).
        MachineCompareKind::Eq => Cond::Eq,
        // Ne: unordered | not-equal.
        // On ARM64 after FCMP, NE is true when Z=0, which includes unordered. Correct!
        MachineCompareKind::Ne => Cond::Ne,
        // Lt: ordered & less. On ARM64: MI (N=1, V=0). Correct for ordered less-than.
        MachineCompareKind::Lt => Cond::Mi,
        // Gt: ordered & greater. On ARM64: GT (Z=0, N=V). Correct.
        MachineCompareKind::Gt => Cond::Gt,
        // Le: ordered & less-or-equal. On ARM64: LS (C=0 || Z=1).
        MachineCompareKind::Le => Cond::Ls,
        // Ge: ordered & greater-or-equal. On ARM64: GE (N=V). Correct.
        MachineCompareKind::Ge => Cond::Ge,
    }
}

fn convert_op_code(op: MachineConvertOp) -> u64 {
    match op {
        MachineConvertOp::I32TruncF32S => 0,
        MachineConvertOp::I32TruncF32U => 1,
        MachineConvertOp::I32TruncF64S => 2,
        MachineConvertOp::I32TruncF64U => 3,
        MachineConvertOp::I64TruncF32S => 4,
        MachineConvertOp::I64TruncF32U => 5,
        MachineConvertOp::I64TruncF64S => 6,
        MachineConvertOp::I64TruncF64U => 7,
        MachineConvertOp::I32TruncSatF32S => 8,
        MachineConvertOp::I32TruncSatF32U => 9,
        MachineConvertOp::I32TruncSatF64S => 10,
        MachineConvertOp::I32TruncSatF64U => 11,
        MachineConvertOp::I64TruncSatF32S => 12,
        MachineConvertOp::I64TruncSatF32U => 13,
        MachineConvertOp::I64TruncSatF64S => 14,
        MachineConvertOp::I64TruncSatF64U => 15,
        _ => u64::MAX,
    }
}

use crate::vm::raw_value::{as_f32, as_f64, from_i32, from_i64};

/// Return type for trapping truncation helpers.
/// On ARM64 C ABI, a 2-field repr(C) struct of u64s is returned in X0 and X1.
#[repr(C)]
pub(crate) struct TruncResult {
    pub status: u64,
    pub value: u64,
}

/// Trapping truncation helper called from generated code.
/// Returns status in X0 (0 = ok) and result in X1 via struct return.
pub(crate) unsafe extern "C" fn arm64_trapping_trunc(
    ctx: *mut crate::vm::native::runtime::context::NativeContext,
    src_bits: u64,
    op_code: u64,
) -> TruncResult {
    let result = match op_code {
        0 => trunc_f32_to_i32_s(src_bits as u32),
        1 => trunc_f32_to_i32_u(src_bits as u32),
        2 => trunc_f64_to_i32_s(src_bits),
        3 => trunc_f64_to_i32_u(src_bits),
        4 => trunc_f32_to_i64_s(src_bits as u32),
        5 => trunc_f32_to_i64_u(src_bits as u32),
        6 => trunc_f64_to_i64_s(src_bits),
        7 => trunc_f64_to_i64_u(src_bits),
        _ => Err(WasmError::trap("invalid trunc op".into())),
    };
    match result {
        Ok(value) => TruncResult { status: 0, value },
        Err(err) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(err);
            }
            TruncResult { status: 1, value: 0 }
        }
    }
}

/// Saturating truncation helper called from generated code.
/// Returns result in X0 (no error possible).
pub(crate) unsafe extern "C" fn arm64_saturating_trunc(src_bits: u64, op_code: u64) -> u64 {
    match op_code {
        8 => trunc_sat_f32_to_i32_s(src_bits as u32),
        9 => trunc_sat_f32_to_i32_u(src_bits as u32),
        10 => trunc_sat_f64_to_i32_s(src_bits),
        11 => trunc_sat_f64_to_i32_u(src_bits),
        12 => trunc_sat_f32_to_i64_s(src_bits as u32),
        13 => trunc_sat_f32_to_i64_u(src_bits as u32),
        14 => trunc_sat_f64_to_i64_s(src_bits),
        15 => trunc_sat_f64_to_i64_u(src_bits),
        _ => 0,
    }
}

// Trapping truncation implementations (matching Wasm spec)

fn trunc_f32_to_i32_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    // Range: -2147483648.0 <= value < 2147483648.0
    if value >= 2147483648.0_f32 || value < -2147483648.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 2147483648.0 || value <= -2147483649.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0_f32 || value < -9223372036854775808.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 18446744073709551616.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0 || value < -9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 18446744073709551616.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

// Saturating truncation implementations

fn trunc_sat_f32_to_i32_s(bits: u32) -> u64 {
    let v = as_f32(bits as u64);
    if v.is_nan() { return 0; }
    if v >= 2147483647.0_f32 { return i32::MAX as u32 as u64; }
    if v <= -2147483648.0_f32 { return i32::MIN as u32 as u64; }
    from_i32(v as i32)
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let v = as_f32(bits as u64);
    if v.is_nan() || v <= -1.0_f32 { return 0; }
    if v >= 4294967295.0_f32 { return u32::MAX as u64; }
    u64::from(v as u32)
}

fn trunc_sat_f64_to_i32_s(bits: u64) -> u64 {
    let v = as_f64(bits);
    if v.is_nan() { return 0; }
    if v >= 2147483647.0 { return i32::MAX as u32 as u64; }
    if v <= -2147483648.0 { return i32::MIN as u32 as u64; }
    from_i32(v as i32)
}

fn trunc_sat_f64_to_i32_u(bits: u64) -> u64 {
    let v = as_f64(bits);
    if v.is_nan() || v <= -1.0 { return 0; }
    if v >= 4294967295.0 { return u32::MAX as u64; }
    u64::from(v as u32)
}

fn trunc_sat_f32_to_i64_s(bits: u32) -> u64 {
    let v = as_f32(bits as u64);
    if v.is_nan() { return 0; }
    if v >= 9223372036854775807.0_f32 { return i64::MAX as u64; }
    if v <= -9223372036854775808.0_f32 { return i64::MIN as u64; }
    from_i64(v as i64)
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let v = as_f32(bits as u64);
    if v.is_nan() || v <= -1.0_f32 { return 0; }
    if v >= 18446744073709551615.0_f32 { return u64::MAX; }
    v as u64
}

fn trunc_sat_f64_to_i64_s(bits: u64) -> u64 {
    let v = as_f64(bits);
    if v.is_nan() { return 0; }
    if v >= 9223372036854775807.0 { return i64::MAX as u64; }
    if v <= -9223372036854775808.0 { return i64::MIN as u64; }
    from_i64(v as i64)
}

fn trunc_sat_f64_to_i64_u(bits: u64) -> u64 {
    let v = as_f64(bits);
    if v.is_nan() || v <= -1.0 { return 0; }
    if v >= 18446744073709551615.0 { return u64::MAX; }
    v as u64
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::compile_module;
    use crate::{
        module::{type_context::TypeContext, type_defs::FunctionType},
        vm::{
            entities::ModuleInst,
            native::{
                code::CompiledNativeModule,
                ir::{
                    machine::{
                        MachineBlock, MachineBlockId, MachineFunction, MachineInst,
                        MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineModule,
                        MachineReg, MachineTerminator, MachineValue,
                    },
                    runtime::{MachineFunctionRuntime, MachineRuntimeContract},
                },
                runtime::context::NativeContext,
            },
            store::Store,
        },
    };

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_simple_add_function_in_arm64_code() {
        let function = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 7,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![
                        MachineInst {
                            kind: MachineInstKind::Move {
                                dst: MachineReg(4),
                                src: MachineValue::Imm64(40),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Move {
                                dst: MachineReg(5),
                                src: MachineValue::Imm64(2),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::IntBinary {
                                width: MachineIntWidth::I32,
                                op: MachineIntBinaryOp::Add,
                                dst: MachineReg(6),
                                lhs: MachineValue::Reg(MachineReg(4)),
                                rhs: MachineValue::Reg(MachineReg(5)),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                addr: crate::vm::native::ir::machine::MachineAddr {
                                    base: crate::vm::native::ir::machine::MACHINE_FP_REG,
                                    offset: 0,
                                },
                                width: crate::vm::native::ir::machine::MachineMemWidth::U64,
                                src: MachineValue::Reg(MachineReg(6)),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::native::arch::NativeBackend::Arm64,
            crate::vm::backend::BackendConfig::new(3, 4),
            MachineModule {
                functions: vec![function],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: crate::vm::native::ir::runtime::MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: crate::vm::native::ir::machine::MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 4,
                    call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                        base_slot: 1,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                        base_slot: 0,
                        slots: 1,
                    }),
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 4];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        ctx.call_depth = 1;
        stack[1] = entry.root_return as u64;
        stack[2] = stack.as_mut_ptr() as u64;
        stack[3] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert_eq!(stack[0], 42);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_store_imm64_directly() {
        // Test: Store { src: Imm64(42) } without a preceding move.
        // This is the pattern produced by constant folding.
        let function = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 7,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::Store {
                            addr: crate::vm::native::ir::machine::MachineAddr {
                                base: crate::vm::native::ir::machine::MACHINE_FP_REG,
                                offset: 0,
                            },
                            width: crate::vm::native::ir::machine::MachineMemWidth::U64,
                            src: MachineValue::Imm64(42),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::native::arch::NativeBackend::Arm64,
            crate::vm::backend::BackendConfig::new(3, 4),
            MachineModule {
                functions: vec![function],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: crate::vm::native::ir::runtime::MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: crate::vm::native::ir::machine::MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 4,
                    call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                        base_slot: 1,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                        base_slot: 0,
                        slots: 1,
                    }),
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 4];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        ctx.call_depth = 1;
        stack[1] = entry.root_return as u64;
        stack[2] = stack.as_mut_ptr() as u64;
        stack[3] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert_eq!(stack[0], 42, "Store with Imm64(42) should write 42 to fp[0]");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_empty_root_function_with_unsupported_neighbor_stub() {
        let supported = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 4,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let unsupported = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(1),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 5,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::FloatUnary {
                            width: crate::vm::native::ir::machine::MachineFloatWidth::F32,
                            op: crate::vm::native::ir::machine::MachineFloatUnaryOp::Abs,
                            dst: MachineReg(4),
                            src: MachineValue::Imm64(0),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::native::arch::NativeBackend::Arm64,
            crate::vm::backend::BackendConfig::new(3, 4),
            MachineModule {
                functions: vec![supported, unsupported],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: crate::vm::native::ir::runtime::MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        ctx.call_depth = 1;
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_multiblock_empty_root_function() {
        let function = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 4,
                blocks: vec![
                    MachineBlock {
                        id: MachineBlockId(0),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Jump(
                            crate::vm::native::ir::machine::MachineEdge {
                                target: MachineBlockId(1),
                                args: vec![],
                            },
                        ),
                    },
                    MachineBlock {
                        id: MachineBlockId(1),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Jump(
                            crate::vm::native::ir::machine::MachineEdge {
                                target: MachineBlockId(2),
                                args: vec![],
                            },
                        ),
                    },
                    MachineBlock {
                        id: MachineBlockId(2),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                ],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::native::arch::NativeBackend::Arm64,
            crate::vm::backend::BackendConfig::new(3, 4),
            MachineModule {
                functions: vec![function],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: crate::vm::native::ir::runtime::MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: crate::vm::native::ir::machine::MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 3,
                    call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                        base_slot: 0,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: None,
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        ctx.call_depth = 1;
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_nonfirst_function_entry() {
        let dummy = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 4,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let target = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(1),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 7,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![
                        MachineInst {
                            kind: MachineInstKind::Move {
                                dst: MachineReg(4),
                                src: MachineValue::Imm64(9),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                addr: crate::vm::native::ir::machine::MachineAddr {
                                    base: crate::vm::native::ir::machine::MACHINE_FP_REG,
                                    offset: 0,
                                },
                                width: crate::vm::native::ir::machine::MachineMemWidth::U64,
                                src: MachineValue::Reg(MachineReg(4)),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::native::arch::NativeBackend::Arm64,
            crate::vm::backend::BackendConfig::new(3, 4),
            MachineModule {
                functions: vec![dummy, target],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: crate::vm::native::ir::runtime::MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 4,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 1,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 1,
                        }),
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[1].clone().expect("entry");

        let mut stack = [0u64; 4];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        ctx.call_depth = 1;
        stack[1] = entry.root_return as u64;
        stack[2] = stack.as_mut_ptr() as u64;
        stack[3] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert_eq!(stack[0], 9);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_empty_root_with_jump_table_neighbor() {
        let empty = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 4,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let jumpy = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(1),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 5,
                blocks: vec![
                    MachineBlock {
                        id: MachineBlockId(0),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::JumpTable {
                            index: MachineValue::Imm64(0),
                            entries: vec![
                                crate::vm::native::ir::machine::MachineEdge {
                                    target: MachineBlockId(1),
                                    args: vec![],
                                },
                                crate::vm::native::ir::machine::MachineEdge {
                                    target: MachineBlockId(2),
                                    args: vec![],
                                },
                            ],
                        },
                    },
                    MachineBlock {
                        id: MachineBlockId(1),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                    MachineBlock {
                        id: MachineBlockId(2),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                ],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::native::arch::NativeBackend::Arm64,
            crate::vm::backend::BackendConfig::new(3, 4),
            MachineModule {
                functions: vec![empty, jumpy],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: crate::vm::native::ir::runtime::MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        ctx.call_depth = 1;
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_empty_root_with_indirect_call_neighbor() {
        let empty = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(0),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 4,
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let indirect = MachineFunction {
            id: crate::vm::native::ir::machine::MachineFuncId(1),
            program: crate::vm::native::ir::machine::MachineProgram {
                entry: MachineBlockId(0),
                reg_count: 5,
                blocks: vec![
                    MachineBlock {
                        id: MachineBlockId(0),
                        params: vec![],
                        ops: vec![MachineInst {
                            kind: MachineInstKind::Move {
                                dst: MachineReg(4),
                                src: MachineValue::Reg(
                                    crate::vm::native::ir::machine::MACHINE_FP_REG,
                                ),
                            },
                        }],
                        terminator: MachineTerminator::CallIndirect {
                            callee_target: MachineValue::Imm64(0),
                            callee_frame_base: MachineReg(4),
                            arg_slots: 0,
                            caller_result_base: 0,
                            continuation: MachineBlockId(1),
                        },
                    },
                    MachineBlock {
                        id: MachineBlockId(1),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                ],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::native::arch::NativeBackend::Arm64,
            crate::vm::backend::BackendConfig::new(3, 4),
            MachineModule {
                functions: vec![empty, indirect],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: crate::vm::native::ir::runtime::MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: crate::vm::native::ir::machine::MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(crate::vm::native::ir::runtime::MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        ctx.call_depth = 1;
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }
}
