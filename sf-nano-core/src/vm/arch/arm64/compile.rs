use alloc::{vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        entities::ModuleInst,
        machine::machine_ir::{
            MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
            MachineCompareKind, MachineConstId, MachineConvertOp, MachineEdge,
            MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth, MachineFuncId,
            MachineFunction, MachineFunctionRuntime, MachineHelperSymbol, MachineInst,
            MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
            MachineLoadExtension, MachineMemWidth, MachineProgram, MachineReg, MachineSign,
            MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
            MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG,
            MACHINE_MEM0_SIZE_REG,
        },
        runtime::{
            code::{Arm64CodePtr, Arm64RootEntry, CompiledNativeModule},
            context::ctx_offset,
            helpers::resolve_helper_entry,
        },
    },
};

use super::{
    abi::{
        emit_shared_epilogue, emit_shared_prologue, fp_machine_reg, inv_map_reg, map_fixed_reg,
        map_reg, max_fp_machine_regs, max_gp_mapped_regs, max_total_machine_regs,
        FP_MACHINE_REG_COUNT, FP_SCRATCH0, FP_SCRATCH1, FP_SCRATCH2, SCRATCH0, SCRATCH1,
    },
    arm64_raise_trap, arm64_raise_unsupported,
    emit::Arm64TextEmitter,
    enc::{self, Cond},
    reg::Arm64Reg,
};

// Re-export items from submodules that are needed by sibling submodules or externally.
use super::compile_fusion::{
    add_sub_imm_inst_32, add_sub_imm_inst_64, cmp_imm_inst, float_compare_branch_fusion,
    indexed_mem_fusion, int_binary_imm_inst, is_fallthrough_edge, logical_imm_inst_32,
    logical_imm_inst_64, mul_imm_inst_32, mul_imm_inst_64, uxtw_mem_fusion, value_is_reg,
    zero_store_pair_fusion,
};
use super::compile_helpers::{
    convert_op_code, convert_result_float_width,
    map_float_cond, map_int_cond, materialize_u64_into, mem_width_bytes, trap_code,
    trap_kind_index, MACHINE_TRAP_KIND_COUNT,
};

// Re-export pub(crate) items from compile_helpers so they remain
// accessible at the same visibility level as before the split.
pub(crate) use super::compile_helpers::{arm64_trapping_trunc, arm64_saturating_trunc, TruncResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LabelKind {
    Block,
    Edge,
    StackOverflow,
    ReturnOk,
    ReturnError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BranchFixupKind {
    B,
    BCond(Cond),
    Cbz(Arm64Reg),
    Cbnz(Arm64Reg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BranchFixup {
    inst_offset: usize,
    label: usize,
    kind: BranchFixupKind,
}

#[derive(Clone, Debug)]
pub(super) struct EdgeStub {
    label: usize,
    target: MachineBlockId,
    params: Vec<MachineBlockParam>,
    args: Vec<MachineValue>,
    arg_float_widths: Vec<Option<MachineFloatWidth>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LocalPtrPatch {
    pub(super) literal_offset: usize,
    pub(super) target_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PendingLocalPtrPatch {
    pub(super) literal_offset: usize,
    pub(super) target_label: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DirectCallPatch {
    pub(super) literal_offset: usize,
    pub(super) callee: MachineFuncId,
}

pub(crate) use crate::vm::debug::ir_dump::DebugRegion;

#[derive(Debug)]
struct FunctionArtifact {
    text: Arm64TextEmitter,
    local_ptr_patches: Vec<LocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    root_return_offset: usize,
    #[cfg(has_guard_pages)]
    return_error_offset: usize,
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

const ARM64_FUNCTION_INFO_SIZE: usize = core::mem::size_of::<Arm64FunctionInfo>();

#[derive(Clone, Debug)]
pub struct CompiledArm64Entry {
    pub entry: Arm64RootEntry,
    pub text_len: usize,
    pub debug_regions: Vec<DebugRegion>,
    pub root_return: Arm64CodePtr,
    #[cfg(has_guard_pages)]
    pub return_error: Arm64CodePtr,
}

#[derive(Debug)]
pub(super) struct FunctionCompiler<'a> {
    pub(super) compiled: &'a CompiledNativeModule,
    pub(super) function: &'a MachineFunction,
    pub(super) text: Arm64TextEmitter,
    pub(super) labels: Vec<Option<usize>>,
    pub(super) fixups: Vec<BranchFixup>,
    pub(super) block_labels: Vec<usize>,
    pub(super) edge_stubs: Vec<EdgeStub>,
    pub(super) resolved_ptr_patches: Vec<LocalPtrPatch>,
    pub(super) local_ptr_patches: Vec<PendingLocalPtrPatch>,
    pub(super) direct_call_patches: Vec<DirectCallPatch>,
    pub(super) function_table_patches: Vec<usize>,
    pub(super) deferred_traps: Vec<(usize, MachineTrapKind)>,
    pub(super) fp_reg_widths: [Option<MachineFloatWidth>; FP_MACHINE_REG_COUNT],
    pub(super) current_block: Option<MachineBlockId>,
    pub(super) current_op_index: Option<usize>,
    pub(super) current_edge_target: Option<MachineBlockId>,
    pub(super) stack_overflow_label: usize,
    pub(super) return_ok_label: usize,
    pub(super) return_error_label: usize,
    pub(super) shared_trap_labels: [Option<usize>; MACHINE_TRAP_KIND_COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IndexedMemFusion {
    Load {
        dst: MachineReg,
        base: MachineReg,
        index: MachineReg,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
        scaled: bool,
        /// Use UXTW addressing: `ldr Rt, [Xn, Wm, UXTW]`. The index register
        /// is treated as a 32-bit value that is zero-extended inline by the
        /// load instruction, eliminating an explicit I64ExtendI32U.
        uxtw: bool,
    },
    Store {
        base: MachineReg,
        index: MachineReg,
        width: MachineMemWidth,
        src: MachineValue,
        scaled: bool,
    },
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
        internal_entry_addrs.push(unsafe {
            base_ptr.add(*base_offset + artifacts[i].internal_entry_offset)
        } as usize);
    }

    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let function_base = base_offsets[index];
        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as u64;
            artifact.text.patch_u64(patch.literal_offset, target_addr);
        }
        for patch in &artifact.direct_call_patches {
            // Direct local calls use internal entry (skip prologue)
            let callee_addr = *internal_entry_addrs
                .get(patch.callee.0 as usize)
                .ok_or_else(|| {
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
        #[cfg(has_guard_pages)]
        let return_error = unsafe { executable.ptr(offset + artifact.return_error_offset) };
        entries.push(Some(CompiledArm64Entry {
            entry,
            root_return,
            #[cfg(has_guard_pages)]
            return_error,
            text_len,
            debug_regions,
        }));
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
                    let symbol =
                        alloc::format!("jit::{}::func{}::{}", module_name, func_idx, region.label);
                    crate::vm::runtime::profiler::record_function(region_start, code_bytes, &symbol);
                }
            }
        }
    }

    #[cfg(has_guard_pages)]
    {
        let ranges: Vec<_> = entries
            .iter()
            .flatten()
            .map(|e| {
                (
                    e.entry as usize,
                    e.entry as usize + e.text_len,
                    e.return_error as usize,
                )
            })
            .collect();
        crate::vm::runtime::trap_signal::register_jit_ranges(&ranges);
    }

    Ok(entries)
}

fn compile_function(
    compiled: &CompiledNativeModule,
    function: &MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    let max_reg = max_gp_mapped_regs();
    let max_total_reg = max_total_machine_regs();
    if function.program.reg_count as usize > max_total_reg {
        return Err(WasmError::invalid(alloc::format!(
            "arm64 MachineIR backend supports at most {} machine regs, got {} in function {}",
            max_total_reg,
            function.program.reg_count,
            function.id.0
        )));
    }
    if function.program.first_fp_reg < MACHINE_FIXED_REG_COUNT
        || function.program.first_fp_reg > function.program.reg_count
    {
        return Err(WasmError::invalid(alloc::format!(
            "arm64 MachineIR backend received invalid first_fp_reg {} for function {}",
            function.program.first_fp_reg,
            function.id.0,
        )));
    }
    if (function.program.reg_count - function.program.first_fp_reg) as usize > max_fp_machine_regs()
    {
        return Err(WasmError::invalid(alloc::format!(
            "arm64 MachineIR backend supports at most {} FP machine regs, got {} in function {}",
            max_fp_machine_regs(),
            function.program.reg_count - function.program.first_fp_reg,
            function.id.0,
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

    let block_layout = compiler.block_layout();
    for (index, block_id) in block_layout.iter().copied().enumerate() {
        let block = compiler
            .function
            .program
            .blocks
            .get(block_id.as_usize())
            .ok_or_else(|| {
                WasmError::internal("arm64 block layout references missing block".into())
            })?;
        let label = compiler.block_label(block.id)?;
        compiler.bind_label(label);
        let block_start = compiler.text.len();
        let fallthrough = block_layout.get(index + 1).copied();
        compiler.emit_block(block, fallthrough)?;
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
        compiler.current_block = None;
        compiler.current_op_index = None;
        compiler.current_edge_target = Some(edge.target);
        compiler.emit_parallel_moves(&edge.params, &edge.args, &edge.arg_float_widths)?;
        compiler.emit_branch_to_block(edge.target)?;
        compiler.current_edge_target = None;
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
    #[cfg(has_guard_pages)]
    let return_error_offset = compiler
        .labels
        .get(compiler.return_error_label)
        .and_then(|offset| *offset)
        .ok_or_else(|| WasmError::internal("arm64 return error label is unresolved".into()))?;
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
        #[cfg(has_guard_pages)]
        return_error_offset,
        internal_entry_offset,
        debug_regions,
    })
}

fn compile_unsupported_stub(
    func_id: MachineFuncId,
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
        #[cfg(has_guard_pages)]
        return_error_offset: 0,
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
        let return_ok_label = labels.len();
        labels.push(None);
        let return_error_label = labels.len();
        labels.push(None);
        let mut shared_trap_labels = [None; MACHINE_TRAP_KIND_COUNT];
        shared_trap_labels[trap_kind_index(MachineTrapKind::StackOverflow)] =
            Some(stack_overflow_label);
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
            fp_reg_widths: {
                let mut widths = [None; FP_MACHINE_REG_COUNT];
                if function.program.fp_reg_init_widths.is_empty() {
                    let fp_bank_count = function
                        .program
                        .reg_count
                        .saturating_sub(function.program.first_fp_reg)
                        as usize;
                    let transient_count = defaulted_fp_transient_count(&function.program);
                    for i in transient_count..fp_bank_count.min(FP_MACHINE_REG_COUNT) {
                        widths[i] = Some(MachineFloatWidth::F64);
                    }
                } else {
                    for (i, width) in function
                        .program
                        .fp_reg_init_widths
                        .iter()
                        .copied()
                        .enumerate()
                    {
                        widths[i] = width;
                    }
                }
                widths
            },
            current_block: None,
            current_op_index: None,
            current_edge_target: None,
            stack_overflow_label,
            return_ok_label,
            return_error_label,
            shared_trap_labels,
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

    fn emit_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.current_block = Some(block.id);
        self.current_edge_target = None;
        self.reset_block_fp_state(block)?;

        // ARM64-specific FloatCompare-and-branch fusion. ARM64's FCMP condition
        // codes handle NaN correctly for a single B.cond (unlike x86_64's UCOMISD
        // which needs multi-flag checks), so this is safe as an ARM64 backend opt.
        let fused_fcmp_cond = float_compare_branch_fusion(block, &self.function.program.blocks);

        let mut index = 0;
        while index < block.ops.len() {
            self.current_op_index = Some(index);
            // Float compare fusion: emit FCMP only (no CSET) for the last op.
            if fused_fcmp_cond.is_some() && index == block.ops.len() - 1 {
                if let MachineInstKind::FloatCompare {
                    width, lhs, rhs, ..
                } = &block.ops[index].kind
                {
                    self.emit_fcmp_values(*width, *lhs, *rhs)?;
                    index += 1;
                    continue;
                }
            }
            if let Some((base, imm7)) = zero_store_pair_fusion(block, index) {
                let base_reg = self.map_gp_reg(base)?;
                self.text
                    .emit_u32(enc::stp_64(Arm64Reg::Xzr, Arm64Reg::Xzr, base_reg, imm7));
                index += 2;
                continue;
            }
            if let Some((fused, skip)) = uxtw_mem_fusion(block, index) {
                self.emit_indexed_mem_fusion(fused)?;
                index += 1 + skip; // skip convert + (add+load fused pair)
                continue;
            }
            if let Some(fused) = indexed_mem_fusion(block, index) {
                self.emit_indexed_mem_fusion(fused)?;
                index += 2;
                continue;
            }
            self.emit_inst(&block.ops[index])?;
            index += 1;
        }
        self.current_op_index = None;
        let result = if let Some(cond) = fused_fcmp_cond {
            // FCMP flags already set. Emit B.cond directly.
            match &block.terminator {
                MachineTerminator::Branch {
                    then_edge,
                    else_edge,
                    ..
                } => self.emit_fused_cond_branch(cond, then_edge, else_edge, fallthrough),
                _ => unreachable!(),
            }
        } else {
            self.emit_terminator(&block.terminator, fallthrough)
        };
        self.current_block = None;
        result
    }

    fn emit_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src, ty } => self.emit_move(*ty, *dst, *src),
            MachineInstKind::FloatConst { width, dst, bits } => {
                self.emit_float_const(*width, *dst, *bits)
            }
            MachineInstKind::Lea { dst, addr } => {
                self.emit_addr_into(self.map_gp_reg(*dst)?, *addr)
            }
            MachineInstKind::Load {
                ty: _,
                dst,
                addr,
                width,
                extension,
            } => self.emit_load(*dst, *addr, *width, *extension),
            MachineInstKind::Store {
                ty: _,
                addr,
                width,
                src,
            } => self.emit_store(*addr, *width, *src),
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
            MachineInstKind::IntMulWide { .. } => Err(WasmError::internal(
                "arm64 backend received IntMulWide; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::Int64PairBinary { .. } => Err(WasmError::internal(
                "arm64 backend received Int64PairBinary; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::Int64PairUnary { .. } => Err(WasmError::internal(
                "arm64 backend received Int64PairUnary; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::Int64PairDivRem { .. } => Err(WasmError::internal(
                "arm64 backend received Int64PairDivRem; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::Int64PairShift { .. } => Err(WasmError::internal(
                "arm64 backend received Int64PairShift; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => self.emit_int_compare(*width, *kind, *sign, *dst, *lhs, *rhs),
            MachineInstKind::Select {
                ty,
                dst,
                on_true,
                on_false,
                cond,
                ..
            } => self.emit_select(*ty, *dst, *on_true, *on_false, *cond),
            MachineInstKind::TrapIf { kind, cond } => self.emit_trap_if(*kind, cond),
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
            MachineInstKind::ConvertI64PairToFloat { .. } => Err(WasmError::internal(
                "arm64 backend received ConvertI64PairToFloat; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::Int64PairCompare { .. } => Err(WasmError::internal(
                "arm64 backend received Int64PairCompare; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::ConvertFloatToI64Pair { .. } => Err(WasmError::internal(
                "arm64 backend received ConvertFloatToI64Pair; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::ReinterpretF64ToI64Pair { .. } => Err(WasmError::internal(
                "arm64 backend received ReinterpretF64ToI64Pair; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
            MachineInstKind::ReinterpretI64PairToF64 { .. } => Err(WasmError::internal(
                "arm64 backend received ReinterpretI64PairToF64; 32-bit legalized MachineIR should not reach arm64 codegen".into(),
            )),
        }
    }

    fn reset_block_fp_state(&mut self, block: &MachineBlock) -> Result<(), WasmError> {
        for i in 0..defaulted_fp_transient_count(&self.function.program) {
            self.fp_reg_widths[i] = None;
        }
        for param in &block.params {
            if let Some(width) = param.ty.float_width() {
                self.set_fp_reg_width(param.reg, width)?;
            }
        }
        Ok(())
    }

    pub(super) fn prepare_float_operand(
        &mut self,
        width: MachineFloatWidth,
        value: MachineValue,
        gp_scratch: Arm64Reg,
        fp_scratch: u32,
    ) -> Result<u32, WasmError> {
        if let MachineValue::Reg(reg) = value {
            if self.is_fp_reg(reg) {
                return Ok(self.map_fp_reg(reg)?);
            }
        }

        let gp = self.materialize_value(gp_scratch, value)?;
        match width {
            MachineFloatWidth::F32 => self.text.emit_u32(enc::fmov_s_from_gp(fp_scratch, gp)),
            MachineFloatWidth::F64 => self.text.emit_u32(enc::fmov_d_from_gp(fp_scratch, gp)),
        };
        Ok(fp_scratch)
    }

    pub(super) fn ensure_trap_label(&mut self, kind: MachineTrapKind) -> usize {
        let slot = trap_kind_index(kind);
        if let Some(label) = self.shared_trap_labels[slot] {
            return label;
        }
        let label = self.new_label(LabelKind::Edge);
        self.shared_trap_labels[slot] = Some(label);
        self.deferred_traps.push((label, kind));
        label
    }

    pub(super) fn runtime_for(
        &self,
        func_id: MachineFuncId,
    ) -> Result<&MachineFunctionRuntime, WasmError> {
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

    pub(super) fn is_fp_reg(&self, reg: MachineReg) -> bool {
        self.function.program.is_fp_reg(reg)
    }

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<Arm64Reg, WasmError> {
        if self.is_fp_reg(reg) {
            return Err(WasmError::invalid(alloc::format!(
                "arm64 MachineIR backend expected GP register, got FP machine reg {}",
                reg.0
            )));
        }
        map_reg(reg)
    }

    pub(super) fn map_fp_reg(&self, reg: MachineReg) -> Result<u32, WasmError> {
        let Some(index) = reg.0.checked_sub(self.function.program.first_fp_reg) else {
            return Err(WasmError::invalid(alloc::format!(
                "arm64 MachineIR backend expected FP register, got machine reg {}",
                reg.0
            )));
        };
        fp_machine_reg(index as usize).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 MachineIR backend has no physical FP mapping for machine reg {}",
                reg.0
            ))
        })
    }

    pub(super) fn set_fp_reg_width(
        &mut self,
        reg: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        let index = reg
            .0
            .checked_sub(self.function.program.first_fp_reg)
            .ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "arm64 MachineIR backend expected FP register, got machine reg {}",
                    reg.0
                ))
            })? as usize;
        let slot = self.fp_reg_widths.get_mut(index).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "arm64 MachineIR backend has no tracked FP slot for machine reg {}",
                reg.0
            ))
        })?;
        *slot = Some(width);
        Ok(())
    }

    pub(super) fn fp_reg_width(&self, reg: MachineReg) -> Result<MachineFloatWidth, WasmError> {
        let index = reg
            .0
            .checked_sub(self.function.program.first_fp_reg)
            .ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "arm64 MachineIR backend expected FP register, got machine reg {}",
                    reg.0
                ))
            })? as usize;
        self.fp_reg_widths
            .get(index)
            .and_then(|width| *width)
            .ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "arm64 MachineIR backend is missing float-width tracking for machine reg {} in function {} at {}",
                    reg.0,
                    self.function.id.0,
                    self.current_location(),
                ))
            })
    }

    fn current_location(&self) -> alloc::string::String {
        if let Some(target) = self.current_edge_target {
            return alloc::format!("edge stub to b{}", target.0);
        }
        if let Some(block) = self.current_block {
            if let Some(op_index) = self.current_op_index {
                return alloc::format!("b{} op{}", block.0, op_index);
            }
            return alloc::format!("b{} terminator", block.0);
        }
        alloc::format!("unknown location")
    }

    fn emit_parallel_moves(
        &mut self,
        params: &[MachineBlockParam],
        args: &[MachineValue],
        arg_float_widths: &[Option<MachineFloatWidth>],
    ) -> Result<(), WasmError> {
        let mut pending = Vec::new();
        for ((&dst, &arg), &float_width) in
            params.iter().zip(args.iter()).zip(arg_float_widths.iter())
        {
            let src = match arg {
                MachineValue::Reg(reg) => ParallelSource::Reg { reg, float_width },
                MachineValue::Imm64(value) => ParallelSource::Imm(value),
            };
            if matches!(src, ParallelSource::Reg { reg, .. } if reg == dst.reg) {
                continue;
            }
            pending.push((dst, src));
        }

        while !pending.is_empty() {
            let mut ready = None;
            for index in 0..pending.len() {
                let dst = pending[index].0.reg;
                let blocked = pending.iter().enumerate().any(|(other_index, (_, src))| {
                    other_index != index
                        && matches!(src, ParallelSource::Reg { reg, .. } if *reg == dst)
                });
                if !blocked {
                    ready = Some(index);
                    break;
                }
            }
            if let Some(index) = ready {
                let (dst, src) = pending.remove(index);
                self.emit_source_move(dst, src)?;
                continue;
            }

            let (dst, src) = pending.remove(0);
            let ParallelSource::Reg {
                reg: src_reg,
                float_width,
            } = src
            else {
                self.emit_source_move(dst, src)?;
                continue;
            };
            if dst.ty.is_fp() {
                let dst_fp = self.map_fp_reg(dst.reg)?;
                let width = dst.ty.float_width().expect("FP param width");
                self.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s(FP_SCRATCH2, dst_fp),
                    MachineFloatWidth::F64 => enc::fmov_d(FP_SCRATCH2, dst_fp),
                });
                self.emit_source_move(
                    dst,
                    ParallelSource::Reg {
                        reg: src_reg,
                        float_width,
                    },
                )?;
            } else {
                let dst_gp = self.map_gp_reg(dst.reg)?;
                let src_gp = self.map_gp_reg(src_reg)?;
                self.text.emit_u32(enc::mov_reg_64(SCRATCH1, dst_gp));
                self.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
            }
            for (_, source) in pending.iter_mut() {
                if matches!(*source, ParallelSource::Reg { reg, .. } if reg == dst.reg) {
                    *source = if dst.ty.is_fp() {
                        ParallelSource::FpTemp(dst.ty.float_width().expect("FP temp width"))
                    } else {
                        ParallelSource::GpTemp
                    };
                }
            }
        }
        Ok(())
    }

    fn emit_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        match src {
            ParallelSource::Reg {
                reg: src_reg,
                float_width: src_float_width,
            } => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)?;
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)?;
                        self.text.emit_u32(match width {
                            MachineFloatWidth::F32 => enc::fmov_s(dst_fp, src_fp),
                            MachineFloatWidth::F64 => enc::fmov_d(dst_fp, src_fp),
                        });
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        self.text.emit_u32(match width {
                            MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, src_gp),
                            MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, src_gp),
                        });
                    }
                    self.set_fp_reg_width(dst.reg, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst.reg)?;
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)?;
                        match src_float_width.ok_or_else(|| {
                            WasmError::invalid(alloc::format!(
                                "arm64 edge move is missing float-width metadata for machine reg {}",
                                src_reg.0
                            ))
                        })? {
                            MachineFloatWidth::F32 => self.text.emit_u32(enc::fmov_gp_from_s(dst_gp, src_fp)),
                            MachineFloatWidth::F64 => self.text.emit_u32(enc::fmov_gp_from_d(dst_gp, src_fp)),
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        self.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
                    }
                }
            }
            ParallelSource::Imm(value) => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)?;
                    self.materialize_u64(SCRATCH0, value);
                    self.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmov_s_from_gp(dst_fp, SCRATCH0),
                        MachineFloatWidth::F64 => enc::fmov_d_from_gp(dst_fp, SCRATCH0),
                    });
                    self.set_fp_reg_width(dst.reg, width)?;
                } else {
                    self.materialize_u64(self.map_gp_reg(dst.reg)?, value);
                }
            }
            ParallelSource::GpTemp => {
                self.text
                    .emit_u32(enc::mov_reg_64(self.map_gp_reg(dst.reg)?, SCRATCH1));
            }
            ParallelSource::FpTemp(width) => {
                let dst_fp = self.map_fp_reg(dst.reg)?;
                self.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmov_s(dst_fp, FP_SCRATCH2),
                    MachineFloatWidth::F64 => enc::fmov_d(dst_fp, FP_SCRATCH2),
                });
                self.set_fp_reg_width(dst.reg, width)?;
            }
        }
        Ok(())
    }

    pub(super) fn materialize_value(
        &mut self,
        scratch: Arm64Reg,
        value: MachineValue,
    ) -> Result<Arm64Reg, WasmError> {
        match value {
            MachineValue::Reg(reg) if self.is_fp_reg(reg) => {
                let src_fp = self.map_fp_reg(reg)?;
                match self.fp_reg_width(reg)? {
                    MachineFloatWidth::F32 => {
                        self.text.emit_u32(enc::fmov_gp_from_s(scratch, src_fp));
                    }
                    MachineFloatWidth::F64 => {
                        self.text.emit_u32(enc::fmov_gp_from_d(scratch, src_fp));
                    }
                };
                Ok(scratch)
            }
            MachineValue::Reg(reg) => self.map_gp_reg(reg),
            MachineValue::Imm64(value) => {
                self.materialize_u64(scratch, value);
                Ok(scratch)
            }
        }
    }

    pub(super) fn materialize_u64(&mut self, dst: Arm64Reg, value: u64) {
        materialize_u64_into(&mut self.text, dst, value);
    }

    /// Check if an edge is a no-op (args match params exactly, no copies needed).
    pub(super) fn is_identity_edge(&self, target: MachineBlockId, args: &[MachineValue]) -> bool {
        let Some(block) = self.function.program.blocks.get(target.as_usize()) else {
            return false;
        };
        if block.params.len() != args.len() {
            return false;
        }
        block
            .params
            .iter()
            .zip(args.iter())
            .all(|(param, arg)| matches!(arg, MachineValue::Reg(r) if *r == param.reg))
    }

    fn block_layout(&self) -> Vec<MachineBlockId> {
        let mut order = Vec::with_capacity(self.function.program.blocks.len());
        let mut seen = vec![false; self.function.program.blocks.len()];
        let mut worklist = vec![self.function.program.entry];

        while let Some(start) = worklist.pop() {
            self.extend_block_trace(start, &mut seen, &mut order, &mut worklist);
        }

        for block in &self.function.program.blocks {
            if seen[block.id.as_usize()] {
                continue;
            }
            worklist.push(block.id);
            while let Some(start) = worklist.pop() {
                self.extend_block_trace(start, &mut seen, &mut order, &mut worklist);
            }
        }

        order
    }

    fn extend_block_trace(
        &self,
        start: MachineBlockId,
        seen: &mut [bool],
        order: &mut Vec<MachineBlockId>,
        worklist: &mut Vec<MachineBlockId>,
    ) {
        let mut current = Some(start);
        while let Some(block_id) = current {
            let Some(block) = self.function.program.blocks.get(block_id.as_usize()) else {
                break;
            };
            if seen[block_id.as_usize()] {
                break;
            }
            seen[block_id.as_usize()] = true;
            order.push(block_id);

            let mut fallthrough = None;
            match &block.terminator {
                MachineTerminator::Jump(edge) => {
                    if self.is_identity_edge(edge.target, &edge.args) {
                        fallthrough = Some(edge.target);
                    } else {
                        worklist.push(edge.target);
                    }
                }
                MachineTerminator::Branch {
                    then_edge,
                    else_edge,
                    ..
                } => {
                    if self.is_identity_edge(else_edge.target, &else_edge.args) {
                        fallthrough = Some(else_edge.target);
                        worklist.push(then_edge.target);
                    } else {
                        worklist.push(else_edge.target);
                        worklist.push(then_edge.target);
                    }
                }
                MachineTerminator::JumpTable { entries, .. } => {
                    for edge in entries.iter().rev() {
                        worklist.push(edge.target);
                    }
                }
                MachineTerminator::CallDirect { continuation, .. }
                | MachineTerminator::CallIndirect { continuation, .. } => {
                    fallthrough = Some(*continuation);
                }
                MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
            }

            current = fallthrough.filter(|target| !seen[target.as_usize()]);
        }
    }

    /// For identity/empty edges, branch directly to the target block.
    /// For edges needing copies, create an edge stub.
    pub(super) fn emit_edge(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        if self.is_identity_edge(target, args) {
            // No copies needed -- branch directly to the target block.
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
        let arg_float_widths = args
            .iter()
            .map(|arg| match arg {
                MachineValue::Reg(reg) if self.is_fp_reg(*reg) => self.fp_reg_width(*reg).map(Some),
                MachineValue::Reg(_) | MachineValue::Imm64(_) => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.edge_stubs.push(EdgeStub {
            label,
            target,
            params: block.params.clone(),
            args: args.to_vec(),
            arg_float_widths,
        });
        Ok(label)
    }

    fn emit_branch_to_block(&mut self, target: MachineBlockId) -> Result<(), WasmError> {
        let label = self.block_label(target)?;
        self.emit_b(label);
        Ok(())
    }

    pub(super) fn block_label(&self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.block_labels
            .get(target.0 as usize)
            .copied()
            .filter(|label| *label != usize::MAX)
            .ok_or_else(|| WasmError::internal("arm64 block label is out of range".into()))
    }

    pub(super) fn new_label(&mut self, _kind: LabelKind) -> usize {
        let label = self.labels.len();
        self.labels.push(None);
        label
    }

    pub(super) fn bind_label(&mut self, label: usize) {
        self.labels[label] = Some(self.text.len());
    }

    pub(super) fn emit_b(&mut self, label: usize) {
        let inst_offset = self.text.emit_u32(enc::b(0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::B,
        });
    }

    pub(super) fn emit_b_cond(&mut self, cond: Cond, label: usize) {
        let inst_offset = self.text.emit_u32(enc::b_cond(cond, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::BCond(cond),
        });
    }

    pub(super) fn emit_cbnz(&mut self, reg: Arm64Reg, label: usize) {
        let inst_offset = self.text.emit_u32(enc::cbnz_64(reg, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::Cbnz(reg),
        });
    }

    pub(super) fn emit_cbz(&mut self, reg: Arm64Reg, label: usize) {
        let inst_offset = self.text.emit_u32(enc::cbz_64(reg, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::Cbz(reg),
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

fn defaulted_fp_transient_count(program: &MachineProgram) -> usize {
    if program.fp_transient_count != 0 {
        return program.fp_transient_count as usize;
    }
    let fp_bank_count = program.reg_count.saturating_sub(program.first_fp_reg) as usize;
    fp_bank_count.min(2)
}

#[derive(Clone, Copy)]
enum ParallelSource {
    Reg {
        reg: MachineReg,
        float_width: Option<MachineFloatWidth>,
    },
    Imm(u64),
    GpTemp,
    FpTemp(MachineFloatWidth),
}

impl From<u64> for ParallelSource {
    fn from(value: u64) -> Self {
        Self::Imm(value)
    }
}
