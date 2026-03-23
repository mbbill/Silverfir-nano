//! x86_64 backend: compile MachineIR to x86_64 machine code.

use alloc::{string::String, vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        entities::ModuleInst,
        machine::machine_ir::{
            MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
            MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp,
            MachineFloatWidth, MachineFunction, MachineHelperSymbol, MachineInst, MachineInstKind,
            MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension,
            MachineMemWidth, MachineProgram, MachineReg, MachineSign, MachineStorageType,
            MachineTerminator, MachineTrapKind, MachineValue, MACHINE_CTX_REG,
            MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG,
            MACHINE_MEM0_SIZE_REG,
        },
        runtime::{
            code::{CompiledNativeModule, X86_64CodePtr, X86_64RootEntry},
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
    emit::X86_64TextEmitter,
    enc::{self, Cc},
    reg::X86Reg,
    x86_64_raise_trap, x86_64_raise_unsupported,
};

// Re-export items from compile_helpers that are used by sibling modules.
pub(super) use super::compile_helpers::{
    convert_op_code, convert_result_float_width, defaulted_fp_transient_count,
    is_fallthrough_edge, map_float_cond, map_int_cond, trap_code, trap_kind_index,
    x86_64_saturating_trunc, x86_64_trapping_trunc, MACHINE_TRAP_KIND_COUNT, ParallelSource,
};

// Call depth checking removed: the stack overflow check alone limits recursion.

// ─── Label & fixup types ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LabelKind {
    Block,
    Edge,
    StackOverflow,
    ReturnOk,
    ReturnError,
}

/// x86_64 branch fixup: we emit a JMP/Jcc with a placeholder rel32, then
/// patch it once the target label is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BranchFixup {
    /// Byte offset of the rel32 field in the text.
    rel32_offset: usize,
    /// Label index to resolve.
    label: usize,
}

#[derive(Clone, Debug)]
struct EdgeStub {
    label: usize,
    target: MachineBlockId,
    params: Vec<MachineBlockParam>,
    args: Vec<MachineValue>,
    arg_float_widths: Vec<Option<MachineFloatWidth>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LocalPtrPatch {
    pub literal_offset: usize,
    pub target_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PendingLocalPtrPatch {
    pub literal_offset: usize,
    pub target_label: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DirectCallPatch {
    pub literal_offset: usize,
    pub callee: crate::vm::machine::machine_ir::MachineFuncId,
}

pub use crate::vm::debug::ir_dump::DebugRegion;

#[derive(Debug)]
struct FunctionArtifact {
    text: X86_64TextEmitter,
    local_ptr_patches: Vec<LocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    root_return_offset: usize,
    #[cfg(has_guard_pages)]
    return_error_offset: usize,
    internal_entry_offset: usize,
    debug_regions: Vec<DebugRegion>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct X86_64FunctionInfo {
    entry: u64,
    total_frame_bytes: u64,
    frame_prefix_slots: u64,
    call_scratch_base_slot: u64,
}

const X86_64_FUNCTION_INFO_SIZE: usize = core::mem::size_of::<X86_64FunctionInfo>();

/// Result of compiling one function to x86_64 machine code.
#[derive(Clone, Debug)]
pub struct CompiledX86_64Entry {
    pub entry: X86_64RootEntry,
    pub text_len: usize,
    pub debug_regions: Vec<DebugRegion>,
    pub root_return: X86_64CodePtr,
    #[cfg(has_guard_pages)]
    pub return_error: X86_64CodePtr,
}

// ─── FunctionCompiler ────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct FunctionCompiler<'a> {
    pub(super) compiled: &'a CompiledNativeModule,
    pub(super) function: &'a MachineFunction,
    pub(super) text: X86_64TextEmitter,
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

// ─── compile_module ──────────────────────────────────────────────────────────

pub fn compile_module(
    module: &ModuleInst,
    compiled: &CompiledNativeModule,
) -> Result<Vec<Option<CompiledX86_64Entry>>, WasmError> {
    let mut artifacts = Vec::with_capacity(compiled.module().functions.len());
    for function in &compiled.module().functions {
        match compile_function(compiled, function) {
            Ok(artifact) => artifacts.push(artifact),
            Err(err) => return Err(err),
        }
    }

    // Compute base offsets for each function in the code buffer.
    let mut base_offsets = Vec::with_capacity(artifacts.len());
    let mut running_offset = 0usize;
    for artifact in &artifacts {
        running_offset = page_align_function(running_offset, artifact.text.len());
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

    // Patch local pointer literals and direct call literals.
    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let function_base = base_offsets[index];
        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as u64;
            artifact.text.patch_u64(patch.literal_offset, target_addr);
        }
        for patch in &artifact.direct_call_patches {
            let callee_addr = *internal_entry_addrs
                .get(patch.callee.0 as usize)
                .ok_or_else(|| {
                    WasmError::internal("x86_64 direct callee address is out of range".into())
                })? as u64;
            artifact.text.patch_u64(patch.literal_offset, callee_addr);
        }
        for &literal_offset in &artifact.function_table_patches {
            artifact.text.patch_u64(literal_offset, unsafe {
                base_ptr.add(function_info_table_offset)
            } as u64);
        }
    }

    // Build function info table.
    let mut function_info_bytes = Vec::with_capacity(artifacts.len() * X86_64_FUNCTION_INFO_SIZE);
    for (func_idx, runtime) in compiled.runtime().functions.iter().enumerate() {
        let info = X86_64FunctionInfo {
            entry: *internal_entry_addrs.get(func_idx).ok_or_else(|| {
                WasmError::internal("x86_64 function entry is out of range".into())
            })? as u64,
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

    // Emit into executable code buffer.
    let mut executable = module
        .native_code_buffer()
        .map_err(|err| WasmError::internal(err.into()))?;
    executable.begin_write();
    executable.reset();

    let written_start = executable.len();
    let mut entries = Vec::with_capacity(artifacts.len());
    for (func_idx, artifact) in artifacts.into_iter().enumerate() {
        // Emit INT3 padding to match the aligned base_offsets.
        let current = executable.len() - written_start;
        let expected = base_offsets[func_idx];
        debug_assert!(expected >= current);
        let padding = expected - current;
        if padding > 0 {
            const INT3: u8 = 0xCC;
            for _ in 0..padding {
                executable.emit_bytes(&[INT3]);
            }
        }
        let text_bytes = artifact.text.finish();
        let text_len = text_bytes.len();
        let debug_regions = artifact.debug_regions;
        let offset = executable.emit_bytes(&text_bytes);
        let entry = unsafe { executable.fn_ptr::<X86_64RootEntry>(offset) };
        let root_return = unsafe { executable.ptr(offset + artifact.root_return_offset) };
        #[cfg(has_guard_pages)]
        let return_error = unsafe { executable.ptr(offset + artifact.return_error_offset) };
        entries.push(Some(CompiledX86_64Entry {
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

    // Record per-block JIT symbols for profiling.
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

// ─── compile_function ────────────────────────────────────────────────────────

fn compile_function(
    compiled: &CompiledNativeModule,
    function: &MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    let max_reg = max_gp_mapped_regs();
    let max_total_reg = max_total_machine_regs();
    if function.program.reg_count as usize > max_total_reg {
        return Err(WasmError::invalid(alloc::format!(
            "x86_64 MachineIR backend supports at most {} machine regs, got {} in function {}",
            max_total_reg,
            function.program.reg_count,
            function.id.0
        )));
    }
    if function.program.first_fp_reg < MACHINE_FIXED_REG_COUNT
        || function.program.first_fp_reg > function.program.reg_count
    {
        return Err(WasmError::invalid(alloc::format!(
            "x86_64 MachineIR backend received invalid first_fp_reg {} for function {}",
            function.program.first_fp_reg,
            function.id.0,
        )));
    }
    if (function.program.reg_count - function.program.first_fp_reg) as usize > max_fp_machine_regs()
    {
        return Err(WasmError::invalid(alloc::format!(
            "x86_64 MachineIR backend supports at most {} FP machine regs, got {} in function {}",
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
            "x86_64 MachineIR backend does not support entry block params yet".into(),
        ));
    }

    let mut compiler = FunctionCompiler::new(compiled, function);
    let mut debug_regions = Vec::new();

    // Prologue
    let prologue_start = compiler.text.len();
    compiler.emit_prologue();
    let internal_entry_offset = compiler.text.len();
    debug_regions.push(DebugRegion {
        offset: prologue_start,
        len: internal_entry_offset - prologue_start,
        label: alloc::format!("prologue"),
    });

    // Blocks
    let block_layout = compiler.block_layout();
    for (index, block_id) in block_layout.iter().copied().enumerate() {
        let block = compiler
            .function
            .program
            .blocks
            .get(block_id.as_usize())
            .ok_or_else(|| {
                WasmError::internal("x86_64 block layout references missing block".into())
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

    // Edge stubs
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

    // Tail: return_ok, stack_overflow, return_error, deferred traps
    let tail_start = compiler.text.len();
    compiler.bind_label(compiler.return_ok_label);
    enc::mov_ri_32(&mut compiler.text, X86Reg::RAX, 0); // status = 0
    compiler.emit_epilogue();

    compiler.bind_label(compiler.stack_overflow_label);
    compiler.emit_trap(MachineTrapKind::StackOverflow);

    compiler.bind_label(compiler.return_error_label);
    compiler.emit_epilogue();

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

    // Patch branch fixups
    compiler.patch_fixups()?;
    let root_return_offset = compiler
        .labels
        .get(compiler.return_ok_label)
        .and_then(|offset| *offset)
        .ok_or_else(|| WasmError::internal("x86_64 root return label is unresolved".into()))?;
    #[cfg(has_guard_pages)]
    let return_error_offset = compiler
        .labels
        .get(compiler.return_error_label)
        .and_then(|offset| *offset)
        .ok_or_else(|| WasmError::internal("x86_64 return error label is unresolved".into()))?;
    let mut local_ptr_patches = compiler.resolved_ptr_patches;
    local_ptr_patches.reserve(compiler.local_ptr_patches.len());
    for patch in compiler.local_ptr_patches {
        let target_offset = compiler
            .labels
            .get(patch.target_label)
            .and_then(|offset| *offset)
            .ok_or_else(|| {
                WasmError::internal("x86_64 local continuation label is unresolved".into())
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

// ─── FunctionCompiler impl ───────────────────────────────────────────────────

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
            text: X86_64TextEmitter::new(),
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

    // ── Prologue / epilogue ──────────────────────────────────────────────────

    fn emit_prologue(&mut self) {
        emit_shared_prologue(&mut self.text);
        // System V AMD64: RDI = ctx, RSI = fp. Move to fixed regs.
        // MOV RBX, RDI  (ctx → MACHINE_CTX_REG)
        enc::mov_rr_64(&mut self.text, map_fixed_reg(MACHINE_CTX_REG), X86Reg::RDI);
        // MOV RBP, RSI  (fp → MACHINE_FP_REG)
        enc::mov_rr_64(&mut self.text, map_fixed_reg(MACHINE_FP_REG), X86Reg::RSI);
        // Load mem0_base from ctx
        enc::load_64(
            &mut self.text,
            map_fixed_reg(MACHINE_MEM0_BASE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_BASE as i32,
        );
        // Load mem0_size from ctx
        enc::load_64(
            &mut self.text,
            map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_SIZE as i32,
        );
    }

    fn emit_epilogue(&mut self) {
        emit_shared_epilogue(&mut self.text);
    }

    // ── Block emission ───────────────────────────────────────────────────────

    fn emit_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.current_block = Some(block.id);
        self.current_edge_target = None;
        self.reset_block_fp_state(block)?;
        let mut index = 0;
        while index < block.ops.len() {
            self.current_op_index = Some(index);
            self.emit_inst(&block.ops[index])?;
            index += 1;
        }
        self.current_op_index = None;
        let result = self.emit_terminator(&block.terminator, fallthrough);
        self.current_block = None;
        result
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

    // ── Instruction emission dispatch ────────────────────────────────────────

    fn emit_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { dst, src, ty } => self.emit_move(*ty, *dst, *src),
            MachineInstKind::FloatConst { width, dst, bits } => {
                self.emit_float_const(*width, *dst, *bits)
            }
            MachineInstKind::Lea { dst, addr } => self.emit_lea(*dst, *addr),
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
                "x86_64 backend received IntMulWide; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairBinary { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairBinary; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairUnary { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairUnary; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairDivRem { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairDivRem; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairShift { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairShift; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
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
                "x86_64 backend received ConvertI64PairToFloat; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairCompare { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairCompare; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ConvertFloatToI64Pair { .. } => Err(WasmError::internal(
                "x86_64 backend received ConvertFloatToI64Pair; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ReinterpretF64ToI64Pair { .. } => Err(WasmError::internal(
                "x86_64 backend received ReinterpretF64ToI64Pair; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ReinterpretI64PairToF64 { .. } => Err(WasmError::internal(
                "x86_64 backend received ReinterpretI64PairToF64; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
        }
    }

    // ── Terminator emission ──────────────────────────────────────────────────

    fn emit_terminator(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => {
                if is_fallthrough_edge(self, edge.target, &edge.args, fallthrough) {
                    return Ok(());
                }
                let label = self.emit_edge(edge.target, &edge.args)?;
                self.emit_jmp(label);
                Ok(())
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.emit_branch(cond, then_edge, else_edge, fallthrough),
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

    // ── Trap emission ────────────────────────────────────────────────────────

    pub(super) fn emit_trap(&mut self, kind: MachineTrapKind) {
        // MOV RDI, ctx (arg0)
        enc::mov_rr_64(&mut self.text, X86Reg::RDI, map_fixed_reg(MACHINE_CTX_REG));
        // MOV RSI, trap_code (arg1)
        self.materialize_u64(X86Reg::RSI, trap_code(kind));
        // MOV R11, x86_64_raise_trap
        self.materialize_u64(SCRATCH1, x86_64_raise_trap as usize as u64);
        // CALL R11
        enc::call_reg(&mut self.text, SCRATCH1);
        // JMP return_error_label
        self.emit_jmp(self.return_error_label);
    }

    fn emit_trap_if(
        &mut self,
        kind: MachineTrapKind,
        cond: &MachineBranchCond,
    ) -> Result<(), WasmError> {
        let trap_label = self.ensure_trap_label(kind);
        self.emit_branch_if(cond, trap_label)
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

    // ── Label / fixup infrastructure ─────────────────────────────────────────

    pub(super) fn new_label(&mut self, _kind: LabelKind) -> usize {
        let label = self.labels.len();
        self.labels.push(None);
        label
    }

    pub(super) fn bind_label(&mut self, label: usize) {
        self.labels[label] = Some(self.text.len());
    }

    pub(super) fn block_label(&self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.block_labels
            .get(target.0 as usize)
            .copied()
            .filter(|label| *label != usize::MAX)
            .ok_or_else(|| WasmError::internal("x86_64 block label is out of range".into()))
    }

    /// Emit JMP rel32 with a fixup to be patched later.
    pub(super) fn emit_jmp(&mut self, label: usize) {
        let rel32_offset = enc::jmp_rel32(&mut self.text);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    /// Emit Jcc rel32 with a fixup to be patched later.
    pub(super) fn emit_jcc(&mut self, cc: Cc, label: usize) {
        let rel32_offset = enc::jcc_rel32(&mut self.text, cc);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target = self
                .labels
                .get(fixup.label)
                .and_then(|value| *value)
                .ok_or_else(|| {
                    WasmError::internal("x86_64 branch target label is unresolved".into())
                })?;
            enc::patch_rel32(&mut self.text, fixup.rel32_offset, target);
        }
        Ok(())
    }

    // ── Register mapping ─────────────────────────────────────────────────────

    pub(super) fn is_fp_reg(&self, reg: MachineReg) -> bool {
        self.function.program.is_fp_reg(reg)
    }

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<X86Reg, WasmError> {
        if self.is_fp_reg(reg) {
            return Err(WasmError::invalid(alloc::format!(
                "x86_64 MachineIR backend expected GP register, got FP machine reg {}",
                reg.0
            )));
        }
        map_reg(reg)
    }

    pub(super) fn map_fp_reg(&self, reg: MachineReg) -> Result<u32, WasmError> {
        let Some(index) = reg.0.checked_sub(self.function.program.first_fp_reg) else {
            return Err(WasmError::invalid(alloc::format!(
                "x86_64 MachineIR backend expected FP register, got machine reg {}",
                reg.0
            )));
        };
        fp_machine_reg(index as usize).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "x86_64 MachineIR backend has no physical FP mapping for machine reg {}",
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
                    "x86_64 MachineIR backend expected FP register, got machine reg {}",
                    reg.0
                ))
            })? as usize;
        let slot = self.fp_reg_widths.get_mut(index).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "x86_64 MachineIR backend has no tracked FP slot for machine reg {}",
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
                    "x86_64 MachineIR backend expected FP register, got machine reg {}",
                    reg.0
                ))
            })? as usize;
        self.fp_reg_widths
            .get(index)
            .and_then(|width| *width)
            .ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "x86_64 MachineIR backend is missing float-width tracking for machine reg {} in function {} at {}",
                    reg.0,
                    self.function.id.0,
                    self.current_location(),
                ))
            })
    }

    fn current_location(&self) -> String {
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

    // ── Value materialization ────────────────────────────────────────────────

    pub(super) fn materialize_value(
        &mut self,
        scratch: X86Reg,
        value: MachineValue,
    ) -> Result<X86Reg, WasmError> {
        match value {
            MachineValue::Reg(reg) if self.is_fp_reg(reg) => {
                let src_fp = self.map_fp_reg(reg)?;
                match self.fp_reg_width(reg)? {
                    MachineFloatWidth::F32 => {
                        enc::movd_r32_xmm(&mut self.text, scratch, src_fp as u8);
                    }
                    MachineFloatWidth::F64 => {
                        enc::movq_r64_xmm(&mut self.text, scratch, src_fp as u8);
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

    pub(super) fn materialize_u64(&mut self, dst: X86Reg, value: u64) {
        if value == 0 {
            // XOR r32, r32 is the canonical zero-register idiom on x86_64.
            enc::xor_rr_32(&mut self.text, dst, dst);
        } else if value <= u32::MAX as u64 {
            enc::mov_ri_32(&mut self.text, dst, value as u32);
        } else {
            enc::mov_ri_64(&mut self.text, dst, value);
        }
    }

    pub(super) fn prepare_float_operand(
        &mut self,
        width: MachineFloatWidth,
        value: MachineValue,
        gp_scratch: X86Reg,
        fp_scratch: u32,
    ) -> Result<u32, WasmError> {
        if let MachineValue::Reg(reg) = value {
            if self.is_fp_reg(reg) {
                return Ok(self.map_fp_reg(reg)?);
            }
        }
        let gp = self.materialize_value(gp_scratch, value)?;
        match width {
            MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.text, fp_scratch as u8, gp),
            MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.text, fp_scratch as u8, gp),
        };
        Ok(fp_scratch)
    }

    // ── Block layout ─────────────────────────────────────────────────────────

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

    // ── Edge stubs ───────────────────────────────────────────────────────────

    pub(super) fn emit_edge(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        if self.is_identity_edge(target, args) {
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
            .ok_or_else(|| {
                WasmError::internal("x86_64 edge target block is out of range".into())
            })?;
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

    pub(super) fn emit_branch_to_block(&mut self, target: MachineBlockId) -> Result<(), WasmError> {
        let label = self.block_label(target)?;
        self.emit_jmp(label);
        Ok(())
    }

    // ── Runtime metadata ─────────────────────────────────────────────────────

    pub(super) fn runtime_for(
        &self,
        func_id: crate::vm::machine::machine_ir::MachineFuncId,
    ) -> Result<&crate::vm::machine::machine_ir::MachineFunctionRuntime, WasmError> {
        self.compiled
            .runtime()
            .functions
            .get(func_id.0 as usize)
            .ok_or_else(|| {
                WasmError::internal(alloc::format!(
                    "x86_64 runtime metadata missing for machine function {}",
                    func_id.0
                ))
            })
    }
}

/// Align a function start to reduce instruction-cache and iTLB pressure.
#[inline]
fn page_align_function(offset: usize, func_size: usize) -> usize {
    let aligned = (offset + 63) & !63;
    if func_size == 0 {
        return aligned;
    }
    const PAGE_SIZE: usize = 16384;
    const MAX_PADDING: usize = 1024;
    if func_size <= PAGE_SIZE {
        let start_page = aligned / PAGE_SIZE;
        let end_page = (aligned + func_size - 1) / PAGE_SIZE;
        if start_page != end_page {
            let next_page = (start_page + 1) * PAGE_SIZE;
            let padding = next_page - aligned;
            if padding <= MAX_PADDING {
                return next_page;
            }
        }
    }
    aligned
}
