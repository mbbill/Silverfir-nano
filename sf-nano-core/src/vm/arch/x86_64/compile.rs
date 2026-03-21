//! x86_64 backend: compile MachineIR to x86_64 machine code.

use alloc::{string::String, vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        entities::ModuleInst,
        machine::mir::{
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

// Call depth checking removed: the stack overflow check alone limits recursion.

// ─── Label & fixup types ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LabelKind {
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
    callee: crate::vm::machine::mir::MachineFuncId,
}

pub use crate::vm::machine::ir_dump::DebugRegion;

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
struct FunctionCompiler<'a> {
    compiled: &'a CompiledNativeModule,
    function: &'a MachineFunction,
    text: X86_64TextEmitter,
    labels: Vec<Option<usize>>,
    fixups: Vec<BranchFixup>,
    block_labels: Vec<usize>,
    edge_stubs: Vec<EdgeStub>,
    resolved_ptr_patches: Vec<LocalPtrPatch>,
    local_ptr_patches: Vec<PendingLocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    deferred_traps: Vec<(usize, MachineTrapKind)>,
    fp_reg_widths: [Option<MachineFloatWidth>; FP_MACHINE_REG_COUNT],
    current_block: Option<MachineBlockId>,
    current_op_index: Option<usize>,
    current_edge_target: Option<MachineBlockId>,
    stack_overflow_label: usize,
    return_ok_label: usize,
    return_error_label: usize,
    shared_trap_labels: [Option<usize>; MACHINE_TRAP_KIND_COUNT],
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
    for artifact in artifacts {
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

    fn emit_trap(&mut self, kind: MachineTrapKind) {
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

    fn ensure_trap_label(&mut self, kind: MachineTrapKind) -> usize {
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

    fn new_label(&mut self, _kind: LabelKind) -> usize {
        let label = self.labels.len();
        self.labels.push(None);
        label
    }

    fn bind_label(&mut self, label: usize) {
        self.labels[label] = Some(self.text.len());
    }

    fn block_label(&self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.block_labels
            .get(target.0 as usize)
            .copied()
            .filter(|label| *label != usize::MAX)
            .ok_or_else(|| WasmError::internal("x86_64 block label is out of range".into()))
    }

    /// Emit JMP rel32 with a fixup to be patched later.
    fn emit_jmp(&mut self, label: usize) {
        let rel32_offset = enc::jmp_rel32(&mut self.text);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    /// Emit Jcc rel32 with a fixup to be patched later.
    fn emit_jcc(&mut self, cc: Cc, label: usize) {
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

    fn is_fp_reg(&self, reg: MachineReg) -> bool {
        self.function.program.is_fp_reg(reg)
    }

    fn map_gp_reg(&self, reg: MachineReg) -> Result<X86Reg, WasmError> {
        if self.is_fp_reg(reg) {
            return Err(WasmError::invalid(alloc::format!(
                "x86_64 MachineIR backend expected GP register, got FP machine reg {}",
                reg.0
            )));
        }
        map_reg(reg)
    }

    fn map_fp_reg(&self, reg: MachineReg) -> Result<u32, WasmError> {
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

    fn set_fp_reg_width(
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

    fn fp_reg_width(&self, reg: MachineReg) -> Result<MachineFloatWidth, WasmError> {
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

    fn materialize_value(
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

    fn materialize_u64(&mut self, dst: X86Reg, value: u64) {
        if value == 0 {
            // XOR r32, r32 is the canonical zero-register idiom on x86_64.
            enc::xor_rr_32(&mut self.text, dst, dst);
        } else if value <= u32::MAX as u64 {
            enc::mov_ri_32(&mut self.text, dst, value as u32);
        } else {
            enc::mov_ri_64(&mut self.text, dst, value);
        }
    }

    fn prepare_float_operand(
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

    fn is_identity_edge(&self, target: MachineBlockId, args: &[MachineValue]) -> bool {
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

    fn emit_edge(
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

    fn emit_branch_to_block(&mut self, target: MachineBlockId) -> Result<(), WasmError> {
        let label = self.block_label(target)?;
        self.emit_jmp(label);
        Ok(())
    }

    // ── Runtime metadata ─────────────────────────────────────────────────────

    fn runtime_for(
        &self,
        func_id: crate::vm::machine::mir::MachineFuncId,
    ) -> Result<&crate::vm::machine::mir::MachineFunctionRuntime, WasmError> {
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

    // ── Move / const ────────────────────────────────────────────────────────

    fn emit_move(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            match src {
                MachineValue::Reg(src_reg) if self.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    let src_width = self.fp_reg_width(src_reg)?;
                    if src_width != width {
                        return Err(WasmError::invalid(alloc::format!(
                            "x86_64 typed float move width mismatch: dst expects {:?}, src {} is {:?}",
                            width,
                            src_reg.0,
                            src_width,
                        )));
                    }
                    if dst_fp != src_fp {
                        match width {
                            MachineFloatWidth::F32 => enc::movss_rr(&mut self.text, dst_fp, src_fp),
                            MachineFloatWidth::F64 => enc::movsd_rr(&mut self.text, dst_fp, src_fp),
                        };
                    }
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    match width {
                        MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.text, dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.text, dst_fp, src_gp),
                    };
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(SCRATCH0, value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.text, dst_fp, SCRATCH0)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.text, dst_fp, SCRATCH0)
                        }
                    };
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
            }
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            match src {
                MachineValue::Reg(src_reg) if self.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    match self.fp_reg_width(src_reg)? {
                        MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, src_fp),
                        MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, src_fp),
                    };
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                    }
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(dst_gp, value);
                    Ok(())
                }
            }
        }
    }

    fn emit_float_const(
        &mut self,
        width: MachineFloatWidth,
        dst: MachineReg,
        bits: u64,
    ) -> Result<(), WasmError> {
        if !self.is_fp_reg(dst) {
            return Err(WasmError::invalid(alloc::format!(
                "x86_64 FloatConst destination {} must be an FP register",
                dst.0
            )));
        }
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let imm = match width {
            MachineFloatWidth::F32 => u64::from(bits as u32),
            MachineFloatWidth::F64 => bits,
        };
        if imm == 0 {
            // XORPS/XORPD xmm, xmm → zero
            match width {
                MachineFloatWidth::F32 => enc::xorps(&mut self.text, dst_fp, dst_fp),
                MachineFloatWidth::F64 => enc::xorpd(&mut self.text, dst_fp, dst_fp),
            };
        } else {
            self.materialize_u64(SCRATCH0, imm);
            match width {
                MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.text, dst_fp, SCRATCH0),
                MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.text, dst_fp, SCRATCH0),
            };
        }
        self.set_fp_reg_width(dst, width)?;
        Ok(())
    }

    // ── LEA / Load / Store ───────────────────────────────────────────────────

    fn emit_lea(&mut self, dst: MachineReg, addr: MachineAddr) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let base = self.map_gp_reg(addr.base)?;
        if addr.offset == 0 {
            if dst_gp != base {
                enc::mov_rr_64(&mut self.text, dst_gp, base);
            }
        } else {
            enc::lea_64(&mut self.text, dst_gp, base, addr.offset);
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
        let base = self.map_gp_reg(addr.base)?;
        let disp = addr.offset;

        // FP register destination
        if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            let tracked_width = match width {
                MachineMemWidth::U32 => MachineFloatWidth::F32,
                MachineMemWidth::U64 => MachineFloatWidth::F64,
                _ => return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support narrow integer loads into FP regs"
                        .into(),
                )),
            };
            match tracked_width {
                MachineFloatWidth::F32 => enc::movss_load(&mut self.text, dst_fp, base, disp),
                MachineFloatWidth::F64 => enc::movsd_load(&mut self.text, dst_fp, base, disp),
            };
            self.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }

        // GP register destination
        let dst_gp = self.map_gp_reg(dst)?;
        match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::load_u8(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::load_s8_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::load_u16(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::load_s16_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::load_32(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::load_s32_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::load_64(&mut self.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support sign-extending 64-bit loads".into(),
                ))
            }
        };
        Ok(())
    }

    fn emit_store(
        &mut self,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        let disp = addr.offset;

        // FP register source
        if let MachineValue::Reg(src_reg) = src {
            if self.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)? as u8;
                match width {
                    MachineMemWidth::U32 => enc::movss_store(&mut self.text, base, disp, src_fp),
                    MachineMemWidth::U64 => enc::movsd_store(&mut self.text, base, disp, src_fp),
                    _ => {
                        return Err(WasmError::invalid(
                            "x86_64 MachineIR backend does not support narrow FP stores".into(),
                        ))
                    }
                };
                return Ok(());
            }
        }

        // Imm64(0) store → store_imm32_64 for U64
        if matches!(src, MachineValue::Imm64(0)) && width == MachineMemWidth::U64 {
            enc::store_imm32_64(&mut self.text, base, disp, 0);
            return Ok(());
        }

        let src_gp = self.materialize_value(SCRATCH0, src)?;
        match width {
            MachineMemWidth::U8 => enc::store_8(&mut self.text, base, disp, src_gp),
            MachineMemWidth::U16 => enc::store_16(&mut self.text, base, disp, src_gp),
            MachineMemWidth::U32 => enc::store_32(&mut self.text, base, disp, src_gp),
            MachineMemWidth::U64 => enc::store_64(&mut self.text, base, disp, src_gp),
        };
        Ok(())
    }

    // ── Integer unary ops ────────────────────────────────────────────────────

    fn emit_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = self.materialize_value(SCRATCH0, src)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntUnaryOp::Eqz) => {
                enc::test_rr_32(&mut self.text, src, src);
                enc::setcc(&mut self.text, Cc::E, dst);
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.text, dst, dst);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Eqz) => {
                enc::test_rr_64(&mut self.text, src, src);
                enc::setcc(&mut self.text, Cc::E, dst);
                enc::movzx_r32_r8(&mut self.text, dst, dst);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_64(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_64(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_64(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r32_r8(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r32_r16(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r64_r8(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r64_r16(&mut self.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
                enc::movsxd_r64_r32(&mut self.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
                // i32.extend32_s is a nop (already 32-bit)
                if dst != src {
                    enc::mov_rr_64(&mut self.text, dst, src);
                }
            }
        }
        Ok(())
    }

    // ── Integer binary ops ───────────────────────────────────────────────────

    fn emit_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;

        // x86 two-operand model: dst = dst OP src. So we need lhs in dst first.
        match op {
            MachineIntBinaryOp::Add
            | MachineIntBinaryOp::Sub
            | MachineIntBinaryOp::And
            | MachineIntBinaryOp::Or
            | MachineIntBinaryOp::Xor => {
                // Try immediate form: dst = lhs OP imm32
                if let MachineValue::Imm64(imm_val) = rhs {
                    let imm = imm_val as i64 as i32;
                    if imm as i64 == imm_val as i64
                        || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                    {
                        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                        if dst != lhs_gp {
                            enc::mov_rr_64(&mut self.text, dst, lhs_gp);
                        }
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_ri_32(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_64(&mut self.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_32(&mut self.text, dst, imm)
                            }
                            _ => unreachable!(),
                        };
                        return Ok(());
                    }
                }
                let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
                // Handle aliasing: if dst == rhs_gp but dst != lhs_gp,
                // mov dst, lhs would clobber rhs before the operation.
                if dst == rhs_gp && dst != lhs_gp {
                    if op == MachineIntBinaryOp::Sub {
                        // Sub is not commutative: compute in scratch
                        enc::mov_rr_64(&mut self.text, SCRATCH0, lhs_gp);
                        match width {
                            MachineIntWidth::I64 => {
                                enc::sub_rr_64(&mut self.text, SCRATCH0, rhs_gp)
                            }
                            MachineIntWidth::I32 => {
                                enc::sub_rr_32(&mut self.text, SCRATCH0, rhs_gp)
                            }
                        };
                        enc::mov_rr_64(&mut self.text, dst, SCRATCH0);
                    } else {
                        // Commutative: swap operands — do dst = rhs OP lhs
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_64(&mut self.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_32(&mut self.text, dst, lhs_gp)
                            }
                            _ => unreachable!(),
                        };
                    }
                } else {
                    if dst != lhs_gp {
                        enc::mov_rr_64(&mut self.text, dst, lhs_gp);
                    }
                    match (width, op) {
                        (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                            enc::add_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                            enc::add_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                            enc::and_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                            enc::and_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                            enc::or_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                            enc::or_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_64(&mut self.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_32(&mut self.text, dst, rhs_gp)
                        }
                        _ => unreachable!(),
                    };
                }
                Ok(())
            }
            MachineIntBinaryOp::Mul => {
                let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
                if dst == rhs_gp && dst != lhs_gp {
                    // IMUL is commutative: dst already has rhs, just mul by lhs
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.text, dst, lhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.text, dst, lhs_gp),
                    };
                } else {
                    if dst != lhs_gp {
                        enc::mov_rr_64(&mut self.text, dst, lhs_gp);
                    }
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.text, dst, rhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.text, dst, rhs_gp),
                    };
                }
                Ok(())
            }
            MachineIntBinaryOp::Shl
            | MachineIntBinaryOp::ShrS
            | MachineIntBinaryOp::ShrU
            | MachineIntBinaryOp::Rotl
            | MachineIntBinaryOp::Rotr => self.emit_shift_op(width, op, dst, lhs, rhs),
            MachineIntBinaryOp::DivS
            | MachineIntBinaryOp::DivU
            | MachineIntBinaryOp::RemS
            | MachineIntBinaryOp::RemU => self.emit_div_rem(width, op, dst, lhs, rhs),
        }
    }

    /// Emit shift/rotate: on x86_64, shift amount must be in CL.
    fn emit_shift_op(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // If dst == RCX, we need special handling: shift amount goes in CL,
        // but dst is also RCX. Strategy: put lhs in SCRATCH0 first, then
        // load shift amount into RCX, shift SCRATCH0, move result to RCX.
        if dst == X86Reg::RCX {
            let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
            if SCRATCH0 != lhs_gp {
                enc::mov_rr_64(&mut self.text, SCRATCH0, lhs_gp);
            }
            let rhs_gp = self.materialize_value(X86Reg::RCX, rhs)?;
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp);
            }
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                    enc::shl_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                    enc::shl_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                    enc::sar_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                    enc::sar_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                    enc::shr_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                    enc::shr_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                    enc::rol_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                    enc::rol_cl_32(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                    enc::ror_cl_64(&mut self.text, SCRATCH0)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                    enc::ror_cl_32(&mut self.text, SCRATCH0)
                }
                _ => unreachable!(),
            };
            enc::mov_rr_64(&mut self.text, X86Reg::RCX, SCRATCH0);
            return Ok(());
        }
        // For immediate shift amounts, use the imm8 form (no RCX needed).
        if let MachineValue::Imm64(amount) = rhs {
            let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.text, dst, lhs_gp);
            }
            let imm = (amount & 0x3F) as u8; // mask to 6 bits (x86 does this anyway)
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_32(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_64(&mut self.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_32(&mut self.text, dst, imm)
                }
                _ => unreachable!(),
            };
            return Ok(());
        }
        // Variable shift: need lhs in dst and rhs in RCX (CL).
        // Careful: moving lhs→dst may clobber rhs, and moving rhs→RCX may clobber lhs.
        // Resolve both source registers first, then do a safe parallel assignment.
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
        let need_save_rcx = dst != X86Reg::RCX;
        if need_save_rcx {
            enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RCX);
        }
        // Parallel assignment: lhs_gp → dst, rhs_gp → RCX.
        // Check for conflicts to determine safe ordering.
        let lhs_conflicts_rcx = lhs_gp == X86Reg::RCX; // moving rhs→RCX clobbers lhs
        let rhs_conflicts_dst = rhs_gp == dst; // moving lhs→dst clobbers rhs
        if lhs_conflicts_rcx && rhs_conflicts_dst {
            // Cycle: lhs is in RCX, rhs is in dst. Swap via SCRATCH0.
            enc::mov_rr_64(&mut self.text, SCRATCH0, lhs_gp); // save lhs
            enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp); // rhs → RCX
            enc::mov_rr_64(&mut self.text, dst, SCRATCH0); // lhs → dst
        } else if rhs_conflicts_dst {
            // Moving lhs→dst would clobber rhs. Do rhs→RCX first.
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp);
            }
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.text, dst, lhs_gp);
            }
        } else {
            // No conflict, or only lhs_conflicts_rcx. Do lhs→dst first.
            if dst != lhs_gp {
                enc::mov_rr_64(&mut self.text, dst, lhs_gp);
            }
            if rhs_gp != X86Reg::RCX {
                enc::mov_rr_64(&mut self.text, X86Reg::RCX, rhs_gp);
            }
        }
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => enc::shl_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => enc::shl_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => enc::sar_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => enc::sar_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => enc::shr_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => enc::shr_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => enc::rol_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => enc::rol_cl_32(&mut self.text, dst),
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => enc::ror_cl_64(&mut self.text, dst),
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => enc::ror_cl_32(&mut self.text, dst),
            _ => unreachable!(),
        };
        if need_save_rcx {
            enc::mov_rr_64(&mut self.text, X86Reg::RCX, SCRATCH1);
        }
        Ok(())
    }

    /// Emit div/rem: x86_64 uses RAX:RDX for dividend, result in RAX (quot) / RDX (rem).
    fn emit_div_rem(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // div/idiv implicitly uses RAX and RDX. RDX is a GP transient that
        // might hold a live value. Save it to SCRATCH1 (R11) and restore after.
        // (RAX = SCRATCH0, not in dynamic pool, so no save needed.)
        let need_save_rdx = dst != X86Reg::RDX;
        if need_save_rdx {
            enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RDX);
        }

        // Put dividend into RAX
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        if lhs_gp != X86Reg::RAX {
            enc::mov_rr_64(&mut self.text, X86Reg::RAX, lhs_gp);
        }
        // Divisor must NOT be RAX or RDX. Use R10 as safe scratch for divisor.
        let rhs_gp = self.materialize_value(X86Reg::R10, rhs)?;
        if rhs_gp == X86Reg::RAX || rhs_gp == X86Reg::RDX {
            enc::mov_rr_64(&mut self.text, X86Reg::R10, rhs_gp);
        }
        let divisor = if rhs_gp == X86Reg::RAX || rhs_gp == X86Reg::RDX {
            X86Reg::R10
        } else {
            rhs_gp
        };

        // Division-by-zero check: divisor == 0 => trap
        enc::test_rr_64(&mut self.text, divisor, divisor);
        let div_zero_label = self.ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
        self.emit_jcc(Cc::E, div_zero_label);

        match op {
            MachineIntBinaryOp::DivS => {
                // Signed overflow check: MIN / -1 => IntegerOverflow trap
                let not_min = self.new_label(LabelKind::Edge);
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.text, X86Reg::RAX, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(X86Reg::RDX, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.text, X86Reg::RAX, X86Reg::RDX);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                // Compare divisor against -1 using matching width
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.text, divisor, -1),
                };
                let overflow_label = self.ensure_trap_label(MachineTrapKind::IntegerOverflow);
                self.emit_jcc(Cc::E, overflow_label);
                self.bind_label(not_min);
                // Sign-extend RAX → RDX:RAX, then IDIV
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.text);
                        enc::idiv_rm_64(&mut self.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.text);
                        enc::idiv_rm_32(&mut self.text, divisor);
                    }
                };
            }
            MachineIntBinaryOp::RemS => {
                // MIN % -1 = 0 (no trap, just skip the div)
                let not_min = self.new_label(LabelKind::Edge);
                let done = self.new_label(LabelKind::Edge);
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.text, X86Reg::RAX, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(X86Reg::RDX, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.text, X86Reg::RAX, X86Reg::RDX);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.text, divisor, -1),
                };
                self.emit_jcc(Cc::NE, not_min);
                // MIN % -1 = 0
                enc::xor_rr_32(&mut self.text, X86Reg::RDX, X86Reg::RDX);
                self.emit_jmp(done);
                self.bind_label(not_min);
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.text);
                        enc::idiv_rm_64(&mut self.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.text);
                        enc::idiv_rm_32(&mut self.text, divisor);
                    }
                };
                self.bind_label(done);
            }
            MachineIntBinaryOp::DivU | MachineIntBinaryOp::RemU => {
                // Zero-extend: XOR RDX, RDX
                enc::xor_rr_32(&mut self.text, X86Reg::RDX, X86Reg::RDX);
                match width {
                    MachineIntWidth::I64 => enc::div_rm_64(&mut self.text, divisor),
                    MachineIntWidth::I32 => enc::div_rm_32(&mut self.text, divisor),
                };
            }
            _ => unreachable!(),
        }

        // Result: quotient in RAX, remainder in RDX
        let result_reg = match op {
            MachineIntBinaryOp::DivS | MachineIntBinaryOp::DivU => X86Reg::RAX,
            MachineIntBinaryOp::RemS | MachineIntBinaryOp::RemU => X86Reg::RDX,
            _ => unreachable!(),
        };
        if dst != result_reg {
            enc::mov_rr_64(&mut self.text, dst, result_reg);
        }
        // Restore RDX if it was saved
        if need_save_rdx {
            enc::mov_rr_64(&mut self.text, X86Reg::RDX, SCRATCH1);
        }
        Ok(())
    }

    // ── Integer compare ──────────────────────────────────────────────────────

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
        let dst = self.map_gp_reg(dst)?;
        let cc = map_int_cond(kind, sign);
        enc::setcc(&mut self.text, cc, dst);
        enc::movzx_r32_r8(&mut self.text, dst, dst);
        Ok(())
    }

    fn emit_cmp_values(
        &mut self,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // Immediate form
        if let MachineValue::Imm64(imm_val) = rhs {
            let imm = imm_val as i64 as i32;
            if imm as i64 == imm_val as i64
                || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
            {
                let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
                match width {
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.text, lhs_gp, imm),
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.text, lhs_gp, imm),
                };
                return Ok(());
            }
        }
        let lhs_gp = self.materialize_value(SCRATCH0, lhs)?;
        let rhs_gp = self.materialize_value(SCRATCH1, rhs)?;
        match width {
            MachineIntWidth::I64 => enc::cmp_rr_64(&mut self.text, lhs_gp, rhs_gp),
            MachineIntWidth::I32 => enc::cmp_rr_32(&mut self.text, lhs_gp, rhs_gp),
        };
        Ok(())
    }

    // ── Select ───────────────────────────────────────────────────────────────

    fn emit_select(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            match cond {
                MachineValue::Imm64(value) => {
                    let selected = if value != 0 { on_true } else { on_false };
                    return self.emit_move(ty, dst, selected);
                }
                MachineValue::Reg(reg) => {
                    let cond_gp = self.map_gp_reg(reg)?;
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    let false_label = self.new_label(LabelKind::Edge);
                    let done = self.new_label(LabelKind::Edge);
                    enc::test_rr_64(&mut self.text, cond_gp, cond_gp);
                    self.emit_jcc(Cc::E, false_label);
                    let true_fp =
                        self.prepare_float_operand(width, on_true, SCRATCH0, FP_SCRATCH0)?;
                    if dst_fp != true_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.text, dst_fp, true_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.text, dst_fp, true_fp as u8)
                            }
                        };
                    }
                    self.emit_jmp(done);
                    self.bind_label(false_label);
                    let false_fp =
                        self.prepare_float_operand(width, on_false, SCRATCH1, FP_SCRATCH1)?;
                    if dst_fp != false_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.text, dst_fp, false_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.text, dst_fp, false_fp as u8)
                            }
                        };
                    }
                    self.bind_label(done);
                    self.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
            }
        } else {
            let dst = self.map_gp_reg(dst)?;
            match cond {
                MachineValue::Imm64(value) => {
                    let selected = if value != 0 { on_true } else { on_false };
                    return self.emit_move(ty, inv_map_reg(dst), selected);
                }
                _ => {}
            }
            // Materialize operands BEFORE testing the condition, because
            // materialize_value may clobber flags (e.g. xor reg,reg for zero).
            let true_reg = self.materialize_value(SCRATCH0, on_true)?;
            let false_reg = self.materialize_value(SCRATCH1, on_false)?;
            let cond_gp = match cond {
                MachineValue::Reg(reg) => self.map_gp_reg(reg)?,
                _ => unreachable!(),
            };
            enc::test_rr_64(&mut self.text, cond_gp, cond_gp);
            if dst == true_reg && dst != false_reg {
                enc::cmovcc_rr_64(&mut self.text, Cc::E, dst, false_reg);
            } else if dst == false_reg {
                enc::cmovcc_rr_64(&mut self.text, Cc::NE, dst, true_reg);
            } else {
                enc::mov_rr_64(&mut self.text, dst, false_reg);
                enc::cmovcc_rr_64(&mut self.text, Cc::NE, dst, true_reg);
            }
            Ok(())
        }
    }

    // ── Convert (integer parts only — float conversions are Phase 4) ─────────

    fn emit_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_float_width = convert_result_float_width(op);
        let src_gp = self.materialize_value(SCRATCH0, src)?;
        match op {
            MachineConvertOp::I32WrapI64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zeroes the upper 32 bits
                enc::mov_rr_32(&mut self.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32S => {
                let dst_gp = self.map_gp_reg(dst)?;
                enc::movsxd_r64_r32(&mut self.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32U => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zero-extends
                enc::mov_rr_32(&mut self.text, dst_gp, src_gp);
            }
            MachineConvertOp::I32ReinterpretF32 | MachineConvertOp::I64ReinterpretF64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                if dst_gp != src_gp {
                    enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                }
            }
            MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
                let width = dst_float_width.expect("float reinterpret width");
                if self.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    match width {
                        MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.text, dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.text, dst_fp, src_gp),
                    };
                    self.set_fp_reg_width(dst, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst)?;
                    if dst_gp != src_gp {
                        enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                    }
                }
            }
            // Float promotion / demotion
            MachineConvertOp::F64PromoteF32 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F32, src, SCRATCH0, FP_SCRATCH0)?;
                if self.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
                    enc::cvtss2sd(&mut self.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtss2sd(&mut self.text, FP_SCRATCH1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movq_r64_xmm(&mut self.text, dst_gp, FP_SCRATCH1 as u8);
                }
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp =
                    self.prepare_float_operand(MachineFloatWidth::F64, src, SCRATCH0, FP_SCRATCH0)?;
                if self.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
                    enc::cvtsd2ss(&mut self.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtsd2ss(&mut self.text, FP_SCRATCH1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movd_r32_xmm(&mut self.text, dst_gp, FP_SCRATCH1 as u8);
                }
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                // CVTSI2SS xmm, r32
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r32(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI32U => {
                // Zero-extend to 64-bit first for unsigned interpretation
                enc::mov_rr_32(&mut self.text, SCRATCH0, src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, SCRATCH0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64U => {
                // x86_64 has no unsigned int-to-float instruction.
                // For values that fit in i64 (bit 63 = 0), use signed conversion.
                // For values with bit 63 set, shift right by 1, convert, then double.
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F32)?;
                enc::test_rr_64(&mut self.text, src_gp, src_gp);
                let large = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::S, large); // JS = sign flag set = bit 63 is 1
                                             // Small path: fits in i64
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, src_gp);
                let done = self.new_label(LabelKind::Edge);
                self.emit_jmp(done);
                // Large path: bit 63 set
                self.bind_label(large);
                enc::mov_rr_64(&mut self.text, SCRATCH0, src_gp);
                enc::mov_rr_64(&mut self.text, SCRATCH1, src_gp);
                enc::shr_imm_64(&mut self.text, SCRATCH0, 1); // src >> 1
                enc::and_ri_32(&mut self.text, SCRATCH1, 1); // src & 1 (preserve LSB)
                enc::or_rr_64(&mut self.text, SCRATCH0, SCRATCH1); // (src >> 1) | (src & 1)
                enc::cvtsi2ss_r64(&mut self.text, dst_fp, SCRATCH0);
                enc::addss(&mut self.text, dst_fp, dst_fp); // double it
                self.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r32(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32U => {
                enc::mov_rr_32(&mut self.text, SCRATCH0, src_gp);
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, SCRATCH0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64S => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64U => {
                let dst_fp = self.dst_float_reg(dst, MachineFloatWidth::F64)?;
                enc::test_rr_64(&mut self.text, src_gp, src_gp);
                let large = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::S, large);
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, src_gp);
                let done = self.new_label(LabelKind::Edge);
                self.emit_jmp(done);
                self.bind_label(large);
                enc::mov_rr_64(&mut self.text, SCRATCH0, src_gp);
                enc::mov_rr_64(&mut self.text, SCRATCH1, src_gp);
                enc::shr_imm_64(&mut self.text, SCRATCH0, 1);
                enc::and_ri_32(&mut self.text, SCRATCH1, 1);
                enc::or_rr_64(&mut self.text, SCRATCH0, SCRATCH1);
                enc::cvtsi2sd_r64(&mut self.text, dst_fp, SCRATCH0);
                enc::addsd(&mut self.text, dst_fp, dst_fp);
                self.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            // Trapping truncations: call Rust helper
            MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.emit_trapping_trunc(op, dst_gp, src_gp)?;
            }
            // Saturating truncations: call Rust helper
            MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.emit_saturating_trunc(op, dst_gp, src_gp)?;
            }
        }
        Ok(())
    }

    // ── Float ops ─────────────────────────────────────────────────────────────

    fn emit_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = self.prepare_float_operand(width, src, SCRATCH0, FP_SCRATCH0)?;
        let result_fp = if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            FP_SCRATCH2 as u8
        };
        match op {
            MachineFloatUnaryOp::Sqrt => {
                match width {
                    MachineFloatWidth::F32 => enc::sqrtss(&mut self.text, result_fp, src_fp as u8),
                    MachineFloatWidth::F64 => enc::sqrtsd(&mut self.text, result_fp, src_fp as u8),
                };
            }
            MachineFloatUnaryOp::Ceil => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_CEIL)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_CEIL)
                    }
                };
            }
            MachineFloatUnaryOp::Floor => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_FLOOR)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_FLOOR)
                    }
                };
            }
            MachineFloatUnaryOp::Trunc => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_TRUNC)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_TRUNC)
                    }
                };
            }
            MachineFloatUnaryOp::Nearest => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::roundss(&mut self.text, result_fp, src_fp as u8, enc::ROUND_NEAREST)
                    }
                    MachineFloatWidth::F64 => {
                        enc::roundsd(&mut self.text, result_fp, src_fp as u8, enc::ROUND_NEAREST)
                    }
                };
            }
            MachineFloatUnaryOp::Abs => {
                // Clear sign bit: AND with mask.
                let mask_xmm = if result_fp != FP_SCRATCH0 as u8 {
                    FP_SCRATCH0 as u8
                } else {
                    FP_SCRATCH2 as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x7FFF_FFFFu64,
                    MachineFloatWidth::F64 => 0x7FFF_FFFF_FFFF_FFFFu64,
                };
                self.materialize_u64(SCRATCH0, mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::andpd(&mut self.text, result_fp, mask_xmm);
            }
            MachineFloatUnaryOp::Neg => {
                // Flip sign bit: XOR with mask.
                let mask_xmm = if result_fp != FP_SCRATCH0 as u8 {
                    FP_SCRATCH0 as u8
                } else {
                    FP_SCRATCH2 as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x8000_0000u64,
                    MachineFloatWidth::F64 => 0x8000_0000_0000_0000u64,
                };
                self.materialize_u64(SCRATCH0, mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::xorpd(&mut self.text, result_fp, mask_xmm);
            }
        }
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, result_fp),
            };
        }
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
        let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
        let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
        let result_fp = if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            FP_SCRATCH2 as u8
        };
        match op {
            MachineFloatBinaryOp::Add
            | MachineFloatBinaryOp::Sub
            | MachineFloatBinaryOp::Mul
            | MachineFloatBinaryOp::Div => {
                // SSE two-operand: dst = dst OP src. Move lhs into result first.
                // If result == rhs and result != lhs, the move would clobber rhs.
                // Save rhs to scratch first in that case.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = FP_SCRATCH2 as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match (width, op) {
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                        enc::addss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                        enc::addsd(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                        enc::subss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                        enc::subsd(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                        enc::mulss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                        enc::mulsd(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                        enc::divss(&mut self.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                        enc::divsd(&mut self.text, result_fp, actual_rhs)
                    }
                    _ => unreachable!(),
                };
            }
            MachineFloatBinaryOp::Min => {
                // Wasm fmin: if either operand is NaN, result is NaN.
                // x86_64 minsd/minss: if either is NaN, returns the SECOND operand.
                // Strategy: result = minsd(lhs, rhs); if unordered, result = addsd(lhs, rhs) (NaN propagation).
                // Guard: if result == rhs and result != lhs, save rhs to scratch first.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = FP_SCRATCH2 as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::minss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::minsd(&mut self.text, result_fp, actual_rhs),
                };
                // Compare for NaN: ucomisd lhs, rhs sets PF=1 if unordered (NaN)
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let done = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::NP, done); // no NaN => minsd result is correct
                                             // NaN case: add propagates NaN
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::addss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::addsd(&mut self.text, result_fp, actual_rhs),
                };
                self.bind_label(done);
            }
            MachineFloatBinaryOp::Max => {
                // Same NaN handling as Min but with maxsd/maxss.
                // Guard: if result == rhs and result != lhs, save rhs to scratch first.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = FP_SCRATCH2 as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::maxss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::maxsd(&mut self.text, result_fp, actual_rhs),
                };
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let done = self.new_label(LabelKind::Edge);
                self.emit_jcc(Cc::NP, done);
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                match width {
                    MachineFloatWidth::F32 => enc::addss(&mut self.text, result_fp, actual_rhs),
                    MachineFloatWidth::F64 => enc::addsd(&mut self.text, result_fp, actual_rhs),
                };
                self.bind_label(done);
            }
            MachineFloatBinaryOp::Copysign => {
                // magnitude of lhs, sign of rhs.
                // Strategy: clear sign of lhs (abs), extract sign of rhs, OR them.
                // Use a mask scratch that doesn't conflict with result_fp or rhs_fp.
                let mask_xmm =
                    if result_fp != FP_SCRATCH0 as u8 && rhs_fp as u8 != FP_SCRATCH0 as u8 {
                        FP_SCRATCH0 as u8
                    } else {
                        FP_SCRATCH2 as u8
                    };
                let sign_mask = match width {
                    MachineFloatWidth::F32 => 0x8000_0000u64,
                    MachineFloatWidth::F64 => 0x8000_0000_0000_0000u64,
                };
                let abs_mask = match width {
                    MachineFloatWidth::F32 => 0x7FFF_FFFFu64,
                    MachineFloatWidth::F64 => 0x7FFF_FFFF_FFFF_FFFFu64,
                };
                // result = lhs & abs_mask (clear sign bit)
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movss_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movsd_rr(&mut self.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                self.materialize_u64(SCRATCH0, abs_mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::andpd(&mut self.text, result_fp, mask_xmm);
                // mask_xmm = rhs & sign_mask (extract sign bit)
                self.materialize_u64(SCRATCH0, sign_mask);
                enc::movq_xmm_r64(&mut self.text, mask_xmm, SCRATCH0);
                enc::andpd(&mut self.text, mask_xmm, rhs_fp as u8);
                // result |= mask_xmm
                enc::orpd(&mut self.text, result_fp, mask_xmm);
            }
        };
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, result_fp),
            };
        }
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
        let dst_gp = self.map_gp_reg(dst)?;
        let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
        // Choose an rhs FP scratch that doesn't conflict with lhs. When lhs
        // already lives in a mapped FP register (not FP_SCRATCH0), reuse
        // FP_SCRATCH0 for rhs to avoid clobbering live FP transients in
        // FP_SCRATCH1/FP_SCRATCH2.
        let rhs_fp_scratch = if lhs_fp != FP_SCRATCH0 as u32 {
            FP_SCRATCH0
        } else {
            FP_SCRATCH2
        };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(&mut self.text, rhs_fp_scratch as u8, rhs_fp_scratch as u8);
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp as u8),
                MachineFloatWidth::F64 => enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp as u8),
            };
        }
        // Wasm float comparisons: unordered (NaN) => 0 for all except Ne.
        // UCOMISD sets: ZF=1,PF=1,CF=1 for unordered; ZF=1,PF=0,CF=0 for equal;
        // ZF=0,PF=0,CF=1 for less-than; ZF=0,PF=0,CF=0 for greater-than.
        match kind {
            MachineCompareKind::Eq => {
                // Ordered and equal: ZF=1 AND PF=0
                // SETE sets dst to ZF, SETNP sets tmp to !PF, AND them.
                enc::setcc(&mut self.text, Cc::E, dst_gp);
                enc::setcc(&mut self.text, Cc::NP, SCRATCH0);
                enc::and_rr_32(&mut self.text, dst_gp, SCRATCH0);
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ne => {
                // Unordered OR not-equal: NE=1 OR PF=1
                enc::setcc(&mut self.text, Cc::NE, dst_gp);
                enc::setcc(&mut self.text, Cc::P, SCRATCH0);
                enc::or_rr_32(&mut self.text, dst_gp, SCRATCH0);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Lt => {
                // Ordered and less: CF=1 AND PF=0 → use JB (CF=1), but need !PF too
                // Actually, for UCOMISD: CF=1 for less-than AND for unordered.
                // So Lt = CF=1 AND PF=0 → SETB AND SETNP
                enc::setcc(&mut self.text, Cc::B, dst_gp);
                enc::setcc(&mut self.text, Cc::NP, SCRATCH0);
                enc::and_rr_32(&mut self.text, dst_gp, SCRATCH0);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Gt => {
                // Ordered and greater: ZF=0, CF=0, PF=0 → JA (CF=0 AND ZF=0)
                // JA already excludes unordered (PF=1 implies CF=1), so SETA is correct.
                enc::setcc(&mut self.text, Cc::A, dst_gp);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Le => {
                // Ordered and less-or-equal: (CF=1 OR ZF=1) AND PF=0
                enc::setcc(&mut self.text, Cc::BE, dst_gp);
                enc::setcc(&mut self.text, Cc::NP, SCRATCH0);
                enc::and_rr_32(&mut self.text, dst_gp, SCRATCH0);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ge => {
                // Ordered and greater-or-equal: CF=0 AND PF=0 → JAE excludes unordered already
                // Actually JAE = !CF. Unordered sets CF=1, so JAE is 0 for unordered. Correct.
                enc::setcc(&mut self.text, Cc::AE, dst_gp);
                enc::movzx_r32_r8(&mut self.text, dst_gp, dst_gp);
            }
        }
        Ok(())
    }

    // ── Helper calls ──────────────────────────────────────────────────────────

    fn emit_call_helper(&mut self, extern_idx: usize, const_idx: usize) -> Result<(), WasmError> {
        let binding = self
            .compiled
            .module()
            .externs
            .get(extern_idx)
            .ok_or_else(|| WasmError::internal("x86_64 helper target is out of range".into()))?;
        let metadata = self
            .compiled
            .const_ptr(crate::vm::machine::mir::MachineConstId(
                const_idx as u32,
            ))
            .ok_or_else(|| WasmError::internal("x86_64 helper metadata is out of range".into()))?;
        // System V AMD64 ABI: RDI=ctx, RSI=fp, RDX=metadata
        enc::mov_rr_64(&mut self.text, X86Reg::RDI, map_fixed_reg(MACHINE_CTX_REG));
        enc::mov_rr_64(&mut self.text, X86Reg::RSI, map_fixed_reg(MACHINE_FP_REG));
        self.materialize_u64(X86Reg::RDX, metadata as u64);
        self.materialize_u64(
            SCRATCH1,
            resolve_helper_entry(binding.symbol) as usize as u64,
        );
        enc::call_reg(&mut self.text, SCRATCH1);
        // Check return: RAX != 0 => error
        enc::test_rr_32(&mut self.text, X86Reg::RAX, X86Reg::RAX);
        self.emit_jcc(Cc::NE, self.return_error_label);
        Ok(())
    }

    // ── Float conversion helpers ────────────────────────────────────────────

    /// Get FP destination register: if dst is FP reg, use it directly; else use scratch.
    fn dst_float_reg(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<u8, WasmError> {
        if self.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.set_fp_reg_width(dst, width)?;
            Ok(dst_fp)
        } else {
            Ok(FP_SCRATCH1 as u8)
        }
    }

    /// If dst is a GP register, move float result from XMM to GP.
    fn store_fp_result_if_gp(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
        fp_reg: u8,
    ) -> Result<(), WasmError> {
        if !self.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.text, dst_gp, fp_reg),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.text, dst_gp, fp_reg),
            };
        }
        Ok(())
    }

    /// Save GP transient registers to the system stack before a C helper call.
    /// Pushes 8 registers (7 transients + padding) for 16-byte alignment.
    fn save_gp_transients(&mut self) {
        // Push 7 GP transients + 1 padding for 16-byte alignment (8 * 8 = 64 bytes)
        enc::push(&mut self.text, X86Reg::RCX);
        enc::push(&mut self.text, X86Reg::RDX);
        enc::push(&mut self.text, X86Reg::RSI);
        enc::push(&mut self.text, X86Reg::RDI);
        enc::push(&mut self.text, X86Reg::R8);
        enc::push(&mut self.text, X86Reg::R9);
        enc::push(&mut self.text, X86Reg::R10);
        enc::push(&mut self.text, X86Reg::R10); // padding for 16-byte alignment
    }

    /// Restore GP transient registers from the system stack after a C helper call.
    fn restore_gp_transients(&mut self) {
        enc::pop(&mut self.text, X86Reg::R10); // padding
        enc::pop(&mut self.text, X86Reg::R10);
        enc::pop(&mut self.text, X86Reg::R9);
        enc::pop(&mut self.text, X86Reg::R8);
        enc::pop(&mut self.text, X86Reg::RDI);
        enc::pop(&mut self.text, X86Reg::RSI);
        enc::pop(&mut self.text, X86Reg::RDX);
        enc::pop(&mut self.text, X86Reg::RCX);
    }

    fn emit_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C helper call clobbers all GP transient registers. Save them.
        self.save_gp_transients();
        // Call: x86_64_trapping_trunc(ctx, src_bits, op_code) -> TruncResult{status, value}
        // System V: RDI=ctx, RSI=src_bits, RDX=op_code
        // Returns: RAX=status, RDX=value (struct in RAX+RDX)
        enc::mov_rr_64(&mut self.text, X86Reg::RDI, map_fixed_reg(MACHINE_CTX_REG));
        enc::mov_rr_64(&mut self.text, X86Reg::RSI, src);
        self.materialize_u64(X86Reg::RDX, convert_op_code(op));
        self.materialize_u64(SCRATCH1, x86_64_trapping_trunc as usize as u64);
        enc::call_reg(&mut self.text, SCRATCH1);
        // Save result before restoring transients (RDX would be clobbered)
        enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RDX); // save result value
        let status = X86Reg::RAX; // save status
        self.restore_gp_transients();
        // Check status
        enc::test_rr_64(&mut self.text, status, status);
        self.emit_jcc(Cc::NE, self.return_error_label);
        if dst != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, dst, SCRATCH1);
        }
        Ok(())
    }

    fn emit_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C helper call clobbers all GP transient registers. Save them.
        self.save_gp_transients();
        // Call: x86_64_saturating_trunc(src_bits, op_code) -> u64
        // System V: RDI=src_bits, RSI=op_code. Returns RAX.
        enc::mov_rr_64(&mut self.text, X86Reg::RDI, src);
        self.materialize_u64(X86Reg::RSI, convert_op_code(op));
        self.materialize_u64(SCRATCH1, x86_64_saturating_trunc as usize as u64);
        enc::call_reg(&mut self.text, SCRATCH1);
        // Save result before restoring transients
        enc::mov_rr_64(&mut self.text, SCRATCH1, X86Reg::RAX);
        self.restore_gp_transients();
        if dst != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, dst, SCRATCH1);
        }
        Ok(())
    }

    // ── Branch / branch_if ─────────────────────────────────────────────────

    fn emit_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &crate::vm::machine::mir::MachineEdge,
        else_edge: &crate::vm::machine::mir::MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let then_fallthrough =
            is_fallthrough_edge(self, then_edge.target, &then_edge.args, fallthrough);
        let else_fallthrough =
            is_fallthrough_edge(self, else_edge.target, &else_edge.args, fallthrough);
        let then_label = (!then_fallthrough)
            .then(|| self.emit_edge(then_edge.target, &then_edge.args))
            .transpose()?;
        let else_label = (!else_fallthrough)
            .then(|| self.emit_edge(else_edge.target, &else_edge.args))
            .transpose()?;
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {
                    if let Some(label) = else_label {
                        self.emit_jmp(label);
                    }
                }
                MachineValue::Imm64(_) => {
                    if let Some(label) = then_label {
                        self.emit_jmp(label);
                    }
                }
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    enc::test_rr_64(&mut self.text, reg, reg);
                    if else_fallthrough {
                        if let Some(label) = then_label {
                            self.emit_jcc(Cc::NE, label);
                        }
                    } else if then_fallthrough {
                        if let Some(label) = else_label {
                            self.emit_jcc(Cc::E, label);
                        }
                    } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                        self.emit_jcc(Cc::NE, then_label);
                        self.emit_jmp(else_label);
                    }
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
                let cc = map_int_cond(kind, sign);
                if else_fallthrough {
                    if let Some(label) = then_label {
                        self.emit_jcc(cc, label);
                    }
                } else if then_fallthrough {
                    if let Some(label) = else_label {
                        self.emit_jcc(cc.invert(), label);
                    }
                } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
                    self.emit_jcc(cc, then_label);
                    self.emit_jmp(else_label);
                }
            }
            MachineBranchCond::FloatCompare {
                width,
                kind,
                lhs,
                rhs,
            } => {
                return self.emit_float_branch(
                    width,
                    kind,
                    lhs,
                    rhs,
                    then_label,
                    else_label,
                    then_fallthrough,
                    else_fallthrough,
                );
            }
        }
        Ok(())
    }

    fn emit_branch_if(
        &mut self,
        cond: &MachineBranchCond,
        trap_label: usize,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(0) => {}
                MachineValue::Imm64(_) => self.emit_jmp(trap_label),
                MachineValue::Reg(reg) => {
                    let reg = self.map_gp_reg(reg)?;
                    enc::test_rr_64(&mut self.text, reg, reg);
                    self.emit_jcc(Cc::NE, trap_label);
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
                self.emit_jcc(map_int_cond(kind, sign), trap_label);
            }
            MachineBranchCond::FloatCompare {
                width,
                kind,
                lhs,
                rhs,
            } => {
                let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
                let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, FP_SCRATCH1)?;
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp as u8)
                    }
                };
                self.emit_jcc(map_float_cond(kind), trap_label);
            }
        }
        Ok(())
    }

    fn emit_float_branch(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        lhs: MachineValue,
        rhs: MachineValue,
        then_label: Option<usize>,
        else_label: Option<usize>,
        then_fallthrough: bool,
        else_fallthrough: bool,
    ) -> Result<(), WasmError> {
        let lhs_fp = self.prepare_float_operand(width, lhs, SCRATCH0, FP_SCRATCH0)?;
        // Same scratch selection as emit_float_compare: avoid clobbering live
        // FP transients in FP_SCRATCH1.
        let rhs_fp_scratch = if lhs_fp != FP_SCRATCH0 as u32 {
            FP_SCRATCH0
        } else {
            FP_SCRATCH2
        };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(&mut self.text, rhs_fp_scratch as u8, rhs_fp_scratch as u8);
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, SCRATCH1, rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => enc::ucomiss(&mut self.text, lhs_fp as u8, rhs_fp as u8),
                MachineFloatWidth::F64 => enc::ucomisd(&mut self.text, lhs_fp as u8, rhs_fp as u8),
            };
        }
        let cc = map_float_cond(kind);
        if else_fallthrough {
            if let Some(label) = then_label {
                self.emit_jcc(cc, label);
            }
        } else if then_fallthrough {
            if let Some(label) = else_label {
                self.emit_jcc(cc.invert(), label);
            }
        } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
            self.emit_jcc(cc, then_label);
            self.emit_jmp(else_label);
        }
        Ok(())
    }

    // ── Return sequence ──────────────────────────────────────────────────────

    fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = *self.runtime_for(self.function.id)?;
        let call_scratch = runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("x86_64 local return requires call scratch".into())
        })?;
        let call_link = self.compiled.runtime().call_link;
        let continuation_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.continuation_offset as i32;
        let caller_frame_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.caller_frame_offset as i32;
        let caller_result_base_offset =
            (call_scratch.base_slot as i32) * 8 + call_link.caller_result_base_offset as i32;

        let fp = map_fixed_reg(MACHINE_FP_REG);
        // Load continuation address into RDI (caller-saved, not SCRATCH0=RAX)
        enc::load_64(&mut self.text, X86Reg::RDI, fp, continuation_offset);
        // Load caller frame pointer
        enc::load_64(&mut self.text, SCRATCH1, fp, caller_frame_offset);
        // Load caller result base offset
        enc::load_64(&mut self.text, SCRATCH0, fp, caller_result_base_offset);
        // result_ptr = caller_fp + result_base
        enc::add_rr_64(&mut self.text, SCRATCH0, SCRATCH1);

        // Copy results
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                enc::load_64(
                    &mut self.text,
                    X86Reg::RCX,
                    fp,
                    (results.base_slot as i32 + index) * 8,
                );
                enc::store_64(&mut self.text, SCRATCH0, index * 8, X86Reg::RCX);
            }
        }

        // Restore caller FP
        enc::mov_rr_64(&mut self.text, fp, SCRATCH1);
        // Jump to continuation
        enc::jmp_reg(&mut self.text, X86Reg::RDI);
        Ok(())
    }

    // ── Call direct ──────────────────────────────────────────────────────────

    fn emit_call_direct(
        &mut self,
        callee: crate::vm::machine::mir::MachineFuncId,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_runtime = self.runtime_for(callee)?;
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("x86_64 direct local call requires callee call scratch".into())
        })?;
        let continuation_slot_offset = (call_scratch.base_slot as i32) * 8
            + self.compiled.runtime().call_link.continuation_offset as i32;
        let callee_fp = self.map_gp_reg(callee_frame_base)?;

        // Load continuation address via movabs (patched after label resolution).
        // mov_ri_64 emits REX.W B8+rd imm64 (2 + 8 = 10 bytes).
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let cont_imm_offset = self.text.len() - 8;
        let continuation_label = self.block_label(continuation)?;
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: cont_imm_offset,
            target_label: continuation_label,
        });

        // Store continuation address into callee's call-link slot
        // (caller_frame and caller_result_base are stored by the lowered MachineIR
        // instructions before this CallDirect terminator)
        enc::store_64(
            &mut self.text,
            callee_fp,
            continuation_slot_offset,
            SCRATCH0,
        );

        // Load callee entry address via movabs (patched after compilation)
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let callee_imm_offset = self.text.len() - 8;
        self.direct_call_patches.push(DirectCallPatch {
            literal_offset: callee_imm_offset,
            callee,
        });

        // Set FP to callee frame
        enc::mov_rr_64(&mut self.text, map_fixed_reg(MACHINE_FP_REG), callee_fp);
        // Jump to callee
        enc::jmp_reg(&mut self.text, SCRATCH0);
        Ok(())
    }

    // ── Call indirect ────────────────────────────────────────────────────────

    fn emit_call_indirect(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        // Load function table base address
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let table_base_offset = self.text.len() - 8;
        self.function_table_patches.push(table_base_offset);

        // Compute table entry: table_base + callee_id * 32 (sizeof X86_64FunctionInfo)
        let callee_id_reg = self.materialize_value(SCRATCH1, callee_target)?;
        if callee_id_reg != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, SCRATCH1, callee_id_reg);
        }
        // callee_id * 32 = callee_id << 5
        enc::shl_imm_64(&mut self.text, SCRATCH1, 5);
        enc::add_rr_64(&mut self.text, SCRATCH0, SCRATCH1);

        // Resolve callee_fp BEFORE loading function info (which clobbers RDI/RSI/RDX/RCX).
        // Save callee_fp to R8 (caller-saved transient, not used by info loads).
        let callee_fp_orig = self.map_gp_reg(callee_frame_base)?;
        let callee_fp = X86Reg::R8;
        if callee_fp_orig != callee_fp {
            enc::mov_rr_64(&mut self.text, callee_fp, callee_fp_orig);
        }

        // Load function info fields: entry(0), total_frame_bytes(8), frame_prefix_slots(16), call_scratch_base(24)
        enc::load_64(&mut self.text, X86Reg::RDI, SCRATCH0, 0); // entry
        enc::load_64(&mut self.text, X86Reg::RSI, SCRATCH0, 8); // total_frame_bytes
        enc::load_64(&mut self.text, X86Reg::RDX, SCRATCH0, 16); // frame_prefix_slots
        enc::load_64(&mut self.text, X86Reg::RCX, SCRATCH0, 24); // call_scratch_base_slot
                                                                 // Stack overflow check: callee_fp + total_frame_bytes > stack_end?
        enc::lea_64(&mut self.text, SCRATCH0, callee_fp, 0);
        enc::add_rr_64(&mut self.text, SCRATCH0, X86Reg::RSI);
        enc::load_64(
            &mut self.text,
            SCRATCH1,
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::STACK_END as i32,
        );
        enc::cmp_rr_64(&mut self.text, SCRATCH0, SCRATCH1);
        self.emit_jcc(Cc::A, self.stack_overflow_label);

        // Zero callee prefix: from callee_fp + arg_slots*8 to callee_fp + frame_prefix_slots*8
        self.emit_zero_dynamic_callee_prefix(callee_fp, arg_slots)?;

        // Load continuation address
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let cont_imm_offset = self.text.len() - 8;
        let continuation_label = self.block_label(continuation)?;
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            literal_offset: cont_imm_offset,
            target_label: continuation_label,
        });

        // call_scratch_base_slot is in RCX (in units of slots)
        // Compute call-link base: callee_fp + call_scratch_base_slot * 8
        enc::shl_imm_64(&mut self.text, X86Reg::RCX, 3);
        enc::add_rr_64(&mut self.text, X86Reg::RCX, callee_fp);

        let call_link = self.compiled.runtime().call_link;
        // Store continuation
        enc::store_64(
            &mut self.text,
            X86Reg::RCX,
            call_link.continuation_offset as i32,
            SCRATCH0,
        );
        // Store caller frame pointer
        enc::store_64(
            &mut self.text,
            X86Reg::RCX,
            call_link.caller_frame_offset as i32,
            map_fixed_reg(MACHINE_FP_REG),
        );
        // Store caller result base
        self.materialize_u64(SCRATCH1, u64::from(caller_result_base) * 8);
        enc::store_64(
            &mut self.text,
            X86Reg::RCX,
            call_link.caller_result_base_offset as i32,
            SCRATCH1,
        );

        // Set FP to callee and jump to entry
        enc::mov_rr_64(&mut self.text, map_fixed_reg(MACHINE_FP_REG), callee_fp);
        enc::jmp_reg(&mut self.text, X86Reg::RDI);
        Ok(())
    }

    fn emit_zero_dynamic_callee_prefix(
        &mut self,
        callee_fp: X86Reg,
        arg_slots: u16,
    ) -> Result<(), WasmError> {
        // Zero from callee_fp + arg_slots*8 up to callee_fp + frame_prefix_slots*8
        // frame_prefix_slots is in RDX (from function table load)
        self.materialize_u64(SCRATCH0, u64::from(arg_slots) * 8);
        enc::add_rr_64(&mut self.text, SCRATCH0, callee_fp);
        // end = callee_fp + frame_prefix_slots * 8
        enc::shl_imm_64(&mut self.text, X86Reg::RDX, 3);
        enc::add_rr_64(&mut self.text, X86Reg::RDX, callee_fp);
        // If start >= end, skip
        enc::cmp_rr_64(&mut self.text, SCRATCH0, X86Reg::RDX);
        let done = self.new_label(LabelKind::Edge);
        self.emit_jcc(Cc::AE, done);
        let loop_label = self.new_label(LabelKind::Edge);
        self.bind_label(loop_label);
        enc::store_imm32_64(&mut self.text, SCRATCH0, 0, 0);
        enc::add_ri_64(&mut self.text, SCRATCH0, 8);
        enc::cmp_rr_64(&mut self.text, SCRATCH0, X86Reg::RDX);
        self.emit_jcc(Cc::B, loop_label);
        self.bind_label(done);
        Ok(())
    }

    // ── Jump table ───────────────────────────────────────────────────────────

    fn emit_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[crate::vm::machine::mir::MachineEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal(
                "x86_64 MachineIR jump table requires at least one entry".into(),
            ));
        }
        if entries.len() == 1 {
            let label = self.emit_edge(entries[0].target, &entries[0].args)?;
            self.emit_jmp(label);
            return Ok(());
        }

        let index_reg = self.materialize_value(SCRATCH1, index)?;
        // Clamp index to (entries.len() - 1)
        self.materialize_u64(X86Reg::RAX, (entries.len() - 1) as u64);
        enc::cmp_rr_64(&mut self.text, index_reg, X86Reg::RAX);
        enc::cmovcc_rr_64(&mut self.text, Cc::A, SCRATCH1, X86Reg::RAX);
        if index_reg != SCRATCH1 {
            enc::mov_rr_64(&mut self.text, SCRATCH1, index_reg);
            enc::cmovcc_rr_64(&mut self.text, Cc::A, SCRATCH1, X86Reg::RAX);
        }

        // Load table base address (absolute, patched later)
        enc::movabs_ri_64(&mut self.text, SCRATCH0, 0); // placeholder (patched later)
        let table_base_imm_offset = self.text.len() - 8;

        // index * 8 for table entry
        enc::shl_imm_64(&mut self.text, SCRATCH1, 3);
        enc::add_rr_64(&mut self.text, SCRATCH0, SCRATCH1);
        // Load target address from table
        enc::load_64(&mut self.text, SCRATCH0, SCRATCH0, 0);
        enc::jmp_reg(&mut self.text, SCRATCH0);

        // Emit jump table entries (each is a u64 absolute address, patched later)
        let table_offset = self.text.len();
        self.resolved_ptr_patches.push(LocalPtrPatch {
            literal_offset: table_base_imm_offset,
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

    // ── Stack overflow / call depth ──────────────────────────────────────────

    // ── Parallel moves ───────────────────────────────────────────────────────

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

            // Cycle detected — break it with a temporary.
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
                let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                let width = dst.ty.float_width().expect("FP param width");
                match width {
                    MachineFloatWidth::F32 => {
                        enc::movss_rr(&mut self.text, FP_SCRATCH2 as u8, dst_fp)
                    }
                    MachineFloatWidth::F64 => {
                        enc::movsd_rr(&mut self.text, FP_SCRATCH2 as u8, dst_fp)
                    }
                };
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
                enc::mov_rr_64(&mut self.text, SCRATCH1, dst_gp);
                enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
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
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match width {
                            MachineFloatWidth::F32 => enc::movss_rr(&mut self.text, dst_fp, src_fp),
                            MachineFloatWidth::F64 => enc::movsd_rr(&mut self.text, dst_fp, src_fp),
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movd_xmm_r32(&mut self.text, dst_fp, src_gp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movq_xmm_r64(&mut self.text, dst_fp, src_gp)
                            }
                        };
                    }
                    self.set_fp_reg_width(dst.reg, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst.reg)?;
                    if self.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match src_float_width.ok_or_else(|| {
                            WasmError::invalid(alloc::format!(
                                "x86_64 edge move is missing float-width metadata for machine reg {}",
                                src_reg.0
                            ))
                        })? {
                            MachineFloatWidth::F32 => {
                                enc::movd_r32_xmm(&mut self.text, dst_gp, src_fp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movq_r64_xmm(&mut self.text, dst_gp, src_fp)
                            }
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        enc::mov_rr_64(&mut self.text, dst_gp, src_gp);
                    }
                }
            }
            ParallelSource::Imm(value) => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    self.materialize_u64(SCRATCH0, value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.text, dst_fp, SCRATCH0)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.text, dst_fp, SCRATCH0)
                        }
                    };
                    self.set_fp_reg_width(dst.reg, width)?;
                } else {
                    self.materialize_u64(self.map_gp_reg(dst.reg)?, value);
                }
            }
            ParallelSource::GpTemp => {
                let dst_gp = self.map_gp_reg(dst.reg)?;
                enc::mov_rr_64(&mut self.text, dst_gp, SCRATCH1);
            }
            ParallelSource::FpTemp(width) => {
                let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                match width {
                    MachineFloatWidth::F32 => {
                        enc::movss_rr(&mut self.text, dst_fp, FP_SCRATCH2 as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::movsd_rr(&mut self.text, dst_fp, FP_SCRATCH2 as u8)
                    }
                };
                self.set_fp_reg_width(dst.reg, width)?;
            }
        }
        Ok(())
    }
}

// ─── Free functions ──────────────────────────────────────────────────────────

fn defaulted_fp_transient_count(program: &MachineProgram) -> usize {
    if program.fp_transient_count != 0 {
        return program.fp_transient_count as usize;
    }
    let fp_bank_count = program.reg_count.saturating_sub(program.first_fp_reg) as usize;
    fp_bank_count.min(2)
}

fn is_fallthrough_edge(
    compiler: &FunctionCompiler<'_>,
    target: MachineBlockId,
    args: &[MachineValue],
    fallthrough: Option<MachineBlockId>,
) -> bool {
    fallthrough == Some(target) && compiler.is_identity_edge(target, args)
}

fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cc {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cc::E,
        (MachineCompareKind::Ne, _) => Cc::NE,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cc::L,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cc::B,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cc::G,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cc::A,
        (MachineCompareKind::Le, MachineSign::Signed) => Cc::LE,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cc::BE,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cc::GE,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cc::AE,
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
        MachineTrapKind::StackOverflow => 7,
        MachineTrapKind::HelperFailure => 8,
    }
}

const MACHINE_TRAP_KIND_COUNT: usize = 9;

fn trap_kind_index(kind: MachineTrapKind) -> usize {
    trap_code(kind) as usize
}

/// Map Wasm float comparison kind to x86_64 condition code.
///
/// x86_64 UCOMISD/UCOMISS set flags:
///   ordered & equal: ZF=1, PF=0, CF=0 → use E (but need to check PF for NaN)
///   unordered (NaN): ZF=1, PF=1, CF=1
///
/// Wasm semantics: NaN is false for all relations except Ne (Ne is true for NaN).
fn map_float_cond(kind: MachineCompareKind) -> Cc {
    match kind {
        // Eq: ordered & equal. Use E with NaN handled by caller if needed.
        // After UCOMISD: E is true when ZF=1 (both equal and unordered).
        // For proper NaN handling, caller must also check PF. But for now,
        // use Cc::E — the lowering ensures NaN is handled separately if needed.
        // Actually, for Wasm: use AE-based pairs. Simplest:
        //   Eq → JE (then JNP to skip NaN case — but we'll handle this in float_branch)
        MachineCompareKind::Eq => Cc::E,
        // Ne → JNE
        MachineCompareKind::Ne => Cc::NE,
        // Lt: ordered & less-than → JB (CF=1, for unsigned/unordered comparison)
        MachineCompareKind::Lt => Cc::B,
        // Gt: ordered & greater-than → JA
        MachineCompareKind::Gt => Cc::A,
        // Le: ordered & less-or-equal → JBE
        MachineCompareKind::Le => Cc::BE,
        // Ge: ordered & greater-or-equal → JAE
        MachineCompareKind::Ge => Cc::AE,
    }
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

fn convert_result_float_width(op: MachineConvertOp) -> Option<MachineFloatWidth> {
    Some(match op {
        MachineConvertOp::F32ConvertI32S
        | MachineConvertOp::F32ConvertI32U
        | MachineConvertOp::F32ConvertI64S
        | MachineConvertOp::F32ConvertI64U
        | MachineConvertOp::F32DemoteF64
        | MachineConvertOp::F32ReinterpretI32 => MachineFloatWidth::F32,
        MachineConvertOp::F64ConvertI32S
        | MachineConvertOp::F64ConvertI32U
        | MachineConvertOp::F64ConvertI64S
        | MachineConvertOp::F64ConvertI64U
        | MachineConvertOp::F64PromoteF32
        | MachineConvertOp::F64ReinterpretI64 => MachineFloatWidth::F64,
        _ => return None,
    })
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
/// On System V AMD64, a 2-field repr(C) struct of u64s is returned in RAX and RDX.
#[repr(C)]
pub(crate) struct TruncResult {
    pub status: u64,
    pub value: u64,
}

/// Trapping truncation helper called from generated code.
/// Returns status in RAX (0 = ok) and result in RDX via struct return.
pub(crate) unsafe extern "C" fn x86_64_trapping_trunc(
    ctx: *mut crate::vm::runtime::context::NativeContext,
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
            TruncResult {
                status: 1,
                value: 0,
            }
        }
    }
}

/// Saturating truncation helper called from generated code.
/// Returns result in RAX (no error possible for sat).
pub(crate) unsafe extern "C" fn x86_64_saturating_trunc(src_bits: u64, op_code: u64) -> u64 {
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
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return 0;
    }
    if value >= 2147483648.0_f32 {
        return from_i32(i32::MAX);
    }
    if value < -2147483648.0_f32 {
        return from_i32(i32::MIN);
    }
    from_i32(value as i32)
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= -1.0_f32 {
        return 0;
    }
    if value >= 4294967296.0_f32 {
        return u64::from(u32::MAX);
    }
    u64::from(value as u32)
}

fn trunc_sat_f64_to_i32_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        return 0;
    }
    if value >= 2147483648.0 {
        return from_i32(i32::MAX);
    }
    if value <= -2147483649.0 {
        return from_i32(i32::MIN);
    }
    from_i32(value as i32)
}

fn trunc_sat_f64_to_i32_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= -1.0 {
        return 0;
    }
    if value >= 4294967296.0 {
        return u64::from(u32::MAX);
    }
    u64::from(value as u32)
}

fn trunc_sat_f32_to_i64_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return 0;
    }
    if value >= 9223372036854775808.0_f32 {
        return from_i64(i64::MAX);
    }
    if value < -9223372036854775808.0_f32 {
        return from_i64(i64::MIN);
    }
    from_i64(value as i64)
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= -1.0_f32 {
        return 0;
    }
    if value >= 18446744073709551616.0_f32 {
        return u64::MAX;
    }
    value as u64
}

fn trunc_sat_f64_to_i64_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        return 0;
    }
    if value >= 9223372036854775808.0 {
        return from_i64(i64::MAX);
    }
    if value < -9223372036854775808.0 {
        return from_i64(i64::MIN);
    }
    from_i64(value as i64)
}

fn trunc_sat_f64_to_i64_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= -1.0 {
        return 0;
    }
    if value >= 18446744073709551616.0 {
        return u64::MAX;
    }
    value as u64
}
