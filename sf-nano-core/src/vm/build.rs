use crate::collections::{self, phase_span, phase_span_with_function};
use tracked_alloc::rc::Rc;

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::module::entities::FunctionSpec;

/// Minimal native stats surface for CLI/debug output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeStatsSnapshot {
    pub groups: usize,
    pub ssa_ops: usize,
    pub mir_ops: usize,
    pub bytes_emitted: usize,
    pub groups_skipped: usize,
    pub ops_skipped: usize,
}

static STATS_GROUPS: AtomicUsize = AtomicUsize::new(0);
static STATS_SSA_OPS: AtomicUsize = AtomicUsize::new(0);
static STATS_MIR_OPS: AtomicUsize = AtomicUsize::new(0);
static STATS_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn set_native_stats(
    groups: usize,
    ssa_ops: usize,
    mir_ops: usize,
    bytes_emitted: usize,
) {
    STATS_GROUPS.store(groups, Ordering::Relaxed);
    STATS_SSA_OPS.store(ssa_ops, Ordering::Relaxed);
    STATS_MIR_OPS.store(mir_ops, Ordering::Relaxed);
    STATS_BYTES.store(bytes_emitted, Ordering::Relaxed);
}

#[inline]
pub fn native_stats_snapshot() -> NativeStatsSnapshot {
    NativeStatsSnapshot {
        groups: STATS_GROUPS.load(Ordering::Relaxed),
        ssa_ops: STATS_SSA_OPS.load(Ordering::Relaxed),
        mir_ops: STATS_MIR_OPS.load(Ordering::Relaxed),
        bytes_emitted: STATS_BYTES.load(Ordering::Relaxed),
        groups_skipped: 0,
        ops_skipped: 0,
    }
}

#[inline]
pub fn native_stats() -> (usize, usize) {
    (
        STATS_GROUPS.load(Ordering::Relaxed),
        STATS_SSA_OPS.load(Ordering::Relaxed),
    )
}

#[inline]
pub const fn native_capacity_skips() -> (usize, usize) {
    (0, 0)
}

#[cfg(sf_ir_dump)]
use crate::vm::debug::ir_dump;
use crate::vm::entities::TableDispatchMode;
use crate::vm::{backend::BackendConfig, entities::ModuleInst};
use crate::{
    error::WasmError,
    vm::{
        arch,
        arch::common::{
            helpers::page_align_function,
            types::{DirectCallPatch, DirectCallPatchSite},
        },
        machine::{
            derive_param_locs_from_types, derive_return_abi,
            lower_module_with_table_dispatch_modes,
            lower_single_function_with_table_dispatch_modes,
            machine_ir::{
                MachineFrameRegion, MachineFuncId, MachineFunctionAbi, MachineModuleAbi,
                MachineParamLoc, MachineReturnAbi,
            },
            optimize_function, optimize_module, ConstPoolBuilder, LowerFunctionInput,
            LowerModuleInput, LoweredMachineModule,
        },
        middle::{
            frame::{plan_frame_layout, FrameLayoutPlan, FrameSpan},
            prepare_function, ModuleFacts, PrepareInput,
        },
        runtime::{
            code::{
                AlignedConstData, CodegenModuleView, CompiledNativeModule, NativeCode,
                NativeCodeCache, NativeRootEntry,
            },
            code_buf::CodeBuffer,
            dispatch_view::{NativeLocalCallInfo32, NativeLocalCallInfo64},
        },
        store::Store,
        template,
        wasm::{context::CompileContext, decode, semantic_ir::SemanticProgram},
    },
};

const COMPILE_EXPANSION_FACTOR: u32 = 160;
const OVER_BUDGET_TEMPLATE_UNSUPPORTED: &str =
    "function exceeds compiler RAM budget and is not supported by template jit";

#[inline]
fn full_optimization_for_bytecode_size(bytecode_size: usize) -> bool {
    let budget = crate::runtime_config().compiler_ram_budget_bytes;
    if budget == u32::MAX {
        return true;
    }
    let Ok(bytecode_size) = u32::try_from(bytecode_size) else {
        return false;
    };
    bytecode_size
        .checked_mul(COMPILE_EXPANSION_FACTOR)
        .map(|estimated| estimated <= budget)
        .unwrap_or(false)
}

#[inline]
fn over_budget_template_unsupported() -> WasmError {
    WasmError::exhaustion(OVER_BUDGET_TEMPLATE_UNSUPPORTED)
}

fn decode_function_semantic(
    module: &ModuleInst,
    store: &Store,
    spec: &FunctionSpec,
) -> Result<SemanticProgram, WasmError> {
    let params = spec.func_type().params().len() as u16;
    let local_count = params.saturating_add(spec.locals().len() as u16);
    let results = spec.func_type().results().len() as u16;
    let mut local_types = collections::Vec::with_capacity(local_count as usize);
    local_types.extend_from_slice(spec.func_type().params());
    local_types.extend_from_slice(spec.locals());
    decode::decode_to_semantic_ir(
        spec.code(),
        CompileContext::with_value_types(
            &module.types,
            store,
            params,
            local_count,
            results,
            &local_types,
            spec.func_type().results(),
        ),
    )
    .map_err(|_err| WasmError::internal("native decode failed for function"))
}

#[derive(Clone, Debug)]
struct FunctionStaticSummary {
    id: MachineFuncId,
    frame: FrameLayoutPlan,
    param_locs: collections::Vec<MachineParamLoc>,
    result_count: u16,
    return_abi: MachineReturnAbi,
}

impl FunctionStaticSummary {
    fn abi(self) -> MachineFunctionAbi {
        let return_results = frame_return_results(self.frame, self.result_count);
        MachineFunctionAbi {
            id: self.id,
            param_locs: self.param_locs,
            frame_prefix_slots: self.frame.frame_prefix_size,
            total_frame_slots: self.frame.total_slots(),
            helper_scratch: self.frame.call_scratch.map(frame_span_region),
            return_results,
            return_abi: self.return_abi,
            init_locals: collections::Vec::new(),
        }
    }
}

fn frame_return_results(frame: FrameLayoutPlan, result_count: u16) -> Option<MachineFrameRegion> {
    (result_count > 0).then(|| frame_span_region(frame.return_results(result_count)))
}

fn frame_return_abi(return_results: Option<MachineFrameRegion>) -> MachineReturnAbi {
    return_results
        .map(|results| MachineReturnAbi::FrameFallback { results })
        .unwrap_or(MachineReturnAbi::None)
}

#[derive(Clone, Debug)]
struct FunctionEmitSummary {
    entry: NativeRootEntry,
    internal_entry_addr: usize,
    text_len: usize,
    #[cfg(sf_has_guard_pages)]
    body_local_error_addr: usize,
    #[cfg(sf_jitdump)]
    debug_regions: collections::Vec<crate::vm::arch::common::types::DebugRegion>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingPatchEncoding {
    #[cfg(sf_backend_riscv32)]
    U32,
    #[cfg(any(
        sf_backend_arm64,
        sf_backend_riscv64,
        sf_backend_x64,
        sf_backend_emu64,
        sf_backend_emu32
    ))]
    U64,
    #[cfg(any(sf_backend_armv7a, sf_backend_thumbm))]
    MovwMovt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingPatch {
    function_base: usize,
    patch: DirectCallPatch,
}

#[derive(Debug)]
struct StreamingCompiledModule {
    backend: BackendConfig,
    abi: MachineModuleAbi,
    aligned_consts: collections::Vec<AlignedConstData>,
}

impl StreamingCompiledModule {
    fn new(backend: BackendConfig, abi: MachineModuleAbi) -> Self {
        Self {
            backend,
            abi,
            aligned_consts: collections::Vec::new(),
        }
    }

    fn sync_consts(&mut self, const_pool: &ConstPoolBuilder) -> Result<(), WasmError> {
        while self.aligned_consts.len() < const_pool.records().len() {
            let record = const_pool
                .records()
                .get(self.aligned_consts.len())
                .ok_or_else(|| WasmError::internal("const pool record is out of range"))?;
            self.aligned_consts.push(AlignedConstData::new(record)?);
        }
        Ok(())
    }

    fn finish(self, active_backend: arch::NativeBackend) -> CompiledNativeModule {
        CompiledNativeModule::from_streamed(
            active_backend,
            self.backend,
            self.abi,
            self.aligned_consts,
        )
    }
}

impl CodegenModuleView for StreamingCompiledModule {
    fn backend(&self) -> BackendConfig {
        self.backend
    }

    fn runtime_for(&self, id: MachineFuncId) -> Option<&MachineFunctionAbi> {
        self.abi.functions.get(id.0 as usize)
    }

    fn const_ptr(&self, id: crate::vm::machine::machine_ir::MachineConstId) -> Option<*const u8> {
        self.aligned_consts
            .get(id.0 as usize)
            .map(AlignedConstData::as_ptr)
    }
}

#[inline]
fn frame_span_region(span: FrameSpan) -> MachineFrameRegion {
    MachineFrameRegion {
        base_slot: span.start.0,
        slots: span.count,
    }
}

fn patch_code_buffer_word(
    active_backend: arch::NativeBackend,
    executable: &mut crate::vm::runtime::code_buf::CodeBuffer,
    code_offset: usize,
    value: usize,
) {
    match patch_encoding(active_backend) {
        #[cfg(sf_backend_riscv32)]
        PendingPatchEncoding::U32 => executable.patch_u32(code_offset, value as u32),
        #[cfg(any(
            sf_backend_arm64,
            sf_backend_riscv64,
            sf_backend_x64,
            sf_backend_emu64,
            sf_backend_emu32
        ))]
        PendingPatchEncoding::U64 => executable.patch_u64(code_offset, value as u64),
        #[cfg(any(sf_backend_armv7a, sf_backend_thumbm))]
        PendingPatchEncoding::MovwMovt => {
            #[cfg(any(sf_backend_armv7a, sf_backend_thumbm))]
            {
                crate::vm::arch::arm32::compile::patch_movw_movt_code_buffer(
                    executable,
                    code_offset,
                    value as u32,
                );
            }
            #[cfg(not(any(sf_backend_armv7a, sf_backend_thumbm)))]
            {
                let _ = (executable, code_offset, value);
                unreachable!("movw/movt patching is unavailable on this target");
            }
        }
    }
}

fn patch_direct_call_code_buffer(
    active_backend: arch::NativeBackend,
    executable: &mut crate::vm::runtime::code_buf::CodeBuffer,
    function_base: usize,
    patch: &DirectCallPatch,
    callee_addr: usize,
) -> Result<(), WasmError> {
    match patch.site {
        DirectCallPatchSite::AddressLiteral { offset } => {
            let code_offset = function_base
                .checked_add(offset)
                .ok_or_else(|| WasmError::internal("direct-call patch offset overflow"))?;
            patch_code_buffer_word(active_backend, executable, code_offset, callee_addr);
            Ok(())
        }
        #[cfg(sf_backend_arm64)]
        DirectCallPatchSite::Arm64Bl {
            inst_offset,
            fallback_veneer_offset,
            fallback_literal_offset,
        } => patch_arm64_direct_branch_code_buffer(
            executable,
            function_base,
            inst_offset,
            fallback_veneer_offset,
            fallback_literal_offset,
            callee_addr,
            true,
        ),
        #[cfg(sf_backend_arm64)]
        DirectCallPatchSite::Arm64B {
            inst_offset,
            fallback_veneer_offset,
            fallback_literal_offset,
        } => patch_arm64_direct_branch_code_buffer(
            executable,
            function_base,
            inst_offset,
            fallback_veneer_offset,
            fallback_literal_offset,
            callee_addr,
            false,
        ),
    }
}

#[cfg(sf_backend_arm64)]
fn patch_arm64_direct_branch_code_buffer(
    executable: &mut crate::vm::runtime::code_buf::CodeBuffer,
    function_base: usize,
    inst_offset: usize,
    fallback_veneer_offset: usize,
    fallback_literal_offset: usize,
    callee_addr: usize,
    link: bool,
) -> Result<(), WasmError> {
    let fallback_literal_code_offset = function_base
        .checked_add(fallback_literal_offset)
        .ok_or_else(|| WasmError::internal("arm64 fallback literal offset overflow"))?;
    executable.patch_u64(fallback_literal_code_offset, callee_addr as u64);

    let inst_code_offset = function_base
        .checked_add(inst_offset)
        .ok_or_else(|| WasmError::internal("arm64 direct branch offset overflow"))?;
    let inst_addr = unsafe { executable.as_ptr().add(inst_code_offset) } as usize;
    let fallback_code_offset = function_base
        .checked_add(fallback_veneer_offset)
        .ok_or_else(|| WasmError::internal("arm64 fallback veneer offset overflow"))?;
    let fallback_addr = unsafe { executable.as_ptr().add(fallback_code_offset) } as usize;
    let target = if arm64_branch_delta_words(inst_addr, callee_addr).is_ok() {
        callee_addr
    } else {
        fallback_addr
    };
    let delta_words = arm64_branch_delta_words(inst_addr, target)?;
    let encoded = if link {
        arch::arm64::enc::bl(delta_words)
    } else {
        arch::arm64::enc::b(delta_words)
    };
    executable.patch_u32(inst_code_offset, encoded);
    Ok(())
}

#[cfg(sf_backend_arm64)]
fn arm64_branch_delta_words(from_addr: usize, to_addr: usize) -> Result<i32, WasmError> {
    let delta_bytes = to_addr as isize - from_addr as isize;
    if delta_bytes & 0b11 != 0 {
        return Err(WasmError::internal(
            "arm64 direct branch target is unaligned",
        ));
    }
    let delta_words = delta_bytes / 4;
    const IMM26_WORD_MIN: isize = -(1 << 25);
    const IMM26_WORD_MAX: isize = (1 << 25) - 1;
    if !(IMM26_WORD_MIN..=IMM26_WORD_MAX).contains(&delta_words) {
        return Err(WasmError::internal(
            "arm64 direct-call veneer is out of range",
        ));
    }
    Ok(delta_words as i32)
}

#[inline]
fn patch_encoding(active_backend: arch::NativeBackend) -> PendingPatchEncoding {
    match active_backend {
        #[cfg(sf_backend_arm64)]
        arch::NativeBackend::Arm64 => PendingPatchEncoding::U64,
        #[cfg(sf_backend_armv7a)]
        arch::NativeBackend::Armv7a => PendingPatchEncoding::MovwMovt,
        #[cfg(sf_backend_thumbm)]
        arch::NativeBackend::ThumbM => PendingPatchEncoding::MovwMovt,
        #[cfg(sf_backend_riscv32)]
        arch::NativeBackend::Riscv32 => PendingPatchEncoding::U32,
        #[cfg(sf_backend_riscv64)]
        arch::NativeBackend::Riscv64 => PendingPatchEncoding::U64,
        #[cfg(sf_backend_x64)]
        arch::NativeBackend::X86_64 => PendingPatchEncoding::U64,
        #[cfg(sf_backend_emu64)]
        arch::NativeBackend::Emu64 => PendingPatchEncoding::U64,
        #[cfg(sf_backend_emu32)]
        arch::NativeBackend::Emu32 => PendingPatchEncoding::U64,
    }
}

#[inline]
const fn is_emulator_backend(_active_backend: arch::NativeBackend) -> bool {
    cfg!(any(sf_backend_emu64, sf_backend_emu32))
}

fn build_static_summaries(
    module: &ModuleInst,
    store: &Store,
    backend: BackendConfig,
) -> Result<(MachineModuleAbi, collections::Vec<bool>), WasmError> {
    let mut abi = MachineModuleAbi {
        functions: (0..module.functions.len())
            .map(|index| MachineFunctionAbi {
                id: MachineFuncId(index as u32),
                ..MachineFunctionAbi::default()
            })
            .collect(),
    };
    let mut is_local_func = collections::vec![false; module.functions.len()];
    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let template_requested = !full_optimization_for_bytecode_size(spec.code().len());
        if template_requested {
            let template_scan_phase =
                phase_span_with_function("template_scan", Some(func_idx as u32));
            match template::scan_function(backend, spec, !module.memories.is_empty()) {
                Ok(scan) => {
                    let frame = plan_frame_layout(
                        scan.local_count,
                        scan.max_stack_height,
                        backend.call_scratch_slots,
                    );
                    let return_results = frame_return_results(frame, scan.result_count);
                    let summary = FunctionStaticSummary {
                        id: MachineFuncId(func_idx as u32),
                        frame,
                        param_locs: derive_param_locs_from_types(
                            spec.func_type().params(),
                            backend,
                        ),
                        result_count: scan.result_count,
                        return_abi: frame_return_abi(return_results),
                    };
                    abi.functions[func_idx] = summary.abi();
                    is_local_func[func_idx] = true;
                    drop(template_scan_phase);
                    continue;
                }
                Err(err) if template::is_template_unsupported(&err) => {
                    drop(template_scan_phase);
                    return Err(over_budget_template_unsupported());
                }
                Err(err) => return Err(err),
            }
        }
        let sem_scan_function_phase = phase_span_with_function("sem_scan", Some(func_idx as u32));
        let semantic = decode_function_semantic(module, store, spec)?;
        let frame = plan_frame_layout(
            semantic.local_count,
            semantic.max_stack_height,
            backend.call_scratch_slots,
        );
        let result_count = spec.func_type().results().len() as u16;
        let return_results = frame_return_results(frame, result_count);
        let summary = FunctionStaticSummary {
            id: MachineFuncId(func_idx as u32),
            frame,
            param_locs: derive_param_locs_from_types(spec.func_type().params(), backend),
            result_count,
            return_abi: derive_return_abi(spec.func_type().results(), return_results, backend),
        };
        abi.functions[func_idx] = summary.abi();
        is_local_func[func_idx] = true;
        drop(sem_scan_function_phase);
    }
    Ok((abi, is_local_func))
}

fn emit_function_info_bytes(
    backend: BackendConfig,
    abi: &MachineModuleAbi,
    emitted: &[Option<FunctionEmitSummary>],
) -> collections::Vec<u8> {
    if backend.gp_unit_bytes == 8 {
        let mut bytes = collections::Vec::with_capacity(
            abi.functions.len() * core::mem::size_of::<NativeLocalCallInfo64>(),
        );
        for (func_idx, runtime) in abi.functions.iter().enumerate() {
            let info = NativeLocalCallInfo64 {
                entry: emitted
                    .get(func_idx)
                    .and_then(|slot| {
                        slot.as_ref()
                            .map(|summary| summary.internal_entry_addr as u64)
                    })
                    .unwrap_or(0),
                total_frame_bytes: u64::from(runtime.total_frame_slots) * 8,
                frame_prefix_slots: u64::from(runtime.frame_prefix_slots),
            };
            bytes.extend_from_slice(&info.entry.to_le_bytes());
            bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
            bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
        }
        return bytes;
    }

    let mut bytes = collections::Vec::with_capacity(
        abi.functions.len() * core::mem::size_of::<NativeLocalCallInfo32>(),
    );
    for (func_idx, runtime) in abi.functions.iter().enumerate() {
        let info = NativeLocalCallInfo32 {
            entry: emitted
                .get(func_idx)
                .and_then(|slot| {
                    slot.as_ref()
                        .map(|summary| summary.internal_entry_addr as u32)
                })
                .unwrap_or(0),
            total_frame_bytes: u32::from(runtime.total_frame_slots) * 8,
            frame_prefix_slots: u32::from(runtime.frame_prefix_slots),
        };
        bytes.extend_from_slice(&info.entry.to_le_bytes());
        bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
        bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
    }
    bytes
}

fn compile_full_streaming_function(
    active_backend: arch::NativeBackend,
    backend: BackendConfig,
    module: &ModuleInst,
    store: &Store,
    spec: &FunctionSpec,
    func_idx: usize,
    func_id: MachineFuncId,
    compiled_view: &mut StreamingCompiledModule,
    const_pool: &mut ConstPoolBuilder,
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    groups: &mut usize,
    ssa_ops: &mut usize,
    mir_ops: &mut usize,
    executable: &mut CodeBuffer,
    #[cfg(sf_has_guard_pages)] use_guard_pages: bool,
    #[cfg(sf_has_guard_pages)] use_stack_guard_pages: bool,
) -> Result<arch::common::types::FunctionArtifact, WasmError> {
    let sem_decode_function_phase = phase_span_with_function("sem_decode", Some(func_idx as u32));
    let semantic = decode_function_semantic(module, store, spec)?;
    drop(sem_decode_function_phase);

    let ssa_lower_function_phase = phase_span_with_function("ssa_lower", Some(func_idx as u32));
    let prepared = prepare_function(
        PrepareInput {
            config: backend,
            function_index: Some(func_idx as u32),
        },
        ModuleFacts { is_local_func },
        semantic,
    )?;
    drop(ssa_lower_function_phase);

    *groups += 1;
    *ssa_ops += prepared
        .ssa
        .blocks
        .iter()
        .map(|block| block.ops.len())
        .sum::<usize>();

    let result_count = spec.func_type().results().len() as u16;
    let (mut machine, _abi) = lower_single_function_with_table_dispatch_modes(
        backend,
        LowerFunctionInput {
            id: func_id,
            frame: prepared.frame,
            ssa: prepared.ssa,
            result_count,
        },
        &mut compiled_view.abi.functions,
        is_local_func,
        table_dispatch_modes,
        const_pool,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages,
        #[cfg(sf_has_guard_pages)]
        use_stack_guard_pages,
    )?;
    optimize_function(&mut machine, backend);
    if backend.is_32bit_gp_target() {
        machine
            .program
            .validate_32bit_gp_target(backend.first_fp_reg(), backend)?;
    }
    *mir_ops += machine
        .program
        .blocks
        .iter()
        .map(|block| block.ops.len())
        .sum::<usize>();

    compiled_view.sync_consts(const_pool)?;

    let arch_lower_function_phase =
        phase_span_with_function("arch_lower_func", Some(func_idx as u32));
    let artifact = arch::dispatch_compile_function_into_buffer(
        active_backend,
        compiled_view,
        &machine,
        executable,
    )?;
    drop(arch_lower_function_phase);
    Ok(artifact)
}

fn finish_native_compile_streaming(
    active_backend: arch::NativeBackend,
    backend: BackendConfig,
    module: &ModuleInst,
    store: &Store,
) -> Result<(), WasmError> {
    let (abi, is_local_func) = build_static_summaries(module, store, backend)?;
    let mut compiled_view = StreamingCompiledModule::new(backend, abi);
    let mut const_pool = ConstPoolBuilder::new();
    let mut groups = 0usize;
    let mut ssa_ops = 0usize;
    let mut mir_ops = 0usize;
    let mut emitted: collections::Vec<Option<FunctionEmitSummary>> =
        collections::vec![None; module.functions.len()];
    let mut pending_direct_patches =
        collections::vec![collections::Vec::new(); module.functions.len()];

    #[cfg(sf_has_guard_pages)]
    let use_guard_pages = module
        .memories
        .first()
        .map(|m| m.has_guard_pages())
        .unwrap_or(false)
        && backend.gp_unit_bytes == 8;
    #[cfg(sf_has_guard_pages)]
    let use_stack_guard_pages = backend.gp_unit_bytes == 8;

    let arch_lower_phase = phase_span("arch_lower");
    let mut executable = CodeBuffer::new().map_err(WasmError::internal)?;
    executable.begin_write();
    executable.reset();
    let base_ptr = executable.as_ptr();

    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };

        let func_id = MachineFuncId(func_idx as u32);
        let stream_base = executable.len();
        let scratch_base = page_align_function(stream_base, 0);
        let initial_padding = scratch_base.saturating_sub(stream_base);
        if initial_padding > 0 {
            arch::dispatch_emit_nop_padding(active_backend, &mut executable, initial_padding);
        }

        let template_requested = !full_optimization_for_bytecode_size(spec.code().len());
        let artifact = if template_requested {
            let template_phase = phase_span_with_function("template_jit", Some(func_idx as u32));
            match template::compile_function_into_buffer(
                active_backend,
                &compiled_view,
                spec,
                func_id,
                &mut executable,
                !module.memories.is_empty(),
            ) {
                Ok(artifact) => {
                    groups += 1;
                    drop(template_phase);
                    artifact
                }
                Err(err) if template::is_template_unsupported(&err) => {
                    executable.set_len(scratch_base);
                    drop(template_phase);
                    return Err(over_budget_template_unsupported());
                }
                Err(err) => return Err(err),
            }
        } else {
            compile_full_streaming_function(
                active_backend,
                backend,
                module,
                store,
                spec,
                func_idx,
                func_id,
                &mut compiled_view,
                &mut const_pool,
                &is_local_func,
                module.table_dispatch_modes(),
                &mut groups,
                &mut ssa_ops,
                &mut mir_ops,
                &mut executable,
                #[cfg(sf_has_guard_pages)]
                use_guard_pages,
                #[cfg(sf_has_guard_pages)]
                use_stack_guard_pages,
            )?
        };

        let text_len = artifact.text.len();
        let function_base = page_align_function(stream_base, text_len);
        let extra_padding = function_base.saturating_sub(scratch_base);
        if extra_padding > 0 {
            executable.move_region(scratch_base, scratch_base + extra_padding, text_len);
            executable.set_len(scratch_base);
            arch::dispatch_emit_nop_padding(active_backend, &mut executable, extra_padding);
            executable.set_len(function_base + text_len);
        }

        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as usize;
            // arm32 Thumb-2 interworking: continuation / local-ptr addresses
            // are branched to via BX, so the LSB must signal Thumb mode.
            // See arm32::thumb_interworking_bit. On A32 / arm64 / x86_64 the
            // helper passes through unchanged.
            #[cfg(sf_arm32_isa_thumb)]
            let target_addr = arch::arm32::thumb_interworking_bit(target_addr);
            patch_code_buffer_word(
                active_backend,
                &mut executable,
                function_base + patch.literal_offset,
                target_addr,
            );
        }

        let internal_entry_addr =
            unsafe { base_ptr.add(function_base + artifact.internal_entry_offset) } as usize;
        // arm32 Thumb-2: internal call targets (reached via BLX reg after
        // MOVW/MOVT) need LSB=1 so the CPU stays in Thumb on entry.
        #[cfg(sf_arm32_isa_thumb)]
        let internal_entry_addr = arch::arm32::thumb_interworking_bit(internal_entry_addr);
        for patch in &artifact.direct_call_patches {
            let callee_idx = patch.callee.0 as usize;
            if callee_idx == func_idx {
                patch_direct_call_code_buffer(
                    active_backend,
                    &mut executable,
                    function_base,
                    patch,
                    internal_entry_addr,
                )?;
                continue;
            }
            if let Some(summary) = emitted.get(callee_idx).and_then(|slot| slot.as_ref()) {
                patch_direct_call_code_buffer(
                    active_backend,
                    &mut executable,
                    function_base,
                    patch,
                    summary.internal_entry_addr,
                )?;
            } else {
                let pending = pending_direct_patches.get_mut(callee_idx).ok_or_else(|| {
                    WasmError::internal("direct callee patch list is out of range")
                })?;
                pending.push(PendingPatch {
                    function_base,
                    patch: *patch,
                });
            }
        }

        let entry = unsafe { executable.fn_ptr::<NativeRootEntry>(function_base) };
        // Rust code (A32) enters this JITted function via `blx reg`; on
        // arm32 Thumb-2 builds, the pointer needs LSB=1 to switch the CPU
        // to Thumb mode on entry.
        #[cfg(sf_arm32_isa_thumb)]
        let entry: NativeRootEntry = unsafe {
            core::mem::transmute::<usize, NativeRootEntry>(arch::arm32::thumb_interworking_bit(
                entry as usize,
            ))
        };
        emitted[func_idx] = Some(FunctionEmitSummary {
            entry,
            internal_entry_addr,
            text_len,
            #[cfg(sf_has_guard_pages)]
            body_local_error_addr: unsafe {
                executable.ptr(function_base + artifact.body_local_error_offset)
            } as usize,
            #[cfg(sf_jitdump)]
            debug_regions: artifact.debug_regions,
        });

        let pending = core::mem::take(
            pending_direct_patches
                .get_mut(func_idx)
                .ok_or_else(|| WasmError::internal("pending direct patch list is out of range"))?,
        );
        for patch in pending {
            let callee_addr = emitted
                .get(func_idx)
                .and_then(|slot| slot.as_ref().map(|summary| summary.internal_entry_addr))
                .ok_or_else(|| WasmError::internal("pending direct callee address is missing"))?;
            patch_direct_call_code_buffer(
                active_backend,
                &mut executable,
                patch.function_base,
                &patch.patch,
                callee_addr,
            )?;
        }
    }

    if pending_direct_patches
        .iter()
        .any(|patches| !patches.is_empty())
    {
        return Err(WasmError::internal(
            "native streaming compile finished with unresolved direct-call patches",
        ));
    }

    let function_info_table_offset = executable.len();
    let function_info_bytes = emit_function_info_bytes(backend, &compiled_view.abi, &emitted);
    executable.emit_bytes(&function_info_bytes);
    let written_len = executable.len();
    executable.finish_write(0, written_len);
    let function_info_base = unsafe { executable.as_ptr().add(function_info_table_offset) };
    drop(arch_lower_phase);

    #[cfg(sf_jitdump)]
    {
        let module_name = &module.name;
        for (func_idx, entry) in emitted.iter().enumerate() {
            let Some(entry) = entry else {
                continue;
            };
            let func_base = entry.entry as *const u8;
            for region in &entry.debug_regions {
                if region.len > 0 {
                    let region_start = unsafe { func_base.add(region.offset) };
                    let code_bytes =
                        unsafe { core::slice::from_raw_parts(region_start, region.len) };
                    let symbol = tracked_alloc::format!(
                        "jit::{}::func{}::{}",
                        module_name,
                        func_idx,
                        region.label
                    );
                    crate::vm::debug::jitdump::record_function(region_start, code_bytes, &symbol);
                }
            }
        }
    }

    #[cfg(sf_has_guard_pages)]
    {
        let ranges: collections::Vec<_> = emitted
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|entry| {
                (
                    entry.entry as usize,
                    entry.entry as usize + entry.text_len,
                    entry.body_local_error_addr,
                )
            })
            .collect();
        crate::vm::runtime::trap_signal::register_jit_ranges(&ranges);
    }

    let bytes: usize = emitted
        .iter()
        .filter_map(|slot| slot.as_ref().map(|entry| entry.text_len))
        .sum();
    set_native_stats(groups, ssa_ops, mir_ops, bytes);

    // Install the freshly-emitted buffer as the module's persistent
    // native buffer. Avoids the old swap pattern (which would allocate
    // a second CodeBuffer inside `native_code_buffer()` just to throw
    // it away after the swap) — significant on MCU targets where the
    // OS layer serves a fixed executable arena.
    module
        .install_native_code_buffer(executable)
        .map_err(WasmError::internal)?;

    let compiled = Rc::new(compiled_view.finish(active_backend));
    compiled.publish_local_call_infos(function_info_base);

    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let mut code = NativeCode::new(Rc::clone(&compiled), MachineFuncId(func_idx as u32));
        code = code.with_entry(
            emitted
                .get(func_idx)
                .and_then(|slot| slot.as_ref().map(|entry| entry.entry)),
        );
        spec.set_native_code(code, NativeCodeCache::compiled());
    }

    Ok(())
}

fn finish_native_compile(
    active_backend: arch::NativeBackend,
    backend: BackendConfig,
    module: &ModuleInst,
    groups: usize,
    ssa_ops: usize,
    lowered: LoweredMachineModule,
    #[cfg(sf_ir_dump)] dump_lir_inputs: Option<&[ir_dump::DumpFunctionLir]>,
) -> Result<(), WasmError> {
    let mir_ops: usize = lowered
        .module
        .functions
        .iter()
        .map(|f| f.program.blocks.iter().map(|b| b.ops.len()).sum::<usize>())
        .sum();
    let arch_lower_phase = phase_span("arch_lower");
    let mut compiled = Rc::new(CompiledNativeModule::new(
        active_backend,
        backend,
        lowered.module,
        lowered.abi,
    )?);
    let entries = arch::dispatch_compile_module(active_backend, module, &compiled)?;
    drop(arch_lower_phase);

    let bytes: usize = entries
        .iter()
        .filter_map(|e| e.as_ref().map(|e| e.text_len))
        .sum();
    set_native_stats(groups, ssa_ops, mir_ops, bytes);

    #[cfg(sf_ir_dump)]
    if ir_dump::dump_enabled() {
        let code_slices: collections::Vec<(u32, &[u8])> = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                entry.as_ref().map(|e| {
                    // On Thumb-2 builds `e.entry` carries the interworking
                    // bit (LSB=1) so callers BLX correctly. The raw text
                    // starts at the masked address; reading from the tagged
                    // pointer would shift every function by one byte.
                    let ptr = (e.entry as usize & !1usize) as *const u8;
                    let len = e.text_len;
                    (idx as u32, unsafe { core::slice::from_raw_parts(ptr, len) })
                })
            })
            .collect();
        let dump_regions: collections::Vec<ir_dump::DumpFunctionRegions> = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                entry.as_ref().map(|e| ir_dump::DumpFunctionRegions {
                    func_idx: idx as u32,
                    regions: e.debug_regions.clone(),
                })
            })
            .collect();
        let _ = ir_dump::write_module_dump(
            &module.name,
            module.functions.len(),
            dump_lir_inputs.unwrap_or(&[]),
            compiled.module(),
            compiled.abi(),
            &code_slices,
            &dump_regions,
        );
    }

    let keep_machine_ir = is_emulator_backend(active_backend);
    if !keep_machine_ir {
        if let Some(compiled_mut) = Rc::get_mut(&mut compiled) {
            compiled_mut.strip_machine_ir_for_runtime();
        }
    }

    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let mut code = NativeCode::new(Rc::clone(&compiled), MachineFuncId(func_idx as u32));
        let native_entry = entries.get(func_idx).and_then(|e| e.as_ref());
        code = code.with_entry(native_entry.map(|e| e.entry));
        spec.set_native_code(code, NativeCodeCache::compiled());
    }

    Ok(())
}

pub(crate) fn ensure_module_compiled(store: &Store) -> Result<(), WasmError> {
    let active_backend = arch::active_native_backend()
        .map_err(|_err| WasmError::invalid("native backend unavailable"))?;
    let backend = arch::active_backend_config()
        .map_err(|_err| WasmError::invalid("native backend unavailable"))?;
    let module = store.module();
    let all_compiled = module
        .functions
        .iter()
        .filter_map(|func| func.spec())
        .all(|spec| {
            spec.get_native_code()
                .map(|code| {
                    code.compiled().backend_kind() == active_backend
                        && code.compiled().backend() == backend
                })
                .unwrap_or(false)
        });
    if all_compiled {
        return Ok(());
    }

    #[cfg(sf_ir_dump)]
    let ir_dump_enabled = ir_dump::dump_enabled();
    #[cfg(not(sf_ir_dump))]
    let ir_dump_enabled = false;
    let use_batch_pipeline = is_emulator_backend(active_backend) || ir_dump_enabled;
    if !use_batch_pipeline {
        return finish_native_compile_streaming(active_backend, backend, module, store);
    }

    let mut groups = 0usize;
    let mut ssa_ops = 0usize;
    // Module facts for the middle's final-SSA cache-signal derivation. A function
    // is local exactly when it has a spec (the same predicate
    // `build_static_summaries` uses); the batch path derives it directly from the
    // has-spec flags rather than re-running that scan. Table modes are the
    // module's current dispatch modes.
    let is_local_func: collections::Vec<bool> = module
        .functions
        .iter()
        .map(|func| func.spec().is_some())
        .collect();
    let mut prepared_functions: collections::Vec<LowerFunctionInput> = collections::Vec::new();
    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let sem_decode_function_phase =
            phase_span_with_function("sem_decode", Some(func_idx as u32));
        let semantic = decode_function_semantic(module, store, spec)?;
        drop(sem_decode_function_phase);

        let ssa_lower_function_phase = phase_span_with_function("ssa_lower", Some(func_idx as u32));
        let prepared = prepare_function(
            PrepareInput {
                config: backend,
                function_index: Some(func_idx as u32),
            },
            ModuleFacts {
                is_local_func: &is_local_func,
            },
            semantic,
        )?;
        drop(ssa_lower_function_phase);
        groups += 1;
        ssa_ops += prepared
            .ssa
            .blocks
            .iter()
            .map(|block| block.ops.len())
            .sum::<usize>();
        let func_id = MachineFuncId(func_idx as u32);
        let result_count = spec.func_type().results().len() as u16;
        prepared_functions.push(LowerFunctionInput {
            id: func_id,
            frame: prepared.frame,
            ssa: prepared.ssa,
            result_count,
        });
    }

    #[cfg(sf_ir_dump)]
    let dump_lir_inputs = if ir_dump::dump_enabled() {
        Some(
            prepared_functions
                .iter()
                .map(|prepared| ir_dump::DumpFunctionLir {
                    func_idx: prepared.id.0,
                    ssa: prepared.ssa.clone(),
                })
                .collect::<collections::Vec<_>>(),
        )
    } else {
        None
    };

    #[cfg(sf_has_guard_pages)]
    let use_guard_pages = module
        .memories
        .first()
        .map(|m| m.has_guard_pages())
        .unwrap_or(false);
    #[cfg(sf_has_guard_pages)]
    let use_guard_pages = use_guard_pages && backend.gp_unit_bytes == 8;
    #[cfg(sf_has_guard_pages)]
    let use_stack_guard_pages = backend.gp_unit_bytes == 8;
    let mut lowered = lower_module_with_table_dispatch_modes(
        LowerModuleInput {
            backend,
            functions: prepared_functions,
            #[cfg(sf_has_guard_pages)]
            use_guard_pages,
            #[cfg(sf_has_guard_pages)]
            use_stack_guard_pages,
        },
        module.table_dispatch_modes(),
    )?;
    let module_opt_phase = phase_span("module_opt");
    optimize_module(&mut lowered.module);
    if backend.is_32bit_gp_target() {
        lowered
            .module
            .validate_32bit_gp_target(backend.first_fp_reg())?;
    }
    drop(module_opt_phase);

    finish_native_compile(
        active_backend,
        backend,
        module,
        groups,
        ssa_ops,
        lowered,
        #[cfg(sf_ir_dump)]
        dump_lir_inputs.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use tracked_alloc::{boxed::Box, rc::Rc, string::String};

    use super::ensure_module_compiled;
    use crate::collections;
    use crate::{
        module::{
            entities::FunctionSpec,
            type_context::TypeContext,
            type_defs::{CompositeType, DefType, FunctionType},
        },
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            entities::{FunctionInst, MemInst, ModuleInst},
            store::Store,
        },
    };

    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    use crate::vm::arch::backend_mode_test_lock;

    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    struct CompiledBackendGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    fn enable_compiled_backend() -> CompiledBackendGuard {
        let lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        CompiledBackendGuard { _lock: lock }
    }

    #[cfg(sf_backend_emu32)]
    fn encode_u32_leb(mut value: u32, out: &mut collections::Vec<u8>) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    #[cfg(sf_backend_emu32)]
    fn local_get_code(index: u32) -> collections::Vec<u8> {
        let mut code = collections::vec![0x20];
        encode_u32_leb(index, &mut code);
        code.push(0x0b);
        code
    }

    #[cfg(sf_backend_emu32)]
    fn long_argument_caller_code(helper_params: &[ValueType]) -> collections::Vec<u8> {
        let mut code = collections::Vec::new();
        for (index, ty) in helper_params.iter().copied().enumerate() {
            if index + 1 == helper_params.len() {
                code.extend_from_slice(&[0x20, 0x00]);
                continue;
            }
            match ty {
                ValueType::I32 => code.extend_from_slice(&[0x41, 0x00]),
                ValueType::I64 => code.extend_from_slice(&[0x42, 0x00]),
                ValueType::F32 => code.extend_from_slice(&[0x43, 0x00, 0x00, 0x00, 0x00]),
                ValueType::F64 => {
                    code.extend_from_slice(&[0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
                }
                other => panic!("unexpected helper param type in regression: {:?}", other),
            }
        }
        code.extend_from_slice(&[0x10, 0x00, 0x0b]);
        code
    }

    fn func_def(
        params: collections::Vec<ValueType>,
        results: collections::Vec<ValueType>,
    ) -> Rc<DefType> {
        Rc::new(DefType {
            composite: CompositeType::Func(Rc::new(FunctionType::new(params, results))),
            supertypes: collections::vec![],
            is_final: true,
            rec_group: None,
        })
    }

    #[test]
    fn compiles_all_local_functions_once() {
        let types = TypeContext::new(collections::vec![
            func_def(collections::vec![], collections::vec![]),
            func_def(collections::vec![ValueType::I32], collections::vec![]),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec0 = FunctionSpec::new(
            Rc::new(FunctionType::new(collections::vec![], collections::vec![])),
            0,
        );
        spec0.set_code((&[0x0b][..]).into());
        let mut spec1 = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![],
            )),
            1,
        );
        spec1.set_code((&[0x20, 0x00, 0x1a, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: spec0,
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: spec1,
            type_index: 1,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("native compile should succeed");

        let first = store.module().functions[0]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("first native code");
        let second = store.module().functions[1]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("second native code");

        assert!(Rc::ptr_eq(first.compiled_rc(), second.compiled_rc()));
        assert_eq!(first.func_id().0, 0);
        assert_eq!(second.func_id().0, 1);
    }

    #[cfg(sf_backend_emu64)]
    #[test]
    fn emu64_compiled_module_publishes_local_call_info_table() {
        let _guard = enable_compiled_backend();

        let types = TypeContext::new(collections::vec![
            func_def(collections::vec![], collections::vec![]),
            func_def(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I32],
            ),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut spec0 = FunctionSpec::new(
            Rc::new(FunctionType::new(collections::vec![], collections::vec![])),
            0,
        );
        spec0.set_code((&[0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: spec0,
            type_index: 0,
        });

        let mut spec1 = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I32],
            )),
            1,
        );
        spec1.set_code((&[0x20, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: spec1,
            type_index: 1,
        });

        let store = Box::new(Store::new(module));
        ensure_module_compiled(&store).expect("emu64 native compile should succeed");

        let compiled = store.module().functions[0]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("first native code")
            .compiled();
        let local_call_infos = compiled.dispatch_metadata().local_call_infos();
        assert_eq!(local_call_infos.len(), 2);
        assert!(
            !local_call_infos.base().is_null(),
            "reference-backend compiled module must publish local call infos for indirect local calls"
        );
    }

    #[test]
    fn compiles_function_with_f64_local() {
        // (func (param f64) (result f64) (local.get 0))
        // Bytecode: local.get 0, end
        let types = TypeContext::new(collections::vec![func_def(
            collections::vec![ValueType::F64],
            collections::vec![ValueType::F64],
        )]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::F64],
                collections::vec![ValueType::F64],
            )),
            0,
        );
        spec.set_code((&[0x20, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f64 local function should compile");
    }

    #[test]
    fn compiles_function_with_f32_local_and_add() {
        // (func (param f32 f32) (result f32) (f32.add (local.get 0) (local.get 1)))
        // Bytecode: local.get 0, local.get 1, f32.add, end
        let types = TypeContext::new(collections::vec![func_def(
            collections::vec![ValueType::F32, ValueType::F32],
            collections::vec![ValueType::F32],
        )]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::F32, ValueType::F32],
                collections::vec![ValueType::F32],
            )),
            0,
        );
        spec.set_code((&[0x20, 0x00, 0x20, 0x01, 0x92, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f32 add function should compile");
    }

    #[test]
    fn compiles_function_with_f32_local_in_if() {
        // (func (param f32) (param i32) (result f32)
        //   (if (result f32) (local.get 1)
        //     (then (local.get 0))
        //     (else (f32.const 0))
        //   ))
        // Bytecode:
        //   local.get 1     ;; 0x20 0x01
        //   if (result f32)  ;; 0x04 0x7d
        //   local.get 0     ;; 0x20 0x00
        //   else             ;; 0x05
        //   f32.const 0     ;; 0x43 0x00 0x00 0x00 0x00
        //   end              ;; 0x0b
        //   end              ;; 0x0b
        let types = TypeContext::new(collections::vec![func_def(
            collections::vec![ValueType::F32, ValueType::I32],
            collections::vec![ValueType::F32],
        )]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::F32, ValueType::I32],
                collections::vec![ValueType::F32],
            )),
            0,
        );
        spec.set_code(
            (&[
                0x20, 0x01, // local.get 1
                0x04, 0x7d, // if (result f32)
                0x20, 0x00, // local.get 0
                0x05, // else
                0x43, 0x00, 0x00, 0x00, 0x00, // f32.const 0
                0x0b, // end
                0x0b, // end (function)
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f32 if/else function should compile");
    }

    #[test]
    fn compiles_f32_kahan_sum_style_loop() {
        // Reduced from `float_exprs.wast` `f32.kahan_sum`.
        //
        // The important shape is:
        // - three F32 locals
        // - nested `local.tee` on F32 values
        // - a loop backedge guarded by `br_if`
        // - a live F32 value carried through the block right up to the
        //   terminator
        //
        // Native ARM64 compilation was previously failing here with:
        // "missing float-width tracking for machine reg ... at bN terminator".
        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I32],
            collections::vec![ValueType::F32],
        ));
        let types = TypeContext::new(collections::vec![func_def(
            collections::vec![ValueType::I32, ValueType::I32],
            collections::vec![ValueType::F32],
        )]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_locals(collections::vec![
            ValueType::F32,
            ValueType::F32,
            ValueType::F32
        ]);
        spec.set_code(
            (&[
                0x02, 0x40, // block
                0x03, 0x40, // loop
                0x20, 0x00, // local.get 0
                0x2a, 0x02, 0x00, // f32.load align=2 offset=0
                0x20, 0x04, // local.get 4
                0x93, // f32.sub
                0x22, 0x04, // local.tee 4
                0x20, 0x03, // local.get 3
                0x92, // f32.add
                0x22, 0x02, // local.tee 2
                0x20, 0x03, // local.get 3
                0x93, // f32.sub
                0x20, 0x04, // local.get 4
                0x93, // f32.sub
                0x21, 0x04, // local.set 4
                0x20, 0x00, // local.get 0
                0x41, 0x04, // i32.const 4
                0x6a, // i32.add
                0x21, 0x00, // local.set 0
                0x20, 0x02, // local.get 2
                0x21, 0x03, // local.set 3
                0x20, 0x01, // local.get 1
                0x41, 0x7f, // i32.const -1
                0x6a, // i32.add
                0x22, 0x01, // local.tee 1
                0x0d, 0x00, // br_if 0
                0x0b, // end loop
                0x0b, // end block
                0x20, 0x02, // local.get 2
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        module.memories.push(
            MemInst::new(Limits::new(1, Some(1)).unwrap())
                .expect("test memory within runtime limits"),
        );
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f32 kahan-style loop should compile");
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn emu32_compiles_long_argument_list_internal_call() {
        let _guard = enable_compiled_backend();

        let helper_params = collections::vec![
            ValueType::F32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F64,
            ValueType::F32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I64,
            ValueType::I64,
            ValueType::I32,
            ValueType::I64,
            ValueType::I64,
            ValueType::F32,
            ValueType::I64,
            ValueType::I64,
            ValueType::I64,
            ValueType::I32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F64,
            ValueType::F32,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F64,
            ValueType::F64,
            ValueType::F32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I64,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F64,
            ValueType::I32,
            ValueType::I32,
            ValueType::I32,
            ValueType::I64,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::I64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::I32,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F32,
            ValueType::I64,
            ValueType::I64,
            ValueType::I32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I64,
            ValueType::F32,
            ValueType::F64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::I32,
            ValueType::F32,
            ValueType::I64,
            ValueType::I32,
        ];
        let helper_type = Rc::new(FunctionType::new(
            helper_params.clone(),
            collections::vec![ValueType::I32],
        ));
        let caller_type = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![
            func_def(helper_params.clone(), collections::vec![ValueType::I32]),
            func_def(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I32],
            ),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let helper_code = local_get_code((helper_params.len() - 1) as u32);
        let mut helper_spec = FunctionSpec::new(Rc::clone(&helper_type), 0);
        helper_spec.set_code(helper_code.as_slice().into());
        module.functions.push(FunctionInst::Local {
            spec: helper_spec,
            type_index: 0,
        });

        let caller_code = long_argument_caller_code(&helper_params);
        let mut caller_spec = FunctionSpec::new(Rc::clone(&caller_type), 1);
        caller_spec.set_code(caller_code.as_slice().into());
        module.functions.push(FunctionInst::Local {
            spec: caller_spec,
            type_index: 1,
        });

        let store = Box::new(Store::new(module));
        ensure_module_compiled(&store).expect(
            "long-argument internal call should compile through the direct 32-bit lowering path",
        );
    }
}
