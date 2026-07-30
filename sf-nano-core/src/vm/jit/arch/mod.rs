use crate::collections;
use tracked_alloc::rc::Rc;

use crate::{
    error::WasmError,
    module::{entities::FunctionSpec, type_defs::FunctionType},
    vm::{
        jit::backend::BackendConfig,
        jit::entities::ModuleInst,
        jit::machine::machine_ir::{MachineFuncId, MachineFunction},
        jit::runtime::{
            code::{CodegenModuleView, CompiledNativeModule, NativeCode, NativeRootEntry},
            code_buf::CodeBuffer,
            StoreAccess,
        },
        value::Value,
    },
};

#[cfg(sf_ir_dump)]
use crate::vm::jit::arch::common::types::DebugRegion;
use crate::vm::jit::arch::common::types::FunctionArtifact;

pub(crate) mod common;
#[cfg(any(sf_backend_riscv64, sf_backend_riscv32))]
mod riscv;
#[cfg(any(sf_backend_arm64, sf_backend_x64, sf_backend_riscv64))]
mod shared_64;

#[cfg(any(sf_backend_armv7a, sf_backend_thumbm))]
pub(crate) mod arm32;
#[cfg(sf_backend_arm64)]
pub(crate) mod arm64;
#[cfg(sf_backend_riscv32)]
pub(crate) mod riscv32;
#[cfg(sf_backend_riscv64)]
pub(crate) mod riscv64;
#[cfg(sf_backend_x64)]
pub(crate) mod x86_64;

/// This build's backend, as one value each ISA defines for itself.
///
/// There is no backend *selection*: exactly one `sf_backend_*` cfg is set, and
/// `build.rs` refuses a target where none would be, so the backend is simply
/// what this target's ISA is. Selection used to be a `NativeBackend` enum
/// threaded through the compile pipeline; with one variant it was a ZST whose
/// every match folded to a single arm, so the enum bought nothing that the cfg
/// did not already say.
///
/// ISA choice stays confined to this module -- callers remain free of
/// `sf_backend_*` cfgs, which is the property that made per-call-site cfg
/// branching the wrong shape (see `backends.alt/inline-per-arch-dispatch`).
#[inline]
pub(crate) fn backend_config() -> BackendConfig {
    // Each backend returns an explicit budget preset. Physical register
    // mapping and ABI constraints stay in the backend-specific ABI/layout
    // code; this selects policy, not hardware facts.
    #[cfg(sf_backend_arm64)]
    {
        arm64::abi::compile_backend_config()
    }
    #[cfg(any(sf_backend_armv7a, sf_backend_thumbm))]
    {
        arm32::abi::compile_backend_config()
    }
    #[cfg(sf_backend_riscv32)]
    {
        riscv32::abi::compile_backend_config()
    }
    #[cfg(sf_backend_riscv64)]
    {
        riscv64::abi::compile_backend_config()
    }
    #[cfg(sf_backend_x64)]
    {
        x86_64::abi::compile_backend_config()
    }
}

/// Which ISA the JIT targets on this build, for embedders that report it.
///
/// Distinct from [`crate::vm::engine::Tier`]: that says *which engine* runs a
/// module, this says which machine the JIT emits for. Infallible -- a build
/// for an ISA with no backend does not exist.
#[inline]
pub const fn active_native_backend_name() -> &'static str {
    #[cfg(sf_backend_arm64)]
    {
        "arm64"
    }
    #[cfg(sf_backend_armv7a)]
    {
        "armv7a"
    }
    #[cfg(sf_backend_thumbm)]
    {
        "thumbm"
    }
    #[cfg(sf_backend_riscv32)]
    {
        "riscv32"
    }
    #[cfg(sf_backend_riscv64)]
    {
        "riscv64"
    }
    #[cfg(sf_backend_x64)]
    {
        "x86_64"
    }
}

/// Normalized view of a backend's per-function compile result.
///
/// Each backend-specific module linker returns an entry type carrying whatever
/// arch-local state it needs. The module-build pipeline in `vm::jit::build` only
/// cares about three facts after compilation is done — the entry pointer, the
/// function's text size, and the optional IR-dump debug region list — so
/// `dispatch_compile_module` projects every backend's entry type into this
/// uniform shape. `sf_jitdump` consumes debug regions earlier in the
/// arch-specific linkers instead of through this normalized handoff.
pub(crate) struct CompiledArchEntry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    #[cfg(sf_ir_dump)]
    pub debug_regions: collections::Vec<DebugRegion>,
}

/// Compile all functions for the active native backend and return the
/// per-function projection the build pipeline consumes.
///
/// Single place in the crate where per-arch `compile_module` calls live.
/// Callers stay free of `sf_backend_*` cfgs.
pub(crate) fn dispatch_compile_module(
    module: &ModuleInst,
    compiled: &Rc<CompiledNativeModule>,
) -> Result<collections::Vec<Option<CompiledArchEntry>>, WasmError> {
    #[cfg(sf_backend_arm64)]
    {
        let entries =
            shared_64::compile_module_64::<arm64::backend::Arm64Backend>(module, compiled)?;
        Ok(entries
            .into_iter()
            .map(|opt| {
                opt.map(|e| CompiledArchEntry {
                    entry: e.entry,
                    text_len: e.text_len,
                    #[cfg(sf_ir_dump)]
                    debug_regions: e.debug_regions,
                })
            })
            .collect())
    }
    #[cfg(sf_backend_armv7a)]
    {
        let entries = arm32::compile::compile_module(module, compiled)?;
        Ok(entries
            .into_iter()
            .map(|opt| {
                opt.map(|e| CompiledArchEntry {
                    entry: e.entry,
                    text_len: e.text_len,
                    #[cfg(sf_ir_dump)]
                    debug_regions: e.debug_regions,
                })
            })
            .collect())
    }
    #[cfg(sf_backend_thumbm)]
    {
        let entries = arm32::compile::compile_module(module, compiled)?;
        Ok(entries
            .into_iter()
            .map(|opt| {
                opt.map(|e| CompiledArchEntry {
                    entry: e.entry,
                    text_len: e.text_len,
                    #[cfg(sf_ir_dump)]
                    debug_regions: e.debug_regions,
                })
            })
            .collect())
    }
    #[cfg(sf_backend_x64)]
    {
        let entries =
            shared_64::compile_module_64::<x86_64::backend::X86_64Backend>(module, compiled)?;
        Ok(entries
            .into_iter()
            .map(|opt| {
                opt.map(|e| CompiledArchEntry {
                    entry: e.entry,
                    text_len: e.text_len,
                    #[cfg(sf_ir_dump)]
                    debug_regions: e.debug_regions,
                })
            })
            .collect())
    }
    #[cfg(sf_backend_riscv32)]
    {
        let entries = riscv32::compile::compile_module(module, compiled)?;
        Ok(entries
            .into_iter()
            .map(|opt| {
                opt.map(|e| CompiledArchEntry {
                    entry: e.entry,
                    text_len: e.text_len,
                    #[cfg(sf_ir_dump)]
                    debug_regions: e.debug_regions,
                })
            })
            .collect())
    }
    #[cfg(sf_backend_riscv64)]
    {
        let entries =
            shared_64::compile_module_64::<riscv64::backend::Riscv64Backend>(module, compiled)?;
        Ok(entries
            .into_iter()
            .map(|opt| {
                opt.map(|e| CompiledArchEntry {
                    entry: e.entry,
                    text_len: e.text_len,
                    #[cfg(sf_ir_dump)]
                    debug_regions: e.debug_regions,
                })
            })
            .collect())
    }
}

pub(crate) fn dispatch_compile_function_into_buffer(
    compiled: &dyn CodegenModuleView,
    function: &MachineFunction,
    executable: &mut CodeBuffer,
) -> Result<FunctionArtifact, WasmError> {
    #[cfg(sf_backend_arm64)]
    {
        common::pipeline::compile_function_into_buffer::<arm64::backend::Arm64Backend>(
            compiled, function, executable,
        )
    }
    #[cfg(sf_backend_armv7a)]
    {
        common::pipeline::compile_function_into_buffer::<arm32::backend::Arm32Backend>(
            compiled, function, executable,
        )
    }
    #[cfg(sf_backend_thumbm)]
    {
        common::pipeline::compile_function_into_buffer::<arm32::backend::Arm32Backend>(
            compiled, function, executable,
        )
    }
    #[cfg(sf_backend_x64)]
    {
        common::pipeline::compile_function_into_buffer::<x86_64::backend::X86_64Backend>(
            compiled, function, executable,
        )
    }
    #[cfg(sf_backend_riscv32)]
    {
        common::pipeline::compile_function_into_buffer::<riscv32::backend::Riscv32Backend>(
            compiled, function, executable,
        )
    }
    #[cfg(sf_backend_riscv64)]
    {
        common::pipeline::compile_function_into_buffer::<riscv64::backend::Riscv64Backend>(
            compiled, function, executable,
        )
    }
}

/// Only the parallel eager-compile path calls this, and that path is
/// `sf_has_std`-gated in `vm::jit::build`.
#[cfg(sf_has_std)]
pub(crate) fn dispatch_compile_function(
    compiled: &dyn CodegenModuleView,
    function: &MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    #[cfg(sf_backend_arm64)]
    {
        common::pipeline::compile_function::<arm64::backend::Arm64Backend>(compiled, function)
    }
    #[cfg(sf_backend_armv7a)]
    {
        common::pipeline::compile_function::<arm32::backend::Arm32Backend>(compiled, function)
    }
    #[cfg(sf_backend_thumbm)]
    {
        common::pipeline::compile_function::<arm32::backend::Arm32Backend>(compiled, function)
    }
    #[cfg(sf_backend_x64)]
    {
        common::pipeline::compile_function::<x86_64::backend::X86_64Backend>(compiled, function)
    }
    #[cfg(sf_backend_riscv32)]
    {
        common::pipeline::compile_function::<riscv32::backend::Riscv32Backend>(compiled, function)
    }
    #[cfg(sf_backend_riscv64)]
    {
        common::pipeline::compile_function::<riscv64::backend::Riscv64Backend>(compiled, function)
    }
}

pub(crate) fn dispatch_compile_template_function_into_buffer(
    compiled: &dyn CodegenModuleView,
    spec: &FunctionSpec,
    func_id: MachineFuncId,
    executable: &mut CodeBuffer,
    has_memory: bool,
) -> Result<FunctionArtifact, WasmError> {
    #[cfg(sf_backend_arm64)]
    {
        crate::vm::jit::template::compile_template_for_backend::<arm64::backend::Arm64Backend>(
            compiled, spec, func_id, executable, has_memory,
        )
    }
    #[cfg(sf_backend_armv7a)]
    {
        crate::vm::jit::template::compile_template_for_backend::<arm32::backend::Arm32Backend>(
            compiled, spec, func_id, executable, has_memory,
        )
    }
    #[cfg(sf_backend_thumbm)]
    {
        crate::vm::jit::template::compile_template_for_backend::<arm32::backend::Arm32Backend>(
            compiled, spec, func_id, executable, has_memory,
        )
    }
    #[cfg(sf_backend_x64)]
    {
        crate::vm::jit::template::compile_template_for_backend::<x86_64::backend::X86_64Backend>(
            compiled, spec, func_id, executable, has_memory,
        )
    }
    #[cfg(sf_backend_riscv32)]
    {
        crate::vm::jit::template::compile_template_for_backend::<riscv32::backend::Riscv32Backend>(
            compiled, spec, func_id, executable, has_memory,
        )
    }
    #[cfg(sf_backend_riscv64)]
    {
        crate::vm::jit::template::compile_template_for_backend::<riscv64::backend::Riscv64Backend>(
            compiled, spec, func_id, executable, has_memory,
        )
    }
}

pub(crate) fn dispatch_emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
    #[cfg(sf_backend_arm64)]
    {
        <arm64::backend::Arm64Backend as common::backend::ArchBackend>::emit_nop_padding(buf, bytes)
    }
    #[cfg(sf_backend_armv7a)]
    {
        <arm32::backend::Arm32Backend as common::backend::ArchBackend>::emit_nop_padding(buf, bytes)
    }
    #[cfg(sf_backend_thumbm)]
    {
        <arm32::backend::Arm32Backend as common::backend::ArchBackend>::emit_nop_padding(buf, bytes)
    }
    #[cfg(sf_backend_x64)]
    {
        <x86_64::backend::X86_64Backend as common::backend::ArchBackend>::emit_nop_padding(
            buf, bytes,
        )
    }
    #[cfg(sf_backend_riscv32)]
    {
        <riscv32::backend::Riscv32Backend as common::backend::ArchBackend>::emit_nop_padding(
            buf, bytes,
        )
    }
    #[cfg(sf_backend_riscv64)]
    {
        <riscv64::backend::Riscv64Backend as common::backend::ArchBackend>::emit_nop_padding(
            buf, bytes,
        )
    }
}

/// Dispatch a JIT-compiled function through the arch-appropriate eval entry.
///
/// Single place in the crate that decides which backend's `eval` implementation
/// runs. Callers (`runtime::native_eval`) pass the active backend enum and stay
/// free of `sf_backend_*` cfgs.
#[inline]
pub(crate) fn dispatch_eval(
    access: &mut StoreAccess<'_>,
    local_index: u32,
    func_type: &FunctionType,
    code: &NativeCode,
    args: &[Value],
) -> Result<collections::Vec<Value>, WasmError> {
    let backend_name = active_native_backend_name();
    #[cfg(sf_backend_arm64)]
    {
        common::eval::eval(access, local_index, func_type, code, args, backend_name)
    }
    #[cfg(sf_backend_armv7a)]
    {
        common::eval::eval(access, local_index, func_type, code, args, backend_name)
    }
    #[cfg(sf_backend_thumbm)]
    {
        common::eval::eval(access, local_index, func_type, code, args, backend_name)
    }
    #[cfg(sf_backend_x64)]
    {
        common::eval::eval(access, local_index, func_type, code, args, backend_name)
    }
    #[cfg(sf_backend_riscv32)]
    {
        common::eval::eval(access, local_index, func_type, code, args, backend_name)
    }
    #[cfg(sf_backend_riscv64)]
    {
        common::eval::eval(access, local_index, func_type, code, args, backend_name)
    }
}
