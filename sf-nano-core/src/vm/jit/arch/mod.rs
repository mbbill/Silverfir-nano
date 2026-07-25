use crate::collections;
use tracked_alloc::rc::Rc;

use crate::{
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        entities::ModuleInst,
        jit::backend::BackendConfig,
        jit::machine::machine_ir::{MachineFuncId, MachineFunction},
        jit::runtime::{
            code::{CodegenModuleView, CompiledNativeModule, NativeCode, NativeRootEntry},
            code_buf::CodeBuffer,
        },
        result_buffer::ResultBuffer,
        store::Store,
        value::Value,
    },
};

#[cfg(sf_ir_dump)]
use crate::vm::jit::arch::common::types::DebugRegion;
use crate::vm::jit::arch::common::types::FunctionArtifact;

pub(crate) mod common;
#[cfg(any(sf_backend_emu64, sf_backend_emu32))]
pub(crate) mod emulator;
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

/// Compiled execution backend implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeBackend {
    #[cfg(sf_backend_arm64)]
    Arm64,
    #[cfg(sf_backend_armv7a)]
    Armv7a,
    #[cfg(sf_backend_thumbm)]
    ThumbM,
    #[cfg(sf_backend_riscv32)]
    Riscv32,
    #[cfg(sf_backend_riscv64)]
    Riscv64,
    #[cfg(sf_backend_x64)]
    X86_64,
    #[cfg(sf_backend_emu64)]
    Emu64,
    #[cfg(sf_backend_emu32)]
    Emu32,
}

#[inline]
pub(crate) fn compile_backend_config(backend: NativeBackend) -> BackendConfig {
    match backend {
        // Each backend returns an explicit budget preset. Physical register
        // mapping and ABI constraints stay in the backend-specific ABI/layout
        // code; this function selects policy, not hardware facts.
        #[cfg(sf_backend_arm64)]
        NativeBackend::Arm64 => arm64::abi::compile_backend_config(),
        #[cfg(sf_backend_armv7a)]
        NativeBackend::Armv7a => arm32::abi::compile_backend_config(),
        #[cfg(sf_backend_thumbm)]
        NativeBackend::ThumbM => arm32::abi::compile_backend_config(),
        #[cfg(sf_backend_riscv32)]
        NativeBackend::Riscv32 => riscv32::abi::compile_backend_config(),
        #[cfg(sf_backend_riscv64)]
        NativeBackend::Riscv64 => riscv64::abi::compile_backend_config(),
        #[cfg(sf_backend_x64)]
        NativeBackend::X86_64 => x86_64::abi::compile_backend_config(),
        #[cfg(sf_backend_emu64)]
        NativeBackend::Emu64 => emulator::config::compile_backend_config(),
        #[cfg(sf_backend_emu32)]
        NativeBackend::Emu32 => emulator::config::compile_backend_config(),
    }
}

impl NativeBackend {
    #[inline]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            #[cfg(sf_backend_arm64)]
            Self::Arm64 => "arm64",
            #[cfg(sf_backend_armv7a)]
            Self::Armv7a => "armv7a",
            #[cfg(sf_backend_thumbm)]
            Self::ThumbM => "thumbm",
            #[cfg(sf_backend_riscv32)]
            Self::Riscv32 => "riscv32",
            #[cfg(sf_backend_riscv64)]
            Self::Riscv64 => "riscv64",
            #[cfg(sf_backend_x64)]
            Self::X86_64 => "x86_64",
            #[cfg(sf_backend_emu64)]
            Self::Emu64 => "emu64",
            #[cfg(sf_backend_emu32)]
            Self::Emu32 => "emu32",
        }
    }
}

#[inline]
const fn compiled_native_backend() -> Option<NativeBackend> {
    #[cfg(sf_backend_arm64)]
    {
        return Some(NativeBackend::Arm64);
    }

    #[cfg(sf_backend_armv7a)]
    {
        return Some(NativeBackend::Armv7a);
    }

    #[cfg(sf_backend_thumbm)]
    {
        return Some(NativeBackend::ThumbM);
    }

    #[cfg(sf_backend_riscv32)]
    {
        return Some(NativeBackend::Riscv32);
    }

    #[cfg(sf_backend_riscv64)]
    {
        return Some(NativeBackend::Riscv64);
    }

    #[cfg(sf_backend_x64)]
    {
        return Some(NativeBackend::X86_64);
    }

    #[cfg(sf_backend_emu64)]
    {
        return Some(NativeBackend::Emu64);
    }

    #[cfg(sf_backend_emu32)]
    {
        return Some(NativeBackend::Emu32);
    }

    #[allow(unreachable_code)]
    None
}

pub(crate) fn active_native_backend() -> Result<NativeBackend, &'static str> {
    if let Some(backend) = compiled_native_backend() {
        return Ok(backend);
    }

    Err("native backend unavailable for this target; rebuild for a supported backend")
}

#[inline]
pub(crate) fn active_backend_config() -> Result<BackendConfig, &'static str> {
    active_native_backend().map(compile_backend_config)
}

#[inline]
pub(crate) fn backend_display_name(backend: NativeBackend) -> &'static str {
    backend.as_str()
}

/// Which ISA the JIT targets on this build, for embedders that report it.
///
/// Distinct from [`crate::vm::engine::Tier`]: that says *which engine*
/// runs a module, this says which machine the JIT emits for.
#[inline]
pub fn active_native_backend_name() -> Result<&'static str, &'static str> {
    active_native_backend().map(backend_display_name)
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
/// Callers stay free of `sf_backend_*` cfgs. Emulator backend builds report an
/// empty native-code vector and execution falls back to the MIR interpreter
/// path.
pub(crate) fn dispatch_compile_module(
    active_backend: NativeBackend,
    module: &ModuleInst,
    compiled: &Rc<CompiledNativeModule>,
) -> Result<collections::Vec<Option<CompiledArchEntry>>, WasmError> {
    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    let _ = (module, compiled);
    match active_backend {
        #[cfg(sf_backend_arm64)]
        NativeBackend::Arm64 => {
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
        NativeBackend::Armv7a => {
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
        NativeBackend::ThumbM => {
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
        NativeBackend::X86_64 => {
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
        NativeBackend::Riscv32 => {
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
        NativeBackend::Riscv64 => {
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
        #[cfg(sf_backend_emu64)]
        NativeBackend::Emu64 => Ok(collections::Vec::new()),
        #[cfg(sf_backend_emu32)]
        NativeBackend::Emu32 => Ok(collections::Vec::new()),
    }
}

pub(crate) fn dispatch_compile_function_into_buffer(
    active_backend: NativeBackend,
    compiled: &dyn CodegenModuleView,
    function: &MachineFunction,
    executable: &mut CodeBuffer,
) -> Result<FunctionArtifact, WasmError> {
    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    let _ = (compiled, function, executable);
    match active_backend {
        #[cfg(sf_backend_arm64)]
        NativeBackend::Arm64 => common::pipeline::compile_function_into_buffer::<
            arm64::backend::Arm64Backend,
        >(compiled, function, executable),
        #[cfg(sf_backend_armv7a)]
        NativeBackend::Armv7a => common::pipeline::compile_function_into_buffer::<
            arm32::backend::Arm32Backend,
        >(compiled, function, executable),
        #[cfg(sf_backend_thumbm)]
        NativeBackend::ThumbM => common::pipeline::compile_function_into_buffer::<
            arm32::backend::Arm32Backend,
        >(compiled, function, executable),
        #[cfg(sf_backend_x64)]
        NativeBackend::X86_64 => common::pipeline::compile_function_into_buffer::<
            x86_64::backend::X86_64Backend,
        >(compiled, function, executable),
        #[cfg(sf_backend_riscv32)]
        NativeBackend::Riscv32 => common::pipeline::compile_function_into_buffer::<
            riscv32::backend::Riscv32Backend,
        >(compiled, function, executable),
        #[cfg(sf_backend_riscv64)]
        NativeBackend::Riscv64 => common::pipeline::compile_function_into_buffer::<
            riscv64::backend::Riscv64Backend,
        >(compiled, function, executable),
        #[cfg(sf_backend_emu64)]
        NativeBackend::Emu64 => Err(WasmError::invalid(
            "emu64 backend does not emit native code artifacts",
        )),
        #[cfg(sf_backend_emu32)]
        NativeBackend::Emu32 => Err(WasmError::invalid(
            "emu32 backend does not emit native code artifacts",
        )),
    }
}

/// Only the parallel eager-compile path calls this, and that path is
/// `sf_has_std`-gated in `vm::jit::build`.
#[cfg(sf_has_std)]
pub(crate) fn dispatch_compile_function(
    active_backend: NativeBackend,
    compiled: &dyn CodegenModuleView,
    function: &MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    let _ = (compiled, function);
    match active_backend {
        #[cfg(sf_backend_arm64)]
        NativeBackend::Arm64 => {
            common::pipeline::compile_function::<arm64::backend::Arm64Backend>(compiled, function)
        }
        #[cfg(sf_backend_armv7a)]
        NativeBackend::Armv7a => {
            common::pipeline::compile_function::<arm32::backend::Arm32Backend>(compiled, function)
        }
        #[cfg(sf_backend_thumbm)]
        NativeBackend::ThumbM => {
            common::pipeline::compile_function::<arm32::backend::Arm32Backend>(compiled, function)
        }
        #[cfg(sf_backend_x64)]
        NativeBackend::X86_64 => {
            common::pipeline::compile_function::<x86_64::backend::X86_64Backend>(compiled, function)
        }
        #[cfg(sf_backend_riscv32)]
        NativeBackend::Riscv32 => common::pipeline::compile_function::<
            riscv32::backend::Riscv32Backend,
        >(compiled, function),
        #[cfg(sf_backend_riscv64)]
        NativeBackend::Riscv64 => common::pipeline::compile_function::<
            riscv64::backend::Riscv64Backend,
        >(compiled, function),
        #[cfg(sf_backend_emu64)]
        NativeBackend::Emu64 => Err(WasmError::invalid(
            "emu64 backend does not emit native code artifacts",
        )),
        #[cfg(sf_backend_emu32)]
        NativeBackend::Emu32 => Err(WasmError::invalid(
            "emu32 backend does not emit native code artifacts",
        )),
    }
}

pub(crate) fn dispatch_compile_template_function_into_buffer(
    active_backend: NativeBackend,
    compiled: &dyn CodegenModuleView,
    spec: &FunctionSpec,
    func_id: MachineFuncId,
    executable: &mut CodeBuffer,
    has_memory: bool,
) -> Result<FunctionArtifact, WasmError> {
    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    let _ = (compiled, spec, func_id, executable, has_memory);
    match active_backend {
        #[cfg(sf_backend_arm64)]
        NativeBackend::Arm64 => crate::vm::jit::template::compile_template_for_backend::<
            arm64::backend::Arm64Backend,
        >(compiled, spec, func_id, executable, has_memory),
        #[cfg(sf_backend_armv7a)]
        NativeBackend::Armv7a => crate::vm::jit::template::compile_template_for_backend::<
            arm32::backend::Arm32Backend,
        >(compiled, spec, func_id, executable, has_memory),
        #[cfg(sf_backend_thumbm)]
        NativeBackend::ThumbM => crate::vm::jit::template::compile_template_for_backend::<
            arm32::backend::Arm32Backend,
        >(compiled, spec, func_id, executable, has_memory),
        #[cfg(sf_backend_x64)]
        NativeBackend::X86_64 => crate::vm::jit::template::compile_template_for_backend::<
            x86_64::backend::X86_64Backend,
        >(compiled, spec, func_id, executable, has_memory),
        #[cfg(sf_backend_riscv32)]
        NativeBackend::Riscv32 => crate::vm::jit::template::compile_template_for_backend::<
            riscv32::backend::Riscv32Backend,
        >(compiled, spec, func_id, executable, has_memory),
        #[cfg(sf_backend_riscv64)]
        NativeBackend::Riscv64 => crate::vm::jit::template::compile_template_for_backend::<
            riscv64::backend::Riscv64Backend,
        >(compiled, spec, func_id, executable, has_memory),
        #[cfg(sf_backend_emu64)]
        NativeBackend::Emu64 => Err(WasmError::internal(
            "template jit unsupported for emulator backend",
        )),
        #[cfg(sf_backend_emu32)]
        NativeBackend::Emu32 => Err(WasmError::internal(
            "template jit unsupported for emulator backend",
        )),
    }
}

pub(crate) fn dispatch_emit_nop_padding(
    active_backend: NativeBackend,
    buf: &mut CodeBuffer,
    bytes: usize,
) {
    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    let _ = (buf, bytes);
    match active_backend {
        #[cfg(sf_backend_arm64)]
        NativeBackend::Arm64 => {
            <arm64::backend::Arm64Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_backend_armv7a)]
        NativeBackend::Armv7a => {
            <arm32::backend::Arm32Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_backend_thumbm)]
        NativeBackend::ThumbM => {
            <arm32::backend::Arm32Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_backend_x64)]
        NativeBackend::X86_64 => {
            <x86_64::backend::X86_64Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_backend_riscv32)]
        NativeBackend::Riscv32 => {
            <riscv32::backend::Riscv32Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_backend_riscv64)]
        NativeBackend::Riscv64 => {
            <riscv64::backend::Riscv64Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_backend_emu64)]
        NativeBackend::Emu64 => {}
        #[cfg(sf_backend_emu32)]
        NativeBackend::Emu32 => {}
    }
}

/// Dispatch a JIT-compiled function through the arch-appropriate eval entry.
///
/// Single place in the crate that decides which backend's `eval` implementation
/// runs. Callers (`runtime::native_eval`) pass the active backend enum and stay
/// free of `sf_backend_*` cfgs.
#[inline]
pub(crate) fn dispatch_eval(
    active_backend: NativeBackend,
    spec: &FunctionSpec,
    code: &NativeCode,
    store: &mut Store,
    args: &[Value],
) -> Result<ResultBuffer, WasmError> {
    let backend_name = backend_display_name(active_backend);
    match active_backend {
        #[cfg(sf_backend_arm64)]
        NativeBackend::Arm64 => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_backend_armv7a)]
        NativeBackend::Armv7a => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_backend_thumbm)]
        NativeBackend::ThumbM => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_backend_x64)]
        NativeBackend::X86_64 => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_backend_riscv32)]
        NativeBackend::Riscv32 => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_backend_riscv64)]
        NativeBackend::Riscv64 => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_backend_emu64)]
        NativeBackend::Emu64 => emulator::eval(spec, code, store, args, backend_name),
        #[cfg(sf_backend_emu32)]
        NativeBackend::Emu32 => emulator::eval(spec, code, store, args, backend_name),
    }
}

#[cfg(all(test, any(sf_backend_emu64, sf_backend_emu32)))]
pub(crate) fn backend_mode_test_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(all(test, any(sf_backend_emu64, sf_backend_emu32)))]
mod tests {
    use super::{
        backend_display_name, backend_mode_test_lock, compile_backend_config, NativeBackend,
    };

    #[cfg(sf_backend_emu64)]
    #[test]
    fn emu64_backend_uses_64bit_budget() {
        let _lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        let config = compile_backend_config(NativeBackend::Emu64);
        assert_eq!(config.gp_unit_bytes, 8);
        assert_eq!(config.gp_dynamic_budget, 12);
        assert_eq!(backend_display_name(NativeBackend::Emu64), "emu64");
    }

    #[cfg(sf_backend_emu32)]
    #[test]
    fn emu32_backend_uses_32bit_budget() {
        let _lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        let config = compile_backend_config(NativeBackend::Emu32);
        assert_eq!(config.gp_unit_bytes, 4);
        assert_eq!(config.gp_dynamic_budget, 9);
        assert_eq!(config.fp_dynamic_budget, 13);
        assert_eq!(backend_display_name(NativeBackend::Emu32), "emu32");
    }
}
