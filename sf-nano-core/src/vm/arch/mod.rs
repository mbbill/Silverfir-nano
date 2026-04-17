use crate::collections;
use tracked_alloc::rc::Rc;

#[cfg(sf_emulator)]
use core::sync::atomic::{AtomicU8, Ordering};

use crate::{
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        backend::BackendConfig,
        entities::ModuleInst,
        machine::machine_ir::MachineFunction,
        result_buffer::ResultBuffer,
        runtime::{
            code::{CodegenModuleView, CompiledNativeModule, NativeCode, NativeRootEntry},
            code_buf::CodeBuffer,
        },
        store::Store,
        value::Value,
    },
};

#[cfg(sf_ir_dump)]
use crate::vm::arch::common::types::DebugRegion;
use crate::vm::arch::common::types::FunctionArtifact;

pub(crate) mod common;
#[cfg(sf_emulator)]
pub(crate) mod emulator;
#[cfg(any(sf_arch_arm64, sf_arch_x64))]
mod shared_64;

#[cfg(any(sf_arch_armv7a, sf_arch_thumbm))]
pub(crate) mod arm32;
#[cfg(sf_arch_arm64)]
pub(crate) mod arm64;
#[cfg(sf_arch_x64)]
pub(crate) mod x86_64;

/// Process-global emulator/reference-backend target mode.
///
/// `Disabled` means "do not force the emulator backend on a host that has a
/// real native backend". When the emulator feature is compiled and it is the
/// only available backend on a target, `Disabled` still falls back to the
/// default `Emu64` profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReferenceBackendMode {
    Disabled = 0,
    Emu64 = 1,
    Emu32 = 2,
}

impl ReferenceBackendMode {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Emu64 => "emu64",
            Self::Emu32 => "emu32",
        }
    }

    #[inline]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

/// Active native backend implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeBackend {
    #[cfg(sf_arch_arm64)]
    Arm64,
    #[cfg(sf_arch_armv7a)]
    Armv7a,
    #[cfg(sf_arch_thumbm)]
    ThumbM,
    #[cfg(sf_arch_x64)]
    X86_64,
    #[cfg(sf_emulator)]
    Reference,
}

#[inline]
pub(crate) fn compile_backend_config(backend: NativeBackend) -> BackendConfig {
    match backend {
        // Each backend returns an explicit budget preset. Physical register
        // mapping and ABI constraints stay in the backend-specific ABI/layout
        // code; this function selects policy, not hardware facts.
        #[cfg(sf_arch_arm64)]
        NativeBackend::Arm64 => arm64::abi::compile_backend_config(),
        #[cfg(sf_arch_armv7a)]
        NativeBackend::Armv7a => arm32::abi::compile_backend_config(),
        #[cfg(sf_arch_thumbm)]
        NativeBackend::ThumbM => arm32::abi::compile_backend_config(),
        #[cfg(sf_arch_x64)]
        NativeBackend::X86_64 => x86_64::abi::compile_backend_config(),
        #[cfg(sf_emulator)]
        NativeBackend::Reference => {
            emulator::config::compile_backend_config(effective_reference_backend_mode())
        }
    }
}

impl NativeBackend {
    #[inline]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            #[cfg(sf_arch_arm64)]
            Self::Arm64 => "arm64",
            #[cfg(sf_arch_armv7a)]
            Self::Armv7a => "armv7a",
            #[cfg(sf_arch_thumbm)]
            Self::ThumbM => "thumbm",
            #[cfg(sf_arch_x64)]
            Self::X86_64 => "x86_64",
            #[cfg(sf_emulator)]
            Self::Reference => "emulator",
        }
    }
}

#[cfg(sf_emulator)]
static REFERENCE_BACKEND_MODE: AtomicU8 = AtomicU8::new(ReferenceBackendMode::Disabled as u8);

#[inline]
#[cfg(sf_emulator)]
fn selected_reference_backend_mode() -> ReferenceBackendMode {
    match REFERENCE_BACKEND_MODE.load(Ordering::Relaxed) {
        x if x == ReferenceBackendMode::Disabled as u8 => ReferenceBackendMode::Disabled,
        x if x == ReferenceBackendMode::Emu64 as u8 => ReferenceBackendMode::Emu64,
        x if x == ReferenceBackendMode::Emu32 as u8 => ReferenceBackendMode::Emu32,
        _ => ReferenceBackendMode::Disabled,
    }
}

#[inline]
#[cfg(not(sf_emulator))]
const fn selected_reference_backend_mode() -> ReferenceBackendMode {
    ReferenceBackendMode::Disabled
}

#[inline]
#[cfg(sf_emulator)]
fn effective_reference_backend_mode() -> ReferenceBackendMode {
    match selected_reference_backend_mode() {
        ReferenceBackendMode::Disabled => ReferenceBackendMode::Emu64,
        mode => mode,
    }
}

#[inline]
fn host_native_backend() -> Option<NativeBackend> {
    #[cfg(sf_arch_arm64)]
    {
        return Some(NativeBackend::Arm64);
    }

    #[cfg(sf_arch_armv7a)]
    {
        return Some(NativeBackend::Armv7a);
    }

    #[cfg(sf_arch_thumbm)]
    {
        return Some(NativeBackend::ThumbM);
    }

    #[cfg(sf_arch_x64)]
    {
        return Some(NativeBackend::X86_64);
    }

    #[allow(unreachable_code)]
    None
}

pub(crate) fn active_native_backend() -> Result<NativeBackend, &'static str> {
    if selected_reference_backend_mode().is_enabled() {
        #[cfg(sf_emulator)]
        return Ok(NativeBackend::Reference);
    }

    if let Some(backend) = host_native_backend() {
        return Ok(backend);
    }

    #[cfg(sf_emulator)]
    {
        Ok(NativeBackend::Reference)
    }

    #[cfg(not(sf_emulator))]
    {
        Err("native backend unavailable for this target; rebuild with emulator feature")
    }
}

#[inline]
pub(crate) fn active_backend_config() -> Result<BackendConfig, &'static str> {
    active_native_backend().map(compile_backend_config)
}

#[inline]
pub(crate) fn backend_display_name(backend: NativeBackend) -> &'static str {
    match backend {
        #[cfg(sf_emulator)]
        NativeBackend::Reference => effective_reference_backend_mode().as_str(),
        _ => backend.as_str(),
    }
}

#[inline]
pub(crate) fn active_native_backend_name() -> Result<&'static str, &'static str> {
    active_native_backend().map(backend_display_name)
}

pub(crate) fn set_reference_backend_mode(mode: ReferenceBackendMode) -> Result<(), &'static str> {
    #[cfg(sf_emulator)]
    {
        REFERENCE_BACKEND_MODE.store(mode as u8, Ordering::Relaxed);
        return Ok(());
    }

    #[cfg(not(sf_emulator))]
    {
        if mode.is_enabled() {
            return Err("reference backend requires emulator feature");
        }
        Ok(())
    }
}

pub(crate) fn set_reference_backend(enabled: bool) -> Result<(), &'static str> {
    set_reference_backend_mode(if enabled {
        ReferenceBackendMode::Emu64
    } else {
        ReferenceBackendMode::Disabled
    })
}

/// Normalized view of a backend's per-function compile result.
///
/// Each backend-specific module linker returns an entry type carrying whatever
/// arch-local state it needs. The module-build pipeline in `vm::build` only
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
/// Callers stay free of `sf_arch_*` cfgs. When the emulator feature is
/// compiled, the reference backend reports an empty native-code vector and
/// execution falls back to the MIR interpreter path.
pub(crate) fn dispatch_compile_module(
    active_backend: NativeBackend,
    module: &ModuleInst,
    compiled: &Rc<CompiledNativeModule>,
) -> Result<collections::Vec<Option<CompiledArchEntry>>, WasmError> {
    match active_backend {
        #[cfg(sf_arch_arm64)]
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
        #[cfg(sf_arch_armv7a)]
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
        #[cfg(sf_arch_thumbm)]
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
        #[cfg(sf_arch_x64)]
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
        #[cfg(sf_emulator)]
        NativeBackend::Reference => Ok(collections::Vec::new()),
    }
}

pub(crate) fn dispatch_compile_function_into_buffer(
    active_backend: NativeBackend,
    compiled: &dyn CodegenModuleView,
    function: &MachineFunction,
    executable: &mut CodeBuffer,
) -> Result<FunctionArtifact, WasmError> {
    match active_backend {
        #[cfg(sf_arch_arm64)]
        NativeBackend::Arm64 => common::pipeline::compile_function_into_buffer::<
            arm64::backend::Arm64Backend,
        >(compiled, function, executable),
        #[cfg(sf_arch_armv7a)]
        NativeBackend::Armv7a => common::pipeline::compile_function_into_buffer::<
            arm32::backend::Arm32Backend,
        >(compiled, function, executable),
        #[cfg(sf_arch_thumbm)]
        NativeBackend::ThumbM => common::pipeline::compile_function_into_buffer::<
            arm32::backend::Arm32Backend,
        >(compiled, function, executable),
        #[cfg(sf_arch_x64)]
        NativeBackend::X86_64 => common::pipeline::compile_function_into_buffer::<
            x86_64::backend::X86_64Backend,
        >(compiled, function, executable),
        #[cfg(sf_emulator)]
        NativeBackend::Reference => Err(WasmError::invalid(
            "reference backend does not emit native code artifacts",
        )),
    }
}

pub(crate) fn dispatch_emit_nop_padding(
    active_backend: NativeBackend,
    buf: &mut CodeBuffer,
    bytes: usize,
) {
    match active_backend {
        #[cfg(sf_arch_arm64)]
        NativeBackend::Arm64 => {
            <arm64::backend::Arm64Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_arch_armv7a)]
        NativeBackend::Armv7a => {
            <arm32::backend::Arm32Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_arch_thumbm)]
        NativeBackend::ThumbM => {
            <arm32::backend::Arm32Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_arch_x64)]
        NativeBackend::X86_64 => {
            <x86_64::backend::X86_64Backend as common::backend::ArchBackend>::emit_nop_padding(
                buf, bytes,
            )
        }
        #[cfg(sf_emulator)]
        NativeBackend::Reference => {}
    }
}

/// Dispatch a JIT-compiled function through the arch-appropriate eval entry.
///
/// Single place in the crate that decides which backend's `eval` implementation
/// runs. Callers (`runtime::native_eval`) pass the active backend enum and stay
/// free of `sf_arch_*` cfgs.
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
        #[cfg(sf_arch_arm64)]
        NativeBackend::Arm64 => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_arch_armv7a)]
        NativeBackend::Armv7a => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_arch_thumbm)]
        NativeBackend::ThumbM => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_arch_x64)]
        NativeBackend::X86_64 => common::eval::eval(spec, code, store, args, backend_name),
        #[cfg(sf_emulator)]
        NativeBackend::Reference => emulator::eval(spec, code, store, args, backend_name),
    }
}

#[cfg(test)]
pub(crate) fn backend_mode_test_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(all(test, sf_emulator))]
mod tests {
    use super::{
        backend_display_name, backend_mode_test_lock, compile_backend_config,
        set_reference_backend_mode, NativeBackend, ReferenceBackendMode,
    };

    #[test]
    fn reference_backend_mode_bool_alias_defaults_to_emu64_budget() {
        let _lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        set_reference_backend_mode(ReferenceBackendMode::Disabled).expect("reset reference mode");
        let config = compile_backend_config(NativeBackend::Reference);
        assert_eq!(config.gp_unit_bytes, 8);
        assert_eq!(config.gp_dynamic_budget, 12);
    }

    #[test]
    fn reference_backend_mode_emu32_uses_its_own_32bit_budget() {
        let _lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        set_reference_backend_mode(ReferenceBackendMode::Emu32).expect("enable emu32");
        let config = compile_backend_config(NativeBackend::Reference);
        assert_eq!(config.gp_unit_bytes, 4);
        assert_eq!(config.gp_dynamic_budget, 9);
        assert_eq!(config.fp_dynamic_budget, 13);
        assert_eq!(backend_display_name(NativeBackend::Reference), "emu32");
        set_reference_backend_mode(ReferenceBackendMode::Disabled).expect("reset reference mode");
    }
}
