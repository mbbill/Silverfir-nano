use tracked_alloc::{boxed::Box, rc::Rc};

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        jit::arch::NativeBackend,
        jit::machine::machine_ir::{
            MachineConstData, MachineConstId, MachineFuncId, MachineFunctionAbi, MachineModule,
            MachineModuleAbi,
        },
        runtime::{context::NativeContext, dispatch_view::NativeDispatchMetadata},
    },
};

/// Native entry point: identical signature across all architectures.
pub(crate) type NativeRootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;

pub(crate) trait CodegenModuleView: core::fmt::Debug {
    fn backend(&self) -> BackendConfig;
    fn runtime_for(&self, id: MachineFuncId) -> Option<&MachineFunctionAbi>;
    fn const_ptr(&self, id: MachineConstId) -> Option<*const u8>;
}

#[derive(Clone, Debug)]
pub(crate) struct AlignedConstData {
    storage: Box<[u64]>,
}

impl AlignedConstData {
    pub(crate) fn new(record: &MachineConstData) -> Result<Self, WasmError> {
        if record.align as usize > core::mem::align_of::<u64>() {
            return Err(WasmError::internal(
                "machine const requires unsupported alignment in the emulator",
            ));
        }

        let words = record.bytes.len().div_ceil(core::mem::size_of::<u64>());
        let mut storage = collections::vec![0u64; words.max(1)].into_boxed_slice();
        unsafe {
            core::ptr::copy_nonoverlapping(
                record.bytes.as_ptr(),
                storage.as_mut_ptr().cast::<u8>(),
                record.bytes.len(),
            );
        }
        Ok(Self { storage })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.storage.as_ptr().cast::<u8>()
    }
}

/// Sizes the guard region under the wasm stack; only the guard-page runtime
/// consumes it.
#[cfg(sf_has_guard_pages)]
#[inline]
fn max_frame_bytes(abi: &MachineModuleAbi) -> usize {
    abi.functions
        .iter()
        .map(|function| usize::from(function.total_frame_slots) * core::mem::size_of::<u64>())
        .max()
        .unwrap_or(0)
}

#[derive(Debug)]
pub(crate) struct CompiledNativeModule {
    backend_kind: NativeBackend,
    backend: BackendConfig,
    module: Option<MachineModule>,
    abi: Option<MachineModuleAbi>,
    aligned_consts: collections::Vec<AlignedConstData>,
    dispatch_metadata: NativeDispatchMetadata,
    #[cfg(sf_has_guard_pages)]
    max_frame_bytes: usize,
}

impl CompiledNativeModule {
    pub(crate) fn new(
        backend_kind: NativeBackend,
        backend: BackendConfig,
        module: MachineModule,
        abi: MachineModuleAbi,
    ) -> Result<Self, WasmError> {
        if backend.is_32bit_gp_target() {
            module.validate_32bit_gp_target(backend.first_fp_reg())?;
        }
        let mut aligned_consts = collections::Vec::with_capacity(module.consts.len());
        for konst in &module.consts {
            aligned_consts.push(AlignedConstData::new(konst)?);
        }
        let dispatch_metadata = NativeDispatchMetadata::new(backend, &abi);
        #[cfg(sf_has_guard_pages)]
        let max_frame_bytes = max_frame_bytes(&abi);
        Ok(Self {
            backend_kind,
            backend,
            module: Some(module),
            abi: Some(abi),
            aligned_consts,
            dispatch_metadata,
            #[cfg(sf_has_guard_pages)]
            max_frame_bytes,
        })
    }

    pub(crate) fn from_streamed(
        backend_kind: NativeBackend,
        backend: BackendConfig,
        abi: MachineModuleAbi,
        aligned_consts: collections::Vec<AlignedConstData>,
    ) -> Self {
        let dispatch_metadata = NativeDispatchMetadata::new(backend, &abi);
        #[cfg(sf_has_guard_pages)]
        let max_frame_bytes = max_frame_bytes(&abi);
        Self {
            backend_kind,
            backend,
            module: None,
            abi: Some(abi),
            aligned_consts,
            dispatch_metadata,
            #[cfg(sf_has_guard_pages)]
            max_frame_bytes,
        }
    }

    #[inline]
    pub(crate) const fn backend(&self) -> BackendConfig {
        self.backend
    }

    #[inline]
    pub(crate) const fn backend_kind(&self) -> NativeBackend {
        self.backend_kind
    }

    #[inline]
    pub(crate) fn module(&self) -> &MachineModule {
        match &self.module {
            Some(module) => module,
            None => panic!("compiled machine module is not retained for this backend"),
        }
    }

    #[inline]
    pub(crate) fn abi(&self) -> &MachineModuleAbi {
        match &self.abi {
            Some(abi) => abi,
            None => panic!("compiled machine ABI is not retained for this backend"),
        }
    }

    #[cfg(sf_has_guard_pages)]
    #[inline]
    pub(crate) const fn max_frame_bytes(&self) -> usize {
        self.max_frame_bytes
    }

    #[inline]
    pub(crate) fn strip_machine_ir_for_runtime(&mut self) {
        self.module = None;
    }

    #[cfg(any(sf_backend_emu64, sf_backend_emu32))]
    #[inline]
    pub(crate) fn function(
        &self,
        id: MachineFuncId,
    ) -> Option<&crate::vm::jit::machine::machine_ir::MachineFunction> {
        self.module().functions.get(id.0 as usize)
    }

    #[inline]
    pub(crate) fn const_ptr(&self, id: MachineConstId) -> Option<*const u8> {
        self.aligned_consts
            .get(id.0 as usize)
            .map(AlignedConstData::as_ptr)
    }

    #[inline]
    pub(crate) const fn dispatch_metadata(&self) -> &NativeDispatchMetadata {
        &self.dispatch_metadata
    }

    #[inline]
    pub(crate) fn publish_local_call_infos(&self, base: *const u8) {
        self.dispatch_metadata.local_call_infos().publish(base);
    }
}

impl CodegenModuleView for CompiledNativeModule {
    #[inline]
    fn backend(&self) -> BackendConfig {
        self.backend
    }

    #[inline]
    fn runtime_for(&self, id: MachineFuncId) -> Option<&MachineFunctionAbi> {
        self.abi
            .as_ref()
            .and_then(|abi| abi.functions.get(id.0 as usize))
    }

    #[inline]
    fn const_ptr(&self, id: MachineConstId) -> Option<*const u8> {
        CompiledNativeModule::const_ptr(self, id)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativeCode {
    compiled: Rc<CompiledNativeModule>,
    func_id: MachineFuncId,
    entry: Option<NativeRootEntry>,
}

impl NativeCode {
    #[inline]
    pub(crate) fn new(compiled: Rc<CompiledNativeModule>, func_id: MachineFuncId) -> Self {
        Self {
            compiled,
            func_id,
            entry: None,
        }
    }

    #[inline]
    pub(crate) fn with_entry(mut self, entry: Option<NativeRootEntry>) -> Self {
        self.entry = entry;
        self
    }

    #[inline]
    pub(crate) const fn func_id(&self) -> MachineFuncId {
        self.func_id
    }

    #[inline]
    pub(crate) fn compiled(&self) -> &CompiledNativeModule {
        &self.compiled
    }

    #[cfg(test)]
    pub(crate) fn compiled_rc(&self) -> &Rc<CompiledNativeModule> {
        &self.compiled
    }

    #[inline]
    pub(crate) fn native_entry(&self) -> Option<NativeRootEntry> {
        self.entry
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeCodeCache {
    compiled: bool,
}

impl NativeCodeCache {
    #[inline]
    pub(crate) const fn is_compiled(self) -> bool {
        self.compiled
    }

    #[inline]
    pub(crate) const fn compiled() -> Self {
        Self { compiled: true }
    }
}
