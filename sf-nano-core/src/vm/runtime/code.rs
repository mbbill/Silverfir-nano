use alloc::{boxed::Box, rc::Rc, vec::Vec};
use core::cell::Cell;

use crate::{
    error::WasmError,
    vm::{
        arch::NativeBackend,
        backend::BackendConfig,
        machine::machine_ir::{
            MachineConstData, MachineConstId, MachineFuncId, MachineModule, MachineModuleAbi,
        },
        runtime::context::NativeContext,
    },
};

/// Native entry point: identical signature across all architectures.
pub(crate) type NativeRootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;
pub(crate) type NativeCodePtr = *const u8;

#[derive(Clone, Debug)]
struct AlignedConstData {
    storage: Box<[u64]>,
}

impl AlignedConstData {
    fn new(record: &MachineConstData) -> Result<Self, WasmError> {
        if record.align as usize > core::mem::align_of::<u64>() {
            return Err(WasmError::internal(alloc::format!(
                "machine const {} requires unsupported alignment {} in the emulator",
                record.id.0,
                record.align,
            )));
        }

        let words = record.bytes.len().div_ceil(core::mem::size_of::<u64>());
        let mut storage = alloc::vec![0u64; words.max(1)].into_boxed_slice();
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
    fn as_ptr(&self) -> *const u8 {
        self.storage.as_ptr().cast::<u8>()
    }
}

#[derive(Debug)]
pub(crate) struct CompiledNativeModule {
    backend_kind: NativeBackend,
    backend: BackendConfig,
    module: MachineModule,
    abi: MachineModuleAbi,
    aligned_consts: Vec<AlignedConstData>,
    local_call_infos_base: Cell<*const u8>,
    local_call_infos_len: Cell<usize>,
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
        let mut aligned_consts = Vec::with_capacity(module.consts.len());
        for konst in &module.consts {
            aligned_consts.push(AlignedConstData::new(konst)?);
        }
        Ok(Self {
            backend_kind,
            backend,
            module,
            abi,
            aligned_consts,
            local_call_infos_base: Cell::new(core::ptr::null()),
            local_call_infos_len: Cell::new(0),
        })
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
    pub(crate) const fn module(&self) -> &MachineModule {
        &self.module
    }

    #[inline]
    pub(crate) const fn abi(&self) -> &MachineModuleAbi {
        &self.abi
    }

    #[inline]
    pub(crate) fn function(
        &self,
        id: MachineFuncId,
    ) -> Option<&super::super::machine::machine_ir::MachineFunction> {
        self.module.functions.get(id.0 as usize)
    }

    #[inline]
    pub(crate) fn const_ptr(&self, id: MachineConstId) -> Option<*const u8> {
        self.aligned_consts
            .get(id.0 as usize)
            .map(AlignedConstData::as_ptr)
    }

    #[inline]
    pub(crate) fn publish_local_call_infos(&self, base: *const u8, len: usize) {
        self.local_call_infos_base.set(base);
        self.local_call_infos_len.set(len);
    }

    #[inline]
    pub(crate) fn local_call_infos_base(&self) -> *const u8 {
        self.local_call_infos_base.get()
    }

    #[inline]
    pub(crate) fn local_call_infos_len(&self) -> usize {
        self.local_call_infos_len.get()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativeCode {
    compiled: Rc<CompiledNativeModule>,
    func_id: MachineFuncId,
    entry: Option<NativeRootEntry>,
    root_return: Option<NativeCodePtr>,
}

impl NativeCode {
    #[inline]
    pub(crate) fn new(compiled: Rc<CompiledNativeModule>, func_id: MachineFuncId) -> Self {
        Self {
            compiled,
            func_id,
            entry: None,
            root_return: None,
        }
    }

    #[inline]
    pub(crate) fn with_entry(
        mut self,
        entry: Option<NativeRootEntry>,
        root_return: Option<NativeCodePtr>,
    ) -> Self {
        self.entry = entry;
        self.root_return = root_return;
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

    #[inline]
    pub(crate) fn native_root_return(&self) -> Option<NativeCodePtr> {
        self.root_return
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
