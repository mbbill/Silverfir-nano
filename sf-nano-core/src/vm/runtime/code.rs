use alloc::{boxed::Box, rc::Rc, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        arch::NativeBackend,
        backend::BackendConfig,
        machine::machine_ir::{
            MachineConstData, MachineConstId, MachineFuncId, MachineFunction, MachineModule,
            MachineRuntimeContract, MACHINE_FIXED_REG_COUNT,
        },
        runtime::context::NativeContext,
    },
};

#[cfg(target_arch = "aarch64")]
pub(crate) type Arm64RootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;
#[cfg(target_arch = "aarch64")]
pub(crate) type Arm64CodePtr = *const u8;

#[cfg(target_arch = "arm")]
pub(crate) type Armv7aRootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;
#[cfg(target_arch = "arm")]
pub(crate) type Armv7aCodePtr = *const u8;

#[cfg(target_arch = "x86_64")]
pub(crate) type X86_64RootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;
#[cfg(target_arch = "x86_64")]
pub(crate) type X86_64CodePtr = *const u8;

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
    runtime: MachineRuntimeContract,
    aligned_consts: Vec<AlignedConstData>,
}

impl CompiledNativeModule {
    pub(crate) fn new(
        backend_kind: NativeBackend,
        backend: BackendConfig,
        module: MachineModule,
        runtime: MachineRuntimeContract,
    ) -> Result<Self, WasmError> {
        if backend.is_32bit_gp_target() {
            let max_gp_regs = MACHINE_FIXED_REG_COUNT
                + backend.gp_local_cache_budget as u16
                + backend.gp_transient_budget as u16;
            module.validate_32bit_gp_target(max_gp_regs)?;
        }
        let mut aligned_consts = Vec::with_capacity(module.consts.len());
        for konst in &module.consts {
            aligned_consts.push(AlignedConstData::new(konst)?);
        }
        Ok(Self {
            backend_kind,
            backend,
            module,
            runtime,
            aligned_consts,
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
    pub(crate) const fn runtime(&self) -> &MachineRuntimeContract {
        &self.runtime
    }

    #[inline]
    pub(crate) fn function(&self, id: MachineFuncId) -> Option<&MachineFunction> {
        self.module.functions.get(id.0 as usize)
    }

    #[inline]
    pub(crate) fn const_ptr(&self, id: MachineConstId) -> Option<*const u8> {
        self.aligned_consts
            .get(id.0 as usize)
            .map(AlignedConstData::as_ptr)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativeCode {
    compiled: Rc<CompiledNativeModule>,
    func_id: MachineFuncId,
    #[cfg(target_arch = "aarch64")]
    arm64_entry: Option<Arm64RootEntry>,
    #[cfg(target_arch = "aarch64")]
    arm64_root_return: Option<Arm64CodePtr>,
    #[cfg(target_arch = "arm")]
    armv7a_entry: Option<Armv7aRootEntry>,
    #[cfg(target_arch = "arm")]
    armv7a_root_return: Option<Armv7aCodePtr>,
    #[cfg(target_arch = "x86_64")]
    x86_64_entry: Option<X86_64RootEntry>,
    #[cfg(target_arch = "x86_64")]
    x86_64_root_return: Option<X86_64CodePtr>,
}

impl NativeCode {
    #[inline]
    pub(crate) fn new(compiled: Rc<CompiledNativeModule>, func_id: MachineFuncId) -> Self {
        Self {
            compiled,
            func_id,
            #[cfg(target_arch = "aarch64")]
            arm64_entry: None,
            #[cfg(target_arch = "aarch64")]
            arm64_root_return: None,
            #[cfg(target_arch = "arm")]
            armv7a_entry: None,
            #[cfg(target_arch = "arm")]
            armv7a_root_return: None,
            #[cfg(target_arch = "x86_64")]
            x86_64_entry: None,
            #[cfg(target_arch = "x86_64")]
            x86_64_root_return: None,
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub(crate) fn with_arm64_entry(
        mut self,
        entry: Option<Arm64RootEntry>,
        root_return: Option<Arm64CodePtr>,
    ) -> Self {
        self.arm64_entry = entry;
        self.arm64_root_return = root_return;
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

    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub(crate) const fn arm64_entry(&self) -> Option<Arm64RootEntry> {
        self.arm64_entry
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub(crate) const fn arm64_root_return(&self) -> Option<Arm64CodePtr> {
        self.arm64_root_return
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    pub(crate) fn with_armv7a_entry(
        mut self,
        entry: Option<Armv7aRootEntry>,
        root_return: Option<Armv7aCodePtr>,
    ) -> Self {
        self.armv7a_entry = entry;
        self.armv7a_root_return = root_return;
        self
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    pub(crate) const fn armv7a_entry(&self) -> Option<Armv7aRootEntry> {
        self.armv7a_entry
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    pub(crate) const fn armv7a_root_return(&self) -> Option<Armv7aCodePtr> {
        self.armv7a_root_return
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(crate) fn with_x86_64_entry(
        mut self,
        entry: Option<X86_64RootEntry>,
        root_return: Option<X86_64CodePtr>,
    ) -> Self {
        self.x86_64_entry = entry;
        self.x86_64_root_return = root_return;
        self
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(crate) const fn x86_64_entry(&self) -> Option<X86_64RootEntry> {
        self.x86_64_entry
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(crate) const fn x86_64_root_return(&self) -> Option<X86_64CodePtr> {
        self.x86_64_root_return
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
