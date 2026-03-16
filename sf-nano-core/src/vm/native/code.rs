use alloc::{boxed::Box, rc::Rc, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        backend::BackendConfig,
        native::{
            arch::NativeBackend,
            ir::{
                machine::{
                    MachineConstData, MachineConstId, MachineFuncId, MachineFunction, MachineModule,
                },
                runtime::MachineRuntimeContract,
            },
            runtime::context::NativeContext,
        },
    },
};

#[cfg(target_arch = "aarch64")]
pub type Arm64RootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;
#[cfg(target_arch = "aarch64")]
pub type Arm64CodePtr = *const u8;

#[cfg(target_arch = "arm")]
pub type Armv7aRootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;
#[cfg(target_arch = "arm")]
pub type Armv7aCodePtr = *const u8;

#[cfg(target_arch = "x86_64")]
pub type X86_64RootEntry = unsafe extern "C" fn(*mut NativeContext, *mut u64) -> u32;
#[cfg(target_arch = "x86_64")]
pub type X86_64CodePtr = *const u8;

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
pub struct CompiledNativeModule {
    backend_kind: NativeBackend,
    backend: BackendConfig,
    module: MachineModule,
    runtime: MachineRuntimeContract,
    aligned_consts: Vec<AlignedConstData>,
}

impl CompiledNativeModule {
    pub fn new(
        backend_kind: NativeBackend,
        backend: BackendConfig,
        module: MachineModule,
        runtime: MachineRuntimeContract,
    ) -> Result<Self, WasmError> {
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
    pub const fn backend(&self) -> BackendConfig {
        self.backend
    }

    #[inline]
    pub const fn backend_kind(&self) -> NativeBackend {
        self.backend_kind
    }

    #[inline]
    pub const fn module(&self) -> &MachineModule {
        &self.module
    }

    #[inline]
    pub const fn runtime(&self) -> &MachineRuntimeContract {
        &self.runtime
    }

    #[inline]
    pub fn function(&self, id: MachineFuncId) -> Option<&MachineFunction> {
        self.module.functions.get(id.0 as usize)
    }

    #[inline]
    pub fn const_ptr(&self, id: MachineConstId) -> Option<*const u8> {
        self.aligned_consts
            .get(id.0 as usize)
            .map(AlignedConstData::as_ptr)
    }
}

#[derive(Clone, Debug)]
pub struct NativeCode {
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
    pub fn new(compiled: Rc<CompiledNativeModule>, func_id: MachineFuncId) -> Self {
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
    pub fn with_arm64_entry(
        mut self,
        entry: Option<Arm64RootEntry>,
        root_return: Option<Arm64CodePtr>,
    ) -> Self {
        self.arm64_entry = entry;
        self.arm64_root_return = root_return;
        self
    }

    #[inline]
    pub const fn func_id(&self) -> MachineFuncId {
        self.func_id
    }

    #[inline]
    pub fn compiled(&self) -> &CompiledNativeModule {
        &self.compiled
    }

    #[inline]
    pub fn compiled_rc(&self) -> &Rc<CompiledNativeModule> {
        &self.compiled
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub const fn arm64_entry(&self) -> Option<Arm64RootEntry> {
        self.arm64_entry
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub const fn arm64_root_return(&self) -> Option<Arm64CodePtr> {
        self.arm64_root_return
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    pub fn with_armv7a_entry(
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
    pub const fn armv7a_entry(&self) -> Option<Armv7aRootEntry> {
        self.armv7a_entry
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    pub const fn armv7a_root_return(&self) -> Option<Armv7aCodePtr> {
        self.armv7a_root_return
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn with_x86_64_entry(
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
    pub const fn x86_64_entry(&self) -> Option<X86_64RootEntry> {
        self.x86_64_entry
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub const fn x86_64_root_return(&self) -> Option<X86_64CodePtr> {
        self.x86_64_root_return
    }

    #[inline]
    pub fn program(&self) -> Option<&MachineFunction> {
        self.compiled.function(self.func_id)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeCodeCache {
    compiled: bool,
}

impl NativeCodeCache {
    #[inline]
    pub const fn is_compiled(self) -> bool {
        self.compiled
    }

    #[inline]
    pub const fn compiled() -> Self {
        Self { compiled: true }
    }
}
