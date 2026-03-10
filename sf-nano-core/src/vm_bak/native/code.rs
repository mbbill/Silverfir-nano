//! Native compiled code storage and cached helper metadata.

use crate::vm::entities::FunctionInst;
use super::bridge::HelperMetadata;
use super::entry::NativeEntry;
use alloc::boxed::Box;

#[derive(Debug, Clone, Copy)]
pub struct DirectCallEntryPatch {
    pub callee: *const FunctionInst,
    pub literal_off: usize,
}

pub struct NativeCode {
    entries: Box<[NativeEntry]>,
    helper_metadata: Box<[HelperMetadata]>,
    direct_call_entry_patches: Box<[DirectCallEntryPatch]>,
}

impl core::fmt::Debug for NativeCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativeCode")
            .field("entry_len", &self.entries.len())
            .field("helper_metadata_len", &self.helper_metadata.len())
            .field("direct_call_entry_patches_len", &self.direct_call_entry_patches.len())
            .finish()
    }
}

impl NativeCode {
    pub fn new(entries: Box<[NativeEntry]>, helper_metadata: Box<[HelperMetadata]>) -> Self {
        Self::new_with_patches(entries, helper_metadata, Box::new([]))
    }

    pub fn new_with_patches(
        entries: Box<[NativeEntry]>,
        helper_metadata: Box<[HelperMetadata]>,
        direct_call_entry_patches: Box<[DirectCallEntryPatch]>,
    ) -> Self {
        Self {
            entries,
            helper_metadata,
            direct_call_entry_patches,
        }
    }

    #[inline]
    pub fn entry(&self) -> Option<NativeEntry> {
        self.entries.first().copied()
    }

    #[inline]
    pub fn entry_at(&self, idx: usize) -> Option<NativeEntry> {
        self.entries.get(idx).copied()
    }

    #[inline]
    pub fn helper_metadata(&self) -> &[HelperMetadata] {
        &self.helper_metadata
    }

    #[inline]
    pub fn helper_metadata_mut(&mut self) -> &mut [HelperMetadata] {
        &mut self.helper_metadata
    }

    #[inline]
    pub fn direct_call_entry_patches(&self) -> &[DirectCallEntryPatch] {
        &self.direct_call_entry_patches
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NativeCodeCache {
    entry: Option<NativeEntry>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
}

impl Default for NativeCodeCache {
    fn default() -> Self {
        Self {
            entry: None,
            params_len: 0,
            locals_len: 0,
            results_len: 0,
        }
    }
}

unsafe impl Send for NativeCodeCache {}
unsafe impl Sync for NativeCodeCache {}

impl NativeCodeCache {
    #[inline(always)]
    pub fn is_compiled(&self) -> bool {
        self.entry.is_some()
    }

    #[inline(always)]
    pub fn entry(&self) -> NativeEntry {
        self.entry.expect("native code cache missing entry")
    }

    #[inline(always)]
    pub fn params_len(&self) -> usize {
        self.params_len
    }

    #[inline(always)]
    pub fn locals_len(&self) -> usize {
        self.locals_len
    }

    #[inline(always)]
    pub fn results_len(&self) -> usize {
        self.results_len
    }
}

impl NativeCode {
    pub fn build_cache(&self, params_len: usize, locals_len: usize, results_len: usize) -> NativeCodeCache {
        NativeCodeCache {
            entry: self.entry(),
            params_len,
            locals_len,
            results_len,
        }
    }
}

pub fn create_native_code(
    entries: Box<[NativeEntry]>,
    helper_metadata: Box<[HelperMetadata]>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
) -> (NativeCode, NativeCodeCache) {
    let native_code = NativeCode::new(entries, helper_metadata);
    let cache = native_code.build_cache(params_len, locals_len, results_len);
    (native_code, cache)
}

pub fn create_native_code_with_patches(
    entries: Box<[NativeEntry]>,
    helper_metadata: Box<[HelperMetadata]>,
    direct_call_entry_patches: Box<[DirectCallEntryPatch]>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
) -> (NativeCode, NativeCodeCache) {
    let native_code =
        NativeCode::new_with_patches(entries, helper_metadata, direct_call_entry_patches);
    let cache = native_code.build_cache(params_len, locals_len, results_len);
    (native_code, cache)
}
