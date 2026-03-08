//! Native compiled code storage and cached helper metadata.

use super::bridge::HelperMetadata;
use super::instruction::{NativeEntry, NativeInst};
use alloc::boxed::Box;

pub struct NativeCode {
    code: Box<[NativeInst]>,
    helper_metadata: Box<[HelperMetadata]>,
}

impl core::fmt::Debug for NativeCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativeCode")
            .field("code_len", &self.code.len())
            .field("helper_metadata_len", &self.helper_metadata.len())
            .finish()
    }
}

impl NativeCode {
    pub fn new(code: Box<[NativeInst]>, helper_metadata: Box<[HelperMetadata]>) -> Self {
        Self { code, helper_metadata }
    }

    #[inline]
    pub fn entry_ptr(&self) -> *mut NativeInst {
        if self.code.is_empty() {
            core::ptr::null_mut()
        } else {
            self.code.as_ptr() as *mut NativeInst
        }
    }

    #[inline]
    pub fn code(&self) -> &[NativeInst] {
        &self.code
    }

    #[inline]
    pub fn helper_metadata(&self) -> &[HelperMetadata] {
        &self.helper_metadata
    }
}

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
            entry: self.code.first().map(|inst| inst.entry),
            params_len,
            locals_len,
            results_len,
        }
    }
}

pub fn create_native_code(
    code: Box<[NativeInst]>,
    helper_metadata: Box<[HelperMetadata]>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
) -> (NativeCode, NativeCodeCache) {
    let native_code = NativeCode::new(code, helper_metadata);
    let cache = native_code.build_cache(params_len, locals_len, results_len);
    (native_code, cache)
}
