//! Native compiled code storage and cached metadata.

use super::instruction::NativeInst;
use alloc::boxed::Box;

pub struct NativeCode {
    code: Box<[NativeInst]>,
}

impl core::fmt::Debug for NativeCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativeCode")
            .field("code_len", &self.code.len())
            .finish()
    }
}

impl NativeCode {
    pub fn new(code: Box<[NativeInst]>) -> Self {
        Self { code }
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
}

#[derive(Debug, Clone, Copy)]
pub struct NativeCodeCache {
    entry: *mut NativeInst,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
}

impl Default for NativeCodeCache {
    fn default() -> Self {
        Self {
            entry: core::ptr::null_mut(),
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
        !self.entry.is_null()
    }

    #[inline(always)]
    pub fn entry(&self) -> *mut NativeInst {
        self.entry
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
            entry: self.entry_ptr(),
            params_len,
            locals_len,
            results_len,
        }
    }
}

pub fn create_native_code(
    code: Box<[NativeInst]>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
) -> (NativeCode, NativeCodeCache) {
    let native_code = NativeCode::new(code);
    let cache = native_code.build_cache(params_len, locals_len, results_len);
    (native_code, cache)
}
