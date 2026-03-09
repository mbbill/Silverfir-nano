//! Fast code storage and cached metadata.

use super::instruction::Instruction;
use alloc::boxed::Box;

/// Compiled fast interpreter code for a function.
pub struct FastCode {
    code: Box<[Instruction]>,
    #[cfg(feature = "lockstep-debug")]
    entry_stub: Option<Box<[Instruction]>>,
}

impl core::fmt::Debug for FastCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FastCode")
            .field("code_len", &self.code.len())
            .finish()
    }
}

impl FastCode {
    pub fn new(code: Box<[Instruction]>) -> Self {
        FastCode {
            code,
            #[cfg(feature = "lockstep-debug")]
            entry_stub: None,
        }
    }

    #[inline]
    pub fn entry_ptr(&self) -> *mut Instruction {
        if self.code.is_empty() {
            core::ptr::null_mut()
        } else {
            self.code.as_ptr() as *mut Instruction
        }
    }

    #[inline]
    pub fn code_len(&self) -> usize {
        self.code.len()
    }

    #[inline]
    pub fn code(&self) -> &[Instruction] {
        &self.code
    }

    pub fn build_cache(&self, params_len: usize, locals_len: usize, results_len: usize) -> FastCodeCache {
        FastCodeCache {
            entry: self.entry_ptr(),
            #[cfg(feature = "lockstep-debug")]
            entry_override: core::ptr::null_mut(),
            params_len,
            locals_len,
            results_len,
        }
    }

    #[cfg(feature = "lockstep-debug")]
    pub fn install_entry_stub(
        &mut self,
        stub: Box<[Instruction]>,
        cache: &mut FastCodeCache,
    ) {
        let ptr = stub.as_ptr() as *mut Instruction;
        self.entry_stub = Some(stub);
        cache.entry_override = ptr;
    }
}

/// Cached fast code metadata for hot-path access.
#[derive(Debug, Clone, Copy)]
pub struct FastCodeCache {
    entry: *mut Instruction,
    #[cfg(feature = "lockstep-debug")]
    entry_override: *mut Instruction,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
}

impl Default for FastCodeCache {
    fn default() -> Self {
        FastCodeCache {
            entry: core::ptr::null_mut(),
            #[cfg(feature = "lockstep-debug")]
            entry_override: core::ptr::null_mut(),
            params_len: 0,
            locals_len: 0,
            results_len: 0,
        }
    }
}

unsafe impl Send for FastCodeCache {}
unsafe impl Sync for FastCodeCache {}

impl FastCodeCache {
    #[inline(always)]
    pub fn is_compiled(&self) -> bool {
        !self.entry.is_null()
    }

    #[inline(always)]
    pub fn entry(&self) -> *mut Instruction {
        #[cfg(feature = "lockstep-debug")]
        if !self.entry_override.is_null() {
            return self.entry_override;
        }
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

/// Create a FastCode and cache from compiled instructions.
pub fn create_fast_code(
    code: Box<[Instruction]>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
) -> (FastCode, FastCodeCache) {
    let fast_code = FastCode::new(code);
    let cache = fast_code.build_cache(params_len, locals_len, results_len);
    (fast_code, cache)
}
