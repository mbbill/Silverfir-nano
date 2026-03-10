//! Fast compiled code object.

use alloc::boxed::Box;

use super::instruction::Instruction;

#[derive(Clone, Debug, Default)]
pub struct FastCode {
    pub instructions: Box<[Instruction]>,
}

impl FastCode {
    #[inline]
    pub fn from_instructions(instructions: alloc::vec::Vec<Instruction>) -> Self {
        Self {
            instructions: instructions.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn entry_ptr(&self) -> *mut Instruction {
        if self.instructions.is_empty() {
            core::ptr::null_mut()
        } else {
            self.instructions.as_ptr() as *mut Instruction
        }
    }

    #[inline]
    pub fn code_len(&self) -> usize {
        self.instructions.len()
    }

    #[inline]
    pub fn code(&self) -> &[Instruction] {
        &self.instructions
    }

    #[inline]
    pub fn build_cache(
        &self,
        params_len: usize,
        locals_len: usize,
        results_len: usize,
    ) -> FastCodeCache {
        FastCodeCache {
            entry: self.entry_ptr(),
            params_len,
            locals_len,
            results_len,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastCodeCache {
    entry: *mut Instruction,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
}

impl Default for FastCodeCache {
    fn default() -> Self {
        Self {
            entry: core::ptr::null_mut(),
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

pub fn create_fast_code(
    code: Box<[Instruction]>,
    params_len: usize,
    locals_len: usize,
    results_len: usize,
) -> (FastCode, FastCodeCache) {
    let fast_code = FastCode { instructions: code };
    let cache = fast_code.build_cache(params_len, locals_len, results_len);
    (fast_code, cache)
}
