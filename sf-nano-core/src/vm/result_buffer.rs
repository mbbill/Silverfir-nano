//! Shared execution-value stack used by active runtimes.

use alloc::vec;
use alloc::vec::Vec;

use crate::vm::raw_value::RawValue;

#[derive(Debug)]
pub(crate) struct ResultBuffer {
    buffer: Vec<RawValue>,
    sp: usize,
}

impl ResultBuffer {
    pub(crate) fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            sp: 0,
        }
    }

    pub(crate) fn with_exact_capacity(total_capacity: usize) -> Self {
        Self {
            buffer: vec![0; total_capacity],
            sp: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, value: RawValue) {
        debug_assert!(
            self.sp < self.buffer.len(),
            "ResultBuffer overflow: ensure_capacity/with_exact_capacity must guarantee space"
        );
        unsafe {
            *self.buffer.get_unchecked_mut(self.sp) = value;
        }
        self.sp += 1;
    }

    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.sp
    }

    #[inline(always)]
    pub(crate) fn peek_at_index(&self, index: usize) -> RawValue {
        debug_assert!(index < self.sp, "Stack index out of bounds");
        unsafe { *self.buffer.get_unchecked(index) }
    }
}

impl Default for ResultBuffer {
    fn default() -> Self {
        Self::new()
    }
}
