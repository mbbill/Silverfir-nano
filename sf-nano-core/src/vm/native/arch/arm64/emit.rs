//! ARM64 text emission helpers.

use alloc::vec::Vec;

use super::{enc, reg::Arm64Reg};

/// Small byte-oriented ARM64 emitter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Arm64TextEmitter {
    text: Vec<u8>,
}

impl Arm64TextEmitter {
    #[inline]
    pub fn new() -> Self {
        Self { text: Vec::new() }
    }

    #[inline]
    pub fn emit_u32(&mut self, inst: u32) {
        self.text.extend_from_slice(&inst.to_le_bytes());
    }

    #[inline]
    pub fn emit_u64(&mut self, value: u64) {
        self.text.extend_from_slice(&value.to_le_bytes());
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    #[inline]
    pub fn patch_u32(&mut self, offset: usize, inst: u32) {
        self.text[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
    }

    /// Emit:
    /// `ldr scratch, .+8`
    /// `br scratch`
    /// `.quad target`
    #[inline]
    pub fn emit_tail_branch_literal(&mut self, scratch: Arm64Reg, target: u64) {
        self.emit_u32(enc::ldr_lit_64(scratch, 2));
        self.emit_u32(enc::br(scratch));
        self.emit_u64(target);
    }

    #[inline]
    pub fn finish(self) -> Vec<u8> {
        self.text
    }
}
