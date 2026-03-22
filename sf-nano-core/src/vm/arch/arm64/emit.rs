use alloc::vec::Vec;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Arm64TextEmitter {
    text: Vec<u8>,
}

impl Arm64TextEmitter {
    #[inline]
    pub(super) fn new() -> Self {
        Self { text: Vec::new() }
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.text.len()
    }

    #[inline]
    pub(super) fn emit_u32(&mut self, inst: u32) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&inst.to_le_bytes());
        offset
    }

    #[inline]
    pub(super) fn patch_u32(&mut self, offset: usize, inst: u32) {
        self.text[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
    }

    #[inline]
    pub(super) fn emit_u64(&mut self, value: u64) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&value.to_le_bytes());
        offset
    }

    #[inline]
    pub(super) fn patch_u64(&mut self, offset: usize, value: u64) {
        self.text[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[inline]
    pub(super) fn finish(self) -> Vec<u8> {
        self.text
    }
}
