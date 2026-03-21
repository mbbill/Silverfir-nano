use alloc::vec::Vec;

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
    pub fn len(&self) -> usize {
        self.text.len()
    }

    #[inline]
    pub fn emit_u32(&mut self, inst: u32) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&inst.to_le_bytes());
        offset
    }

    #[inline]
    pub fn patch_u32(&mut self, offset: usize, inst: u32) {
        self.text[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
    }

    #[inline]
    pub fn emit_u64(&mut self, value: u64) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&value.to_le_bytes());
        offset
    }

    #[inline]
    pub fn patch_u64(&mut self, offset: usize, value: u64) {
        self.text[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[inline]
    pub fn finish(self) -> Vec<u8> {
        self.text
    }
}
