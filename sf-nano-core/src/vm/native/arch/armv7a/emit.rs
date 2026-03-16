use alloc::vec::Vec;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Arm32TextEmitter {
    text: Vec<u8>,
}

impl Arm32TextEmitter {
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
    pub fn byte(&self, offset: usize) -> u8 {
        self.text[offset]
    }

    #[inline]
    pub fn finish(self) -> Vec<u8> {
        self.text
    }
}
