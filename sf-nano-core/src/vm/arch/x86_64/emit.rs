use alloc::vec::Vec;

/// Variable-length instruction buffer for x86_64 code emission.
///
/// Unlike the ARM64 emitter which emits fixed 4-byte instructions, x86_64
/// instructions are variable length (1-15 bytes). This emitter works with
/// raw byte slices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct X86_64TextEmitter {
    text: Vec<u8>,
}

impl X86_64TextEmitter {
    #[inline]
    pub fn new() -> Self {
        Self { text: Vec::new() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Emit a single byte.
    #[inline]
    pub fn emit_u8(&mut self, byte: u8) -> usize {
        let offset = self.text.len();
        self.text.push(byte);
        offset
    }

    /// Emit a little-endian u16.
    #[inline]
    pub fn emit_u16(&mut self, value: u16) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&value.to_le_bytes());
        offset
    }

    /// Emit a little-endian u32.
    #[inline]
    pub fn emit_u32(&mut self, value: u32) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&value.to_le_bytes());
        offset
    }

    /// Emit a little-endian u64.
    #[inline]
    pub fn emit_u64(&mut self, value: u64) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&value.to_le_bytes());
        offset
    }

    /// Emit a raw byte slice.
    #[inline]
    pub fn emit_bytes(&mut self, bytes: &[u8]) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(bytes);
        offset
    }

    /// Patch a u8 at the given offset.
    #[inline]
    pub fn patch_u8(&mut self, offset: usize, value: u8) {
        self.text[offset] = value;
    }

    /// Patch a little-endian i32 at the given offset (used for rel32 fixups).
    #[inline]
    pub fn patch_i32(&mut self, offset: usize, value: i32) {
        self.text[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Patch a little-endian u64 at the given offset.
    #[inline]
    pub fn patch_u64(&mut self, offset: usize, value: u64) {
        self.text[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[inline]
    pub fn finish(self) -> Vec<u8> {
        self.text
    }
}
