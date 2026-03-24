use alloc::vec::Vec;

/// Unified code emitter: a growable byte buffer with typed emit/patch operations.
///
/// Used by all arch backends to build function text sections.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TextEmitter {
    text: Vec<u8>,
}

impl TextEmitter {
    #[inline]
    pub(crate) fn new() -> Self {
        Self { text: Vec::new() }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.text.len()
    }

    // ── Emit ─────────────────────────────────────────────────────────────

    #[inline]
    pub(crate) fn emit_u8(&mut self, byte: u8) -> usize {
        let offset = self.text.len();
        self.text.push(byte);
        offset
    }

    #[inline]
    pub(crate) fn emit_u32(&mut self, inst: u32) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&inst.to_le_bytes());
        offset
    }

    #[inline]
    pub(crate) fn emit_u64(&mut self, value: u64) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(&value.to_le_bytes());
        offset
    }

    #[inline]
    pub(crate) fn emit_bytes(&mut self, bytes: &[u8]) -> usize {
        let offset = self.text.len();
        self.text.extend_from_slice(bytes);
        offset
    }

    // ── Patch ────────────────────────────────────────────────────────────

    #[inline]
    pub(crate) fn patch_u8(&mut self, offset: usize, byte: u8) {
        self.text[offset] = byte;
    }

    #[inline]
    pub(crate) fn patch_u32(&mut self, offset: usize, inst: u32) {
        self.text[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
    }

    #[inline]
    pub(crate) fn patch_i32(&mut self, offset: usize, value: i32) {
        self.text[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    #[inline]
    pub(crate) fn patch_u64(&mut self, offset: usize, value: u64) {
        self.text[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    // ── Read ─────────────────────────────────────────────────────────────

    #[inline]
    pub(crate) fn byte(&self, offset: usize) -> u8 {
        self.text[offset]
    }

    // ── Finish ───────────────────────────────────────────────────────────

    #[inline]
    pub(crate) fn finish(self) -> Vec<u8> {
        self.text
    }
}
