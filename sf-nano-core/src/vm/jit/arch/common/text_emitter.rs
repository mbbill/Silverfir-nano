/// Unified code emitter: either an owned growable byte buffer or a direct
/// writer into the shared executable code buffer.
///
/// Used by all arch backends to build function text sections.
use crate::collections;
use crate::vm::jit::runtime::code_buf::CodeBuffer;

pub(crate) struct TextEmitter {
    storage: TextStorage,
}

enum TextStorage {
    Owned(collections::Vec<u8>),
    CodeBuffer {
        buf: *mut CodeBuffer,
        start: usize,
        len: usize,
    },
}

impl core::fmt::Debug for TextEmitter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.storage {
            TextStorage::Owned(text) => f
                .debug_struct("TextEmitter")
                .field("kind", &"owned")
                .field("len", &text.len())
                .finish(),
            TextStorage::CodeBuffer { start, len, .. } => f
                .debug_struct("TextEmitter")
                .field("kind", &"code_buffer")
                .field("start", start)
                .field("len", len)
                .finish(),
        }
    }
}

impl Default for TextEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextEmitter {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            storage: TextStorage::Owned(collections::Vec::new()),
        }
    }

    #[inline]
    // Its one caller (`into_artifact`) exists only in std builds.
    #[cfg(sf_has_std)]
    pub(crate) fn from_owned(text: collections::Vec<u8>) -> Self {
        Self {
            storage: TextStorage::Owned(text),
        }
    }

    #[inline]
    pub(crate) fn new_in_code_buffer(buf: &mut CodeBuffer) -> Self {
        Self {
            storage: TextStorage::CodeBuffer {
                buf,
                start: buf.len(),
                len: 0,
            },
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        match &self.storage {
            TextStorage::Owned(text) => text.len(),
            TextStorage::CodeBuffer { len, .. } => *len,
        }
    }

    /// Value to align against when padding the next instruction to a
    /// boundary. For the shared code buffer this is the true runtime
    /// address of the next byte; an owned buffer has no runtime address
    /// yet, so its own length stands in (artifact paths re-base
    /// page-aligned).
    #[inline]
    pub(crate) fn next_addr_for_alignment(&self) -> usize {
        match &self.storage {
            TextStorage::Owned(text) => text.len(),
            TextStorage::CodeBuffer { buf, .. } => unsafe { (**buf).next_write_addr() },
        }
    }

    // ── Emit ─────────────────────────────────────────────────────────────

    #[cfg(sf_backend_x64)]
    #[inline]
    pub(crate) fn emit_u8(&mut self, byte: u8) -> usize {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                let offset = text.len();
                text.push(byte);
                offset
            }
            TextStorage::CodeBuffer { buf, len, .. } => {
                let offset = *len;
                unsafe { (&mut **buf).emit_u8(byte) };
                *len += 1;
                offset
            }
        }
    }

    #[cfg(sf_arm32_isa_thumb)]
    #[inline]
    pub(crate) fn emit_u16(&mut self, value: u16) -> usize {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                let offset = text.len();
                text.extend_from_slice(&value.to_le_bytes());
                offset
            }
            TextStorage::CodeBuffer { buf, len, .. } => {
                let offset = *len;
                unsafe {
                    (&mut **buf).emit_u8(value as u8);
                    (&mut **buf).emit_u8((value >> 8) as u8);
                };
                *len += 2;
                offset
            }
        }
    }

    #[inline]
    pub(crate) fn emit_u32(&mut self, inst: u32) -> usize {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                let offset = text.len();
                text.extend_from_slice(&inst.to_le_bytes());
                offset
            }
            TextStorage::CodeBuffer { buf, len, .. } => {
                let offset = *len;
                unsafe { (&mut **buf).emit_u32(inst) };
                *len += 4;
                offset
            }
        }
    }

    #[cfg(any(sf_backend_arm64, sf_backend_x64, sf_backend_riscv64))]
    #[inline]
    pub(crate) fn emit_u64(&mut self, value: u64) -> usize {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                let offset = text.len();
                text.extend_from_slice(&value.to_le_bytes());
                offset
            }
            TextStorage::CodeBuffer { buf, len, .. } => {
                let offset = *len;
                unsafe { (&mut **buf).emit_u64(value) };
                *len += 8;
                offset
            }
        }
    }

    #[inline]
    #[cfg(sf_backend_x64)]
    pub(crate) fn emit_bytes(&mut self, bytes: &[u8]) -> usize {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                let offset = text.len();
                text.extend_from_slice(bytes);
                offset
            }
            TextStorage::CodeBuffer { buf, len, .. } => {
                let offset = *len;
                unsafe { (&mut **buf).emit_bytes(bytes) };
                *len += bytes.len();
                offset
            }
        }
    }

    // ── Patch ────────────────────────────────────────────────────────────

    #[cfg(not(sf_backend_x64))]
    #[inline]
    pub(crate) fn patch_u32(&mut self, offset: usize, inst: u32) {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                text[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
            }
            TextStorage::CodeBuffer { buf, start, len } => {
                assert!(offset + 4 <= *len, "patch beyond written region");
                unsafe { (&mut **buf).patch_u32(*start + offset, inst) };
            }
        }
    }

    #[cfg(sf_backend_x64)]
    #[inline]
    pub(crate) fn patch_i32(&mut self, offset: usize, value: i32) {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                text[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            TextStorage::CodeBuffer { buf, start, len } => {
                assert!(offset + 4 <= *len, "patch beyond written region");
                unsafe { (&mut **buf).patch_u32(*start + offset, value as u32) };
            }
        }
    }

    #[cfg(any(sf_backend_arm64, sf_backend_x64, sf_backend_riscv64))]
    #[inline]
    pub(crate) fn patch_u64(&mut self, offset: usize, value: u64) {
        match &mut self.storage {
            TextStorage::Owned(text) => {
                text[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
            TextStorage::CodeBuffer { buf, start, len } => {
                assert!(offset + 8 <= *len, "patch beyond written region");
                unsafe { (&mut **buf).patch_u64(*start + offset, value) };
            }
        }
    }

    // ── Read ─────────────────────────────────────────────────────────────

    #[inline]
    pub(crate) fn byte(&self, offset: usize) -> u8 {
        match &self.storage {
            TextStorage::Owned(text) => text[offset],
            TextStorage::CodeBuffer { buf, start, len } => {
                assert!(offset < *len, "read beyond written region");
                unsafe { (&**buf).byte(*start + offset) }
            }
        }
    }

    #[inline]
    pub(crate) fn read_u32(&self, offset: usize) -> u32 {
        let bytes = [
            self.byte(offset),
            self.byte(offset + 1),
            self.byte(offset + 2),
            self.byte(offset + 3),
        ];
        u32::from_le_bytes(bytes)
    }

    // ── Finish ───────────────────────────────────────────────────────────

    #[inline]
    pub(crate) fn finish(self) -> collections::Vec<u8> {
        match self.storage {
            TextStorage::Owned(text) => text,
            TextStorage::CodeBuffer { .. } => {
                panic!("cannot finish a CodeBuffer-backed TextEmitter into owned bytes")
            }
        }
    }

    #[inline]
    pub(crate) fn into_owned(self) -> Option<collections::Vec<u8>> {
        match self.storage {
            TextStorage::Owned(text) => Some(text),
            TextStorage::CodeBuffer { .. } => None,
        }
    }
}
