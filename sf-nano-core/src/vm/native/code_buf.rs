//! Executable memory ownership for the native backend.

/// Placeholder executable code buffer.
///
/// The old stable top-level VM surface still expects modules to be able to own
/// a lazily allocated native code buffer. The real native backend will replace
/// this with its proper executable-memory implementation.
#[derive(Debug, Default)]
pub struct CodeBuffer;

impl CodeBuffer {
    #[inline]
    pub fn new() -> Result<Self, &'static str> {
        Ok(Self)
    }
}
