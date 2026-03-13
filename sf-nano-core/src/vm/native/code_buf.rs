/// Placeholder executable code buffer while the machine backend is being
/// rebuilt.
#[derive(Debug, Default)]
pub struct CodeBuffer;

impl CodeBuffer {
    #[inline]
    pub fn new() -> Result<Self, &'static str> {
        Ok(Self)
    }

    #[inline]
    pub fn with_capacity(_capacity: usize) -> Result<Self, &'static str> {
        Ok(Self)
    }
}
