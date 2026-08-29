use crate::utils::leb128;
use core::fmt;

#[derive(Debug)]
pub enum PayloadError {
    UnexpectedEndOfInput(&'static str),
    InvalidData(&'static str),
    InvalidLEB128(leb128::ReadError),
    RewindOutOfBounds(&'static str),
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayloadError::UnexpectedEndOfInput(s) => write!(f, "UnexpectedEndOfInput: {}", s),
            PayloadError::InvalidData(s) => write!(f, "InvalidData: {}", s),
            PayloadError::InvalidLEB128(e) => write!(f, "InvalidLEB128: {}", e),
            PayloadError::RewindOutOfBounds(s) => write!(f, "RewindOutOfBounds: {}", s),
        }
    }
}

impl From<leb128::ReadError> for PayloadError {
    fn from(e: leb128::ReadError) -> Self {
        PayloadError::InvalidLEB128(e)
    }
}

#[derive(Clone)]
pub struct Payload<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> From<&'a [u8]> for Payload<'a> {
    fn from(data: &'a [u8]) -> Self {
        Payload { data, position: 0 }
    }
}

impl<'a> Payload<'a> {
    pub fn is_empty(&self) -> bool {
        self.data.len() == self.position
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn read_leb128_u32(&mut self) -> Result<u32, PayloadError> {
        let (value, consumed) = leb128::read_leb128_u32(&self.data[self.position..])?;
        self.position += consumed;
        Ok(value)
    }

    pub fn read_leb128_i32(&mut self) -> Result<i32, PayloadError> {
        let (value, consumed) = leb128::read_leb128_i32(&self.data[self.position..])?;
        self.position += consumed;
        Ok(value)
    }

    pub fn read_leb128_i64(&mut self) -> Result<i64, PayloadError> {
        let (value, consumed) = leb128::read_leb128_i64(&self.data[self.position..])?;
        self.position += consumed;
        Ok(value)
    }

    pub fn read_leb128_u64(&mut self) -> Result<u64, PayloadError> {
        let (value, consumed) = leb128::read_leb128_u64(&self.data[self.position..])?;
        self.position += consumed;
        Ok(value)
    }

    pub fn read_u8(&mut self) -> Result<u8, PayloadError> {
        if self.is_empty() {
            return Err(PayloadError::UnexpectedEndOfInput("read_u8"));
        }
        let byte = self.data[self.position];
        self.position += 1;
        Ok(byte)
    }

    pub fn peek_u8(&self) -> Result<u8, PayloadError> {
        if self.is_empty() {
            return Err(PayloadError::UnexpectedEndOfInput("peek_u8"));
        }
        Ok(self.data[self.position])
    }

    pub fn read_f32(&mut self) -> Result<f32, PayloadError> {
        if self.position + 4 > self.data.len() {
            return Err(PayloadError::UnexpectedEndOfInput("read_f32"));
        }
        let bytes = &self.data[self.position..self.position + 4];
        self.position += 4;
        Ok(f32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| PayloadError::InvalidData("read_f32"))?,
        ))
    }

    pub fn read_f64(&mut self) -> Result<f64, PayloadError> {
        if self.position + 8 > self.data.len() {
            return Err(PayloadError::UnexpectedEndOfInput("read_f64"));
        }
        let bytes = &self.data[self.position..self.position + 8];
        self.position += 8;
        Ok(f64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| PayloadError::InvalidData("read_f64"))?,
        ))
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&[u8], PayloadError> {
        if self.position + len > self.data.len() {
            return Err(PayloadError::UnexpectedEndOfInput("read_bytes"));
        }
        let bytes = &self.data[self.position..self.position + len];
        self.position += len;
        Ok(bytes)
    }

    pub fn read_length_prefixed_utf8(&mut self) -> Result<&str, PayloadError> {
        let len = self.read_leb128_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        core::str::from_utf8(bytes)
            .map_err(|_| PayloadError::InvalidData("read_length_prefixed_utf8"))
    }

    pub fn rewind(&mut self, len: usize) -> Result<(), PayloadError> {
        if len > self.position {
            return Err(PayloadError::RewindOutOfBounds("rewind"));
        }
        self.position -= len;
        Ok(())
    }

    pub fn remaining_slice(&self) -> &'a [u8] {
        &self.data[self.position..]
    }

    /// Advance after a caller has already proved that `len` bytes remain.
    ///
    /// The interpreter predecoder probes a borrowed suffix before committing
    /// a common-op fast decode. Keeping the proof on the probe lets the commit
    /// be one cursor update rather than decoding the immediates a second time.
    #[cfg(sf_interp)]
    #[inline]
    pub(crate) fn advance_known_valid(&mut self, len: usize) {
        debug_assert!(len <= self.data.len() - self.position);
        self.position += len;
    }

    pub fn advance_and_split_at(&mut self, len: usize) -> Result<&'a [u8], PayloadError> {
        if self.position + len > self.data.len() {
            return Err(PayloadError::UnexpectedEndOfInput("advance_and_split_at"));
        }
        let slice = &self.data[self.position..self.position + len];
        self.position += len;
        Ok(slice)
    }
}
