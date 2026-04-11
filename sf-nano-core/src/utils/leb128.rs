// MIT License
// Copyright (c) 2021 Mohanson
// Based on https://github.com/mohanson/leb128

use crate::collections;

use core::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum ReadError {
    InsufficientData,
    ValueTooLong,
    UnusedBitsSet,
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::InsufficientData => write!(f, "Unexpected end of input"),
            ReadError::ValueTooLong => write!(f, "Value too long"),
            ReadError::UnusedBitsSet => write!(f, "Unused bits set"),
        }
    }
}

pub(crate) fn read_leb128_u32(data: &[u8]) -> Result<(u32, usize), ReadError> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    let mut ptr = data.iter();
    let mut consumed = 0;
    let mut b: u8;
    loop {
        if shift >= 32 {
            return Err(ReadError::ValueTooLong);
        }
        b = match ptr.next() {
            Some(b) => *b,
            None => return Err(ReadError::InsufficientData),
        };
        consumed += 1;
        result |= ((b & 0x7f) as u32) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    if consumed == 5 && b & 0b01110000 != 0 {
        return Err(ReadError::UnusedBitsSet);
    }
    Ok((result, consumed))
}

pub(crate) fn read_leb128_i32(data: &[u8]) -> Result<(i32, usize), ReadError> {
    let mut result: i32 = 0;
    let mut shift: u32 = 0;
    let mut ptr = data.iter();
    let mut consumed = 0;
    let mut b: u8;
    loop {
        if shift >= 32 {
            return Err(ReadError::ValueTooLong);
        }
        b = match ptr.next() {
            Some(b) => *b,
            None => return Err(ReadError::InsufficientData),
        };
        consumed += 1;
        result |= ((b & 0x7f) as i32) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    if shift < 32 && (b & 0x40) != 0 {
        result |= (-1i32) << shift;
    }
    if consumed == 5 {
        match b & 0b01111000 {
            0b00000000 | 0b01111000 => {}
            _ => return Err(ReadError::UnusedBitsSet),
        }
    }
    Ok((result, consumed))
}

pub(crate) fn read_leb128_i64(data: &[u8]) -> Result<(i64, usize), ReadError> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    let mut ptr = data.iter();
    let mut consumed = 0;
    let mut b: u8;
    loop {
        if shift >= 64 {
            return Err(ReadError::ValueTooLong);
        }
        b = match ptr.next() {
            Some(b) => *b,
            None => return Err(ReadError::InsufficientData),
        };
        consumed += 1;
        result |= ((b & 0x7f) as i64) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    if shift < 64 && (b & 0x40) != 0 {
        result |= (-1i64) << shift;
    }
    if consumed == 10 {
        match b & 0b01111111 {
            0b00000000 | 0b01111111 => {}
            _ => return Err(ReadError::UnusedBitsSet),
        }
    }
    Ok((result, consumed))
}

pub(crate) fn read_leb128_u64(data: &[u8]) -> Result<(u64, usize), ReadError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut ptr = data.iter();
    let mut consumed = 0;
    let mut b: u8;
    loop {
        if shift >= 64 {
            return Err(ReadError::ValueTooLong);
        }
        b = match ptr.next() {
            Some(b) => *b,
            None => return Err(ReadError::InsufficientData),
        };
        consumed += 1;
        result |= ((b & 0x7f) as u64) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    if consumed == 10 && (b & 0b01111110) != 0 {
        return Err(ReadError::UnusedBitsSet);
    }
    Ok((result, consumed))
}

/// Write a signed 32-bit value as LEB128
pub(crate) fn write_leb128_i32(value: i32) -> collections::Vec<u8> {
    let mut result = collections::Vec::new();
    let mut value = value;
    let mut more = true;
    while more {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            byte |= 0x80;
        }
        result.push(byte);
    }
    result
}
