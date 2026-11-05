//! Per-function FP constant pool.
//!
//! A float constant outside the `fmov #imm8` set used to be built in a GP
//! register (up to four MOVZ/MOVK) and moved across with FMOV at every use.
//! It is now one PC-relative `LDR Dt/St, literal` from a pool of distinct
//! words appended after the function body, next to the direct-call
//! veneers. Zero is `MOVI Dd, #0`.

use crate::{
    collections,
    error::WasmError,
    vm::jit::{arch::common::text_emitter::TextEmitter, machine::machine_ir::MachineFloatWidth},
};

use super::{enc, reg::Arm64FpReg};

/// LDR (literal) encodes a signed 19-bit word offset: ±1 MiB.
const LDR_LIT_BYTE_MAX: isize = (1 << 20) - 4;

#[derive(Debug, Default)]
pub(super) struct FpLiteralPool {
    /// Distinct 64-bit pool words in first-use order. An f32 constant
    /// occupies the low half of its own word (little-endian).
    words: collections::Vec<u64>,
    /// `(LDR offset, word index, width)` patched when the pool is placed.
    loads: collections::Vec<(usize, usize, MachineFloatWidth)>,
}

impl FpLiteralPool {
    /// Emit a load of the constant `bits` (in `width`) into `dst`.
    pub(super) fn emit_load(
        &mut self,
        text: &mut TextEmitter,
        width: MachineFloatWidth,
        bits: u64,
        dst: Arm64FpReg,
    ) {
        let word = match width {
            MachineFloatWidth::F32 => u64::from(bits as u32),
            MachineFloatWidth::F64 => bits,
        };
        if word == 0 {
            text.emit_u32(enc::movi_d_zero(dst));
            return;
        }
        let index = match self.words.iter().position(|&existing| existing == word) {
            Some(index) => index,
            None => {
                self.words.push(word);
                self.words.len() - 1
            }
        };
        let offset = text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::ldr_lit_s(dst, 0),
            MachineFloatWidth::F64 => enc::ldr_lit_d(dst, 0),
        });
        self.loads.push((offset, index, width));
    }

    /// Place the pool at the current (unreachable) end-of-body position and
    /// patch every load. Leaves the pool empty for the next function.
    pub(super) fn flush(&mut self, text: &mut TextEmitter) -> Result<(), WasmError> {
        if self.loads.is_empty() {
            self.words.clear();
            return Ok(());
        }
        while text.len() % 8 != 0 {
            text.emit_u32(0);
        }
        let word_offsets: collections::Vec<usize> =
            self.words.iter().map(|&word| text.emit_u64(word)).collect();
        for &(load_offset, index, width) in &self.loads {
            let delta = word_offsets[index] as isize - load_offset as isize;
            if delta & 0b11 != 0 || !(-LDR_LIT_BYTE_MAX..=LDR_LIT_BYTE_MAX).contains(&delta) {
                return Err(WasmError::internal(
                    "arm64 FP literal pool is out of LDR (literal) range",
                ));
            }
            let words = (delta / 4) as i32;
            let dst = Arm64FpReg::from_raw((text.read_u32(load_offset) & 0x1F) as u8);
            text.patch_u32(
                load_offset,
                match width {
                    MachineFloatWidth::F32 => enc::ldr_lit_s(dst, words),
                    MachineFloatWidth::F64 => enc::ldr_lit_d(dst, words),
                },
            );
        }
        self.words.clear();
        self.loads.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn deduplicates_words_and_patches_forward_loads() {
        let mut text = TextEmitter::new();
        let mut pool = FpLiteralPool::default();
        let d3 = Arm64FpReg::from_raw(3);
        let s17 = Arm64FpReg::from_raw(17);
        pool.emit_load(&mut text, MachineFloatWidth::F64, 0x3EB0_C6F7_A0B5_ED8D, d3);
        pool.emit_load(&mut text, MachineFloatWidth::F32, 0x3DCC_CCCD, s17);
        pool.emit_load(
            &mut text,
            MachineFloatWidth::F64,
            0x3EB0_C6F7_A0B5_ED8D,
            s17,
        );
        pool.emit_load(&mut text, MachineFloatWidth::F64, 0, d3);
        pool.flush(&mut text).unwrap();
        let bytes = text.finish();
        // Three loads and one MOVI, then padding to 8, then two words.
        assert_eq!(word(&bytes, 0), 0x5C00_0000 | (4 << 5) | 3); // ldr d3, +16
        assert_eq!(word(&bytes, 4), 0x1C00_0000 | (5 << 5) | 17); // ldr s17, +20
        assert_eq!(word(&bytes, 8), 0x5C00_0000 | (2 << 5) | 17); // ldr d17, +8
        assert_eq!(word(&bytes, 12), 0x2F00_E400 | 3); // movi d3, #0
        assert_eq!(bytes.len(), 32);
        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            0x3EB0_C6F7_A0B5_ED8D
        );
        assert_eq!(
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            0x3DCC_CCCD
        );
    }

    #[test]
    fn empty_pool_emits_nothing() {
        let mut text = TextEmitter::new();
        let mut pool = FpLiteralPool::default();
        pool.emit_load(
            &mut text,
            MachineFloatWidth::F32,
            0,
            Arm64FpReg::from_raw(0),
        );
        pool.flush(&mut text).unwrap();
        assert_eq!(text.finish().len(), 4);
    }
}
