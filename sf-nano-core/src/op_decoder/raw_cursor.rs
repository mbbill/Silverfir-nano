//! Allocation-free, single-step decoder used by decoder experiments and tests.
//!
//! This deliberately stays behind `cfg(test)`: it is an oracle/prototyping
//! surface, not a second production decoder. Every successful `next` commits
//! exactly one instruction; errors leave the cursor at the original byte.

use super::{decode_block_type, decode_mem_arg, BlockType, Immediate};
use crate::{
    error::WasmError,
    opcodes::{Opcode, OpcodeFB, OpcodeFC, OpcodeFD, WasmOpcode},
    utils::{
        leb128,
        payload::{Payload, PayloadError},
    },
    value_type::{HeapType, RefType, ValueType},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawBlockType {
    Empty,
    Value(ValueType),
    TypeIndex(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawU32Range<'a> {
    bytes: &'a [u8],
    count: u32,
}

impl<'a> RawU32Range<'a> {
    pub(crate) const fn len(self) -> u32 {
        self.count
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.count == 0
    }

    pub(crate) const fn encoded(self) -> &'a [u8] {
        self.bytes
    }

    pub(crate) const fn iter(self) -> RawU32Iter<'a> {
        RawU32Iter {
            bytes: self.bytes,
            pc: 0,
            remaining: self.count,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawU32Iter<'a> {
    bytes: &'a [u8],
    pc: usize,
    remaining: u32,
}

impl Iterator for RawU32Iter<'_> {
    type Item = Result<u32, WasmError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let decoded = leb128::read_leb128_u32(&self.bytes[self.pc..]);
        match decoded {
            Ok((value, consumed)) => {
                self.pc += consumed;
                self.remaining -= 1;
                Some(Ok(value))
            }
            Err(error) => {
                self.remaining = 0;
                Some(Err(PayloadError::InvalidLEB128(error).into()))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RawU32Iter<'_> {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RawImmediate<'a> {
    None,
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
    Block(RawBlockType),
    RefType(ValueType),
    BrTable {
        labels: RawU32Range<'a>,
        default: u32,
    },
    LabelIndex(u32),
    FunctionIndex(u32),
    LocalIndex(u32),
    GlobalIndex(u32),
    TableIndex(u32),
    DataIndex(u32),
    ElementIndex(u32),
    MemoryIndex(u32),
    MemoryInit {
        dataidx: u32,
        memidx: u32,
    },
    MemoryCopy {
        dstidx: u32,
        srcidx: u32,
    },
    CallIndirect {
        typeidx: u32,
        tableidx: u32,
    },
    MemArg {
        align: u32,
        offset: u64,
        memidx: u32,
    },
    TableInit {
        elemidx: u32,
        tableidx: u32,
    },
    TableCopy {
        dstidx: u32,
        srcidx: u32,
    },
    TypeIndex(u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RawOp<'a> {
    pub(crate) wasm_op: WasmOpcode,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) imm: RawImmediate<'a>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RawDecodeError {
    Decode(WasmError),
    Unsupported { opcode: WasmOpcode, offset: usize },
    InvalidPc { pc: usize, len: usize },
}

impl From<WasmError> for RawDecodeError {
    fn from(error: WasmError) -> Self {
        Self::Decode(error)
    }
}

impl From<PayloadError> for RawDecodeError {
    fn from(error: PayloadError) -> Self {
        Self::Decode(error.into())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawOpCursor<'a> {
    bytes: &'a [u8],
    pc: usize,
}

impl<'a> RawOpCursor<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pc: 0 }
    }

    pub(crate) const fn at(bytes: &'a [u8], pc: usize) -> Self {
        Self { bytes, pc }
    }

    pub(crate) const fn position(&self) -> usize {
        self.pc
    }

    pub(crate) fn remaining(&self) -> &'a [u8] {
        self.bytes.get(self.pc..).unwrap_or(&[])
    }

    pub(crate) fn next(&mut self) -> Result<Option<RawOp<'a>>, RawDecodeError> {
        if self.pc > self.bytes.len() {
            return Err(RawDecodeError::InvalidPc {
                pc: self.pc,
                len: self.bytes.len(),
            });
        }
        if self.pc == self.bytes.len() {
            return Ok(None);
        }

        let start = self.pc;
        let mut payload = Payload::from(&self.bytes[start..]);
        let op = Opcode::from_repr(payload.read_u8()?)
            .ok_or_else(|| WasmError::malformed("invalid opcode"))?;
        let (wasm_op, imm) = self.decode_op(start, op, &mut payload)?;
        let end = start + payload.position();
        self.pc = end;
        Ok(Some(RawOp {
            wasm_op,
            start,
            end,
            imm,
        }))
    }

    fn unsupported<T>(&self, opcode: WasmOpcode, offset: usize) -> Result<T, RawDecodeError> {
        Err(RawDecodeError::Unsupported { opcode, offset })
    }

    fn decode_op(
        &self,
        start: usize,
        op: Opcode,
        payload: &mut Payload<'a>,
    ) -> Result<(WasmOpcode, RawImmediate<'a>), RawDecodeError> {
        use crate::opcodes::{Opcode::*, OpcodeFC::*, WasmOpcode::*};

        let decoded = match op {
            BLOCK | LOOP | IF => {
                let block = match decode_block_type(payload)? {
                    BlockType::Empty => RawBlockType::Empty,
                    BlockType::ValueType(value) => RawBlockType::Value(value),
                    BlockType::TypeIndex(index) => RawBlockType::TypeIndex(index),
                };
                (OP(op), RawImmediate::Block(block))
            }
            BR | BR_IF | BR_ON_NULL | BR_ON_NON_NULL => {
                (OP(op), RawImmediate::LabelIndex(payload.read_leb128_u32()?))
            }
            BR_TABLE => {
                let count = payload.read_leb128_u32()?;
                let labels_start = start + payload.position();
                for _ in 0..count {
                    payload.read_leb128_u32()?;
                }
                let labels_end = start + payload.position();
                let default = payload.read_leb128_u32()?;
                (
                    OP(op),
                    RawImmediate::BrTable {
                        labels: RawU32Range {
                            bytes: &self.bytes[labels_start..labels_end],
                            count,
                        },
                        default,
                    },
                )
            }
            CALL | RETURN_CALL | REF_FUNC => (
                OP(op),
                RawImmediate::FunctionIndex(payload.read_leb128_u32()?),
            ),
            CALL_INDIRECT | RETURN_CALL_INDIRECT => (
                OP(op),
                RawImmediate::CallIndirect {
                    typeidx: payload.read_leb128_u32()?,
                    tableidx: payload.read_leb128_u32()?,
                },
            ),
            CALL_REF | RETURN_CALL_REF => {
                (OP(op), RawImmediate::TypeIndex(payload.read_leb128_u32()?))
            }
            LOCAL_GET | LOCAL_SET | LOCAL_TEE => {
                (OP(op), RawImmediate::LocalIndex(payload.read_leb128_u32()?))
            }
            GLOBAL_GET | GLOBAL_SET => (
                OP(op),
                RawImmediate::GlobalIndex(payload.read_leb128_u32()?),
            ),
            TABLE_GET | TABLE_SET => (OP(op), RawImmediate::TableIndex(payload.read_leb128_u32()?)),
            MEMORY_SIZE | MEMORY_GROW => (
                OP(op),
                RawImmediate::MemoryIndex(payload.read_leb128_u32()?),
            ),
            I32_CONST => (OP(op), RawImmediate::I32(payload.read_leb128_i32()?)),
            I64_CONST => (OP(op), RawImmediate::I64(payload.read_leb128_i64()?)),
            F32_CONST => (OP(op), RawImmediate::F32(payload.read_f32()?.to_bits())),
            F64_CONST => (OP(op), RawImmediate::F64(payload.read_f64()?.to_bits())),
            I32_LOAD | I32_LOAD8_S | I32_LOAD8_U | I32_LOAD16_S | I32_LOAD16_U | I64_LOAD
            | I64_LOAD8_S | I64_LOAD8_U | I64_LOAD16_S | I64_LOAD16_U | I64_LOAD32_S
            | I64_LOAD32_U | F32_LOAD | F64_LOAD | I32_STORE | I32_STORE8 | I32_STORE16
            | I64_STORE | I64_STORE8 | I64_STORE16 | I64_STORE32 | F32_STORE | F64_STORE => {
                let Immediate::MemArg {
                    align,
                    offset,
                    memidx,
                } = decode_mem_arg(payload)?
                else {
                    unreachable!();
                };
                (
                    OP(op),
                    RawImmediate::MemArg {
                        align,
                        offset,
                        memidx,
                    },
                )
            }
            REF_NULL => {
                let heap_type = HeapType::parse(payload)?;
                (
                    OP(op),
                    RawImmediate::RefType(ValueType::Ref(RefType::new(true, heap_type))),
                )
            }
            PREFIX_FC => {
                let ext: OpcodeFC = payload.read_leb128_u32()?.try_into()?;
                let imm = match ext {
                    MEMORY_INIT => RawImmediate::MemoryInit {
                        dataidx: payload.read_leb128_u32()?,
                        memidx: payload.read_leb128_u32()?,
                    },
                    MEMORY_COPY => RawImmediate::MemoryCopy {
                        dstidx: payload.read_leb128_u32()?,
                        srcidx: payload.read_leb128_u32()?,
                    },
                    MEMORY_FILL => RawImmediate::MemoryIndex(payload.read_leb128_u32()?),
                    DATA_DROP => RawImmediate::DataIndex(payload.read_leb128_u32()?),
                    ELEM_DROP => RawImmediate::ElementIndex(payload.read_leb128_u32()?),
                    TABLE_GROW | TABLE_SIZE | TABLE_FILL => {
                        RawImmediate::TableIndex(payload.read_leb128_u32()?)
                    }
                    TABLE_INIT => RawImmediate::TableInit {
                        elemidx: payload.read_leb128_u32()?,
                        tableidx: payload.read_leb128_u32()?,
                    },
                    TABLE_COPY => RawImmediate::TableCopy {
                        dstidx: payload.read_leb128_u32()?,
                        srcidx: payload.read_leb128_u32()?,
                    },
                    I32_TRUNC_SAT_F32_S | I32_TRUNC_SAT_F32_U | I32_TRUNC_SAT_F64_S
                    | I32_TRUNC_SAT_F64_U | I64_TRUNC_SAT_F32_S | I64_TRUNC_SAT_F32_U
                    | I64_TRUNC_SAT_F64_S | I64_TRUNC_SAT_F64_U => RawImmediate::None,
                };
                (FC(ext), imm)
            }
            THROW | THROW_REF | TRY_TABLE | SELECT_T => return self.unsupported(OP(op), start),
            PREFIX_FB => {
                let ext: OpcodeFB = payload.read_leb128_u32()?.try_into()?;
                return self.unsupported(FB(ext), start);
            }
            PREFIX_FD => {
                let ext: OpcodeFD = payload.read_leb128_u32()?.try_into()?;
                return self.unsupported(FD(ext), start);
            }
            UNREACHABLE | NOP | ELSE | END | RETURN | DROP | SELECT | REF_IS_NULL | REF_EQ
            | REF_AS_NON_NULL | I32_EQZ | I32_EQ | I32_NE | I32_LT_S | I32_LT_U | I32_GT_S
            | I32_GT_U | I32_LE_S | I32_LE_U | I32_GE_S | I32_GE_U | I64_EQZ | I64_EQ | I64_NE
            | I64_LT_S | I64_LT_U | I64_GT_S | I64_GT_U | I64_LE_S | I64_LE_U | I64_GE_S
            | I64_GE_U | F32_EQ | F32_NE | F32_LT | F32_GT | F32_LE | F32_GE | F64_EQ | F64_NE
            | F64_LT | F64_GT | F64_LE | F64_GE | I32_CLZ | I32_CTZ | I32_POPCNT | I32_ADD
            | I32_SUB | I32_MUL | I32_DIV_S | I32_DIV_U | I32_REM_S | I32_REM_U | I32_AND
            | I32_OR | I32_XOR | I32_SHL | I32_SHR_S | I32_SHR_U | I32_ROTL | I32_ROTR
            | I64_CLZ | I64_CTZ | I64_POPCNT | I64_ADD | I64_SUB | I64_MUL | I64_DIV_S
            | I64_DIV_U | I64_REM_S | I64_REM_U | I64_AND | I64_OR | I64_XOR | I64_SHL
            | I64_SHR_S | I64_SHR_U | I64_ROTL | I64_ROTR | F32_ABS | F32_NEG | F32_CEIL
            | F32_FLOOR | F32_TRUNC | F32_NEAREST | F32_SQRT | F32_ADD | F32_SUB | F32_MUL
            | F32_DIV | F32_MIN | F32_MAX | F32_COPYSIGN | F64_ABS | F64_NEG | F64_CEIL
            | F64_FLOOR | F64_TRUNC | F64_NEAREST | F64_SQRT | F64_ADD | F64_SUB | F64_MUL
            | F64_DIV | F64_MIN | F64_MAX | F64_COPYSIGN | I32_WRAP_I64 | I32_TRUNC_F32_S
            | I32_TRUNC_F32_U | I32_TRUNC_F64_S | I32_TRUNC_F64_U | I64_EXTEND_I32_S
            | I64_EXTEND_I32_U | I64_TRUNC_F32_S | I64_TRUNC_F32_U | I64_TRUNC_F64_S
            | I64_TRUNC_F64_U | F32_CONVERT_I32_S | F32_CONVERT_I32_U | F32_CONVERT_I64_S
            | F32_CONVERT_I64_U | F32_DEMOTE_F64 | F64_CONVERT_I32_S | F64_CONVERT_I32_U
            | F64_CONVERT_I64_S | F64_CONVERT_I64_U | F64_PROMOTE_F32 | I32_REINTERPRET_F32
            | I64_REINTERPRET_F64 | F32_REINTERPRET_I32 | F64_REINTERPRET_I64 | I32_EXTEND8_S
            | I32_EXTEND16_S | I64_EXTEND8_S | I64_EXTEND16_S | I64_EXTEND32_S => {
                (OP(op), RawImmediate::None)
            }
        };
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;
    use crate::op_decoder::{DecodedOp, Decoder};
    use crate::opcodes::{Opcode, OpcodeFC};
    use std::vec::Vec;

    fn decode_generic_at(bytes: &[u8], pc: usize) -> Result<DecodedOp, WasmError> {
        let mut decoder = Decoder::new(&bytes[pc..]);
        decoder.decode_one()?;
        let mut decoded = decoder.current.clone();
        decoded.op_offset += pc;
        decoded.next_op_offset += pc;
        Ok(decoded)
    }

    fn assert_equivalent(raw: RawOp<'_>, generic: &DecodedOp) {
        assert_eq!(raw.wasm_op, generic.wasm_op);
        assert_eq!(raw.start, generic.op_offset);
        assert_eq!(raw.end, generic.next_op_offset);
        match (raw.imm, &generic.imm) {
            (RawImmediate::None, Immediate::None) => {}
            (RawImmediate::I32(a), Immediate::I32(b)) => assert_eq!(a, *b),
            (RawImmediate::I64(a), Immediate::I64(b)) => assert_eq!(a, *b),
            (RawImmediate::F32(a), Immediate::F32(b)) => assert_eq!(a, b.to_bits()),
            (RawImmediate::F64(a), Immediate::F64(b)) => assert_eq!(a, b.to_bits()),
            (RawImmediate::Block(a), Immediate::Block(b)) => match (a, b) {
                (RawBlockType::Empty, BlockType::Empty) => {}
                (RawBlockType::Value(a), BlockType::ValueType(b)) => assert_eq!(a, *b),
                (RawBlockType::TypeIndex(a), BlockType::TypeIndex(b)) => assert_eq!(a, *b),
                _ => panic!("block mismatch: raw={a:?}, generic={b:?}"),
            },
            (RawImmediate::RefType(a), Immediate::RefType(b)) => assert_eq!(a, *b),
            (RawImmediate::LabelIndex(a), Immediate::LabelIndex(b)) => assert_eq!(a, *b),
            (RawImmediate::FunctionIndex(a), Immediate::FunctionIndex(b)) => assert_eq!(a, *b),
            (RawImmediate::LocalIndex(a), Immediate::LocalIndex(b)) => assert_eq!(a, *b),
            (RawImmediate::GlobalIndex(a), Immediate::GlobalIndex(b)) => assert_eq!(a, *b),
            (RawImmediate::TableIndex(a), Immediate::TableIndex(b)) => assert_eq!(a, *b),
            (RawImmediate::DataIndex(a), Immediate::DataIndex(b)) => assert_eq!(a, *b),
            (RawImmediate::ElementIndex(a), Immediate::ElementIndex(b)) => assert_eq!(a, *b),
            (RawImmediate::MemoryIndex(a), Immediate::MemoryIndex(b)) => assert_eq!(a, *b),
            (
                RawImmediate::MemoryInit { dataidx, memidx },
                Immediate::MemoryInitArgs {
                    dataidx: b_data,
                    memidx: b_mem,
                },
            ) => assert_eq!((dataidx, memidx), (*b_data, *b_mem)),
            (
                RawImmediate::MemoryCopy { dstidx, srcidx },
                Immediate::MemoryCopyArgs {
                    dstidx: b_dst,
                    srcidx: b_src,
                },
            ) => assert_eq!((dstidx, srcidx), (*b_dst, *b_src)),
            (
                RawImmediate::CallIndirect { typeidx, tableidx },
                Immediate::CallIndirectArgs {
                    typeidx: b_type,
                    tableidx: b_table,
                },
            ) => assert_eq!((typeidx, tableidx), (*b_type, *b_table)),
            (
                RawImmediate::MemArg {
                    align,
                    offset,
                    memidx,
                },
                Immediate::MemArg {
                    align: b_align,
                    offset: b_offset,
                    memidx: b_memidx,
                },
            ) => assert_eq!((align, offset, memidx), (*b_align, *b_offset, *b_memidx)),
            (
                RawImmediate::TableInit { elemidx, tableidx },
                Immediate::TableInitArgs {
                    elemidx: b_elem,
                    tableidx: b_table,
                },
            ) => assert_eq!((elemidx, tableidx), (*b_elem, *b_table)),
            (
                RawImmediate::TableCopy { dstidx, srcidx },
                Immediate::TableCopyArgs {
                    dstidx: b_dst,
                    srcidx: b_src,
                },
            ) => assert_eq!((dstidx, srcidx), (*b_dst, *b_src)),
            (RawImmediate::TypeIndex(a), Immediate::TypeIndex(b)) => assert_eq!(a, *b),
            (
                RawImmediate::BrTable { labels, default },
                Immediate::BrLabels(b_labels, b_default),
            ) => {
                assert_eq!(labels.len() as usize, b_labels.len());
                assert_eq!(labels.is_empty(), b_labels.is_empty());
                let decoded: Vec<u32> = labels.iter().collect::<Result<_, _>>().unwrap();
                assert_eq!(decoded, b_labels.as_slice());
                assert_eq!(default, *b_default);
                assert!(!labels.encoded().is_empty() || labels.is_empty());
            }
            (a, b) => panic!("immediate mismatch: raw={a:?}, generic={b:?}"),
        }
    }

    fn assert_stream_matches(bytes: &[u8]) {
        let mut raw = RawOpCursor::new(bytes);
        while raw.position() < bytes.len() {
            let pc = raw.position();
            let generic = decode_generic_at(bytes, pc).expect("generic decode");
            let decoded = raw.next().expect("raw decode").expect("one raw op");
            assert_equivalent(decoded, &generic);
        }
        assert!(raw.next().unwrap().is_none());
        assert!(raw.remaining().is_empty());
    }

    fn assert_one_matches_or_is_explicitly_unsupported(bytes: &[u8], pc: usize) -> bool {
        let generic = decode_generic_at(bytes, pc);
        let mut raw = RawOpCursor::at(bytes, pc);
        match (raw.next(), generic) {
            (Ok(Some(raw)), Ok(generic)) => {
                assert_equivalent(raw, &generic);
                true
            }
            (Err(RawDecodeError::Decode(raw_error)), Err(generic)) => {
                assert_eq!(raw_error, generic, "pc={pc}, bytes={bytes:x?}");
                assert_eq!(raw.position(), pc);
                true
            }
            (Err(RawDecodeError::Unsupported { opcode, offset }), Ok(generic)) => {
                assert_eq!(offset, pc);
                assert_eq!(opcode, generic.wasm_op);
                assert_eq!(raw.position(), pc);
                false
            }
            // A non-SIMD build rejects a well-formed FD instruction before
            // exposing it. The raw cursor still classifies the prefix as the
            // deliberately unsupported SIMD family.
            (
                Err(RawDecodeError::Unsupported {
                    opcode: WasmOpcode::FD(_),
                    offset,
                }),
                Err(generic),
            ) => {
                assert_eq!(offset, pc);
                assert_eq!(generic.class(), "invalid");
                assert_eq!(raw.position(), pc);
                false
            }
            (raw, generic) => {
                panic!("raw/generic result mismatch at pc={pc}: raw={raw:?}, generic={generic:?}")
            }
        }
    }

    #[test]
    fn raw_cursor_matches_generic_for_mvp_and_bulk_ops() {
        let code = [
            Opcode::BLOCK as u8,
            0x40,
            Opcode::LOCAL_GET as u8,
            0x80,
            0x00,
            Opcode::GLOBAL_SET as u8,
            0x01,
            Opcode::I32_CONST as u8,
            0x7f,
            Opcode::I64_CONST as u8,
            0x80,
            0x7f,
            Opcode::I32_ADD as u8,
            Opcode::I32_LOAD as u8,
            0x02,
            0x80,
            0x01,
            Opcode::CALL as u8,
            0x03,
            Opcode::CALL_INDIRECT as u8,
            0x01,
            0x00,
            Opcode::BR_TABLE as u8,
            0x02,
            0x00,
            0x81,
            0x00,
            0x02,
            Opcode::PREFIX_FC as u8,
            OpcodeFC::MEMORY_COPY as u8,
            0x00,
            0x00,
            Opcode::END as u8,
        ];
        assert_stream_matches(&code);
    }

    #[test]
    fn raw_cursor_error_and_unsupported_paths_are_transactional() {
        let malformed = [Opcode::LOCAL_GET as u8, 0x80];
        let mut raw = RawOpCursor::new(&malformed);
        let generic = decode_generic_at(&malformed, 0).unwrap_err();
        assert_eq!(raw.next(), Err(RawDecodeError::Decode(generic)));
        assert_eq!(raw.position(), 0);

        let unsupported = [Opcode::TRY_TABLE as u8, 0x40, 0x00];
        let mut raw = RawOpCursor::new(&unsupported);
        assert_eq!(
            raw.next(),
            Err(RawDecodeError::Unsupported {
                opcode: WasmOpcode::OP(Opcode::TRY_TABLE),
                offset: 0,
            })
        );
        assert_eq!(raw.position(), 0);

        let mut raw = RawOpCursor::at(&[], 1);
        assert_eq!(raw.next(), Err(RawDecodeError::InvalidPc { pc: 1, len: 0 }));
    }

    #[test]
    fn every_primary_opcode_byte_matches_or_is_explicitly_unsupported() {
        let mut supported = 0usize;
        let mut unsupported = 0usize;
        for opcode in 0u16..=u8::MAX as u16 {
            let mut bytes = [0u8; 40];
            bytes[0] = opcode as u8;
            bytes[39] = Opcode::END as u8;
            if assert_one_matches_or_is_explicitly_unsupported(&bytes, 0) {
                supported += 1;
            } else {
                unsupported += 1;
            }
        }
        assert!(supported >= 190, "supported primary bytes={supported}");
        assert_eq!(unsupported, 6, "EH/typed-select/GC/SIMD families");
    }

    #[test]
    fn every_fc_opcode_and_immediate_matches_generic() {
        use crate::opcodes::OPCODE_FC_CONSTANTS as fc;

        for (ext, immediates) in [
            (fc::I32_TRUNC_SAT_F32_S, &[][..]),
            (fc::I32_TRUNC_SAT_F32_U, &[][..]),
            (fc::I32_TRUNC_SAT_F64_S, &[][..]),
            (fc::I32_TRUNC_SAT_F64_U, &[][..]),
            (fc::I64_TRUNC_SAT_F32_S, &[][..]),
            (fc::I64_TRUNC_SAT_F32_U, &[][..]),
            (fc::I64_TRUNC_SAT_F64_S, &[][..]),
            (fc::I64_TRUNC_SAT_F64_U, &[][..]),
            (fc::MEMORY_INIT, &[0x80, 0x00, 0x01][..]),
            (fc::DATA_DROP, &[0x80, 0x00][..]),
            (fc::MEMORY_COPY, &[0x00, 0x01][..]),
            (fc::MEMORY_FILL, &[0x01][..]),
            (fc::TABLE_INIT, &[0x80, 0x00, 0x01][..]),
            (fc::ELEM_DROP, &[0x01][..]),
            (fc::TABLE_COPY, &[0x00, 0x01][..]),
            (fc::TABLE_GROW, &[0x01][..]),
            (fc::TABLE_SIZE, &[0x01][..]),
            (fc::TABLE_FILL, &[0x01][..]),
        ] {
            let mut bytes = Vec::with_capacity(2 + immediates.len());
            bytes.push(Opcode::PREFIX_FC as u8);
            bytes.push(ext as u8);
            bytes.extend_from_slice(immediates);
            let generic = decode_generic_at(&bytes, 0).expect("generic FC");
            let raw = RawOpCursor::new(&bytes)
                .next()
                .expect("raw FC")
                .expect("one FC op");
            assert_equivalent(raw, &generic);
        }
    }

    #[test]
    fn noncanonical_and_malformed_leb_match_generic_errors_and_boundaries() {
        for bytes in [
            &[Opcode::LOCAL_GET as u8, 0x80, 0x00][..],
            &[Opcode::LOCAL_SET as u8, 0xff, 0x00][..],
            &[Opcode::I32_CONST as u8, 0xff, 0x7f][..],
            &[Opcode::I64_CONST as u8, 0x80, 0x00][..],
            &[Opcode::I32_LOAD as u8, 0x80, 0x00, 0x80, 0x00][..],
            &[Opcode::BR_TABLE as u8, 0x81, 0x00, 0x80, 0x00, 0x00][..],
        ] {
            assert!(assert_one_matches_or_is_explicitly_unsupported(bytes, 0));
        }

        for bytes in [
            &[Opcode::LOCAL_GET as u8, 0x80][..],
            &[Opcode::LOCAL_GET as u8, 0x80, 0x80, 0x80, 0x80, 0x80][..],
            &[Opcode::I32_CONST as u8, 0x80][..],
            &[Opcode::I64_CONST as u8, 0x80][..],
            &[Opcode::I32_LOAD as u8, 0x00, 0x80][..],
            &[Opcode::BR_TABLE as u8, 0x01, 0x80][..],
            &[Opcode::PREFIX_FC as u8, 0x80][..],
        ] {
            assert!(assert_one_matches_or_is_explicitly_unsupported(bytes, 0));
        }
    }

    fn compare_module_bodies(name: &str, wasm: &[u8]) -> (usize, usize, usize) {
        let module = Module::new(name, wasm).expect("parse corpus module");
        let mut supported = 0usize;
        let mut unsupported = 0usize;
        let mut resumed = 0usize;
        let mut random = 0x4d59_5df4_d0f3_3173u64;
        for function in module.functions() {
            let Some(spec) = function.spec() else {
                continue;
            };
            let bytes = spec.code();
            let mut pc = 0usize;
            let mut starts = Vec::new();
            while pc < bytes.len() {
                let generic = decode_generic_at(bytes, pc).expect("valid corpus body");
                if assert_one_matches_or_is_explicitly_unsupported(bytes, pc) {
                    supported += 1;
                } else {
                    unsupported += 1;
                }
                starts.push(pc);
                assert!(generic.next_op_offset > pc);
                pc = generic.next_op_offset;
            }
            assert_eq!(pc, bytes.len());

            let samples = starts.len().min(16);
            for i in 0..samples {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let chosen = i + (random as usize % (starts.len() - i));
                starts.swap(i, chosen);
                assert_one_matches_or_is_explicitly_unsupported(bytes, starts[i]);
                resumed += 1;
            }
        }
        (supported, unsupported, resumed)
    }

    #[test]
    fn wasi_and_synthetic_bodies_match_generic_at_every_op_and_random_resume() {
        let fib = include_bytes!("../../../benchmarks/wasi/fib/fib_min.wasm");
        let coremark = include_bytes!("../../../benchmarks/wasi/coremark/coremark.wasm");
        let (fib_supported, fib_unsupported, fib_resumed) = compare_module_bodies("fib", fib);
        let (core_supported, core_unsupported, core_resumed) =
            compare_module_bodies("coremark", coremark);
        assert!(fib_supported + core_supported > 1_000);
        assert_eq!(fib_unsupported + core_unsupported, 0);
        assert!(fib_resumed + core_resumed > 100);
    }

    #[test]
    fn eh_gc_typed_select_and_simd_are_explicitly_unsupported() {
        for (bytes, expected) in [
            (
                &[Opcode::THROW as u8, 0x00][..],
                WasmOpcode::OP(Opcode::THROW),
            ),
            (
                &[Opcode::THROW_REF as u8][..],
                WasmOpcode::OP(Opcode::THROW_REF),
            ),
            (
                &[Opcode::TRY_TABLE as u8, 0x40, 0x00][..],
                WasmOpcode::OP(Opcode::TRY_TABLE),
            ),
            (
                &[Opcode::SELECT_T as u8, 0x00][..],
                WasmOpcode::OP(Opcode::SELECT_T),
            ),
            (
                &[Opcode::PREFIX_FB as u8, 0x00, 0x00][..],
                WasmOpcode::FB(OpcodeFB::STRUCT_NEW),
            ),
            (
                &[Opcode::PREFIX_FD as u8, 0x00, 0x00, 0x00][..],
                WasmOpcode::FD(OpcodeFD::V128_LOAD),
            ),
        ] {
            let mut cursor = RawOpCursor::new(bytes);
            assert_eq!(
                cursor.next(),
                Err(RawDecodeError::Unsupported {
                    opcode: expected,
                    offset: 0,
                })
            );
            assert_eq!(cursor.position(), 0);
        }
    }
}
