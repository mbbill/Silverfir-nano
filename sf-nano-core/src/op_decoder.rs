use crate::collections;

use core::cell::{Cell, RefCell};
use core::fmt;

use crate::{
    error::WasmError,
    opcodes::{self, *},
    utils::payload::Payload,
    value_type::{HeapType, RefType, ValueType},
};

pub trait OpcodeHandler {
    fn on_decode_begin(&mut self) -> Result<(), WasmError>;

    /// Streaming interface: consumers can fetch one or multiple ops from `stream`.
    fn on_stream<'x, 'y, 'z>(&mut self, stream: &mut OpStream<'x, 'y, 'z>)
        -> Result<(), WasmError>;

    fn on_decode_end(&mut self) -> Result<(), WasmError>;
}

#[derive(Debug, Clone)]
pub enum BlockType {
    Empty,
    ValueType(ValueType),
    TypeIndex(usize),
}

impl fmt::Display for BlockType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BlockType::*;
        match self {
            Empty => write!(f, "empty"),
            ValueType(valtype) => write!(f, "valtype({})", valtype),
            TypeIndex(idx) => write!(f, "typeidx({})", idx),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum Immediate {
    None,
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Block(BlockType),
    RefType(ValueType),
    BrLabels(collections::Vec<u32>, u32),
    LabelIndex(u32),
    FunctionIndex(u32),
    LocalIndex(u32),
    GlobalIndex(u32),
    TableIndex(u32),
    DataIndex(u32),
    ElementIndex(u32),
    MemoryIndex(u32),
    SelectTypes(collections::Vec<ValueType>),
    MemoryInitArgs {
        dataidx: u32,
        memidx: u32,
    },
    MemoryCopyArgs {
        dstidx: u32,
        srcidx: u32,
    },
    CallIndirectArgs {
        typeidx: u32,
        tableidx: u32,
    },
    MemArg {
        align: u32,
        offset: u64,
        memidx: u32,
    },
    TableInitArgs {
        elemidx: u32,
        tableidx: u32,
    },
    TableCopyArgs {
        dstidx: u32,
        srcidx: u32,
    },
    TypeIndex(u32),
    StructFieldArgs {
        typeidx: u32,
        fieldidx: u32,
    },
    ArrayNewFixed {
        typeidx: u32,
        n: u32,
    },
    TwoTypeIndices {
        type1: u32,
        type2: u32,
    },
    TwoU32s {
        value1: u32,
        value2: u32,
    },
    BrOnCast {
        flags: u8,
        label_idx: u32,
        rt1: ValueType,
        rt2: ValueType,
    },
}

#[macro_export]
macro_rules! extract_imm {
    ($enum:expr, $variant:path) => {
        if let $variant(value) = $enum {
            value
        } else {
            unreachable!()
        }
    };
    ($enum:expr, $variant:path, tuple) => {
        if let $variant(value1, value2) = $enum {
            (value1, value2)
        } else {
            unreachable!()
        }
    };
}

impl fmt::Display for Immediate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Immediate::*;
        match self {
            None => write!(f, ""),
            I32(imm) => write!(f, "i32({})", imm),
            I64(imm) => write!(f, "i64({})", imm),
            F32(imm) => write!(f, "f32({:e})", imm),
            F64(imm) => write!(f, "f64({:e})", imm),
            Block(imm) => write!(f, "blocktype({})", imm),
            RefType(imm) => write!(f, "reftype({})", imm),
            BrLabels(v_idx, default_idx) => {
                write!(f, "table({:?}), default_idx({})", v_idx, default_idx)
            }
            LabelIndex(imm) => write!(f, "lableidx({})", imm),
            FunctionIndex(imm) => write!(f, "funcidx({})", imm),
            LocalIndex(imm) => write!(f, "localidx({})", imm),
            GlobalIndex(imm) => write!(f, "globalidx({})", imm),
            TableIndex(imm) => write!(f, "tableidx({})", imm),
            DataIndex(imm) => write!(f, "dataidx({})", imm),
            ElementIndex(imm) => write!(f, "elementidx({})", imm),
            MemoryIndex(imm) => write!(f, "memidx({})", imm),
            SelectTypes(vec) => write!(f, "types({:?})", vec),
            MemoryInitArgs { dataidx, memidx } => {
                write!(f, "dataidx({}), memidx({})", dataidx, memidx)
            }
            MemoryCopyArgs { dstidx, srcidx } => {
                write!(f, "dstidx({}), srcidx({})", dstidx, srcidx)
            }
            CallIndirectArgs { typeidx, tableidx } => {
                write!(f, "typeidx({}), tableidx({})", typeidx, tableidx)
            }
            MemArg {
                align,
                offset,
                memidx,
            } => write!(
                f,
                "align({}), offset({}), memidx({})",
                align, offset, memidx
            ),
            TableInitArgs { elemidx, tableidx } => {
                write!(f, "elemidx({}), tableidx({})", elemidx, tableidx)
            }
            TableCopyArgs { dstidx, srcidx } => write!(f, "dstidx({}), srcidx({})", dstidx, srcidx),
            TypeIndex(typeidx) => write!(f, "typeidx({})", typeidx),
            StructFieldArgs { typeidx, fieldidx } => {
                write!(f, "typeidx({}), fieldidx({})", typeidx, fieldidx)
            }
            ArrayNewFixed { typeidx, n } => write!(f, "typeidx({}), n({})", typeidx, n),
            TwoTypeIndices { type1, type2 } => write!(f, "type1({}), type2({})", type1, type2),
            TwoU32s { value1, value2 } => write!(f, "value1({}), value2({})", value1, value2),
            BrOnCast {
                flags,
                label_idx,
                rt1,
                rt2,
            } => write!(
                f,
                "flags({}), labelidx({}), rt1({}), rt2({})",
                flags, label_idx, rt1, rt2
            ),
        }
    }
}

/// A single decoded instruction with its immediate and byte offsets.
#[derive(Debug, Clone)]
pub struct DecodedOp {
    pub wasm_op: WasmOpcode,
    pub op_offset: usize,
    pub next_op_offset: usize,
    pub imm: Immediate,
}

pub struct Decoder<'a, 'b> {
    handlers: collections::Vec<&'a mut dyn OpcodeHandler>,
    // Lazy decode state (interior mutability to avoid borrow conflicts with handlers)
    payload: RefCell<Payload<'b>>,
    decoded_ops: RefCell<collections::Vec<DecodedOp>>,
    decoded_base: Cell<usize>,
    end_reached: Cell<bool>,
    retain_decoded_ops: Cell<bool>,
}

impl<'a, 'b> Decoder<'a, 'b> {
    pub fn new(code: &'b [u8]) -> Self {
        Decoder {
            handlers: collections::Vec::new(),
            payload: RefCell::new(Payload::from(code)),
            decoded_ops: RefCell::new(collections::Vec::new()),
            decoded_base: Cell::new(0),
            end_reached: Cell::new(false),
            retain_decoded_ops: Cell::new(false),
        }
    }

    pub fn add_handler(&mut self, handler: &'a mut dyn OpcodeHandler) {
        self.handlers.push(handler);
    }

    fn notify_handlers<F>(&mut self, f: F) -> Result<(), WasmError>
    where
        F: FnMut(&mut &mut dyn OpcodeHandler) -> Result<(), WasmError>,
    {
        self.handlers.iter_mut().try_for_each(f)
    }

    pub fn decode_function(&mut self) -> Result<(), WasmError> {
        self.notify_handlers(|h| h.on_decode_begin())?;

        // Drive each handler with a fresh stream that can lazily decode as needed
        // Move out the handler references to avoid borrow conflicts while streaming over `self`.
        let mut handlers = core::mem::take(&mut self.handlers);
        self.retain_decoded_ops.set(handlers.len() > 1);
        for handler in handlers.iter_mut() {
            let mut stream = OpStream {
                decoder: self,
                cursor: 0,
            };
            handler.on_stream(&mut stream)?;
        }
        // Put handlers back
        self.handlers = handlers;

        self.notify_handlers(|h| h.on_decode_end())?;
        Ok(())
    }
}

impl<'a, 'b> Decoder<'a, 'b> {
    fn decode_one(&self) -> Result<bool, WasmError> {
        use crate::opcodes::{Opcode::*, OpcodeFB::*, OpcodeFC::*, WasmOpcode::*};

        if self.end_reached.get() {
            return Ok(false);
        }
        let mut payload = self.payload.borrow_mut();
        if payload.is_empty() {
            return Err(WasmError::malformed("Unexpected end of code"));
        }
        let op_offset = payload.position();
        let op: Opcode = payload.read_u8()?.try_into()?;
        match op {
            END => {
                let wasm_op = OP(op);
                let imm = Immediate::None;
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
                if payload.is_empty() {
                    self.end_reached.set(true);
                }
            }
            UNREACHABLE => {
                let wasm_op = OP(op);
                let imm = Immediate::None;
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            BLOCK | LOOP | IF => {
                let byte1 = payload.read_u8()?;
                let block_type = match byte1 {
                    0x40 => BlockType::Empty,

                    // Structured reference types (0x64 = ref ht, 0x63 = ref null ht)
                    0x64 | 0x63 => {
                        payload.rewind(1)?;
                        let value_type = ValueType::parse(&mut payload)?;
                        BlockType::ValueType(value_type)
                    }

                    _ => match ValueType::try_from(byte1) {
                        Ok(value_type) => BlockType::ValueType(value_type),
                        Err(_) => {
                            payload.rewind(1)?;
                            let value = payload.read_leb128_i32()?;
                            if value >= 0 {
                                BlockType::TypeIndex(value as usize)
                            } else {
                                return Err(WasmError::malformed(
                                    "Invalid block type index".into(),
                                ));
                            }
                        }
                    },
                };
                let wasm_op = OP(op);
                let imm = Immediate::Block(block_type.clone());
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            BR | BR_IF => {
                let imm1 = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::LabelIndex(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            BR_TABLE => {
                let imm1 = payload.read_leb128_u32()?;
                let v_idx = (0..imm1)
                    .map(|_| payload.read_leb128_u32())
                    .collect::<Result<collections::Vec<u32>, _>>()?;
                let default_idx = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::BrLabels(v_idx.clone(), default_idx);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            MEMORY_SIZE | MEMORY_GROW => {
                let memidx = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::MemoryIndex(memidx);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            REF_NULL => {
                let heap_type = HeapType::parse(&mut payload)?;
                let ref_type = RefType::new(true, heap_type);
                let value_type = ValueType::Ref(ref_type);
                let wasm_op = OP(op);
                let imm = Immediate::RefType(value_type);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            CALL => {
                let imm1 = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::FunctionIndex(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            LOCAL_GET | LOCAL_SET | LOCAL_TEE => {
                let imm1 = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::LocalIndex(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            GLOBAL_GET | GLOBAL_SET => {
                let imm1 = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::GlobalIndex(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            TABLE_GET | TABLE_SET => {
                let imm1 = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::TableIndex(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            REF_FUNC => {
                let imm1 = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::FunctionIndex(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            REF_EQ | REF_AS_NON_NULL => {
                let wasm_op = OP(op);
                let imm = Immediate::None;
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            BR_ON_NULL | BR_ON_NON_NULL => {
                let label = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::LabelIndex(label);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            I32_CONST => {
                let imm1 = payload.read_leb128_i32()?;
                let wasm_op = OP(op);
                let imm = Immediate::I32(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            I64_CONST => {
                let imm1 = payload.read_leb128_i64()?;
                let wasm_op = OP(op);
                let imm = Immediate::I64(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            F32_CONST => {
                let imm1 = payload.read_f32()?;
                let wasm_op = OP(op);
                let imm = Immediate::F32(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            F64_CONST => {
                let imm1 = payload.read_f64()?;
                let wasm_op = OP(op);
                let imm = Immediate::F64(imm1);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            CALL_INDIRECT => {
                let typeidx = payload.read_leb128_u32()?;
                let tableidx = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::CallIndirectArgs { typeidx, tableidx };
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            CALL_REF => {
                let typeidx = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::TypeIndex(typeidx);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            RETURN_CALL => {
                let func_idx = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::FunctionIndex(func_idx);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            RETURN_CALL_INDIRECT => {
                let typeidx = payload.read_leb128_u32()?;
                let tableidx = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::CallIndirectArgs { typeidx, tableidx };
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            RETURN_CALL_REF => {
                let typeidx = payload.read_leb128_u32()?;
                let wasm_op = OP(op);
                let imm = Immediate::TypeIndex(typeidx);
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            SELECT_T => {
                let len = payload.read_leb128_u32()?;
                let mut vec = collections::Vec::new();
                for _ in 0..len {
                    let valtype: ValueType = payload.read_u8()?.try_into()?;
                    vec.push(valtype);
                }
                let wasm_op = OP(op);
                let imm = Immediate::SelectTypes(vec.clone());
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            I32_LOAD | I32_LOAD8_S | I32_LOAD8_U | I32_LOAD16_S | I32_LOAD16_U | I64_LOAD
            | I64_LOAD8_S | I64_LOAD8_U | I64_LOAD16_S | I64_LOAD16_U | I64_LOAD32_S
            | I64_LOAD32_U | F32_LOAD | F64_LOAD | I32_STORE | I32_STORE8 | I32_STORE16
            | I64_STORE | I64_STORE8 | I64_STORE16 | I64_STORE32 | F32_STORE | F64_STORE => {
                let align_flag = payload.read_leb128_u32()?;
                let (align, memidx) = if align_flag < 64 {
                    (align_flag, 0)
                } else {
                    let memidx = payload.read_leb128_u32()?;
                    (align_flag - 64, memidx)
                };
                if align >= 32 {
                    return Err(WasmError::malformed("malformed memop flags"));
                }
                let offset = payload.read_leb128_u64()?;
                let wasm_op = OP(op);
                let imm = Immediate::MemArg {
                    align,
                    offset,
                    memidx,
                };
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
            PREFIX_FB => {
                let op_ext: OpcodeFB = payload.read_leb128_u32()?.try_into()?;
                match op_ext {
                    STRUCT_NEW | STRUCT_NEW_DEFAULT | ARRAY_NEW | ARRAY_NEW_DEFAULT
                    | ARRAY_FILL => {
                        let typeidx = payload.read_leb128_u32()?;
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::TypeIndex(typeidx);
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    STRUCT_GET | STRUCT_SET | STRUCT_GET_S | STRUCT_GET_U => {
                        let typeidx = payload.read_leb128_u32()?;
                        let fieldidx = payload.read_leb128_u32()?;
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::StructFieldArgs { typeidx, fieldidx };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    ARRAY_GET | ARRAY_SET | ARRAY_GET_S | ARRAY_GET_U => {
                        let typeidx = payload.read_leb128_u32()?;
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::TypeIndex(typeidx);
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    ARRAY_NEW_FIXED => {
                        let typeidx = payload.read_leb128_u32()?;
                        let n = payload.read_leb128_u32()?;
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::ArrayNewFixed { typeidx, n };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    ARRAY_LEN | REF_I31 | I31_GET_S | I31_GET_U | ANY_CONVERT_EXTERN
                    | EXTERN_CONVERT_ANY => {
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::None;
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    ARRAY_COPY => {
                        let type1 = payload.read_leb128_u32()?;
                        let type2 = payload.read_leb128_u32()?;
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::TwoTypeIndices { type1, type2 };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    ARRAY_INIT_DATA | ARRAY_INIT_ELEM | ARRAY_NEW_DATA | ARRAY_NEW_ELEM => {
                        let value1 = payload.read_leb128_u32()?;
                        let value2 = payload.read_leb128_u32()?;
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::TwoU32s { value1, value2 };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    REF_TEST | REF_TEST_NULL | REF_CAST | REF_CAST_NULL => {
                        let heap_type = HeapType::parse(&mut payload)?;
                        let nullable = matches!(op_ext, REF_TEST_NULL | REF_CAST_NULL);
                        let ref_type = RefType::new(nullable, heap_type);
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::RefType(ValueType::Ref(ref_type));
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    BR_ON_CAST | BR_ON_CAST_FAIL => {
                        let flags = payload.read_u8()?;
                        let label_idx = payload.read_leb128_u32()?;
                        if flags & !0b11 != 0 {
                            return Err(WasmError::invalid("invalid br_on_cast flags"));
                        }
                        let from_heap = HeapType::parse(&mut payload)?;
                        let to_heap = HeapType::parse(&mut payload)?;
                        let rt1 = ValueType::Ref(RefType::new((flags & 0b01) != 0, from_heap));
                        let rt2 = ValueType::Ref(RefType::new((flags & 0b10) != 0, to_heap));
                        let wasm_op = FB(op_ext);
                        let imm = Immediate::BrOnCast {
                            flags,
                            label_idx,
                            rt1,
                            rt2,
                        };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                }
            }
            PREFIX_FC => {
                let op_ext: OpcodeFC = payload.read_leb128_u32()?.try_into()?;
                match op_ext {
                    MEMORY_INIT => {
                        let dataidx = payload.read_leb128_u32()?;
                        let memidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::MemoryInitArgs { dataidx, memidx };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    MEMORY_COPY => {
                        let dstidx = payload.read_leb128_u32()?;
                        let srcidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::MemoryCopyArgs { dstidx, srcidx };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    MEMORY_FILL => {
                        let memidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::MemoryIndex(memidx);
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    DATA_DROP => {
                        let dataidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::DataIndex(dataidx);
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    ELEM_DROP => {
                        let elemidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::ElementIndex(elemidx);
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    TABLE_GROW | TABLE_SIZE | TABLE_FILL => {
                        let tableidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::TableIndex(tableidx);
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    TABLE_INIT => {
                        let elemidx = payload.read_leb128_u32()?;
                        let tableidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::TableInitArgs { elemidx, tableidx };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    TABLE_COPY => {
                        let dstidx = payload.read_leb128_u32()?;
                        let srcidx = payload.read_leb128_u32()?;
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::TableCopyArgs { dstidx, srcidx };
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                    I32_TRUNC_SAT_F32_S | I32_TRUNC_SAT_F32_U | I32_TRUNC_SAT_F64_S
                    | I32_TRUNC_SAT_F64_U | I64_TRUNC_SAT_F32_S | I64_TRUNC_SAT_F32_U
                    | I64_TRUNC_SAT_F64_S | I64_TRUNC_SAT_F64_U => {
                        let wasm_op = FC(op_ext);
                        let imm = Immediate::None;
                        let next_off = payload.position();
                        self.decoded_ops.borrow_mut().push(DecodedOp {
                            wasm_op,
                            op_offset,
                            next_op_offset: next_off,
                            imm,
                        });
                    }
                }
            }
            PREFIX_FD => {
                return Err(WasmError::invalid("SIMD opcodes are not yet supported"));
            }
            NOP | ELSE | RETURN | DROP | SELECT | I32_EQZ | I32_EQ | I32_NE | I32_LT_S
            | I32_LT_U | I32_GT_S | I32_GT_U | I32_LE_S | I32_LE_U | I32_GE_S | I32_GE_U
            | I64_EQZ | I64_EQ | I64_NE | I64_LT_S | I64_LT_U | I64_GT_S | I64_GT_U | I64_LE_S
            | I64_LE_U | I64_GE_S | I64_GE_U | F32_EQ | F32_NE | F32_LT | F32_GT | F32_LE
            | F32_GE | F64_EQ | F64_NE | F64_LT | F64_GT | F64_LE | F64_GE | I32_CLZ | I32_CTZ
            | I32_POPCNT | I32_ADD | I32_SUB | I32_MUL | I32_DIV_S | I32_DIV_U | I32_REM_S
            | I32_REM_U | I32_AND | I32_OR | I32_XOR | I32_SHL | I32_SHR_S | I32_SHR_U
            | I32_ROTL | I32_ROTR | I64_CLZ | I64_CTZ | I64_POPCNT | I64_ADD | I64_SUB
            | I64_MUL | I64_DIV_S | I64_DIV_U | I64_REM_S | I64_REM_U | I64_AND | I64_OR
            | I64_XOR | I64_SHL | I64_SHR_S | I64_SHR_U | I64_ROTL | I64_ROTR | F32_ABS
            | F32_NEG | F32_CEIL | F32_FLOOR | F32_TRUNC | F32_NEAREST | F32_SQRT | F32_ADD
            | F32_SUB | F32_MUL | F32_DIV | F32_MIN | F32_MAX | F32_COPYSIGN | F64_ABS
            | F64_NEG | F64_CEIL | F64_FLOOR | F64_TRUNC | F64_NEAREST | F64_SQRT | F64_ADD
            | F64_SUB | F64_MUL | F64_DIV | F64_MIN | F64_MAX | F64_COPYSIGN | I32_WRAP_I64
            | I32_TRUNC_F32_S | I32_TRUNC_F32_U | I32_TRUNC_F64_S | I32_TRUNC_F64_U
            | I64_EXTEND_I32_S | I64_EXTEND_I32_U | I64_TRUNC_F32_S | I64_TRUNC_F32_U
            | I64_TRUNC_F64_S | I64_TRUNC_F64_U | F32_CONVERT_I32_S | F32_CONVERT_I32_U
            | F32_CONVERT_I64_S | F32_CONVERT_I64_U | F32_DEMOTE_F64 | F64_CONVERT_I32_S
            | F64_CONVERT_I32_U | F64_CONVERT_I64_S | F64_CONVERT_I64_U | F64_PROMOTE_F32
            | I32_REINTERPRET_F32 | I64_REINTERPRET_F64 | F32_REINTERPRET_I32
            | F64_REINTERPRET_I64 | I32_EXTEND8_S | I32_EXTEND16_S | I64_EXTEND8_S
            | I64_EXTEND16_S | I64_EXTEND32_S | REF_IS_NULL => {
                let wasm_op = OP(op);
                let imm = Immediate::None;
                let next_off = payload.position();
                self.decoded_ops.borrow_mut().push(DecodedOp {
                    wasm_op,
                    op_offset,
                    next_op_offset: next_off,
                    imm,
                });
            }
        }
        Ok(true)
    }
}

pub struct OpStream<'d, 'a, 'b> {
    decoder: &'d mut Decoder<'a, 'b>,
    cursor: usize,
}

impl<'d, 'a, 'b> OpStream<'d, 'a, 'b> {
    const COMPACT_THRESHOLD: usize = 256;

    fn compact_consumed_prefix(&mut self) {
        if self.decoder.retain_decoded_ops.get() {
            return;
        }
        let base = self.decoder.decoded_base.get();
        let consumed = self.cursor.saturating_sub(base);
        if consumed < Self::COMPACT_THRESHOLD {
            return;
        }
        self.decoder.decoded_ops.borrow_mut().drain(..consumed);
        self.decoder.decoded_base.set(self.cursor);
    }

    fn ensure(&mut self, need: usize) -> Result<(), WasmError> {
        self.compact_consumed_prefix();
        while !self.decoder.end_reached.get()
            && self.decoder.decoded_base.get() + self.decoder.decoded_ops.borrow().len()
                < self.cursor + need
        {
            self.decoder.decode_one()?;
        }
        Ok(())
    }

    /// Consume and return the next op, advancing the cursor.
    pub fn next(&mut self) -> Result<Option<&DecodedOp>, WasmError> {
        self.ensure(1)?;
        let base = self.decoder.decoded_base.get();
        let relative = self.cursor.saturating_sub(base);
        if relative >= self.decoder.decoded_ops.borrow().len() {
            return Ok(None);
        }
        let op = unsafe { &*self.decoder.decoded_ops.borrow().as_ptr().add(relative) };
        self.cursor += 1;
        Ok(Some(op))
    }

    /// Peek at current op without consuming it (equivalent to peek_at(0)).
    pub fn peek(&mut self) -> Result<Option<&DecodedOp>, WasmError> {
        self.peek_at(0)
    }

    /// Peek at op at offset from cursor (0 = current, 1 = next, etc.) without consuming.
    /// Used for lookahead in instruction fusion.
    pub fn peek_at(&mut self, offset: usize) -> Result<Option<&DecodedOp>, WasmError> {
        self.ensure(offset + 1)?;
        let base = self.decoder.decoded_base.get();
        let relative = self.cursor + offset - base;
        if relative >= self.decoder.decoded_ops.borrow().len() {
            return Ok(None);
        }
        let op = unsafe { &*self.decoder.decoded_ops.borrow().as_ptr().add(relative) };
        Ok(Some(op))
    }

    /// Skip n ops (consume without returning). Used after fusion pattern match.
    pub fn skip(&mut self, n: usize) {
        self.cursor += n;
    }
}

pub struct OpcodePrinter {
    indent: usize,
}

impl OpcodePrinter {
    pub fn new() -> Self {
        OpcodePrinter { indent: 1 }
    }
}

impl OpcodeHandler for OpcodePrinter {
    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        use crate::opcodes::Opcode::*;
        while let Some(decoded) = stream.next()? {
            if let opcodes::WasmOpcode::OP(op) = decoded.wasm_op {
                if let END | ELSE = op {
                    if self.indent > 0 {
                        self.indent -= 1
                    };
                }
                if let BLOCK | LOOP | IF | ELSE = op {
                    self.indent += 1;
                }
            }
        }
        Ok(())
    }

    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Decoder, OpStream, OpcodeHandler};
    use crate::{
        error::WasmError,
        opcodes::OPCODE_CONSTANTS::{END, PREFIX_FD},
    };

    struct DrainHandler;

    impl OpcodeHandler for DrainHandler {
        fn on_decode_begin(&mut self) -> Result<(), WasmError> {
            Ok(())
        }

        fn on_stream<'x, 'y, 'z>(
            &mut self,
            stream: &mut OpStream<'x, 'y, 'z>,
        ) -> Result<(), WasmError> {
            while stream.next()?.is_some() {}
            Ok(())
        }

        fn on_decode_end(&mut self) -> Result<(), WasmError> {
            Ok(())
        }
    }

    #[test]
    fn simd_prefix_reports_invalid_instead_of_panicking() {
        let code = [PREFIX_FD, 0x00, END];
        let mut decoder = Decoder::new(&code);
        let mut handler = DrainHandler;
        decoder.add_handler(&mut handler);

        let err = decoder
            .decode_function()
            .expect_err("SIMD opcodes should be rejected cleanly");

        assert_eq!(
            err,
            WasmError::invalid("SIMD opcodes are not yet supported")
        );
    }
}
