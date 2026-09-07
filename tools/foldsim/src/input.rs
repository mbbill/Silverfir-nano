//! Input for the standalone simulator. wasmparser owns binary decoding and
//! validation; the byte classifications below describe only the cost model.

use wasmparser::{
    BinaryReader, BlockType, CompositeInnerType, Encoding, FuncType, FunctionBody, Operator,
    OperatorsReader, Parser, Payload, TypeRef, Validator, WasmFeatures,
};

pub(super) type Error = Box<dyn std::error::Error>;
pub(super) type Opcode = u8;

pub(super) struct ModuleInfo<'a> {
    pub types: Vec<Option<FuncType>>,
    pub functions: Vec<u32>,
    pub bodies: Vec<FunctionBody<'a>>,
    pub imported_functions: usize,
}

impl<'a> ModuleInfo<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        Validator::new_with_features(WasmFeatures::all()).validate_all(bytes)?;
        let mut module = Self {
            types: Vec::new(),
            functions: Vec::new(),
            bodies: Vec::new(),
            imported_functions: 0,
        };
        for payload in Parser::new(0).parse_all(bytes) {
            match payload? {
                Payload::Version { encoding, .. } if encoding != Encoding::Module => {
                    return Err("foldsim expects a core Wasm module".into());
                }
                Payload::TypeSection(reader) => {
                    for group in reader {
                        for ty in group?.into_types() {
                            module.types.push(match ty.composite_type.inner {
                                CompositeInnerType::Func(ty) => Some(ty),
                                _ => None,
                            });
                        }
                    }
                }
                Payload::ImportSection(reader) => {
                    for import in reader.into_imports() {
                        if let TypeRef::Func(ty) | TypeRef::FuncExact(ty) = import?.ty {
                            module.functions.push(ty);
                            module.imported_functions += 1;
                        }
                    }
                }
                Payload::FunctionSection(reader) => {
                    for ty in reader {
                        module.functions.push(ty?);
                    }
                }
                Payload::CodeSectionEntry(body) => module.bodies.push(body),
                _ => {}
            }
        }
        Ok(module)
    }

    pub fn function_type(&self, index: usize) -> Option<&FuncType> {
        self.types
            .get(*self.functions.get(index)? as usize)?
            .as_ref()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum WasmOpcode {
    Plain(Opcode),
    Bulk(u32),
    Other,
}

#[derive(Clone)]
pub(super) enum Immediate {
    None,
    Block(BlockType),
    LocalIndex(u32),
    LabelIndex(u32),
    BrLabels(Vec<u32>, u32),
    FunctionIndex(u32),
    CallIndirectArgs { typeidx: u32 },
}

pub(super) struct Decoded<'a> {
    pub wasm_op: WasmOpcode,
    pub imm: Immediate,
    pub op_offset: usize,
    pub operator: Operator<'a>,
}

pub(super) struct Stream<'a> {
    reader: OperatorsReader<'a>,
    bytes: &'a [u8],
    code_start: usize,
}

impl<'a> Stream<'a> {
    pub fn new(body: &FunctionBody<'a>, bytes: &'a [u8]) -> Result<Self, Error> {
        let reader = body.get_operators_reader()?;
        let code_start = reader.original_position();
        Ok(Self {
            reader,
            bytes,
            code_start,
        })
    }

    pub fn next(&mut self) -> Result<Option<Decoded<'a>>, Error> {
        if self.reader.eof() {
            return Ok(None);
        }
        let offset = self.reader.original_position();
        let operator = self.reader.read()?;
        let wasm_op = match self.bytes[offset] {
            0xfc => {
                let subopcode =
                    BinaryReader::new(&self.bytes[offset + 1..], offset + 1).read_var_u32()?;
                WasmOpcode::Bulk(subopcode)
            }
            0xfb | 0xfd | 0xfe => WasmOpcode::Other,
            opcode => WasmOpcode::Plain(opcode),
        };
        let imm = match &operator {
            Operator::Block { blockty } | Operator::Loop { blockty } | Operator::If { blockty } => {
                Immediate::Block(*blockty)
            }
            Operator::LocalGet { local_index }
            | Operator::LocalSet { local_index }
            | Operator::LocalTee { local_index } => Immediate::LocalIndex(*local_index),
            Operator::Br { relative_depth } | Operator::BrIf { relative_depth } => {
                Immediate::LabelIndex(*relative_depth)
            }
            Operator::BrTable { targets } => Immediate::BrLabels(
                targets.targets().collect::<Result<_, _>>()?,
                targets.default(),
            ),
            Operator::Call { function_index } => Immediate::FunctionIndex(*function_index),
            Operator::CallIndirect { type_index, .. } => Immediate::CallIndirectArgs {
                typeidx: *type_index,
            },
            _ => Immediate::None,
        };
        Ok(Some(Decoded {
            wasm_op,
            imm,
            op_offset: offset - self.code_start,
            operator,
        }))
    }
}

// Standard binary opcode values used by this tool's statistical model.
pub(super) mod op {
    pub(crate) const BLOCK: u8 = 0x02;
    pub(crate) const BR: u8 = 0x0c;
    pub(crate) const BR_IF: u8 = 0x0d;
    pub(crate) const BR_TABLE: u8 = 0x0e;
    pub(crate) const CALL: u8 = 0x10;
    pub(crate) const CALL_INDIRECT: u8 = 0x11;
    pub(crate) const CALL_REF: u8 = 0x14;
    pub(crate) const DROP: u8 = 0x1a;
    pub(crate) const ELSE: u8 = 0x05;
    pub(crate) const END: u8 = 0x0b;
    pub(crate) const GLOBAL_GET: u8 = 0x23;
    pub(crate) const GLOBAL_SET: u8 = 0x24;
    pub(crate) const I32_ADD: u8 = 0x6a;
    pub(crate) const I32_EQZ: u8 = 0x45;
    pub(crate) const I32_SUB: u8 = 0x6b;
    pub(crate) const I64_ADD: u8 = 0x7c;
    pub(crate) const I64_EQZ: u8 = 0x50;
    pub(crate) const I64_SUB: u8 = 0x7d;
    pub(crate) const IF: u8 = 0x04;
    pub(crate) const LOCAL_GET: u8 = 0x20;
    pub(crate) const LOCAL_SET: u8 = 0x21;
    pub(crate) const LOCAL_TEE: u8 = 0x22;
    pub(crate) const LOOP: u8 = 0x03;
    pub(crate) const MEMORY_GROW: u8 = 0x40;
    pub(crate) const MEMORY_SIZE: u8 = 0x3f;
    pub(crate) const NOP: u8 = 0x01;
    pub(crate) const REF_FUNC: u8 = 0xd2;
    pub(crate) const REF_IS_NULL: u8 = 0xd1;
    pub(crate) const REF_NULL: u8 = 0xd0;
    pub(crate) const RETURN: u8 = 0x0f;
    pub(crate) const RETURN_CALL: u8 = 0x12;
    pub(crate) const RETURN_CALL_INDIRECT: u8 = 0x13;
    pub(crate) const RETURN_CALL_REF: u8 = 0x15;
    pub(crate) const SELECT: u8 = 0x1b;
    pub(crate) const SELECT_T: u8 = 0x1c;
    pub(crate) const TABLE_GET: u8 = 0x25;
    pub(crate) const TABLE_SET: u8 = 0x26;
    pub(crate) const UNREACHABLE: u8 = 0x00;
}
pub(super) mod bulk {
    pub(crate) const DATA_DROP: u32 = 0x09;
    pub(crate) const ELEM_DROP: u32 = 0x0d;
    pub(crate) const I32_TRUNC_SAT_F32_S: u32 = 0x00;
    pub(crate) const I32_TRUNC_SAT_F32_U: u32 = 0x01;
    pub(crate) const I32_TRUNC_SAT_F64_S: u32 = 0x02;
    pub(crate) const I32_TRUNC_SAT_F64_U: u32 = 0x03;
    pub(crate) const I64_TRUNC_SAT_F32_S: u32 = 0x04;
    pub(crate) const I64_TRUNC_SAT_F32_U: u32 = 0x05;
    pub(crate) const I64_TRUNC_SAT_F64_S: u32 = 0x06;
    pub(crate) const I64_TRUNC_SAT_F64_U: u32 = 0x07;
    pub(crate) const MEMORY_COPY: u32 = 0x0a;
    pub(crate) const MEMORY_FILL: u32 = 0x0b;
    pub(crate) const MEMORY_INIT: u32 = 0x08;
    pub(crate) const TABLE_COPY: u32 = 0x0e;
    pub(crate) const TABLE_FILL: u32 = 0x11;
    pub(crate) const TABLE_GROW: u32 = 0x0f;
    pub(crate) const TABLE_INIT: u32 = 0x0c;
    pub(crate) const TABLE_SIZE: u32 = 0x10;
}

#[cfg(test)]
mod tests {
    use super::ModuleInfo;

    #[test]
    fn signatures_keep_gc_slots_recursion_groups_and_import_indices() {
        let bytes = wat::parse_str(
            r#"(module
                (rec (type (struct)) (type $f (func (param i32) (result i64))))
                (import "host" "f" (func $f (type $f)))
                (func (export "call") (param i32) (result i64)
                    local.get 0 call $f))"#,
        )
        .unwrap();
        let module = ModuleInfo::parse(&bytes).unwrap();
        assert!(module.types[0].is_none());
        assert_eq!(module.imported_functions, 1);
        assert_eq!(module.bodies.len(), 1);
        for index in [0, 1] {
            let ty = module.function_type(index).unwrap();
            assert_eq!(ty.params(), &[wasmparser::ValType::I32]);
            assert_eq!(ty.results(), &[wasmparser::ValType::I64]);
        }
    }

    #[test]
    fn invalid_code_does_not_produce_an_analysis_model() {
        let bytes = wat::parse_str("(module (func (result i32) f64.const 0))").unwrap();
        assert!(ModuleInfo::parse(&bytes).is_err());
    }
}
