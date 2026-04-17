use crate::collections;
use crate::{
    error::WasmError,
    module::{entities::ConstExpr, Module},
    opcodes::{
        Opcode::{self, *},
        OpcodeFB,
    },
    utils::payload::Payload,
    value_type::{AbstractHeapType, HeapType, RefType, ValueType},
};
/// Context for validating constant expressions
#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    is_passive: bool,
    only_imported_globals: bool,
    validating_global_index: Option<usize>,
}

impl ValidationContext {
    pub fn passive() -> Self {
        Self {
            is_passive: true,
            ..Default::default()
        }
    }

    pub fn active() -> Self {
        Self {
            is_passive: false,
            ..Default::default()
        }
    }

    pub fn global(index: usize) -> Self {
        Self {
            is_passive: false,
            only_imported_globals: false,
            validating_global_index: Some(index),
        }
    }

    pub fn table_init() -> Self {
        Self {
            is_passive: false,
            only_imported_globals: true,
            validating_global_index: None,
        }
    }
}

impl ConstExpr {
    /// Validate a constant expression with a specific validation context
    pub fn validate_in_context(
        &self,
        module: &Module,
        ctx: &ValidationContext,
    ) -> Result<ValueType, WasmError> {
        let mut code: Payload = Payload::from(self.as_ref());

        let mut stack = collections::vec![];
        while !code.is_empty() {
            let op: Opcode = code.read_u8()?.try_into()?;
            if ctx.is_passive
                && !matches!(
                    op,
                    REF_NULL
                        | REF_FUNC
                        | PREFIX_FB
                        | END
                        | I32_CONST
                        | I64_CONST
                        | F32_CONST
                        | F64_CONST
                        | GLOBAL_GET
                )
            {
                return Err(WasmError::invalid("Invalid opcode in passive code"));
            }
            match op {
                I32_CONST => {
                    code.read_leb128_i32()?;
                    stack.push(ValueType::I32);
                }
                I64_CONST => {
                    code.read_leb128_i64()?;
                    stack.push(ValueType::I64);
                }
                F32_CONST => {
                    code.read_f32()?;
                    stack.push(ValueType::F32);
                }
                F64_CONST => {
                    code.read_f64()?;
                    stack.push(ValueType::F64);
                }
                REF_NULL => {
                    let heap_type = HeapType::parse(&mut code)?;
                    let reftype = ValueType::Ref(RefType::new(true, heap_type));
                    stack.push(reftype);
                }
                REF_FUNC => {
                    let function_index = code.read_leb128_u32()? as usize;
                    if module.functions().len() <= function_index {
                        return Err(WasmError::invalid(
                            "Invalid function index for ref.func".into(),
                        ));
                    }
                    let type_idx = module.functions()[function_index].type_index();
                    stack.push(ValueType::Ref(RefType::non_nullable_concrete(type_idx)));
                }
                GLOBAL_GET => {
                    let global_index = code.read_leb128_u32()? as usize;
                    if module.globals().len() <= global_index {
                        return Err(WasmError::invalid("unknown global"));
                    }
                    let global = &module.globals()[global_index];
                    if global.mutable() {
                        return Err(WasmError::invalid(
                            "constant expression cannot reference mutable global".into(),
                        ));
                    }
                    if ctx.only_imported_globals && !global.is_import() {
                        return Err(WasmError::invalid("unknown global"));
                    }
                    if let Some(current_global_idx) = ctx.validating_global_index {
                        if global_index >= current_global_idx {
                            return Err(WasmError::invalid("unknown global"));
                        }
                    }
                    stack.push(global.value_type());
                }
                // Extended constant expressions (binary arithmetic)
                I32_ADD | I32_SUB | I32_MUL => {
                    if stack.len() < 2 {
                        return Err(WasmError::invalid(
                            "Not enough operands for binary operation".into(),
                        ));
                    }
                    let right = stack.pop().unwrap();
                    let left = stack.pop().unwrap();
                    if left != ValueType::I32 || right != ValueType::I32 {
                        return Err(WasmError::invalid(
                            "Type mismatch in i32 binary operation".into(),
                        ));
                    }
                    stack.push(ValueType::I32);
                }
                I64_ADD | I64_SUB | I64_MUL => {
                    if stack.len() < 2 {
                        return Err(WasmError::invalid(
                            "Not enough operands for binary operation".into(),
                        ));
                    }
                    let right = stack.pop().unwrap();
                    let left = stack.pop().unwrap();
                    if left != ValueType::I64 || right != ValueType::I64 {
                        return Err(WasmError::invalid(
                            "Type mismatch in i64 binary operation".into(),
                        ));
                    }
                    stack.push(ValueType::I64);
                }
                PREFIX_FB => {
                    let fb_opcode_value = code.read_leb128_u32()?;
                    let fb_opcode: OpcodeFB = fb_opcode_value.try_into()?;
                    use OpcodeFB::*;
                    match fb_opcode {
                        REF_I31 => {
                            if stack.is_empty() {
                                return Err(WasmError::invalid("Not enough operands for ref.i31"));
                            }
                            let operand = stack.pop().unwrap();
                            if operand != ValueType::I32 {
                                return Err(WasmError::invalid("ref.i31 expects i32 operand"));
                            }
                            let i31_ref = ValueType::Ref(RefType::new(
                                false,
                                HeapType::Abstract(AbstractHeapType::I31),
                            ));
                            stack.push(i31_ref);
                        }
                        ANY_CONVERT_EXTERN => {
                            if stack.is_empty() {
                                return Err(WasmError::invalid(
                                    "Not enough operands for any.convert_extern",
                                ));
                            }
                            let operand = stack.pop().unwrap();
                            match operand {
                                ValueType::Ref(ref_type)
                                    if matches!(
                                        ref_type.heap_type,
                                        HeapType::Abstract(AbstractHeapType::Extern)
                                    ) =>
                                {
                                    let any_ref = ValueType::Ref(RefType::new(
                                        ref_type.nullable,
                                        HeapType::Abstract(AbstractHeapType::Any),
                                    ));
                                    stack.push(any_ref);
                                }
                                _ => {
                                    return Err(WasmError::invalid(
                                        "any.convert_extern expects externref operand",
                                    ));
                                }
                            }
                        }
                        EXTERN_CONVERT_ANY => {
                            if stack.is_empty() {
                                return Err(WasmError::invalid(
                                    "Not enough operands for extern.convert_any",
                                ));
                            }
                            let operand = stack.pop().unwrap();
                            match operand {
                                ValueType::Ref(ref_type)
                                    if matches!(
                                        ref_type.heap_type,
                                        HeapType::Abstract(AbstractHeapType::Any)
                                    ) =>
                                {
                                    let extern_ref = ValueType::Ref(RefType::new(
                                        ref_type.nullable,
                                        HeapType::Abstract(AbstractHeapType::Extern),
                                    ));
                                    stack.push(extern_ref);
                                }
                                _ => {
                                    return Err(WasmError::invalid(
                                        "extern.convert_any expects anyref operand",
                                    ));
                                }
                            }
                        }
                        STRUCT_NEW => {
                            let typeidx = code.read_leb128_u32()?;
                            let def_type = module
                                .types()
                                .get(typeidx)
                                .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                            let struct_type = match &def_type.composite {
                                crate::module::type_defs::CompositeType::Struct(s) => s,
                                _ => return Err(WasmError::invalid("Expected struct type")),
                            };
                            for field in struct_type.fields.iter().rev() {
                                if stack.is_empty() {
                                    return Err(WasmError::invalid(
                                        "Not enough operands for struct.new",
                                    ));
                                }
                                let value = stack.pop().unwrap();
                                let expected_type = field.storage.to_valtype();
                                if value != expected_type {
                                    return Err(WasmError::invalid(
                                        "struct.new field type mismatch",
                                    ));
                                }
                            }
                            let struct_ref =
                                ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx)));
                            stack.push(struct_ref);
                        }
                        STRUCT_NEW_DEFAULT => {
                            let typeidx = code.read_leb128_u32()?;
                            let def_type = module
                                .types()
                                .get(typeidx)
                                .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                            if !matches!(
                                &def_type.composite,
                                crate::module::type_defs::CompositeType::Struct(_)
                            ) {
                                return Err(WasmError::invalid("Expected struct type"));
                            }
                            let struct_ref =
                                ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx)));
                            stack.push(struct_ref);
                        }
                        ARRAY_NEW | ARRAY_NEW_DEFAULT | ARRAY_NEW_FIXED => {
                            let typeidx = code.read_leb128_u32()?;
                            let def_type = module
                                .types()
                                .get(typeidx)
                                .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                            let array_type = match &def_type.composite {
                                crate::module::type_defs::CompositeType::Array(a) => a,
                                _ => return Err(WasmError::invalid("Expected array type")),
                            };
                            match fb_opcode {
                                ARRAY_NEW | ARRAY_NEW_DEFAULT => {
                                    if stack.is_empty() {
                                        return Err(WasmError::invalid(
                                            "Not enough operands for array.new",
                                        ));
                                    }
                                    let len = stack.pop().unwrap();
                                    if len != ValueType::I32 {
                                        return Err(WasmError::invalid(
                                            "array.new length expects i32",
                                        ));
                                    }
                                    if matches!(fb_opcode, ARRAY_NEW) {
                                        if stack.is_empty() {
                                            return Err(WasmError::invalid(
                                                "Not enough operands for array.new",
                                            ));
                                        }
                                        let init = stack.pop().unwrap();
                                        let expected_type = array_type.element.storage.to_valtype();
                                        if init != expected_type {
                                            return Err(WasmError::invalid(
                                                "array.new element type mismatch",
                                            ));
                                        }
                                    }
                                }
                                ARRAY_NEW_FIXED => {
                                    let count = code.read_leb128_u32()? as usize;
                                    let expected_type = array_type.element.storage.to_valtype();
                                    for _ in 0..count {
                                        if stack.is_empty() {
                                            return Err(WasmError::invalid(
                                                "Not enough operands for array.new_fixed",
                                            ));
                                        }
                                        let value = stack.pop().unwrap();
                                        if value != expected_type {
                                            return Err(WasmError::invalid(
                                                "array.new_fixed element type mismatch",
                                            ));
                                        }
                                    }
                                }
                                _ => unreachable!(),
                            }
                            let array_ref =
                                ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx)));
                            stack.push(array_ref);
                        }
                        _ => {
                            return Err(WasmError::invalid(
                                "Unsupported GC opcode in constant expression",
                            ));
                        }
                    }
                }
                END => {
                    if stack.len() != 1 {
                        return Err(WasmError::invalid(
                            "Invalid stack length at the end of the code".into(),
                        ));
                    }
                    return Ok(stack.pop().unwrap());
                }
                _ => {
                    return Err(WasmError::invalid("Invalid opcode"));
                }
            }
        }
        Err(WasmError::invalid("Unexpected end of input"))
    }
}
