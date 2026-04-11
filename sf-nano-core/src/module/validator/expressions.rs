use crate::collections;
use crate::{
    error::WasmError,
    module::{entities::ConstExpr, Module},
    opcodes::Opcode::{self, *},
    utils::payload::Payload,
    value_type::{HeapType, RefType, ValueType},
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
            only_imported_globals: true,
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
                        | END
                        | I32_CONST
                        | I64_CONST
                        | F32_CONST
                        | F64_CONST
                        | GLOBAL_GET
                )
            {
                return Err(WasmError::invalid("Invalid opcode in passive code".into()));
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
                        return Err(WasmError::invalid("unknown global".into()));
                    }
                    let global = &module.globals()[global_index];
                    if global.mutable() {
                        return Err(WasmError::invalid(
                            "constant expression cannot reference mutable global".into(),
                        ));
                    }
                    if ctx.only_imported_globals && !global.is_import() {
                        return Err(WasmError::invalid("unknown global".into()));
                    }
                    if let Some(current_global_idx) = ctx.validating_global_index {
                        if global_index >= current_global_idx {
                            return Err(WasmError::invalid("unknown global".into()));
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
                END => {
                    if stack.len() != 1 {
                        return Err(WasmError::invalid(
                            "Invalid stack length at the end of the code".into(),
                        ));
                    }
                    return Ok(stack.pop().unwrap());
                }
                _ => {
                    return Err(WasmError::invalid("Invalid opcode".into()));
                }
            }
        }
        Err(WasmError::invalid("Unexpected end of input".into()))
    }
}
