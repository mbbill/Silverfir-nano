//! The constant-expression evaluator both engines instantiate through.
//!
//! There is one grammar for Wasm constant expressions — `t.const`,
//! `ref.null`, `ref.func`, `global.get`, the extended-const integer ops,
//! `v128.const`, and the GC constructors — and it lives here. What differs
//! per engine is only *resolution*: how a function index becomes a
//! reference value, where a previously-initialized global is read from,
//! and whether GC objects can be allocated at all. Engines provide that
//! through [`ConstResolver`]; the JIT's implementation allocates in its
//! `Store`, the interpreter's reads its flat state and refuses GC.

use crate::collections;
use crate::error::WasmError;
use crate::module::type_context::TypeContext;
use crate::module::type_defs::CompositeType;
use crate::opcodes::{Opcode, OpcodeFB, OpcodeFD};
use crate::utils::payload::Payload;
use crate::value_type::{AbstractHeapType, HeapType, RefType};
use crate::vm::value::{RefValue, Value};

/// Engine-provided resolution for the constant-expression operators whose
/// meaning depends on runtime state.
pub(crate) trait ConstResolver {
    /// `ref.func` — produce the engine's reference value for a validated
    /// function index, including its concrete type.
    fn func_ref(&mut self, func_idx: u32) -> Result<Value, WasmError>;
    /// `global.get` — read an already-initialized global.
    fn global_get(&mut self, global_idx: u32) -> Result<Value, WasmError>;
    /// `ref.i31` — intern an i31 payload.
    fn ref_i31(&mut self, value: i32) -> Result<Value, WasmError>;
    /// `struct.new` family — allocate a struct with the given fields.
    fn alloc_struct(
        &mut self,
        type_idx: u32,
        fields: collections::Vec<Value>,
    ) -> Result<Value, WasmError>;
    /// `array.new` family — allocate an array with the given elements.
    fn alloc_array(
        &mut self,
        type_idx: u32,
        elements: collections::Vec<Value>,
    ) -> Result<Value, WasmError>;
}

pub(crate) fn eval_const_expr(
    bytes: &[u8],
    types: &TypeContext,
    resolver: &mut impl ConstResolver,
) -> Result<Value, WasmError> {
    let mut code: Payload = bytes.into();
    let mut stack = collections::vec![];

    while !code.is_empty() {
        let op: Opcode = code.read_u8()?.try_into()?;
        match op {
            Opcode::I32_CONST => stack.push(Value::I32(code.read_leb128_i32()?)),
            Opcode::I64_CONST => stack.push(Value::I64(code.read_leb128_i64()?)),
            Opcode::F32_CONST => stack.push(Value::F32(code.read_f32()?)),
            Opcode::F64_CONST => stack.push(Value::F64(code.read_f64()?)),
            Opcode::REF_NULL => {
                let heap_type = HeapType::parse(&mut code)?;
                stack.push(Value::Ref(RefValue::null(), RefType::new(true, heap_type)));
            }
            Opcode::REF_FUNC => {
                let func_idx = code.read_leb128_u32()?;
                stack.push(resolver.func_ref(func_idx)?);
            }
            Opcode::GLOBAL_GET => {
                let global_idx = code.read_leb128_u32()?;
                stack.push(resolver.global_get(global_idx)?);
            }
            Opcode::I32_ADD => eval_i32_binary(&mut stack, i32::wrapping_add)?,
            Opcode::I32_SUB => eval_i32_binary(&mut stack, i32::wrapping_sub)?,
            Opcode::I32_MUL => eval_i32_binary(&mut stack, i32::wrapping_mul)?,
            Opcode::I64_ADD => eval_i64_binary(&mut stack, i64::wrapping_add)?,
            Opcode::I64_SUB => eval_i64_binary(&mut stack, i64::wrapping_sub)?,
            Opcode::I64_MUL => eval_i64_binary(&mut stack, i64::wrapping_mul)?,
            Opcode::PREFIX_FD => {
                let fd_opcode: OpcodeFD = code.read_leb128_u32()?.try_into()?;
                match fd_opcode {
                    OpcodeFD::V128_CONST => {
                        #[cfg(not(sf_has_simd))]
                        {
                            return Err(WasmError::invalid("SIMD is not supported on this CPU"));
                        }
                        #[cfg(sf_has_simd)]
                        {
                            let mut value = [0u8; 16];
                            for byte in &mut value {
                                *byte = code.read_u8()?;
                            }
                            stack.push(Value::V128(value));
                        }
                    }
                    _ => {
                        return Err(WasmError::invalid("unsupported opcode in const expression"));
                    }
                }
            }
            Opcode::PREFIX_FB => {
                let fb_opcode: OpcodeFB = code.read_leb128_u32()?.try_into()?;
                match fb_opcode {
                    OpcodeFB::REF_I31 => {
                        let value = pop_i32(&mut stack, "ref.i31 expects i32 operand")?;
                        stack.push(resolver.ref_i31(value)?);
                    }
                    OpcodeFB::ANY_CONVERT_EXTERN => {
                        let (handle, ref_type) =
                            pop_ref(&mut stack, "any.convert_extern expects externref")?;
                        if !matches!(
                            ref_type.heap_type,
                            HeapType::Abstract(AbstractHeapType::Extern)
                        ) {
                            return Err(WasmError::invalid("any.convert_extern expects externref"));
                        }
                        let handle = handle.to_any().map_err(|_| {
                            WasmError::invalid("unsupported any.convert_extern in const expression")
                        })?;
                        stack.push(Value::Ref(
                            handle,
                            RefType::new(
                                ref_type.nullable,
                                HeapType::Abstract(AbstractHeapType::Any),
                            ),
                        ));
                    }
                    OpcodeFB::EXTERN_CONVERT_ANY => {
                        let (handle, ref_type) =
                            pop_ref(&mut stack, "extern.convert_any expects anyref")?;
                        match ref_type.heap_type {
                            HeapType::Abstract(AbstractHeapType::Any)
                            | HeapType::Abstract(AbstractHeapType::Eq)
                            | HeapType::Abstract(AbstractHeapType::I31)
                            | HeapType::Abstract(AbstractHeapType::Struct)
                            | HeapType::Abstract(AbstractHeapType::Array)
                            | HeapType::Concrete(_) => {}
                            _ => {
                                return Err(WasmError::invalid("extern.convert_any expects anyref"))
                            }
                        }
                        let handle = handle.to_extern().map_err(|_| {
                            WasmError::invalid("unsupported extern.convert_any in const expression")
                        })?;
                        stack.push(Value::Ref(
                            handle,
                            RefType::new(
                                ref_type.nullable,
                                HeapType::Abstract(AbstractHeapType::Extern),
                            ),
                        ));
                    }
                    OpcodeFB::STRUCT_NEW | OpcodeFB::STRUCT_NEW_DEFAULT => {
                        let type_idx = code.read_leb128_u32()?;
                        let fields = {
                            let def_type = types.get(type_idx).ok_or_else(|| {
                                WasmError::invalid("struct type index out of range")
                            })?;
                            let struct_type = match &def_type.composite {
                                CompositeType::Struct(struct_type) => struct_type,
                                _ => return Err(WasmError::invalid("expected struct type")),
                            };
                            if matches!(fb_opcode, OpcodeFB::STRUCT_NEW) {
                                let mut fields =
                                    collections::Vec::with_capacity(struct_type.fields.len());
                                for _ in 0..struct_type.fields.len() {
                                    fields.push(stack.pop().ok_or_else(|| {
                                        WasmError::invalid("not enough operands for struct.new")
                                    })?);
                                }
                                fields.reverse();
                                fields
                            } else {
                                struct_type
                                    .fields
                                    .iter()
                                    .map(|field| {
                                        Value::default_for_type(field.storage.to_valtype())
                                    })
                                    .collect()
                            }
                        };
                        stack.push(resolver.alloc_struct(type_idx, fields)?);
                    }
                    OpcodeFB::ARRAY_NEW
                    | OpcodeFB::ARRAY_NEW_DEFAULT
                    | OpcodeFB::ARRAY_NEW_FIXED => {
                        let type_idx = code.read_leb128_u32()?;
                        let elements = {
                            let def_type = types.get(type_idx).ok_or_else(|| {
                                WasmError::invalid("array type index out of range")
                            })?;
                            let array_type = match &def_type.composite {
                                CompositeType::Array(array_type) => array_type,
                                _ => return Err(WasmError::invalid("expected array type")),
                            };
                            match fb_opcode {
                                OpcodeFB::ARRAY_NEW | OpcodeFB::ARRAY_NEW_DEFAULT => {
                                    let len = pop_i32(&mut stack, "array.new length must be i32")?;
                                    if len < 0 {
                                        return Err(WasmError::invalid(
                                            "array.new length must be non-negative",
                                        ));
                                    }
                                    let init = if matches!(fb_opcode, OpcodeFB::ARRAY_NEW) {
                                        stack.pop().ok_or_else(|| {
                                            WasmError::invalid("not enough operands for array.new")
                                        })?
                                    } else {
                                        Value::default_for_type(
                                            array_type.element.storage.to_valtype(),
                                        )
                                    };
                                    let mut elements =
                                        collections::Vec::with_capacity(len as usize);
                                    elements.resize(len as usize, init);
                                    elements
                                }
                                OpcodeFB::ARRAY_NEW_FIXED => {
                                    let count = code.read_leb128_u32()? as usize;
                                    let mut elements = collections::Vec::with_capacity(count);
                                    for _ in 0..count {
                                        elements.push(stack.pop().ok_or_else(|| {
                                            WasmError::invalid(
                                                "not enough operands for array.new_fixed",
                                            )
                                        })?);
                                    }
                                    elements.reverse();
                                    elements
                                }
                                _ => unreachable!(),
                            }
                        };
                        stack.push(resolver.alloc_array(type_idx, elements)?);
                    }
                    _ => return Err(WasmError::invalid("unsupported opcode in const expression")),
                }
            }
            Opcode::END => {
                return stack
                    .pop()
                    .ok_or_else(|| WasmError::invalid("empty const expr"));
            }
            _ => return Err(WasmError::invalid("unsupported opcode in const expression")),
        }
    }

    Err(WasmError::invalid("unexpected end of const expression"))
}

fn eval_i32_binary(
    stack: &mut collections::Vec<Value>,
    f: impl FnOnce(i32, i32) -> i32,
) -> Result<(), WasmError> {
    let rhs = pop_i32(stack, "type mismatch in i32 binary expression")?;
    let lhs = pop_i32(stack, "type mismatch in i32 binary expression")?;
    stack.push(Value::I32(f(lhs, rhs)));
    Ok(())
}

fn eval_i64_binary(
    stack: &mut collections::Vec<Value>,
    f: impl FnOnce(i64, i64) -> i64,
) -> Result<(), WasmError> {
    let rhs = pop_i64(stack, "type mismatch in i64 binary expression")?;
    let lhs = pop_i64(stack, "type mismatch in i64 binary expression")?;
    stack.push(Value::I64(f(lhs, rhs)));
    Ok(())
}

fn pop_i32(stack: &mut collections::Vec<Value>, message: &'static str) -> Result<i32, WasmError> {
    match stack.pop() {
        Some(Value::I32(value)) => Ok(value),
        _ => Err(WasmError::invalid(message)),
    }
}

fn pop_i64(stack: &mut collections::Vec<Value>, message: &'static str) -> Result<i64, WasmError> {
    match stack.pop() {
        Some(Value::I64(value)) => Ok(value),
        _ => Err(WasmError::invalid(message)),
    }
}

fn pop_ref(
    stack: &mut collections::Vec<Value>,
    message: &'static str,
) -> Result<(RefValue, RefType), WasmError> {
    match stack.pop() {
        Some(Value::Ref(handle, ref_type)) => Ok((handle, ref_type)),
        _ => Err(WasmError::invalid(message)),
    }
}
