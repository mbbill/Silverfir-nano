//! Constant expression evaluator for WebAssembly 2.0 module instantiation.

use crate::collections;

use crate::error::WasmError;
use crate::module::entities::ConstExpr;
use crate::opcodes::Opcode;
use crate::utils::payload::Payload;
use crate::value_type::{HeapType, RefType};
use crate::vm::entities::{FunctionInst, ModuleInst};
use crate::vm::value::{RefHandle, Value};

pub(crate) fn eval_const_expr(expr: &ConstExpr, module: &ModuleInst) -> Result<Value, WasmError> {
    let bytes: &[u8] = expr;
    let mut code: Payload = bytes.into();

    let mut stack = collections::vec![];
    while !code.is_empty() {
        let op: Opcode = code.read_u8()?.try_into()?;
        match op {
            Opcode::I32_CONST => {
                let value = code.read_leb128_i32()?;
                stack.push(Value::I32(value));
            }
            Opcode::I64_CONST => {
                let value = code.read_leb128_i64()?;
                stack.push(Value::I64(value));
            }
            Opcode::F32_CONST => {
                let value = code.read_f32()?;
                stack.push(Value::F32(value));
            }
            Opcode::F64_CONST => {
                let value = code.read_f64()?;
                stack.push(Value::F64(value));
            }
            Opcode::REF_NULL => {
                let heap_type = HeapType::parse(&mut code)?;
                let reftype = RefType::new(true, heap_type);
                stack.push(Value::Ref(RefHandle::null(), reftype));
            }
            Opcode::REF_FUNC => {
                let func_idx = code.read_leb128_u32()? as usize;
                if func_idx >= module.functions.len() {
                    return Err(WasmError::invalid(
                        "ref.func: function index out of range".into(),
                    ));
                }
                let ref_handle = RefHandle::new(func_idx);
                let type_idx = match &module.functions[func_idx] {
                    FunctionInst::Local { type_index, .. } => *type_index,
                    FunctionInst::External { .. } => 0,
                };
                let heap_type = HeapType::Concrete(type_idx);
                let ref_type = RefType::new(false, heap_type);
                stack.push(Value::Ref(ref_handle, ref_type));
            }
            Opcode::GLOBAL_GET => {
                let global_idx = code.read_leb128_u32()? as usize;
                if global_idx >= module.globals.len() {
                    return Err(WasmError::invalid("global.get: index out of range"));
                }
                stack.push(module.globals[global_idx].value());
            }
            Opcode::I32_ADD => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                match (l, r) {
                    (Value::I32(a), Value::I32(b)) => stack.push(Value::I32(a.wrapping_add(b))),
                    _ => return Err(WasmError::invalid("type mismatch in i32.add")),
                }
            }
            Opcode::I32_SUB => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                match (l, r) {
                    (Value::I32(a), Value::I32(b)) => stack.push(Value::I32(a.wrapping_sub(b))),
                    _ => return Err(WasmError::invalid("type mismatch in i32.sub")),
                }
            }
            Opcode::I32_MUL => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                match (l, r) {
                    (Value::I32(a), Value::I32(b)) => stack.push(Value::I32(a.wrapping_mul(b))),
                    _ => return Err(WasmError::invalid("type mismatch in i32.mul")),
                }
            }
            Opcode::I64_ADD => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                match (l, r) {
                    (Value::I64(a), Value::I64(b)) => stack.push(Value::I64(a.wrapping_add(b))),
                    _ => return Err(WasmError::invalid("type mismatch in i64.add")),
                }
            }
            Opcode::I64_SUB => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                match (l, r) {
                    (Value::I64(a), Value::I64(b)) => stack.push(Value::I64(a.wrapping_sub(b))),
                    _ => return Err(WasmError::invalid("type mismatch in i64.sub")),
                }
            }
            Opcode::I64_MUL => {
                let r = stack.pop().unwrap();
                let l = stack.pop().unwrap();
                match (l, r) {
                    (Value::I64(a), Value::I64(b)) => stack.push(Value::I64(a.wrapping_mul(b))),
                    _ => return Err(WasmError::invalid("type mismatch in i64.mul")),
                }
            }
            Opcode::END => {
                return stack
                    .pop()
                    .ok_or_else(|| WasmError::invalid("empty const expr"));
            }
            _ => {
                return Err(WasmError::invalid(
                    "unsupported opcode in const expression".into(),
                ));
            }
        }
    }
    Err(WasmError::invalid(
        "unexpected end of const expression".into(),
    ))
}
