//! Driving the interpreter's host boundary from ordinary `&[Import]`.
//!
//! The interpreter calls imports through one flat dispatcher: raw `u64`
//! operands, linear memory as a slice, the import identified by
//! `(module, name)`. An embedder's imports are declared the other way --
//! typed [`Value`] callbacks bound per name. This module is the adapter,
//! and it lives here rather than in each embedder because otherwise every
//! one of them writes it again (both the CLI and the Pico 2 firmware did).
//!
//! Conversion is numeric only. A host import taking or returning a
//! reference is rejected at bind time rather than at the call, so an
//! unsupported signature is an instantiation error and not a surprise
//! mid-run.

use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::entities::FunctionDef;
use crate::module::type_defs::FunctionType;
use crate::module::Module;
use crate::value_type::ValueType;
use crate::vm::entities::{Caller, HostCallback};
use crate::vm::interpreter::InterpInstance;
use crate::vm::value::Value;
use tracked_alloc::string::String;

use super::{Import, ImportValue, ImportedFunction};

/// One resolved host import: the module's declared signature plus the
/// callback the embedder bound to that name.
struct Bound {
    module: String,
    name: String,
    func_type: FunctionType,
    callback: HostCallback,
}

/// Build the interpreter's dispatcher from the module's declared function
/// imports and the embedder's import list.
pub(super) fn bind(
    module: &Module,
    imports: &[Import],
) -> Result<
    impl FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'static,
    WasmError,
> {
    let mut bound: Vec<Bound> = Vec::new();

    for func in module.functions() {
        let FunctionDef::Import {
            module: m, name: n, ..
        } = func.def()
        else {
            continue;
        };
        let func_type = func.func_type().clone();
        for ty in func_type.params().iter().chain(func_type.results().iter()) {
            if raw_kind(*ty).is_none() {
                return Err(WasmError::invalid(
                    "interp: host import signature is not numeric",
                ));
            }
        }

        let Some(callback) = imports.iter().find_map(|imp| {
            if imp.module != *m || imp.name != *n {
                return None;
            }
            match &imp.value {
                ImportValue::Func(ImportedFunction::Host { callback, .. }) => {
                    Some(callback.clone())
                }
                _ => None,
            }
        }) else {
            // Left unbound on purpose: a module may declare an import it
            // never calls, and rejecting that here would refuse modules
            // the JIT path accepts.
            continue;
        };

        bound.push(Bound {
            module: m.clone(),
            name: n.clone(),
            func_type,
            callback,
        });
    }

    // A linear scan, not a map: import lists are short, the keys are
    // already interned by the parser, and a hash map would pull an
    // allocation-heavy structure into a no_std engine for no measurable
    // gain at this call frequency.
    Ok(
        move |m: &str,
              n: &str,
              mem: &mut [u8],
              args: &[u64],
              results: &mut [u64]|
              -> Result<(), WasmError> {
            let entry = bound
                .iter()
                .find(|b| b.module == m && b.name == n)
                .ok_or(WasmError::invalid("interp: unlinked host import"))?;

            let mut params: Vec<Value> = Vec::with_capacity(args.len());
            for (ty, &raw) in entry.func_type.params().iter().zip(args.iter()) {
                params.push(raw_to_value(*ty, raw)?);
            }
            let mut vresults: Vec<Value> = Vec::with_capacity(results.len());
            for ty in entry.func_type.results().iter() {
                vresults.push(raw_to_value(*ty, 0)?);
            }

            let mut caller = Caller::new(Some(mem));
            entry.callback.call(&mut caller, &params, &mut vresults)?;

            for (dst, v) in results.iter_mut().zip(vresults.iter()) {
                *dst = value_to_raw(v)?;
            }
            Ok(())
        },
    )
}

/// Call an export by name with typed values, the way the JIT's instance
/// does, on top of the interpreter's index-and-raw-word call path.
pub(super) fn invoke_by_name(
    inst: &mut InterpInstance,
    name: &str,
    args: &[Value],
) -> Result<Vec<Value>, WasmError> {
    let idx = inst
        .find_export(name)
        .ok_or(WasmError::invalid("exported function not found"))?;
    let func_type = inst
        .module()
        .functions()
        .get(idx)
        .ok_or(WasmError::invalid("exported function not found"))?
        .func_type()
        .clone();

    if args.len() != func_type.params().len() {
        return Err(WasmError::invalid("argument arity mismatch"));
    }

    let mut raw_args: Vec<u64> = Vec::with_capacity(args.len());
    for v in args {
        raw_args.push(value_to_raw(v)?);
    }
    let mut raw_results = crate::collections::vec![0u64; func_type.results().len()];

    inst.invoke(idx, &raw_args, &mut raw_results)?;

    let mut out: Vec<Value> = Vec::with_capacity(raw_results.len());
    for (ty, &raw) in func_type.results().iter().zip(raw_results.iter()) {
        out.push(raw_to_value(*ty, raw)?);
    }
    Ok(out)
}

/// Call a resolved export by index, writing results into a caller-owned
/// slice.
pub(super) fn call_by_index(
    inst: &mut InterpInstance,
    index: usize,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let mut raw_args: Vec<u64> = Vec::with_capacity(args.len());
    for v in args {
        raw_args.push(value_to_raw(v)?);
    }
    let mut raw_results = crate::collections::vec![0u64; results.len()];

    inst.invoke(index, &raw_args, &mut raw_results)?;

    // Read the signature only after the call: holding it across the
    // invocation would need a clone of the type, which on a short function
    // costs more than the call itself.
    let func_type = inst
        .module()
        .functions()
        .get(index)
        .ok_or(WasmError::invalid("function index out of range"))?
        .func_type();
    for (i, ty) in func_type.results().iter().enumerate() {
        results[i] = raw_to_value(*ty, raw_results[i])?;
    }
    Ok(())
}

/// The numeric value types this boundary carries, or `None` for a type
/// that has no raw-word representation here.
#[inline]
fn raw_kind(ty: ValueType) -> Option<ValueType> {
    matches!(
        ty,
        ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64
    )
    .then_some(ty)
}

#[inline]
fn raw_to_value(ty: ValueType, raw: u64) -> Result<Value, WasmError> {
    Ok(match ty {
        ValueType::I32 => Value::I32(raw as u32 as i32),
        ValueType::I64 => Value::I64(raw as i64),
        ValueType::F32 => Value::F32(f32::from_bits(raw as u32)),
        ValueType::F64 => Value::F64(f64::from_bits(raw)),
        _ => {
            return Err(WasmError::invalid(
                "interp: non-numeric value at the host boundary",
            ))
        }
    })
}

#[inline]
fn value_to_raw(v: &Value) -> Result<u64, WasmError> {
    Ok(match v {
        Value::I32(x) => *x as u32 as u64,
        Value::I64(x) => *x as u64,
        Value::F32(x) => x.to_bits() as u64,
        Value::F64(x) => x.to_bits(),
        _ => {
            return Err(WasmError::invalid(
                "interp: non-numeric value at the host boundary",
            ))
        }
    })
}
