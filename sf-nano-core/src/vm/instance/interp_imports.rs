//! Driving the interpreter's host boundary from ordinary `&[Import]`.
//!
//! The interpreter calls imports through one flat dispatcher: raw `u64`
//! operands, linear memory as a slice, the import identified by
//! `(module, name)`. An embedder's imports are declared the other way --
//! typed [`Value`] callbacks bound per name. This module is the adapter,
//! and it lives here rather than in each embedder because otherwise every
//! one of them writes it again (both the CLI and the Pico 2 firmware did).
//!
//! Numeric values and references both have raw slot forms. The live
//! [`InterpInstance`] retags funcrefs at its frame boundaries; this adapter
//! only translates the slot bits to and from typed [`Value`]s. A type with no
//! raw slot form is rejected at bind time, so it remains an instantiation
//! error rather than a surprise mid-run.

use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::entities::FunctionDef;
use crate::module::type_context::{
    check_function_types_equivalent, concrete_type_matches_cross_context,
};
use crate::module::type_defs::FunctionType;
use crate::module::Module;
use crate::value_type::ValueType;
use crate::vm::entities::{Caller, HostCallback};
use crate::vm::interpreter::{InterpInstance, InterpInstanceAccess};
use crate::vm::link::InstanceToken;
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
    impl for<'a> FnMut(&str, &str, &mut Caller<'a>, &[u64], &mut [u64]) -> Result<(), WasmError>
        + 'static,
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
                    "interp: host import signature has no raw slot form",
                ));
            }
        }

        // Linking is checked here, at instantiation, exactly as the JIT
        // does it: a declared import with no provider is unlinkable, and so
        // is one whose provider has the wrong shape. Deferring either to the
        // first call would let a module instantiate that the JIT refuses,
        // which is a difference in more than how code is run.
        let Some(provided) = imports
            .iter()
            .find(|imp| imp.module == *m && imp.name == *n)
        else {
            return Err(WasmError::unlinkable("missing function import"));
        };
        let (provided_type, provided_type_index, type_ctx) = match &provided.value {
            ImportValue::Func(ImportedFunction::Host {
                func_type: provided_type,
                type_index: provided_type_index,
                type_ctx,
                ..
            }) => (
                provided_type.as_ref(),
                *provided_type_index,
                type_ctx.as_ref(),
            ),
            ImportValue::Func(ImportedFunction::Linked {
                func_type,
                type_index,
                type_ctx,
                ..
            }) => (Some(func_type), *type_index, type_ctx.as_ref()),
            _ => return Err(WasmError::unlinkable("incompatible import type")),
        };

        if let Some(provided_type) = provided_type {
            // With a type context, compare through it rather than
            // structurally: two `func` types in the same rec group are
            // structurally identical yet distinct identities, and only the
            // context can tell them apart. This is the same comparison the
            // JIT makes.
            let compatible = match (type_ctx, provided_type_index) {
                // Both halves present: decide by IDENTITY. Two `(func)` types
                // differing only in their position within a rec group are
                // distinct, and nothing structural separates them -- only
                // the index within each context does.
                (Some(ctx), idx) if idx != u32::MAX => {
                    concrete_type_matches_cross_context(ctx, idx, module.types(), func.type_index())
                }
                (Some(ctx), _) => check_function_types_equivalent(provided_type, &func_type, ctx),
                (None, _) => {
                    provided_type.params() == func_type.params()
                        && provided_type.results() == func_type.results()
                }
            };
            if !compatible {
                return Err(WasmError::unlinkable("incompatible import type"));
            }
        }

        let callback = match &provided.value {
            ImportValue::Func(ImportedFunction::Host { callback, .. }) => callback.clone(),
            // The world identity is retained in InterpInstance's function
            // table. A direct call is intercepted there and delegated through
            // FuncRefHost, or rejected with the named no-hook trap.
            ImportValue::Func(ImportedFunction::Linked { .. }) => continue,
            _ => unreachable!("function provider was checked above"),
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
              caller: &mut Caller<'_>,
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

            entry.callback.call(caller, &params, &mut vresults)?;

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
    token: InstanceToken,
    name: &str,
    args: &[Value],
) -> Result<Vec<Value>, WasmError> {
    let mut access = InterpInstanceAccess::checked_out(token);
    let (idx, func_type) = access.with_instance(|inst| {
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
        Ok::<(usize, FunctionType), WasmError>((idx, func_type))
    })??;

    if args.len() != func_type.params().len() {
        return Err(WasmError::invalid("argument arity mismatch"));
    }

    let raw_args = access.with_instance(|inst| {
        let mut raw_args: Vec<u64> = Vec::with_capacity(args.len());
        for (&value_type, &value) in func_type.params().iter().zip(args) {
            let value = inst.localize_value_for_type(value, value_type);
            raw_args.push(value_to_raw(&value)?);
        }
        Ok::<Vec<u64>, WasmError>(raw_args)
    })??;
    let mut raw_results = crate::collections::vec![0u64; func_type.results().len()];

    InterpInstance::invoke_access(&mut access, idx, &raw_args, &mut raw_results)?;

    access.with_instance(|inst| {
        let mut out: Vec<Value> = Vec::with_capacity(raw_results.len());
        for (ty, &raw) in func_type.results().iter().zip(raw_results.iter()) {
            let value = raw_to_value(*ty, raw)?;
            out.push(inst.absolutize_value_for_type(value, *ty));
        }
        Ok(out)
    })?
}

/// Call a resolved export by index, writing results into a caller-owned
/// slice.
pub(super) fn call_by_index(
    token: InstanceToken,
    index: usize,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let mut access = InterpInstanceAccess::checked_out(token);
    let raw_args = access.with_instance(|inst| {
        let mut raw_args: Vec<u64> = Vec::with_capacity(args.len());
        let func_type = inst
            .module()
            .functions()
            .get(index)
            .ok_or(WasmError::invalid("function index out of range"))?
            .func_type();
        for (&value_type, &value) in func_type.params().iter().zip(args) {
            let value = inst.localize_value_for_type(value, value_type);
            raw_args.push(value_to_raw(&value)?);
        }
        Ok::<Vec<u64>, WasmError>(raw_args)
    })??;
    let mut raw_results = crate::collections::vec![0u64; results.len()];

    InterpInstance::invoke_access(&mut access, index, &raw_args, &mut raw_results)?;

    // Read the signature only after the call: holding it across the
    // invocation would need a clone of the type, which on a short function
    // costs more than the call itself.
    access.with_instance(|inst| {
        let func_type = inst
            .module()
            .functions()
            .get(index)
            .ok_or(WasmError::invalid("function index out of range"))?
            .func_type();
        for (i, ty) in func_type.results().iter().enumerate() {
            let value = raw_to_value(*ty, raw_results[i])?;
            results[i] = inst.absolutize_value_for_type(value, *ty);
        }
        Ok(())
    })?
}

/// Call an export by index with typed values.
pub(super) fn invoke_by_index(
    token: InstanceToken,
    idx: usize,
    args: &[Value],
) -> Result<Vec<Value>, WasmError> {
    let mut access = InterpInstanceAccess::checked_out(token);
    let (n_params, n_results) = access
        .with_instance(|inst| inst.func_arity(idx))?
        .ok_or(WasmError::invalid("function index out of range"))?;
    if args.len() != n_params {
        return Err(WasmError::invalid("argument arity mismatch"));
    }
    let mut out = crate::collections::vec![Value::I32(0); n_results];
    let raw_args = access.with_instance(|inst| {
        let mut raw_args: Vec<u64> = Vec::with_capacity(args.len());
        let func_type = inst
            .module()
            .functions()
            .get(idx)
            .ok_or(WasmError::invalid("function index out of range"))?
            .func_type();
        for (&value_type, &value) in func_type.params().iter().zip(args) {
            let value = inst.localize_value_for_type(value, value_type);
            raw_args.push(value_to_raw(&value)?);
        }
        Ok::<Vec<u64>, WasmError>(raw_args)
    })??;
    let mut raw_results = crate::collections::vec![0u64; n_results];
    InterpInstance::invoke_access(&mut access, idx, &raw_args, &mut raw_results)?;
    access.with_instance(|inst| {
        let func_type = inst
            .module()
            .functions()
            .get(idx)
            .ok_or(WasmError::invalid("function index out of range"))?
            .func_type();
        for (i, ty) in func_type.results().iter().enumerate() {
            let value = raw_to_value(*ty, raw_results[i])?;
            out[i] = inst.absolutize_value_for_type(value, *ty);
        }
        Ok::<(), WasmError>(())
    })??;
    Ok(out)
}

/// A global's value by index, typed from the module's declaration.
pub(super) fn global_at(inst: &InterpInstance, idx: usize) -> Result<Option<Value>, WasmError> {
    let Some(raw) = inst.global_at(idx) else {
        return Ok(None);
    };
    let ty = inst
        .module()
        .globals()
        .get(idx)
        .ok_or(WasmError::invalid("global index out of range"))?
        .value_type();
    let value = raw_to_value(ty, raw)?;
    Ok(Some(inst.absolutize_value_for_type(value, ty)))
}

/// Overwrite a global by index, checked against its declared type.
pub(super) fn replace_global_at(
    inst: &mut InterpInstance,
    idx: usize,
    value: Value,
) -> Result<(), WasmError> {
    let raw = value_to_raw(&value)?;
    inst.set_global_at(idx, raw)
}

/// The numeric value types this boundary carries, or `None` for a type
/// that has no raw-word representation here.
#[inline]
/// Whether a host-boundary value type has a raw slot form.
///
/// References do: a slot carries the `RefHandle` verbatim, the same thing the
/// JIT hands across this boundary. Only v128 has no 8-byte form, which is why
/// SIMD is excluded from this engine rather than merely unimplemented.
fn raw_kind(ty: ValueType) -> Option<ValueType> {
    matches!(
        ty,
        ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64 | ValueType::Ref(_)
    )
    .then_some(ty)
}

#[inline]
fn raw_to_value(ty: ValueType, raw: u64) -> Result<Value, WasmError> {
    super::super::interpreter::raw_to_value_for_interp(raw, ty)
}

#[inline]
fn value_to_raw(v: &Value) -> Result<u64, WasmError> {
    super::super::interpreter::value_to_raw_for_interp(v)
}
