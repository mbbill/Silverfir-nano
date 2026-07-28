//! What the emitted code runs inside.
//!
//! Executable memory, the native context and frame layout, traps, and the
//! runtime-call boundary back into Rust. The whole subtree is reached only
//! through the JIT, so the per-module `sf_jit` attributes this file used to
//! carry are now stated once by its position under [`super`].

use crate::error::WasmError;
use crate::value_type::ValueType;
use crate::vm::entities::FunctionInst;
use crate::vm::jit::result_buffer::ResultBuffer;
use crate::vm::jit::store::Store;
use crate::vm::jit::value_encoding::normalize_machine_raw_in_store;
use crate::vm::value::Value;

pub(crate) mod code;
pub(crate) mod code_buf;
pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod dispatch_view;
#[cfg(sf_has_guard_pages)]
pub(crate) mod guard_pages;
pub(crate) mod layout;
pub(crate) mod os;
pub(crate) mod preserved;
pub(crate) mod runtime_call;
pub(crate) mod trap;
#[cfg(sf_has_guard_pages)]
pub(crate) mod trap_signal;

mod native_eval;

/// Run one function through the native engine.
#[inline]
pub(crate) fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<ResultBuffer, WasmError> {
    native_eval::eval(func_inst, store, args)
}

#[inline]
pub(crate) unsafe fn collect_native_results_from_stack(
    stack_base: *const u64,
    result_types: &[ValueType],
    gp_unit_bytes: u8,
    store: &mut Store,
) -> Result<ResultBuffer, WasmError> {
    let mut out = ResultBuffer::with_exact_capacity(result_types.len());
    for (index, ty) in result_types.iter().enumerate() {
        let raw = normalize_machine_raw_in_store(
            unsafe { *stack_base.add(index) },
            *ty,
            gp_unit_bytes,
            store,
        )?;
        out.push(raw);
    }
    Ok(out)
}
