//! What the emitted code runs inside.
//!
//! Executable memory, the native context and frame layout, traps, and the
//! runtime-call boundary back into Rust. The whole subtree is reached only
//! through the JIT, so the per-module `sf_jit` attributes this file used to
//! carry are now stated once by its position under [`super`].

use crate::error::WasmError;
use crate::value_type::ValueType;
use crate::vm::jit::instance::JitInstance;
use crate::vm::jit::value_encoding::try_machine_raw_to_value_in_store;
use crate::vm::link::{InstanceId, InstanceToken};
use crate::vm::value::Value;
use core::marker::PhantomData;
use core::ptr::NonNull;

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

/// Scoped access to the store whose function is currently being evaluated.
///
/// An occupied instance carries its checkout token across calls. During
/// instantiation the slot is deliberately vacant, so the unpublished boxed
/// store is represented by a stable pointer tied to the caller's exclusive
/// borrow instead. Neither variant exposes a store reference beyond a
/// `with_store` closure.
pub(crate) enum StoreAccess<'a> {
    CheckedOut(InstanceToken),
    Initializing {
        pointer: NonNull<JitInstance>,
        id: InstanceId,
        lifetime: PhantomData<&'a mut JitInstance>,
    },
}

impl StoreAccess<'static> {
    #[inline]
    pub(crate) fn checked_out(token: InstanceToken) -> Self {
        Self::CheckedOut(token)
    }
}

impl<'a> StoreAccess<'a> {
    /// Recreate the initializing capability inside a native runtime call.
    ///
    /// # Safety
    ///
    /// `pointer` must identify a live, stable store for the lifetime of the
    /// returned capability, and no store materialization may be live when
    /// this capability is constructed. Production callers use this for the
    /// unpublished initializing store; standalone runtime tests provide the
    /// same guarantees through their owning box.
    #[inline]
    pub(crate) unsafe fn initializing_raw(pointer: *mut JitInstance, id: InstanceId) -> Self {
        Self::Initializing {
            pointer: NonNull::new(pointer).expect("initializing store pointer must be non-null"),
            id,
            lifetime: PhantomData,
        }
    }

    #[inline]
    pub(crate) const fn id(&self) -> InstanceId {
        match self {
            Self::CheckedOut(token) => token.id(),
            Self::Initializing { id, .. } => *id,
        }
    }

    /// Return the capability's stable pointer without materializing a store.
    #[inline]
    pub(crate) fn store_pointer(&self) -> Result<*mut JitInstance, WasmError> {
        match self {
            Self::CheckedOut(token) => token
                .jit_pointer()
                .ok_or_else(|| WasmError::internal("instance is not a JIT store")),
            Self::Initializing { pointer, .. } => Ok(pointer.as_ptr()),
        }
    }

    /// Materialize for one non-reentrant read scope.
    ///
    /// The closure must not invoke guest code, call a host callback, or
    /// return a pointer derived from the store for later dereference.
    #[inline]
    pub(crate) fn with_store<R>(
        &self,
        use_store: impl for<'store> FnOnce(&'store JitInstance) -> R,
    ) -> Result<R, WasmError> {
        let store = match self {
            Self::CheckedOut(token) => token
                .jit()
                .ok_or_else(|| WasmError::internal("instance is not a JIT store"))?,
            Self::Initializing { pointer, .. } => unsafe { pointer.as_ref() },
        };
        Ok(use_store(store))
    }

    /// Materialize for one non-reentrant mutation scope.
    ///
    /// The closure must not invoke guest code, call a host callback, or
    /// return a pointer derived from the store for later dereference.
    #[inline]
    pub(crate) fn with_store_mut<R>(
        &mut self,
        use_store: impl for<'store> FnOnce(&'store mut JitInstance) -> R,
    ) -> Result<R, WasmError> {
        let store = match self {
            Self::CheckedOut(token) => token
                .jit_mut()
                .ok_or_else(|| WasmError::internal("instance is not a JIT store"))?,
            Self::Initializing { pointer, .. } => unsafe { pointer.as_mut() },
        };
        Ok(use_store(store))
    }
}

/// Recover the capability for the store named by a live native context.
///
/// An occupied instance can be checked out again while its outer evaluation
/// token remains live. A start function instead names the unpublished store
/// whose reserved slot is still vacant, so that case retains the initializing
/// raw capability rather than requiring checkout to succeed.
pub(crate) fn current_store_access(
    ctx: &context::NativeContext,
) -> Result<StoreAccess<'static>, WasmError> {
    let store_ptr = ctx.store;
    let (handle, id) = {
        let store = ctx
            .store()
            .ok_or_else(|| WasmError::internal("native context is missing its store"))?;
        (
            store.instance_backref().clone(),
            store.instance_backref().self_id(),
        )
    };
    if let Some(token) = handle.checkout(id) {
        return Ok(StoreAccess::checked_out(token));
    }

    if let Some(table) = handle.table() {
        if table.in_use(id) != Some(0) {
            return Err(WasmError::internal(
                "native context current instance checkout failed",
            ));
        }
    }

    // A live native context keeps the current store alive. An occupied store
    // has an outer checkout for the active evaluation, so a failed self
    // checkout with the reserved generation still unused means this is the
    // unpublished, vacant-slot initialization path. Tests may use a standalone
    // store whose weak table handle has already expired; their owning Box
    // provides the same liveness guarantee for this call.
    Ok(unsafe { StoreAccess::<'static>::initializing_raw(store_ptr, id) })
}

/// Run one function in an occupied instance through the native engine.
#[inline]
pub(crate) fn eval(
    token: InstanceToken,
    local_index: u32,
    args: &[Value],
) -> Result<crate::collections::Vec<Value>, WasmError> {
    let mut access = StoreAccess::checked_out(token);
    eval_access(&mut access, local_index, args)
}

/// Run a start function while its store remains unpublished in a vacant slot.
///
/// # Safety
///
/// `store` must point to the live boxed store reserved as `id`, and no store
/// materialization may be live when this function is called. The box must not
/// move or be dropped until evaluation returns.
#[inline]
pub(crate) unsafe fn eval_initializing(
    store: *mut JitInstance,
    id: InstanceId,
    local_index: u32,
    args: &[Value],
) -> Result<crate::collections::Vec<Value>, WasmError> {
    let mut access = unsafe { StoreAccess::initializing_raw(store, id) };
    eval_access(&mut access, local_index, args)
}

#[inline]
pub(crate) fn eval_access(
    access: &mut StoreAccess<'_>,
    local_index: u32,
    args: &[Value],
) -> Result<crate::collections::Vec<Value>, WasmError> {
    native_eval::eval(access, local_index, args)
}

#[inline]
pub(crate) unsafe fn collect_native_results_from_stack(
    stack_base: *const u64,
    result_types: &[ValueType],
    gp_unit_bytes: u8,
    store: &mut JitInstance,
) -> Result<crate::collections::Vec<Value>, WasmError> {
    let mut out = crate::collections::Vec::with_capacity(result_types.len());
    for (index, ty) in result_types.iter().enumerate() {
        let value = try_machine_raw_to_value_in_store(
            unsafe { *stack_base.add(index) },
            *ty,
            gp_unit_bytes,
            store,
        )?;
        out.push(value);
    }
    Ok(out)
}
