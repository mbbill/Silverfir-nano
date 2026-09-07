//! Import declarations: what an embedder hands to an instance.
//!
//! These describe what a module's imports are bound to, not how any engine
//! calls them. Both engines are driven from the same list -- the JIT links
//! it into its entity model, the interpreter adapts it in
//! `instance::interp_imports` -- so it belongs to neither of them.

use tracked_alloc::string::{String, ToString};

use crate::error::WasmError;
use crate::module::type_context::TypeContext;
use crate::module::type_defs::FunctionType;
use crate::utils::limits::Limits;
use crate::vm::entities::{Caller, GlobalInst, HostCallback, MemInst, TableInst};
use crate::vm::link::InstanceId;
use crate::vm::tag::TagIdentity;
use crate::vm::value::RefValue;
use crate::Value;

/// A named host or linked-instance import. Construct it with the associated
/// functions; runtime state and type-context representation are private.
#[derive(Clone)]
pub struct Import {
    pub(crate) module: String,
    pub(crate) name: String,
    pub(crate) value: ImportValue,
    pub(crate) source: Option<InstanceId>,
}

#[derive(Clone)]
/// Shared table state obtained from an instance for linking another instance.
/// Cloning preserves the identity of the underlying table.
pub(crate) struct ImportedTableState {
    pub(crate) table: TableInst,
    pub(crate) type_ctx: Option<TypeContext>,
}

#[derive(Clone)]
/// Shared global state obtained from an instance for linking another instance.
/// Cloning preserves the identity of the underlying global.
pub(crate) struct ImportedGlobalState {
    pub(crate) global: GlobalInst,
    pub(crate) type_ctx: Option<TypeContext>,
}

#[derive(Clone)]
pub(crate) enum ImportedGlobal {
    Value(Value),
    State(ImportedGlobalState),
}

#[derive(Clone)]
pub(crate) enum ImportedFunction {
    Host {
        callback: HostCallback,
        func_type: Option<FunctionType>,
        /// Source-context type index for `func_type`, or `u32::MAX` for a
        /// host function with no wasm type index. Paired with `type_ctx` it
        /// makes cross-module rec-group identity decidable -- two `(func)`
        /// types differing only in their position within a rec group are
        /// distinct identities, and nothing structural can separate them.
        type_index: u32,
        type_ctx: Option<TypeContext>,
    },
    Linked {
        handle: RefValue,
        func_type: FunctionType,
        type_index: u32,
        type_ctx: Option<TypeContext>,
    },
}

#[derive(Clone)]
pub(crate) struct ImportedTagState {
    pub(crate) handle: TagIdentity,
    pub(crate) func_type: FunctionType,
    /// Source-context type index for the tag's function type, or
    /// `u32::MAX` for host-minted tags that have no wasm type index.
    /// Enables cross-module rec-group identity checks at link time.
    pub(crate) type_index: u32,
    pub(crate) type_ctx: Option<TypeContext>,
}

#[derive(Clone)]
pub(crate) enum ImportValue {
    Func(ImportedFunction),
    Global(ImportedGlobal, bool),
    Memory(Limits, Option<MemInst>),
    Table(Limits, Option<ImportedTableState>),
    Tag(ImportedTagState),
}

/// An exported function, memory, table, global or exception tag.
///
/// Obtained from Instance::get_export or Instance::exports. Cloning preserves
/// identity and type information. Bind it with Import::new within the same
/// RuntimeWorld; the source instance's private type context cannot be replaced.
#[derive(Clone)]
pub struct Extern {
    pub(crate) value: ImportValue,
    pub(crate) source: InstanceId,
}

impl Import {
    /// Bind an exported object under the module/name requested by an importer.
    /// Both instances must belong to the same RuntimeWorld.
    pub fn new(module: &str, name: &str, value: Extern) -> Self {
        Self {
            module: module.to_string(),
            name: name.to_string(),
            value: value.value,
            source: Some(value.source),
        }
    }

    /// Module name requested by the Wasm import declaration.
    pub fn module(&self) -> &str {
        &self.module
    }

    /// Name requested within the imported module.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Bind the same object under another module/name, preserving its type,
    /// identity, captured host state and originating runtime world.
    pub fn alias(&self, module: &str, name: &str) -> Self {
        Self {
            module: module.to_string(),
            name: name.to_string(),
            value: self.value.clone(),
            source: self.source,
        }
    }

    /// Bind a host callback using the Wasm import's declared signature.
    /// Every result slot must be assigned a value of the declared type before
    /// returning `Ok(())`. Missing or mistyped results trap before Wasm resumes.
    pub fn func<F>(module: &str, name: &str, f: F) -> Self
    where
        F: for<'a, 'b, 'c, 'd> Fn(
                &'a mut Caller<'b>,
                &'c [Value],
                &'d mut [Value],
            ) -> Result<(), WasmError>
            + 'static,
    {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: HostCallback::from_host(f),
                func_type: None,
                type_index: u32::MAX,
                type_ctx: None,
            }),
        }
    }

    /// Bind a host callback with a signature checked during linking.
    /// Result initialization and validation follow [`Self::func`].
    pub fn func_typed<F>(module: &str, name: &str, f: F, func_type: FunctionType) -> Self
    where
        F: for<'a, 'b, 'c, 'd> Fn(
                &'a mut Caller<'b>,
                &'c [Value],
                &'d mut [Value],
            ) -> Result<(), WasmError>
            + 'static,
    {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: HostCallback::from_host(f),
                func_type: Some(func_type),
                type_index: u32::MAX,
                type_ctx: None,
            }),
        }
    }

    pub fn global(module: &str, name: &str, value: Value, mutable: bool) -> Self {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Global(ImportedGlobal::Value(value), mutable),
        }
    }

    pub fn memory(
        module: &str,
        name: &str,
        initial_pages: usize,
        max_pages: Option<usize>,
    ) -> Self {
        Self::memory_with_limits(
            module,
            name,
            Limits::new(initial_pages, max_pages).expect("invalid imported memory limits"),
        )
    }

    pub fn memory64(
        module: &str,
        name: &str,
        initial_pages: usize,
        max_pages: Option<usize>,
    ) -> Self {
        Self::memory_with_limits(
            module,
            name,
            Limits::new_64(initial_pages, max_pages).expect("invalid imported memory limits"),
        )
    }

    pub fn memory_with_limits(module: &str, name: &str, limits: Limits) -> Self {
        Self::memory_with_state(module, name, limits, None)
    }

    pub(crate) fn memory_with_state(
        module: &str,
        name: &str,
        limits: Limits,
        memory: Option<MemInst>,
    ) -> Self {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Memory(limits, memory),
        }
    }

    pub fn table(module: &str, name: &str, initial_size: usize, max_size: Option<usize>) -> Self {
        Self::table_with_limits(
            module,
            name,
            Limits::new(initial_size, max_size).expect("invalid imported table limits"),
        )
    }

    pub fn table64(module: &str, name: &str, initial_size: usize, max_size: Option<usize>) -> Self {
        Self::table_with_limits(
            module,
            name,
            Limits::new_64(initial_size, max_size).expect("invalid imported table limits"),
        )
    }

    pub fn table_with_limits(module: &str, name: &str, limits: Limits) -> Self {
        Self::table_with_state(module, name, limits, None)
    }

    pub(crate) fn table_with_state(
        module: &str,
        name: &str,
        limits: Limits,
        state: Option<ImportedTableState>,
    ) -> Self {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Table(limits, state),
        }
    }

    /// Create a fresh host tag with numeric or abstract-reference parameters.
    /// Concrete type indices require a typed module export: a host tag has no
    /// module type context, so using those indices fails at instantiation.
    pub fn tag_typed(module: &str, name: &str, func_type: FunctionType) -> Self {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Tag(ImportedTagState {
                handle: TagIdentity::mint_fresh(),
                func_type,
                type_index: u32::MAX,
                type_ctx: None,
            }),
        }
    }

    /// Host-allocates-a-fresh-identity channel that also returns the minted
    /// handle, so host code can later recover the identity for
    /// `Caller::throw(tag, args)` or cross-import reuse.
    /// Parameter types follow the same restrictions as [`Self::tag_typed`].
    /// Use [`Self::alias`] to reuse the tag under another import name.
    pub fn tag_typed_with_handle(
        module: &str,
        name: &str,
        func_type: FunctionType,
    ) -> (Self, TagIdentity) {
        let handle = TagIdentity::mint_fresh();
        let import = Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Tag(ImportedTagState {
                handle,
                func_type,
                type_index: u32::MAX,
                type_ctx: None,
            }),
        };
        (import, handle)
    }
}

pub(crate) fn table_export(state: ImportedTableState) -> Result<ImportValue, WasmError> {
    let limits = state.table.current_limits()?;
    Ok(ImportValue::Table(limits, Some(state)))
}

pub(crate) fn memory_export(memory: MemInst) -> Result<ImportValue, WasmError> {
    let limits = memory.current_limits()?;
    Ok(ImportValue::Memory(limits, Some(memory)))
}

#[cfg(test)]
mod test_support;
