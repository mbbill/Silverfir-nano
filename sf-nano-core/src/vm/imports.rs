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
use crate::vm::entities::{Caller, GlobalInst, HostCallback, HostFn, MemInst, TableInst};
use crate::vm::tag::TagIdentity;
use crate::vm::value::{RefValue, Value};

pub struct Import {
    pub module: String,
    pub name: String,
    pub value: ImportValue,
}

#[derive(Clone)]
pub struct ImportedTableState {
    pub table: TableInst,
    pub type_ctx: Option<TypeContext>,
}

#[derive(Clone)]
pub struct ImportedGlobalState {
    pub global: GlobalInst,
    pub type_ctx: Option<TypeContext>,
}

#[derive(Clone)]
pub struct ImportedGlobalValue {
    pub value: Value,
    pub linked_function: Option<(HostFn, FunctionType)>,
}

pub enum ImportedGlobal {
    Value(ImportedGlobalValue),
    State(ImportedGlobalState),
}

pub enum ImportedFunction {
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
pub struct ImportedTagState {
    pub handle: TagIdentity,
    pub func_type: FunctionType,
    /// Source-context type index for the tag's function type, or
    /// `u32::MAX` for host-minted tags that have no wasm type index.
    /// Enables cross-module rec-group identity checks at link time.
    pub type_index: u32,
    pub type_ctx: Option<TypeContext>,
}

pub enum ImportValue {
    Func(ImportedFunction),
    Global(ImportedGlobal, bool),
    Memory(Limits, Option<MemInst>),
    Table(Limits, Option<ImportedTableState>),
    Tag(ImportedTagState),
}

impl Import {
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
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: HostCallback::new(f),
                func_type: None,
                type_index: u32::MAX,
                type_ctx: None,
            }),
        }
    }

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
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: HostCallback::new(f),
                func_type: Some(func_type),
                type_index: u32::MAX,
                type_ctx: None,
            }),
        }
    }

    pub fn func_typed_with_context<F>(
        module: &str,
        name: &str,
        f: F,
        func_type: FunctionType,
        type_ctx: TypeContext,
    ) -> Self
    where
        F: for<'a, 'b, 'c, 'd> Fn(
                &'a mut Caller<'b>,
                &'c [Value],
                &'d mut [Value],
            ) -> Result<(), WasmError>
            + 'static,
    {
        Self::func_typed_with_context_and_index(module, name, f, func_type, u32::MAX, type_ctx)
    }

    /// As above, with the source-context type index, so a cross-module
    /// rec-group identity check has both halves it needs.
    pub fn func_typed_with_context_and_index<F>(
        module: &str,
        name: &str,
        f: F,
        func_type: FunctionType,
        type_index: u32,
        type_ctx: TypeContext,
    ) -> Self
    where
        F: for<'a, 'b, 'c, 'd> Fn(
                &'a mut Caller<'b>,
                &'c [Value],
                &'d mut [Value],
            ) -> Result<(), WasmError>
            + 'static,
    {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: HostCallback::new(f),
                func_type: Some(func_type),
                type_index,
                type_ctx: Some(type_ctx),
            }),
        }
    }

    pub fn linked_func_typed(
        module: &str,
        name: &str,
        handle: RefValue,
        func_type: FunctionType,
    ) -> Self {
        Self::linked_func_typed_with_context_and_index(
            module,
            name,
            handle,
            func_type,
            u32::MAX,
            TypeContext::empty(),
        )
    }

    pub fn linked_func_typed_with_context(
        module: &str,
        name: &str,
        handle: RefValue,
        func_type: FunctionType,
        type_ctx: TypeContext,
    ) -> Self {
        Self::linked_func_typed_with_context_and_index(
            module,
            name,
            handle,
            func_type,
            u32::MAX,
            type_ctx,
        )
    }

    pub fn linked_func_typed_with_context_and_index(
        module: &str,
        name: &str,
        handle: RefValue,
        func_type: FunctionType,
        type_index: u32,
        type_ctx: TypeContext,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Linked {
                handle,
                func_type,
                type_index,
                type_ctx: Some(type_ctx),
            }),
        }
    }

    pub fn global(module: &str, name: &str, value: Value, mutable: bool) -> Self {
        Self::global_with_linked_function(module, name, value, mutable, None)
    }

    pub fn global_with_linked_function(
        module: &str,
        name: &str,
        value: Value,
        mutable: bool,
        linked_function: Option<(HostFn, FunctionType)>,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Global(
                ImportedGlobal::Value(ImportedGlobalValue {
                    value,
                    linked_function,
                }),
                mutable,
            ),
        }
    }

    pub fn global_with_state(module: &str, name: &str, state: ImportedGlobalState) -> Self {
        let mutable = state.global.mutable;
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Global(ImportedGlobal::State(state), mutable),
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

    pub fn memory_with_state(
        module: &str,
        name: &str,
        limits: Limits,
        memory: Option<MemInst>,
    ) -> Self {
        Import {
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

    pub fn table_with_state(
        module: &str,
        name: &str,
        limits: Limits,
        state: Option<ImportedTableState>,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Table(limits, state),
        }
    }

    pub fn tag_typed(module: &str, name: &str, func_type: FunctionType) -> Self {
        Import {
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

    pub fn tag_typed_with_context(
        module: &str,
        name: &str,
        func_type: FunctionType,
        type_ctx: TypeContext,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Tag(ImportedTagState {
                handle: TagIdentity::mint_fresh(),
                func_type,
                type_index: u32::MAX,
                type_ctx: Some(type_ctx),
            }),
        }
    }

    /// Host-allocates-a-fresh-identity channel that also returns the minted
    /// handle, so host code can later recover the identity for
    /// `Err(WasmError::HostThrow { tag, .. })` or cross-import reuse.
    pub fn tag_typed_with_handle(
        module: &str,
        name: &str,
        func_type: FunctionType,
    ) -> (Self, TagIdentity) {
        let handle = TagIdentity::mint_fresh();
        let import = Import {
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

    /// Cross-module tag linking: import the tag using an explicit `TagIdentity`
    /// obtained via `JitInstance::tag_identity(...)` or a previous
    /// `tag_typed_with_handle`/`linked_tag_typed*` call. Preserves tag
    /// identity across module boundaries.
    pub fn linked_tag_typed(
        module: &str,
        name: &str,
        handle: TagIdentity,
        func_type: FunctionType,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Tag(ImportedTagState {
                handle,
                func_type,
                type_index: u32::MAX,
                type_ctx: None,
            }),
        }
    }

    pub fn linked_tag_typed_with_context(
        module: &str,
        name: &str,
        handle: TagIdentity,
        func_type: FunctionType,
        type_ctx: TypeContext,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Tag(ImportedTagState {
                handle,
                func_type,
                type_index: u32::MAX,
                type_ctx: Some(type_ctx),
            }),
        }
    }

    /// Full cross-module tag linking with explicit type_index, for proper
    /// rec-group identity checks. Host callers without a wasm type_index
    /// should use `linked_tag_typed_with_context` (which passes
    /// `u32::MAX`).
    pub fn linked_tag_typed_with_context_and_index(
        module: &str,
        name: &str,
        handle: TagIdentity,
        func_type: FunctionType,
        type_index: u32,
        type_ctx: TypeContext,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Tag(ImportedTagState {
                handle,
                func_type,
                type_index,
                type_ctx: Some(type_ctx),
            }),
        }
    }
}
