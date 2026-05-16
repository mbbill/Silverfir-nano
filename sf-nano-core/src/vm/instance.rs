//! Public API for sf-nano: parse, instantiate, and invoke WebAssembly modules.

use crate::collections;

use tracked_alloc::{
    boxed::Box,
    string::{String, ToString},
};

use crate::error::WasmError;
use crate::module::entities::{
    ConstExpr, Data, Element, ElementInit, FunctionDef, FunctionSpec, GlobalDef, MemoryDef,
    TableDef, TagDef,
};
use crate::module::type_context::{
    check_function_types_equivalent, concrete_type_matches_cross_context,
    value_types_equivalent_cross_module, TypeContext,
};
use crate::module::type_defs::FunctionType;
use crate::module::Module;
#[cfg(sf_jit)]
use crate::op_decoder::{Decoder, OpStream, OpcodeHandler};
#[cfg(sf_jit)]
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};
use crate::utils::limits::{Limitable, Limits};
use crate::value_type::{HeapType, ValueType};
#[cfg(sf_jit)]
use crate::vm::entities::TableDispatchMode;
use crate::vm::entities::{
    DataInst, ElementInst, FunctionInst, GlobalInst, HostFn, MemInst, ModuleInst, TableInst,
    TagInst,
};
use crate::vm::expr_eval::eval_const_expr;
use crate::vm::runtime;
use crate::vm::store::{LinkRegistry, Store};
use crate::vm::tag::TagHandle;
use crate::vm::value::{RefHandle, Value};
use crate::vm::value_encoding::{try_raw_to_value_in_store, value_to_raw_in_store};

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
        callback: HostFn,
        func_type: Option<FunctionType>,
        type_ctx: Option<TypeContext>,
    },
    Linked {
        handle: RefHandle,
        func_type: FunctionType,
        type_index: u32,
        type_ctx: Option<TypeContext>,
    },
}

#[derive(Clone)]
pub struct ImportedTagState {
    pub handle: TagHandle,
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
    pub fn func(module: &str, name: &str, f: HostFn) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: f,
                func_type: None,
                type_ctx: None,
            }),
        }
    }

    pub fn func_typed(module: &str, name: &str, f: HostFn, func_type: FunctionType) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: f,
                func_type: Some(func_type),
                type_ctx: None,
            }),
        }
    }

    pub fn func_typed_with_context(
        module: &str,
        name: &str,
        f: HostFn,
        func_type: FunctionType,
        type_ctx: TypeContext,
    ) -> Self {
        Import {
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: f,
                func_type: Some(func_type),
                type_ctx: Some(type_ctx),
            }),
        }
    }

    pub fn linked_func_typed(
        module: &str,
        name: &str,
        handle: RefHandle,
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
        handle: RefHandle,
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
        handle: RefHandle,
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
                handle: TagHandle::mint_fresh(),
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
                handle: TagHandle::mint_fresh(),
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
    ) -> (Self, TagHandle) {
        let handle = TagHandle::mint_fresh();
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

    /// Cross-module tag linking: import the tag using an explicit `TagHandle`
    /// obtained via `Instance::tag_handle(...)` or a previous
    /// `tag_typed_with_handle`/`linked_tag_typed*` call. Preserves tag
    /// identity across module boundaries.
    pub fn linked_tag_typed(
        module: &str,
        name: &str,
        handle: TagHandle,
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
        handle: TagHandle,
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
        handle: TagHandle,
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

pub struct Instance {
    store: Box<Store>,
    exports: collections::Vec<(String, ExportKind, usize)>,
}

pub enum InstanceInstantiationError {
    Complete(WasmError),
    Partial {
        instance: Instance,
        error: WasmError,
    },
}

impl From<WasmError> for InstanceInstantiationError {
    fn from(error: WasmError) -> Self {
        InstanceInstantiationError::Complete(error)
    }
}

impl InstanceInstantiationError {
    pub fn error(&self) -> &WasmError {
        match self {
            InstanceInstantiationError::Complete(error) => error,
            InstanceInstantiationError::Partial { error, .. } => error,
        }
    }

    pub fn into_parts(self) -> (Option<Instance>, WasmError) {
        match self {
            InstanceInstantiationError::Complete(error) => (None, error),
            InstanceInstantiationError::Partial { instance, error } => (Some(instance), error),
        }
    }
}

#[derive(Clone, Copy)]
enum ExportKind {
    Func,
    Table,
    Memory,
    Global,
    Tag,
}

fn global_value_types_equivalent_cross_context(
    actual_type: &ValueType,
    expected_type: &ValueType,
    actual_ctx: Option<&TypeContext>,
    expected_ctx: &TypeContext,
) -> bool {
    match (actual_type, expected_type) {
        (ValueType::I32, ValueType::I32)
        | (ValueType::I64, ValueType::I64)
        | (ValueType::F32, ValueType::F32)
        | (ValueType::F64, ValueType::F64)
        | (ValueType::V128, ValueType::V128)
        | (ValueType::Unknown, ValueType::Unknown) => true,
        (ValueType::Ref(actual_ref), ValueType::Ref(expected_ref))
            if actual_ref.nullable == expected_ref.nullable =>
        {
            match (&actual_ref.heap_type, &expected_ref.heap_type) {
                (HeapType::Abstract(actual), HeapType::Abstract(expected)) => actual == expected,
                (HeapType::Concrete(actual_idx), HeapType::Concrete(expected_idx)) => actual_ctx
                    .is_some_and(|actual_ctx| {
                        concrete_type_matches_cross_context(
                            actual_ctx,
                            *actual_idx,
                            expected_ctx,
                            *expected_idx,
                        ) && concrete_type_matches_cross_context(
                            expected_ctx,
                            *expected_idx,
                            actual_ctx,
                            *actual_idx,
                        )
                    }),
                _ => false,
            }
        }
        _ => false,
    }
}

fn global_value_type_can_initialize_cross_context(
    actual_type: &ValueType,
    expected_type: &ValueType,
    actual_ctx: Option<&TypeContext>,
    expected_ctx: &TypeContext,
) -> bool {
    match (actual_type, expected_type) {
        (ValueType::I32, ValueType::I32)
        | (ValueType::I64, ValueType::I64)
        | (ValueType::F32, ValueType::F32)
        | (ValueType::F64, ValueType::F64)
        | (ValueType::V128, ValueType::V128)
        | (ValueType::Unknown, ValueType::Unknown) => true,
        (ValueType::Ref(actual_ref), ValueType::Ref(expected_ref))
            if !actual_ref.nullable || expected_ref.nullable =>
        {
            match (&actual_ref.heap_type, &expected_ref.heap_type) {
                (HeapType::Abstract(actual), HeapType::Abstract(expected)) => {
                    actual.is_subtype_of(expected)
                }
                (HeapType::Concrete(actual_idx), HeapType::Concrete(expected_idx)) => actual_ctx
                    .is_some_and(|actual_ctx| {
                        concrete_type_matches_cross_context(
                            actual_ctx,
                            *actual_idx,
                            expected_ctx,
                            *expected_idx,
                        )
                    }),
                (HeapType::Concrete(actual_idx), HeapType::Abstract(expected)) => actual_ctx
                    .is_some_and(|actual_ctx| {
                        HeapType::Concrete(*actual_idx)
                            .is_subtype_of(&HeapType::Abstract(*expected), actual_ctx)
                    }),
                (HeapType::Abstract(actual), HeapType::Concrete(expected_idx)) => {
                    HeapType::Abstract(*actual)
                        .is_subtype_of(&HeapType::Concrete(*expected_idx), expected_ctx)
                }
            }
        }
        _ => false,
    }
}

#[cfg(sf_jit)]
struct TableMutationScan {
    has_mutation: bool,
}

#[cfg(sf_jit)]
impl OpcodeHandler for TableMutationScan {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        self.has_mutation = false;
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(op) = stream.next()? {
            match op.wasm_op {
                WasmOpcode::OP(Opcode::TABLE_SET)
                | WasmOpcode::FC(OpcodeFC::TABLE_INIT)
                | WasmOpcode::FC(OpcodeFC::TABLE_COPY)
                | WasmOpcode::FC(OpcodeFC::TABLE_GROW)
                | WasmOpcode::FC(OpcodeFC::TABLE_FILL) => {
                    self.has_mutation = true;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        Ok(())
    }
}

#[cfg(sf_jit)]
fn function_has_table_mutation(spec: &FunctionSpec) -> Result<bool, WasmError> {
    let mut scan = TableMutationScan {
        has_mutation: false,
    };
    {
        let mut decoder = Decoder::new(spec.code());
        decoder.add_handler(&mut scan);
        decoder.decode_function()?;
    }
    Ok(scan.has_mutation)
}

#[cfg(sf_jit)]
fn element_init_is_local_only(module: &Module, init: &ElementInit) -> bool {
    match init {
        ElementInit::FunctionIndexes(indices) => indices.iter().all(|idx| {
            module
                .functions()
                .get(*idx)
                .map(|func| !func.is_import())
                .unwrap_or(false)
        }),
        ElementInit::InitExprs { .. } => false,
    }
}

#[cfg(sf_jit)]
fn table_active_elements_are_local_only(module: &Module, table_idx: usize) -> bool {
    module.elements().iter().all(|element| match element {
        Element::Active {
            table_index, init, ..
        } if *table_index == table_idx => element_init_is_local_only(module, init),
        _ => true,
    })
}

#[cfg(sf_jit)]
fn module_declares_type_subtyping(module: &Module) -> bool {
    module
        .types()
        .as_slice()
        .iter()
        .any(|ty| !ty.supertypes.is_empty())
}

#[cfg(sf_jit)]
fn compute_static_table_dispatch_modes(
    module: &Module,
) -> Result<collections::Vec<TableDispatchMode>, WasmError> {
    let has_table_mutation = module
        .functions()
        .iter()
        .filter_map(|func| func.spec())
        .try_fold(false, |seen, spec| {
            function_has_table_mutation(spec).map(|has_mutation| seen || has_mutation)
        })?;
    let mut modes = collections::Vec::with_capacity(module.tables().len());
    for (table_idx, table) in module.tables().iter().enumerate() {
        let fixed_size = table.limits().max() == Some(table.limits().min());
        let private_local = !table.is_import() && table.export_names().is_empty();
        let default_null = table.spec().init_expr().is_none();
        let local_only = table_active_elements_are_local_only(module, table_idx);
        let exact_type_checks = !module_declares_type_subtyping(module);
        let fixed_len = u32::try_from(table.limits().min()).ok();
        let mode = if !has_table_mutation
            && fixed_size
            && private_local
            && default_null
            && local_only
            && exact_type_checks
        {
            fixed_len
                .map(|len| TableDispatchMode::FixedLocalOnly { len })
                .unwrap_or(TableDispatchMode::Generic)
        } else {
            TableDispatchMode::Generic
        };
        modes.push(mode);
    }
    Ok(modes)
}

impl Instance {
    pub fn new(wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        let module = Module::new("main", wasm_bytes)?;
        Self::from_module(module, imports)
    }

    pub fn from_module(module: Module, imports: &[Import]) -> Result<Self, WasmError> {
        let registry = LinkRegistry::new();
        Self::from_module_with_registry(module, imports, &registry)
            .map_err(|err| err.into_parts().1)
    }

    pub fn from_module_with_registry(
        module: Module,
        imports: &[Import],
        registry: &LinkRegistry,
    ) -> Result<Self, InstanceInstantiationError> {
        module
            .ensure_simd_supported()
            .map_err(InstanceInstantiationError::Complete)?;

        #[cfg(sf_module_validator)]
        {
            use crate::module::validator::Validator;
            let mut validator = Validator::new(&module);
            validator
                .validate()
                .map_err(InstanceInstantiationError::Complete)?;
        }

        let mut exports = collections::Vec::new();
        for (i, f) in module.functions().iter().enumerate() {
            for name in f.export_names() {
                exports.push((name.clone(), ExportKind::Func, i));
            }
        }
        for (i, t) in module.tables().iter().enumerate() {
            for name in t.export_names() {
                exports.push((name.clone(), ExportKind::Table, i));
            }
        }
        for (i, m) in module.memories().iter().enumerate() {
            for name in m.export_names() {
                exports.push((name.clone(), ExportKind::Memory, i));
            }
        }
        for (i, g) in module.globals().iter().enumerate() {
            for name in g.export_names() {
                exports.push((name.clone(), ExportKind::Global, i));
            }
        }
        for (i, t) in module.tags().iter().enumerate() {
            for name in t.export_names() {
                exports.push((name.clone(), ExportKind::Tag, i));
            }
        }

        let start_func_index = module.start_function_index();
        #[cfg(sf_jit)]
        let table_dispatch_modes = compute_static_table_dispatch_modes(&module)
            .map_err(InstanceInstantiationError::Complete)?;
        let (
            types,
            mod_functions,
            mod_tables,
            mod_memories,
            mod_globals,
            mod_tags,
            mod_elements,
            mod_data,
            _start,
        ) = module.into_parts();

        let mut functions: collections::Vec<FunctionInst> =
            collections::Vec::with_capacity(mod_functions.len());
        for func in mod_functions {
            let type_index = func.type_index();
            let (_export_names, def) = func.into_parts();
            match def {
                FunctionDef::Local(spec) => {
                    functions.push(FunctionInst::Local { spec, type_index });
                }
                FunctionDef::Import {
                    module: mod_name,
                    name,
                    func_type,
                    ..
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == mod_name && i.name == name);
                    match import {
                        Some(Import {
                            value: ImportValue::Func(imported_func),
                            ..
                        }) => {
                            let (actual_type, actual_type_idx, import_type_ctx) =
                                match imported_func {
                                    ImportedFunction::Host {
                                        func_type,
                                        type_ctx,
                                        ..
                                    } => (func_type.as_ref(), u32::MAX, type_ctx.as_ref()),
                                    ImportedFunction::Linked {
                                        func_type,
                                        type_index,
                                        type_ctx,
                                        ..
                                    } => (Some(func_type), *type_index, type_ctx.as_ref()),
                                };
                            if let Some(actual_type) = actual_type {
                                let compatible = if actual_type_idx != u32::MAX {
                                    import_type_ctx.is_some_and(|type_ctx| {
                                        concrete_type_matches_cross_context(
                                            type_ctx,
                                            actual_type_idx,
                                            &types,
                                            type_index,
                                        )
                                    })
                                } else if actual_type == &*func_type {
                                    true
                                } else if let Some(type_ctx) = import_type_ctx {
                                    check_function_types_equivalent(
                                        actual_type,
                                        &func_type,
                                        type_ctx,
                                    )
                                } else {
                                    actual_type.params() == func_type.params()
                                        && actual_type.results() == func_type.results()
                                };
                                if !compatible {
                                    return Err(WasmError::unlinkable(
                                        "incompatible import type: .",
                                    )
                                    .into());
                                }
                            }
                            match imported_func {
                                ImportedFunction::Host { callback, .. } => {
                                    functions.push(FunctionInst::Host {
                                        func_type,
                                        callback: *callback,
                                    });
                                }
                                ImportedFunction::Linked { handle, .. } => {
                                    functions.push(FunctionInst::Linked {
                                        func_type,
                                        handle: *handle,
                                    });
                                }
                            }
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing function import: .").into());
                        }
                    }
                }
            }
        }

        let mut tables: collections::Vec<TableInst> =
            collections::Vec::with_capacity(mod_tables.len());
        for table in &mod_tables {
            match table.def() {
                TableDef::Local(_spec) => {
                    tables.push(TableInst::new(table.limits().clone(), table.value_type()));
                }
                TableDef::Import {
                    module: mod_name,
                    name,
                    ..
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == *mod_name && i.name == *name);
                    match import {
                        Some(Import {
                            value: ImportValue::Table(import_limits, state),
                            ..
                        }) => {
                            if let Some(state) = state {
                                let actual_type = &state.table.value_type;
                                let declared_type = &table.value_type();
                                let compatible = if actual_type == declared_type {
                                    true
                                } else if let Some(type_ctx) = state.type_ctx.as_ref() {
                                    value_types_equivalent_cross_module(
                                        actual_type,
                                        declared_type,
                                        type_ctx,
                                    )
                                } else {
                                    false
                                };
                                if !compatible {
                                    return Err(WasmError::unlinkable(
                                        "incompatible import type: .",
                                    )
                                    .into());
                                }
                            }
                            let (actual_min, actual_max, actual_is64) =
                                if let Some(state) = state.as_ref() {
                                    (
                                        state.table.size(),
                                        state.table.limits.max(),
                                        state.table.limits.is64,
                                    )
                                } else {
                                    (import_limits.min(), import_limits.max(), import_limits.is64)
                                };
                            let declared_limits = table.limits();
                            if actual_is64 != declared_limits.is64 {
                                return Err(
                                    WasmError::unlinkable("incompatible import type: .").into()
                                );
                            }
                            if actual_min < declared_limits.min() {
                                return Err(
                                    WasmError::unlinkable("incompatible import type: .").into()
                                );
                            }
                            if let Some(d_max) = declared_limits.max() {
                                match actual_max {
                                    Some(a_max) if a_max <= d_max => {}
                                    _ => {
                                        return Err(WasmError::unlinkable(
                                            "incompatible import type: .",
                                        )
                                        .into());
                                    }
                                }
                            }
                            let table_inst = if let Some(state) = state {
                                let shared = state.table.clone_shared_elements();
                                TableInst::from_shared(
                                    state.table.limits,
                                    table.value_type(),
                                    shared,
                                )
                            } else {
                                TableInst::new(*import_limits, table.value_type())
                            };
                            tables.push(table_inst);
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing table import: .").into());
                        }
                    }
                }
            }
        }

        let mut memories: collections::Vec<MemInst> =
            collections::Vec::with_capacity(mod_memories.len());
        for mem in &mod_memories {
            match mem.def() {
                MemoryDef::Local(_spec) => {
                    #[cfg(sf_has_guard_pages)]
                    {
                        memories.push(MemInst::new_guarded(mem.limits().clone())?);
                    }
                    #[cfg(all(sf_jit, not(sf_has_guard_pages)))]
                    {
                        memories.push(MemInst::new_unallocated(mem.limits().clone())?);
                    }
                    #[cfg(all(not(sf_jit), not(sf_has_guard_pages)))]
                    {
                        memories.push(MemInst::new(mem.limits().clone())?);
                    }
                }
                MemoryDef::Import {
                    module: mod_name,
                    name,
                    ..
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == *mod_name && i.name == *name);
                    match import {
                        Some(Import {
                            value: ImportValue::Memory(import_limits, imported_memory),
                            ..
                        }) => {
                            let (actual_min, actual_max, actual_is64) =
                                if let Some(imported_memory) = imported_memory.as_ref() {
                                    (
                                        imported_memory.current_pages(),
                                        imported_memory.limits.max(),
                                        imported_memory.limits.is64,
                                    )
                                } else {
                                    (import_limits.min(), import_limits.max(), import_limits.is64)
                                };
                            let declared_limits = mem.limits();
                            if actual_is64 != declared_limits.is64 {
                                return Err(
                                    WasmError::unlinkable("incompatible import type: .").into()
                                );
                            }
                            if actual_min < declared_limits.min() {
                                return Err(
                                    WasmError::unlinkable("incompatible import type: .").into()
                                );
                            }
                            if let Some(d_max) = declared_limits.max() {
                                match actual_max {
                                    Some(a_max) if a_max <= d_max => {}
                                    _ => {
                                        return Err(WasmError::unlinkable(
                                            "incompatible import type: .",
                                        )
                                        .into());
                                    }
                                }
                            }
                            let mem_inst = if let Some(imported_memory) = imported_memory {
                                MemInst::from_shared(
                                    imported_memory.limits,
                                    imported_memory.clone_shared_backing(),
                                )
                            } else {
                                #[cfg(sf_has_guard_pages)]
                                {
                                    MemInst::new_guarded(*import_limits)?
                                }
                                #[cfg(all(sf_jit, not(sf_has_guard_pages)))]
                                {
                                    MemInst::new_unallocated(*import_limits)?
                                }
                                #[cfg(all(not(sf_jit), not(sf_has_guard_pages)))]
                                {
                                    MemInst::new(*import_limits)?
                                }
                            };
                            memories.push(mem_inst);
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing memory import: .").into());
                        }
                    }
                }
            }
        }

        let mut globals: collections::Vec<GlobalInst> =
            collections::Vec::with_capacity(mod_globals.len());
        #[cfg(sf_has_simd)]
        let simd_registry = registry.simd_registry_shared();
        for global in &mod_globals {
            match global.def() {
                GlobalDef::Local(_spec) => {
                    let initial = Value::default_for_type(global.value_type());
                    #[cfg(sf_has_simd)]
                    let raw = match initial {
                        Value::V128(value) => simd_registry.intern(value),
                        _ => initial.to_raw(),
                    };
                    #[cfg(not(sf_has_simd))]
                    let raw = initial.to_raw();
                    globals.push(GlobalInst::new_raw(
                        raw,
                        global.mutable(),
                        global.value_type(),
                    ));
                }
                GlobalDef::Import {
                    module: mod_name,
                    name,
                    value_type,
                    mutable,
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == *mod_name && i.name == *name);
                    match import {
                        Some(Import {
                            value: ImportValue::Global(imported, imp_mutable),
                            ..
                        }) => {
                            if *imp_mutable != *mutable {
                                return Err(
                                    WasmError::unlinkable("incompatible import type: .").into()
                                );
                            }

                            match imported {
                                ImportedGlobal::Value(imported) => {
                                    let mut val = imported.value;
                                    if let Some((callback, func_type)) = &imported.linked_function {
                                        if let Value::Ref(_, ref_type) = val {
                                            let func_idx = functions.len();
                                            functions.push(FunctionInst::Host {
                                                func_type: tracked_alloc::rc::Rc::new(
                                                    func_type.clone(),
                                                ),
                                                callback: *callback,
                                            });
                                            val = Value::Ref(RefHandle::new(func_idx), ref_type);
                                        }
                                    }
                                    let val_type = val.value_type();
                                    if *mutable {
                                        if val_type != *value_type {
                                            return Err(WasmError::unlinkable(
                                                "incompatible import type: .",
                                            )
                                            .into());
                                        }
                                    } else if !val_type.can_initialize(value_type, &types) {
                                        return Err(WasmError::unlinkable(
                                            "incompatible import type: .",
                                        )
                                        .into());
                                    }
                                    #[cfg(sf_has_simd)]
                                    let raw = match val {
                                        Value::V128(value) => simd_registry.intern(value),
                                        _ => val.to_raw(),
                                    };
                                    #[cfg(not(sf_has_simd))]
                                    let raw = val.to_raw();
                                    globals.push(GlobalInst::new_raw(raw, *mutable, *value_type));
                                }
                                ImportedGlobal::State(state) => {
                                    let actual_type = state.global.value_type;
                                    let actual_ctx = state.type_ctx.as_ref();
                                    let compatible = if *mutable {
                                        global_value_types_equivalent_cross_context(
                                            &actual_type,
                                            value_type,
                                            actual_ctx,
                                            &types,
                                        )
                                    } else {
                                        global_value_type_can_initialize_cross_context(
                                            &actual_type,
                                            value_type,
                                            actual_ctx,
                                            &types,
                                        )
                                    };
                                    if !compatible {
                                        return Err(WasmError::unlinkable(
                                            "incompatible import type: .",
                                        )
                                        .into());
                                    }
                                    globals.push(GlobalInst::from_shared(
                                        state.global.clone_shared_cell(),
                                        *mutable,
                                        *value_type,
                                    ));
                                }
                            }
                        }
                        _ => {
                            return Err(WasmError::unlinkable("missing global import: .").into());
                        }
                    }
                }
            }
        }

        let mut tag_insts: collections::Vec<TagInst> =
            collections::Vec::with_capacity(mod_tags.len());
        for tag in &mod_tags {
            match tag.def() {
                TagDef::Local(spec) => {
                    tag_insts.push(TagInst {
                        handle: TagHandle::mint_fresh(),
                        type_index: spec.type_index(),
                        is_import: false,
                    });
                }
                TagDef::Import {
                    module: mod_name,
                    name,
                    func_type,
                    type_index,
                } => {
                    let import = imports
                        .iter()
                        .find(|i| i.module == *mod_name && i.name == *name);
                    match import {
                        Some(Import {
                            value: ImportValue::Tag(state),
                            ..
                        }) => {
                            let actual_type = &state.func_type;
                            let actual_type_ctx = state.type_ctx.as_ref();
                            let actual_type_idx = state.type_index;
                            let compatible = if actual_type_idx != u32::MAX {
                                actual_type_ctx.is_some_and(|type_ctx| {
                                    concrete_type_matches_cross_context(
                                        type_ctx,
                                        actual_type_idx,
                                        &types,
                                        *type_index,
                                    )
                                })
                            } else if actual_type == tag.func_type() {
                                true
                            } else if let Some(type_ctx) = actual_type_ctx {
                                check_function_types_equivalent(
                                    actual_type,
                                    tag.func_type(),
                                    type_ctx,
                                )
                            } else {
                                actual_type.params() == func_type.params()
                                    && actual_type.results() == func_type.results()
                            };
                            if !compatible {
                                return Err(
                                    WasmError::unlinkable("incompatible import type: .").into()
                                );
                            }
                            tag_insts.push(TagInst {
                                handle: state.handle,
                                type_index: *type_index,
                                is_import: true,
                            });
                        }
                        Some(_) => {
                            return Err(WasmError::unlinkable("incompatible import type: .").into());
                        }
                        None => {
                            return Err(WasmError::unlinkable("missing tag import: .").into());
                        }
                    }
                }
            }
        }

        let elements: collections::Vec<ElementInst> = mod_elements
            .iter()
            .map(|e| ElementInst::new(collections::Vec::new(), e.value_type()))
            .collect();

        let data: collections::Vec<DataInst> = mod_data
            .iter()
            .map(|d| DataInst::new(d.get_init().to_vec().into()))
            .collect();

        let mut module_inst = ModuleInst::new("main".to_string(), types);
        module_inst.functions = functions;
        module_inst.tables = tables;
        module_inst.memories = memories;
        module_inst.globals = globals;
        module_inst.tags = tag_insts;
        module_inst.elements = elements;
        module_inst.data = data;
        #[cfg(sf_jit)]
        {
            module_inst.table_dispatch_modes = table_dispatch_modes;
        }
        let mut store = Box::new(Store::new_with_registries(
            module_inst,
            registry.function_registry_shared(),
            registry.ref_registry_shared(),
            #[cfg(sf_has_simd)]
            registry.simd_registry_shared(),
        ));
        for func_idx in 0..store.module().functions.len() {
            if let Some(handle) = store.module().functions[func_idx].linked_handle() {
                store.module_mut().set_function_handle(func_idx, handle);
            } else {
                let _ = store.register_local_function(func_idx);
            }
        }

        let init_result = (|| -> Result<(), WasmError> {
            #[cfg(sf_jit)]
            crate::vm::build::ensure_module_compiled(&store)?;

            for (i, global) in mod_globals.iter().enumerate() {
                if let GlobalDef::Local(spec) = global.def() {
                    let value = eval_const_expr(spec.init_expr(), &mut store)?;
                    let raw = value_to_raw_in_store(value, &mut store);
                    store.global_mut(i).set_raw(raw);
                }
            }

            for (i, table) in mod_tables.iter().enumerate() {
                if table.is_import() {
                    continue;
                }

                let Some(init_expr) = table.spec().init_expr() else {
                    continue;
                };

                let value = eval_const_expr(init_expr, &mut store)?;
                let init_ref = match value {
                    Value::Ref(handle, _) => handle,
                    _ => {
                        return Err(WasmError::invalid(
                            "table initializer must evaluate to a reference",
                        ))
                    }
                };

                let table_inst = store.table_mut(i);
                let mut elements = table_inst.elements_mut();
                for slot in elements.iter_mut() {
                    *slot = init_ref;
                }
            }

            for (i, element) in mod_elements.iter().enumerate() {
                match element {
                    Element::Active {
                        table_index,
                        offset_expr,
                        init,
                    } => {
                        if *table_index >= store.module().tables.len() {
                            return Err(WasmError::unlinkable("unknown table"));
                        }
                        let offset = eval_offset(offset_expr, store.module())?;
                        let refs = materialize_element_init(init, &mut store)?;

                        let table = store.table_mut(*table_index);
                        let mut elements = table.elements_mut();
                        if offset + refs.len() > elements.len() {
                            return Err(WasmError::unlinkable("out of bounds table access"));
                        }
                        elements[offset..offset + refs.len()].copy_from_slice(&refs);
                        drop(elements);
                        store.module_mut().elements[i].drop_segment();
                    }
                    Element::Passive { init } => {
                        let refs = materialize_element_init(init, &mut store)?;
                        store.module_mut().elements[i] =
                            ElementInst::new(refs, element.value_type());
                    }
                    Element::Declarative { .. } => {
                        store.module_mut().elements[i].drop_segment();
                    }
                }
            }

            #[cfg(sf_jit)]
            {
                for memory in &store.module().memories {
                    memory.ensure_allocated()?;
                }
            }

            for data_seg in &mod_data {
                match data_seg {
                    Data::Active {
                        memory_index,
                        offset_expr,
                        init,
                    } => {
                        let offset = eval_offset(offset_expr, store.module())?;
                        let mem = store.memory_mut(*memory_index);
                        let mem_len = mem.memory_len();
                        if offset + init.len() > mem_len {
                            return Err(WasmError::unlinkable("out of bounds memory access"));
                        }
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                mem.memory_ptr().add(offset),
                                init.len(),
                            )
                        };
                        dst.copy_from_slice(init);
                    }
                    Data::Passive { .. } => {}
                }
            }

            if let Some(start_idx) = start_func_index {
                let func_ptr = &store.module().functions[start_idx] as *const FunctionInst;
                let func_ref = unsafe { &*func_ptr };
                runtime::eval(func_ref, &mut store, &[])?;
            }

            Ok(())
        })();

        match init_result {
            Ok(()) => Ok(Instance { store, exports }),
            Err(error) => Err(InstanceInstantiationError::Partial {
                instance: Instance { store, exports },
                error,
            }),
        }
    }

    pub fn invoke(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        let (_, _, idx) = self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Func) && n == name)
            .ok_or_else(|| WasmError::invalid("exported function not found"))?;
        let idx = *idx;

        let func_ptr = &self.store.module().functions[idx] as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let result_stack = runtime::eval(func_ref, &mut self.store, args)?;
        let ft = func_ref.func_type();
        let result_types = ft.results();

        let mut results = collections::Vec::with_capacity(result_types.len());
        for (i, ty) in result_types.iter().enumerate() {
            let raw = result_stack.peek_at_index(i);
            results.push(try_raw_to_value_in_store(raw, *ty, &self.store)?);
        }
        Ok(results)
    }

    pub fn invoke_function_index(
        &mut self,
        idx: usize,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        let func_ptr = self
            .store
            .module()
            .functions
            .get(idx)
            .ok_or_else(|| WasmError::invalid("function index out of range"))?
            as *const FunctionInst;
        let func_ref = unsafe { &*func_ptr };

        let result_stack = runtime::eval(func_ref, &mut self.store, args)?;
        let ft = func_ref.func_type();
        let result_types = ft.results();

        let mut results = collections::Vec::with_capacity(result_types.len());
        for (i, ty) in result_types.iter().enumerate() {
            let raw = result_stack.peek_at_index(i);
            results.push(try_raw_to_value_in_store(raw, *ty, &self.store)?);
        }
        Ok(results)
    }

    pub fn has_function_export(&self, name: &str) -> bool {
        self.exports
            .iter()
            .any(|(n, k, _)| matches!(k, ExportKind::Func) && n == name)
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }

    pub fn get_global(&self, name: &str) -> Result<Option<Value>, WasmError> {
        Ok(self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Global) && n == name)
            .map(|(_, _, idx)| self.store.global(*idx).value(&self.store))
            .transpose()?)
    }

    pub fn global_at(&self, idx: usize) -> Result<Option<Value>, WasmError> {
        Ok(self
            .store
            .module()
            .globals
            .get(idx)
            .map(|g| g.value(&self.store))
            .transpose()?)
    }

    pub fn replace_global_at(&mut self, idx: usize, value: Value) -> Result<(), WasmError> {
        let raw = value_to_raw_in_store(value, &mut self.store);
        let global = self
            .store
            .module_mut()
            .globals
            .get_mut(idx)
            .ok_or_else(|| WasmError::invalid("global index out of range"))?;
        global.set_raw(raw);
        Ok(())
    }

    pub fn set_global(&mut self, name: &str, value: Value) -> Result<(), WasmError> {
        let idx = self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Global) && n == name)
            .map(|(_, _, idx)| *idx)
            .ok_or_else(|| WasmError::invalid("global not found"))?;
        let raw = value_to_raw_in_store(value, &mut self.store);
        let global = self.store.global_mut(idx);
        if !global.mutable {
            return Err(WasmError::invalid("cannot set immutable global"));
        }
        global.set_raw(raw);
        Ok(())
    }

    pub fn memory(&self) -> Option<&[u8]> {
        if self.store.module().memories.is_empty() {
            None
        } else {
            let mem = self.store.memory(0);
            let len = mem.memory_len();
            Some(unsafe { core::slice::from_raw_parts(mem.memory_ptr(), len) })
        }
    }

    pub fn memory_mut(&mut self) -> Option<&mut [u8]> {
        if self.store.module().memories.is_empty() {
            None
        } else {
            let mem = self.store.memory_mut(0);
            let len = mem.memory_len();
            Some(unsafe { core::slice::from_raw_parts_mut(mem.memory_ptr(), len) })
        }
    }

    pub fn memory_pages(&self, name: &str) -> Option<usize> {
        for (n, kind, idx) in &self.exports {
            if n == name && matches!(kind, ExportKind::Memory) {
                return Some(self.store.memory(*idx).current_pages());
            }
        }
        None
    }

    pub fn memory_bytes_at(&self, idx: usize) -> Option<&[u8]> {
        self.store.module().memories.get(idx).map(|mem| {
            let len = mem.memory_len();
            unsafe { core::slice::from_raw_parts(mem.memory_ptr(), len) }
        })
    }

    pub fn replace_memory_bytes_at(&mut self, idx: usize, bytes: &[u8]) -> Result<(), WasmError> {
        let mem = self
            .store
            .module_mut()
            .memories
            .get_mut(idx)
            .ok_or_else(|| WasmError::invalid("memory index out of range"))?;
        let len = mem.memory_len();
        if len != bytes.len() {
            return Err(WasmError::invalid("memory size mismatch"));
        }
        let dst = unsafe { core::slice::from_raw_parts_mut(mem.memory_ptr(), len) };
        dst.copy_from_slice(bytes);
        Ok(())
    }

    pub fn table_size(&self, name: &str) -> Option<usize> {
        for (n, kind, idx) in &self.exports {
            if n == name && matches!(kind, ExportKind::Table) {
                return Some(self.store.table(*idx).size());
            }
        }
        None
    }

    pub fn table_elements_at(&self, idx: usize) -> Option<&[RefHandle]> {
        self.store.module().tables.get(idx).map(|table| {
            let elements = table.elements();
            unsafe { core::slice::from_raw_parts(elements.as_ptr(), elements.len()) }
        })
    }

    pub fn replace_table_elements_at(
        &mut self,
        idx: usize,
        elements: &[RefHandle],
    ) -> Result<(), WasmError> {
        let table = self
            .store
            .module_mut()
            .tables
            .get_mut(idx)
            .ok_or_else(|| WasmError::invalid("table index out of range"))?;
        let mut table_elements = table.elements_mut();
        if table_elements.len() != elements.len() {
            return Err(WasmError::invalid("table size mismatch"));
        }
        table_elements.copy_from_slice(elements);
        drop(table_elements);
        #[cfg(sf_jit)]
        {
            for mode in &mut self.store.module_mut().table_dispatch_modes {
                *mode = TableDispatchMode::Generic;
            }
            self.store.module().clear_native_code();
        }
        Ok(())
    }

    pub fn function_type_at(&self, idx: usize) -> Option<FunctionType> {
        self.store
            .module()
            .functions
            .get(idx)
            .map(|func| func.func_type().clone())
    }

    pub fn function_type_index_at(&self, idx: usize) -> Option<u32> {
        self.store
            .module()
            .functions
            .get(idx)
            .map(FunctionInst::type_index)
    }

    pub fn function_handle_at(&self, idx: usize) -> Option<RefHandle> {
        self.store.module().function_handle(idx)
    }

    /// Resolve an exported tag to its runtime identity. Required for
    /// cross-module tag linking via `Import::linked_tag_typed(...)`.
    pub fn tag_handle(&self, name: &str) -> Option<TagHandle> {
        let (_, _, idx) = self
            .exports
            .iter()
            .find(|(n, k, _)| matches!(k, ExportKind::Tag) && n == name)?;
        self.store.module().tags.get(*idx).map(|t| t.handle)
    }

    pub fn shared_table_state_at(&self, idx: usize) -> Option<ImportedTableState> {
        self.store
            .module()
            .tables
            .get(idx)
            .cloned()
            .map(|table| ImportedTableState {
                table,
                type_ctx: Some(self.store.module().types.clone()),
            })
    }

    pub fn shared_global_state_at(&self, idx: usize) -> Option<ImportedGlobalState> {
        self.store
            .module()
            .globals
            .get(idx)
            .cloned()
            .map(|global| ImportedGlobalState {
                global,
                type_ctx: Some(self.store.module().types.clone()),
            })
    }

    pub fn shared_memory_at(&self, idx: usize) -> Option<MemInst> {
        self.store.module().memories.get(idx).cloned()
    }

    pub fn clone_function_registry(&self) -> LinkRegistry {
        #[cfg(sf_has_simd)]
        {
            LinkRegistry::from_shared(
                self.store.clone_function_registry(),
                self.store.clone_ref_registry(),
                self.store.clone_simd_registry(),
            )
        }
        #[cfg(not(sf_has_simd))]
        LinkRegistry::from_shared(
            self.store.clone_function_registry(),
            self.store.clone_ref_registry(),
        )
    }

    pub fn append_host_function(&mut self, func_type: FunctionType, callback: HostFn) -> usize {
        let idx = self.store.module().functions.len();
        self.store.module_mut().functions.push(FunctionInst::Host {
            func_type: tracked_alloc::rc::Rc::new(func_type),
            callback,
        });
        let _ = self.store.register_local_function(idx);
        idx
    }
}

fn eval_offset(expr: &ConstExpr, module: &ModuleInst) -> Result<usize, WasmError> {
    use crate::opcodes::Opcode;
    use crate::utils::payload::Payload;

    let bytes: &[u8] = expr;
    let mut code: Payload = bytes.into();
    let mut stack = collections::vec![];
    while !code.is_empty() {
        let op: Opcode = code.read_u8()?.try_into()?;
        match op {
            Opcode::I32_CONST => stack.push(Value::I32(code.read_leb128_i32()?)),
            Opcode::I64_CONST => stack.push(Value::I64(code.read_leb128_i64()?)),
            Opcode::GLOBAL_GET => {
                let idx = code.read_leb128_u32()? as usize;
                let global = module
                    .globals
                    .get(idx)
                    .ok_or_else(|| WasmError::invalid("global.get: index out of range"))?;
                stack.push(Value::from_raw(global.raw(), global.value_type));
            }
            Opcode::I32_ADD => {
                let rhs = match stack.pop() {
                    Some(Value::I32(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i32.add")),
                };
                let lhs = match stack.pop() {
                    Some(Value::I32(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i32.add")),
                };
                stack.push(Value::I32(lhs.wrapping_add(rhs)));
            }
            Opcode::I32_SUB => {
                let rhs = match stack.pop() {
                    Some(Value::I32(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i32.sub")),
                };
                let lhs = match stack.pop() {
                    Some(Value::I32(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i32.sub")),
                };
                stack.push(Value::I32(lhs.wrapping_sub(rhs)));
            }
            Opcode::I32_MUL => {
                let rhs = match stack.pop() {
                    Some(Value::I32(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i32.mul")),
                };
                let lhs = match stack.pop() {
                    Some(Value::I32(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i32.mul")),
                };
                stack.push(Value::I32(lhs.wrapping_mul(rhs)));
            }
            Opcode::I64_ADD => {
                let rhs = match stack.pop() {
                    Some(Value::I64(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i64.add")),
                };
                let lhs = match stack.pop() {
                    Some(Value::I64(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i64.add")),
                };
                stack.push(Value::I64(lhs.wrapping_add(rhs)));
            }
            Opcode::I64_SUB => {
                let rhs = match stack.pop() {
                    Some(Value::I64(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i64.sub")),
                };
                let lhs = match stack.pop() {
                    Some(Value::I64(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i64.sub")),
                };
                stack.push(Value::I64(lhs.wrapping_sub(rhs)));
            }
            Opcode::I64_MUL => {
                let rhs = match stack.pop() {
                    Some(Value::I64(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i64.mul")),
                };
                let lhs = match stack.pop() {
                    Some(Value::I64(value)) => value,
                    _ => return Err(WasmError::invalid("type mismatch in i64.mul")),
                };
                stack.push(Value::I64(lhs.wrapping_mul(rhs)));
            }
            Opcode::END => break,
            _ => {
                return Err(WasmError::invalid(
                    "offset must be a numeric const expression",
                ))
            }
        }
    }
    let value = stack
        .pop()
        .ok_or_else(|| WasmError::invalid("empty offset const expression"))?;
    match value {
        Value::I32(v) => {
            if v < 0 {
                Err(WasmError::unlinkable("offset is negative"))
            } else {
                Ok(v as usize)
            }
        }
        Value::I64(v) => {
            if v < 0 {
                Err(WasmError::unlinkable("offset is negative"))
            } else {
                Ok(v as usize)
            }
        }
        _ => Err(WasmError::invalid("offset must be i32 or i64")),
    }
}

fn materialize_element_init(
    init: &ElementInit,
    store: &mut Store,
) -> Result<collections::Vec<RefHandle>, WasmError> {
    match init {
        ElementInit::FunctionIndexes(indices) => indices
            .iter()
            .map(|&idx| {
                store
                    .module()
                    .function_handle(idx)
                    .ok_or_else(|| WasmError::invalid("element function index out of range".into()))
            })
            .collect(),
        ElementInit::InitExprs { exprs, .. } => exprs
            .iter()
            .map(|expr| {
                let value = eval_const_expr(expr, store)?;
                match value {
                    Value::Ref(handle, _) => Ok(handle),
                    _ => Err(WasmError::invalid(
                        "element init must be a reference".into(),
                    )),
                }
            })
            .collect(),
    }
}

// NOTE: Old debug_coremark_direct_coverage test removed — it referenced the
// legacy native::ir types (NativeProgram, NativeInstKind, etc.) that no longer
// exist after the native/ → machine/ + arch/ + runtime/ restructuring.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        module::{
            builder::ModuleBuilder,
            entities::{Memory, Table},
            Module,
        },
        utils::limits::Limits,
        value_type::ValueType,
    };

    fn importer_with_memory(limits: Limits) -> Module {
        let mut builder = ModuleBuilder::new();
        builder.with_name("importer");
        builder.with_binary_version(1);
        builder.append_memory(
            Memory::new_import("env".into(), "mem".into(), limits)
                .expect("memory import limits should stay valid"),
        );
        builder.build()
    }

    fn importer_with_table(limits: Limits, value_type: ValueType) -> Module {
        let mut builder = ModuleBuilder::new();
        builder.with_name("importer");
        builder.with_binary_version(1);
        builder.append_table(
            Table::new_import("env".into(), "table".into(), value_type, limits)
                .expect("table import limits should stay valid"),
        );
        builder.build()
    }

    #[cfg(sf_jit)]
    #[test]
    fn fixed_private_local_table_enables_local_only_dispatch_mode() {
        let wasm = wat::parse_str(
            r#"
            (module
              (type $t (func))
              (func $f)
              (table 1 1 funcref)
              (elem (i32.const 0) $f)
            )
            "#,
        )
        .expect("wat should encode");
        let module = Module::new("test", &wasm).expect("module should parse");
        let modes =
            compute_static_table_dispatch_modes(&module).expect("table facts should compute");
        assert_eq!(
            modes,
            collections::vec![TableDispatchMode::FixedLocalOnly { len: 1 }]
        );
    }

    #[cfg(sf_jit)]
    #[test]
    fn table_mutation_keeps_dispatch_mode_generic() {
        let wasm = wat::parse_str(
            r#"
            (module
              (type $t (func))
              (func $f)
              (func $set (param i32 funcref)
                local.get 0
                local.get 1
                table.set 0)
              (table 1 1 funcref)
              (elem (i32.const 0) $f)
            )
            "#,
        )
        .expect("wat should encode");
        let module = Module::new("test", &wasm).expect("module should parse");
        let modes =
            compute_static_table_dispatch_modes(&module).expect("table facts should compute");
        assert_eq!(modes, collections::vec![TableDispatchMode::Generic]);
    }

    #[cfg(sf_jit)]
    #[test]
    fn subtype_module_keeps_dispatch_mode_generic() {
        let wasm = wat::parse_str(
            r#"
            (module
              (type $super (func))
              (type $sub (sub $super (func)))
              (func $f (type $sub))
              (table 1 1 funcref)
              (elem (i32.const 0) $f)
            )
            "#,
        )
        .expect("wat should encode");
        let module = Module::new("test", &wasm).expect("module should parse");
        let modes =
            compute_static_table_dispatch_modes(&module).expect("table facts should compute");
        assert_eq!(modes, collections::vec![TableDispatchMode::Generic]);
    }

    fn grow_shared_memory_for_test(memory: &MemInst, new_pages: usize) {
        let old_pages = memory.current_pages();
        assert!(new_pages >= old_pages);

        let mut backing = memory.backing_mut();
        #[cfg(sf_has_guard_pages)]
        if let Some(guard) = backing.guard.as_mut() {
            let delta_pages = new_pages - old_pages;
            guard
                .grow(delta_pages)
                .expect("shared guard memory grow should succeed");
            return;
        }
        backing
            .data
            .resize(new_pages * crate::constants::WASM_PAGE_SIZE, 0);
    }

    fn grow_shared_table_for_test(table: &TableInst, new_len: usize) {
        let mut elements = table.elements_mut();
        elements.resize(new_len, RefHandle::null());
    }

    #[test]
    fn shared_memory_import_uses_live_size_and_shared_cap() {
        let shared_memory = MemInst::new(Limits::new(1, Some(2)).unwrap())
            .expect("test memory within runtime limits");
        grow_shared_memory_for_test(&shared_memory, 2);
        let module = importer_with_memory(Limits::new(2, Some(2)).unwrap());
        let import = Import::memory_with_state(
            "env",
            "mem",
            Limits::new(2, Some(3)).unwrap(),
            Some(shared_memory),
        );

        let instance = Instance::from_module(module, &[import]).expect("shared import should link");

        assert_eq!(instance.store.memory(0).current_pages(), 2);
        assert_eq!(instance.store.memory(0).limits.max(), Some(2));
    }

    #[test]
    fn shared_table_import_uses_live_size_and_shared_cap() {
        let shared_table = TableInst::new(Limits::new(1, Some(2)).unwrap(), ValueType::funcref());
        grow_shared_table_for_test(&shared_table, 2);
        let module = importer_with_table(Limits::new(2, Some(2)).unwrap(), ValueType::funcref());
        let import = Import::table_with_state(
            "env",
            "table",
            Limits::new(2, Some(3)).unwrap(),
            Some(ImportedTableState {
                table: shared_table,
                type_ctx: None,
            }),
        );

        let instance = Instance::from_module(module, &[import]).expect("shared import should link");

        assert_eq!(instance.store.table(0).size(), 2);
        assert_eq!(instance.store.table(0).limits.max(), Some(2));
    }

    #[test]
    fn shared_global_import_aliases_mutations() {
        let source_wasm = wat::parse_str(
            r#"
            (module
              (global $g (mut i32) (i32.const 0))
              (export "g1" (global $g))
              (export "g2" (global $g)))
            "#,
        )
        .expect("source module should encode");
        let source = Instance::new(&source_wasm, &[]).expect("source module should instantiate");
        let shared = source
            .shared_global_state_at(0)
            .expect("source global should be shareable");

        let importer_wasm = wat::parse_str(
            r#"
            (module
              (import "env" "g1" (global $g1 (mut i32)))
              (import "env" "g2" (global $g2 (mut i32)))
              (func (export "set_then_get") (result i32)
                i32.const 7
                global.set $g1
                global.get $g2))
            "#,
        )
        .expect("importer module should encode");
        let mut importer = Instance::new(
            &importer_wasm,
            &[
                Import::global_with_state("env", "g1", shared.clone()),
                Import::global_with_state("env", "g2", shared),
            ],
        )
        .expect("importer module should instantiate");

        let result = importer
            .invoke("set_then_get", &[])
            .expect("global aliasing invocation should succeed");
        assert_eq!(result.as_slice(), &[Value::I32(7)]);
        assert_eq!(
            source.get_global("g1").expect("global read should succeed"),
            Some(Value::I32(7))
        );
    }

    #[test]
    fn shared_global_import_rejects_same_index_different_concrete_types() {
        let mutable_source_wasm = wat::parse_str(
            r#"
            (module
              (type $a (struct (field i32)))
              (global $g (mut (ref null $a)) (ref.null $a))
              (export "g" (global $g)))
            "#,
        )
        .expect("mutable source module should encode");
        let mutable_source =
            Instance::new(&mutable_source_wasm, &[]).expect("source module should instantiate");
        let mutable_shared = mutable_source
            .shared_global_state_at(0)
            .expect("source global should be shareable");
        let mutable_importer_wasm = wat::parse_str(
            r#"
            (module
              (type $b (array (mut i32)))
              (import "env" "g" (global $g (mut (ref null $b)))))
            "#,
        )
        .expect("mutable importer module should encode");

        assert!(Instance::new(
            &mutable_importer_wasm,
            &[Import::global_with_state("env", "g", mutable_shared)]
        )
        .is_err());

        let immutable_source_wasm = wat::parse_str(
            r#"
            (module
              (type $a (struct (field i32)))
              (global $g (ref null $a) (ref.null $a))
              (export "g" (global $g)))
            "#,
        )
        .expect("immutable source module should encode");
        let immutable_source =
            Instance::new(&immutable_source_wasm, &[]).expect("source module should instantiate");
        let immutable_shared = immutable_source
            .shared_global_state_at(0)
            .expect("source global should be shareable");
        let immutable_importer_wasm = wat::parse_str(
            r#"
            (module
              (type $b (array (mut i32)))
              (import "env" "g" (global $g (ref null $b))))
            "#,
        )
        .expect("immutable importer module should encode");

        assert!(Instance::new(
            &immutable_importer_wasm,
            &[Import::global_with_state("env", "g", immutable_shared)]
        )
        .is_err());
    }

    #[test]
    fn instantiation_rejects_builder_modules_with_simd_types() {
        let mut builder = ModuleBuilder::new();
        builder.with_name("simd-builder");
        builder.with_binary_version(1);
        builder.with_function_types(crate::collections::vec![tracked_alloc::rc::Rc::new(
            crate::FunctionType::new(
                crate::collections::vec![ValueType::V128],
                crate::collections::Vec::new(),
            ),
        )]);

        #[cfg(not(sf_has_simd))]
        {
            let err = match Instance::from_module(builder.build(), &[]) {
                Ok(_) => panic!("instantiation should reject unsupported SIMD-shaped modules"),
                Err(err) => err,
            };
            assert_eq!(
                err,
                crate::WasmError::invalid("SIMD is not supported on this CPU")
            );
        }

        #[cfg(sf_has_simd)]
        {
            Instance::from_module(builder.build(), &[])
                .expect("SIMD-enabled builds should allow unused v128 type definitions");
        }
    }

    #[cfg(all(sf_has_simd, sf_jit))]
    #[test]
    fn invoke_rejects_unsupported_live_simd_ops_during_lowering() {
        let wasm = wat::parse_str(
            r#"
            (module
              (func (export "not") (param v128) (result v128)
                local.get 0
                v128.not))
            "#,
        )
        .expect("wat should encode a SIMD module");

        let mut instance = Instance::new(&wasm, &[]).expect("instantiation should succeed");
        let results = instance
            .invoke("not", &[crate::Value::V128([0; 16])])
            .expect("live SIMD unary ops should lower and execute");
        assert_eq!(results.as_slice(), &[crate::Value::V128([u8::MAX; 16])]);
    }

    #[cfg(all(sf_has_simd, sf_jit))]
    #[test]
    fn invoke_returns_live_v128_const() {
        let wasm = wat::parse_str(
            r#"
            (module
              (func (export "const") (result v128)
                (v128.const i32x4 1 2 3 4)))
            "#,
        )
        .expect("wat should encode a SIMD const module");

        let mut instance = Instance::new(&wasm, &[]).expect("instantiation should succeed");
        let expected = crate::Value::V128([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);
        let results = instance
            .invoke("const", &[])
            .expect("v128.const should lower and execute");
        assert_eq!(results.as_slice(), &[expected]);
    }
}
