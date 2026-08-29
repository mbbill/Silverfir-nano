//! An instantiated module, on whichever engine this build selected.
//!
//! Which engine is underneath is not the embedder's problem. The same
//! bytes, the same `&[Import]`, and the same `invoke(name, &[Value])` work
//! on the JIT and on the interpreter, and the code that calls them needs no
//! `cfg`. What differs -- native code versus a threaded dispatch chain,
//! a `JitInstance` full of entities versus a flat predecoded frame -- differs
//! below this line.
//!
//! The inner variants are gated on the engine features, so a build with one
//! engine has a univariant wrapper: no discriminant, and every match here
//! folds to its single arm. See [`crate::vm::engine`].
//!
//! Deep introspection of the JIT's entity model -- the store, shared
//! entity handles, tag handles, per-index accessors -- has no counterpart
//! on the interpreter and stays where it lives, reachable through
//! [`Instance::as_jit`].

use crate::collections;
use crate::error::WasmError;
use crate::module::type_defs::FunctionType;
use crate::module::Module;
use crate::vm::engine::{Engine, Tier};
use crate::vm::entities::{Caller, MemInst};
use crate::vm::imports::ImportedGlobalState;
use crate::vm::link::{InstanceFreeError, InstanceId, LinkRegistry, WorldAccess};
use crate::vm::tag::TagIdentity;
use crate::vm::value::{RefValue, Value};

#[cfg(all(sf_interp, sf_module_validator))]
use crate::vm::interpreter::ValidatedBaselinePlan;
#[cfg(sf_interp)]
use crate::vm::interpreter::{InterpInstance, InterpInstanceLease};
#[cfg(sf_jit)]
use crate::vm::jit::instantiate::JitInstanceLease;

// One import model for both engines. The interpreter's raw host-dispatch
// boundary is an implementation detail that `interp_imports` drives from
// these same declarations.
pub use crate::vm::imports::{
    Import, ImportValue, ImportedFunction, ImportedTableState, ImportedTagState,
};

#[cfg(sf_interp)]
mod interp_imports;

enum Inner {
    #[cfg(sf_jit)]
    Jit(JitInstanceLease),
    #[cfg(sf_interp)]
    Interp(InterpInstanceLease),
}

/// Reject a module the spec says is invalid.
///
/// The ordinary validation path. Validator-enabled interpreter construction
/// uses the composite validator instead, so the same function-body decode
/// also produces its retained baseline artifact.
#[inline]
#[cfg(any(sf_jit, not(sf_module_validator)))]
fn validate(module: &Module) -> Result<(), WasmError> {
    module.ensure_simd_supported()?;
    #[cfg(sf_module_validator)]
    {
        use crate::module::validator::Validator;
        Validator::new(module).validate()?;
    }
    Ok(())
}

/// A module instantiated on one engine.
pub struct Instance {
    inner: Inner,
    registry: LinkRegistry,
}

/// A set of instances that can name each other's functions.
///
/// Instances live in generational slots, so an id identifies one instance for
/// as long as that instance lives and stops resolving once it does not. A
/// reference held across a free fails its generation check rather than
/// reaching whatever occupied the slot next.
///
/// `invoke` checks the callee out of the table and lets its own borrow of the
/// world end before running it. A cross-instance call made from inside that
/// callee therefore re-enters through a fresh checkout instead of nesting a
/// borrow, which is what makes the nested case work:
///
/// ```ignore
/// let mut world = RuntimeWorld::new();
/// let a = world.instantiate(&engine, module_a, &[])?;
/// let b = world.instantiate(&engine, module_b, &imports_naming(a))?;
/// // `run_b` calls a funcref owned by `a`, mid-execution.
/// let result = world.invoke(b, "run_b", &args)?;
/// ```
///
/// [`Instance::from_module`] remains the convenience for embedders that want
/// one module: it owns a private one-slot world.
pub struct RuntimeWorld {
    registry: LinkRegistry,
    instances: collections::Vec<(InstanceId, Option<Instance>)>,
    /// A world runs one engine. The first instantiation fixes the tier and
    /// later ones must match: instances in a world resolve each other's
    /// function identities out of one address space, and the two engines do
    /// not share a call path for them.
    tier: Option<Tier>,
}

/// The result of an instantiation that did not produce a usable facade.
///
/// A partial failure leaves its slot occupied because initialization may
/// already have published references into another instance's storage.
#[derive(Debug, Clone, PartialEq)]
pub enum InstanceInstantiationError {
    Complete(WasmError),
    Partial { id: InstanceId, error: WasmError },
}

impl From<WasmError> for InstanceInstantiationError {
    fn from(error: WasmError) -> Self {
        Self::Complete(error)
    }
}

impl InstanceInstantiationError {
    pub fn error(&self) -> &WasmError {
        match self {
            Self::Complete(error) | Self::Partial { error, .. } => error,
        }
    }

    pub fn into_parts(self) -> (Option<InstanceId>, WasmError) {
        match self {
            Self::Complete(error) => (None, error),
            Self::Partial { id, error } => (Some(id), error),
        }
    }
}

/// An export resolved once, so calling it again does not look it up again.
///
/// [`Instance::invoke`] searches the export list by name and allocates a
/// result vector on every call. A caller that runs the same function
/// repeatedly -- a render loop, a benchmark harness -- resolves it once
/// with [`Instance::get_func`] and calls it through [`Instance::call`],
/// which writes results into a slice the caller already owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Func {
    index: usize,
    params: usize,
    results: usize,
}

impl Func {
    /// How many arguments this function takes, and how many it returns.
    #[inline]
    pub const fn arity(&self) -> (usize, usize) {
        (self.params, self.results)
    }
}

impl Instance {
    fn from_module_in_registry(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        registry: &LinkRegistry,
    ) -> Result<Self, InstanceInstantiationError> {
        match engine.tier() {
            #[cfg(sf_jit)]
            Tier::Jit => {
                validate(&module)?;
                JitInstanceLease::from_module_with_registry(engine, module, imports, registry).map(
                    |inst| Self {
                        inner: Inner::Jit(inst),
                        registry: registry.clone(),
                    },
                )
            }
            #[cfg(sf_interp)]
            Tier::Interp => {
                #[cfg(sf_module_validator)]
                let baseline = ValidatedBaselinePlan::validate(&module)?;
                #[cfg(not(sf_module_validator))]
                validate(&module)?;
                let dispatch = interp_imports::bind(&module, imports)?;
                match InterpInstance::new_partial_with_registry(
                    engine,
                    module,
                    #[cfg(sf_module_validator)]
                    baseline,
                    Some(InterpInstance::boxed_caller_host(dispatch)),
                    imports,
                    None,
                    registry,
                ) {
                    Ok(inst) => Ok(Self {
                        inner: Inner::Interp(inst),
                        registry: registry.clone(),
                    }),
                    Err((Some(partial), error)) => {
                        let id = partial.into_occupied_id();
                        Err(InstanceInstantiationError::Partial { id, error })
                    }
                    Err((None, error)) => Err(InstanceInstantiationError::Complete(error)),
                }
            }
        }
    }

    /// Instantiate in `engine`, which decides the tier and the budgets.
    pub fn new(engine: &Engine, wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        Self::from_module(engine, Module::new("main", wasm_bytes)?, imports)
    }

    #[cfg(all(test, sf_interp, sf_module_validator))]
    pub(crate) fn from_module_full_fold_for_test(
        engine: &Engine,
        module: Module,
        imports: &[Import],
    ) -> Result<Self, WasmError> {
        if engine.tier() != Tier::Interp {
            return Err(WasmError::invalid(
                "full-fold oracle requires interpreter tier",
            ));
        }
        let registry = LinkRegistry::new();
        let baseline =
            ValidatedBaselinePlan::validate(&module)?.force_all_full_fold_for_test(&module);
        let dispatch = interp_imports::bind(&module, imports)?;
        match InterpInstance::new_partial_with_registry(
            engine,
            module,
            baseline,
            Some(InterpInstance::boxed_caller_host(dispatch)),
            imports,
            None,
            &registry,
        ) {
            Ok(inst) => Ok(Self {
                inner: Inner::Interp(inst),
                registry,
            }),
            Err((_, error)) => Err(error),
        }
    }

    /// Instantiate against a shared link registry, so instances can
    /// resolve each other's exports.
    ///
    /// Both engines use the registry for linked functions and reference
    /// identities. The interpreter enters registry-known peers directly
    /// unless an installed [`crate::FuncRefHost`] overrides that routing.
    pub fn from_module_with_registry(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        registry: &crate::vm::link::LinkRegistry,
    ) -> Result<Self, InstanceInstantiationError> {
        Self::from_module_in_registry(engine, module, imports, registry)
    }

    /// Instantiate with a hook for cross-instance function references, and
    /// keep the instance when a data segment traps.
    #[cfg(sf_interp)]
    pub fn from_module_with_funcref_host(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        funcref_host: crate::vm::interpreter::FuncRefHost,
    ) -> Result<Self, (Option<Self>, WasmError)> {
        let registry = crate::vm::link::LinkRegistry::new();
        Self::from_module_with_registry_and_funcref_host(
            engine,
            module,
            imports,
            &registry,
            funcref_host,
        )
    }

    /// Instantiate with a shared link registry and a hook for cross-instance
    /// function references, retaining a partial instance when a data segment
    /// traps.
    ///
    /// The hook takes precedence over the interpreter's engine-native path,
    /// while the registry preserves reference identity and payloads across
    /// instances.
    #[cfg(sf_interp)]
    pub fn from_module_with_registry_and_funcref_host(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        registry: &crate::vm::link::LinkRegistry,
        funcref_host: crate::vm::interpreter::FuncRefHost,
    ) -> Result<Self, (Option<Self>, WasmError)> {
        #[cfg(sf_module_validator)]
        let baseline = ValidatedBaselinePlan::validate(&module).map_err(|e| (None, e))?;
        #[cfg(not(sf_module_validator))]
        validate(&module).map_err(|e| (None, e))?;
        let dispatch = interp_imports::bind(&module, imports).map_err(|e| (None, e))?;
        match InterpInstance::new_partial_with_registry(
            engine,
            module,
            #[cfg(sf_module_validator)]
            baseline,
            Some(InterpInstance::boxed_caller_host(dispatch)),
            imports,
            Some(funcref_host),
            registry,
        ) {
            Ok(inst) => Ok(Self {
                inner: Inner::Interp(inst),
                registry: registry.clone(),
            }),
            Err((partial, error)) => Err((
                partial.map(|inst| Self {
                    inner: Inner::Interp(inst),
                    registry: registry.clone(),
                }),
                error,
            )),
        }
    }

    /// Instantiate an already-parsed module in `engine`.
    pub fn from_module(
        engine: &Engine,
        module: Module,
        imports: &[Import],
    ) -> Result<Self, WasmError> {
        let mut world = RuntimeWorld::new();
        let id = world
            .instantiate(engine, module, imports)
            .map_err(|error| error.into_parts().1)?;
        world
            .take(id)
            .ok_or_else(|| WasmError::invalid("new instance missing from its private world"))
    }

    /// Which tier is running this instance.
    #[inline]
    pub fn tier(&self) -> Tier {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(_) => Tier::Jit,
            #[cfg(sf_interp)]
            Inner::Interp(_) => Tier::Interp,
        }
    }

    fn instance_id(&self) -> InstanceId {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(instance) => instance.instance_id(),
            #[cfg(sf_interp)]
            Inner::Interp(instance) => instance.instance_id(),
        }
    }

    fn has_exclusive_lease(&self) -> bool {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(instance) => instance.has_exclusive_lease(),
            #[cfg(sf_interp)]
            Inner::Interp(instance) => instance.has_exclusive_lease(),
        }
    }

    /// Call an exported function by name.
    #[inline]
    pub fn invoke(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        RuntimeWorld::from_registry(self.registry.clone()).invoke(self.instance_id(), name, args)
    }

    /// Resolve an exported function once for repeated calls.
    pub fn get_func(&self, name: &str) -> Option<Func> {
        let (index, params, results) = match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => {
                let index = inst.function_index_of_export(name)?;
                let ft = inst.function_type_at(index)?;
                (index, ft.params().len(), ft.results().len())
            }
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| {
                let index = inst.find_export(name)?;
                let (params, results) = inst.func_arity(index)?;
                Some((index, params, results))
            })?,
        };
        Some(Func {
            index,
            params,
            results,
        })
    }

    /// Call a resolved export, writing results into `results`.
    ///
    /// No name lookup and no allocation for the return values, so a hot
    /// call loop pays for neither.
    pub fn call(
        &mut self,
        func: &Func,
        args: &[Value],
        results: &mut [Value],
    ) -> Result<(), WasmError> {
        if args.len() != func.params || results.len() != func.results {
            return Err(WasmError::invalid("argument/result arity mismatch"));
        }
        match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.call_function_index(func.index, args, results),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => interp_imports::call_by_index(
                inst.checkout_for_invocation()?,
                func.index,
                args,
                results,
            ),
        }
    }

    pub fn has_function_export(&self, name: &str) -> bool {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.has_function_export(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.find_export(name).is_some()),
        }
    }

    /// The first linear memory's contents, if the module defines one.
    pub fn memory(&self) -> Option<&[u8]> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.memory(),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.memory(),
        }
    }

    pub fn memory_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.memory_mut(),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.memory_mut(),
        }
    }

    // --- The surface the spec runner drives both engines through ---
    //
    // Each of these delegates to the JIT unchanged. The interpreter answers
    // what it can and returns a named error otherwise, so the shared runner
    // reports a real failure rather than quietly skipping -- which is the
    // only way the gap between the engines stays visible.

    /// Call an exported function by index.
    pub fn invoke_function_index(
        &mut self,
        idx: usize,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.invoke_function_index(idx, args),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => {
                interp_imports::invoke_by_index(inst.checkout_for_invocation()?, idx, args)
            }
        }
    }

    /// A global's value by index.
    pub fn global_at(&self, idx: usize) -> Result<Option<Value>, WasmError> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.global_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| interp_imports::global_at(inst, idx)),
        }
    }

    /// Overwrite a global by index.
    pub fn replace_global_at(&mut self, idx: usize, value: Value) -> Result<(), WasmError> {
        match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.replace_global_at(idx, value),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => {
                inst.with_instance_mut(|inst| interp_imports::replace_global_at(inst, idx, value))
            }
        }
    }

    /// An exported global's value by name.
    pub fn get_global(&self, name: &str) -> Result<Option<Value>, WasmError> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.get_global(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| match inst.find_export_global(name) {
                Some(idx) => interp_imports::global_at(inst, idx),
                None => Ok(None),
            }),
        }
    }

    /// The declared type of a function by index.
    pub fn function_type_at(&self, idx: usize) -> Option<FunctionType> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_type_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| {
                inst.module()
                    .functions()
                    .get(idx)
                    .map(|f| f.func_type().clone())
            }),
        }
    }

    /// The payload of an exception, by the handle a `WasmError::Exception`
    /// carried out.
    ///
    /// An uncaught exception surfaces with a reference handle, its tag, and
    /// the module's name for that tag. The handle alone is opaque to an
    /// embedder, so this resolves it to the values the throw carried.
    /// Exception objects are registry-owned, so this keeps working after the
    /// instance that threw has been dropped.
    pub fn exception_fields(&self, exn: RefValue) -> Option<collections::Vec<Value>> {
        self.registry
            .arenas()
            .resolve_exn(exn)
            .map(|instance| instance.fields.clone())
    }

    /// An absolute reference handle for a function, suitable for crossing
    /// instance boundaries.
    pub fn function_handle_at(&self, idx: usize) -> Option<RefValue> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_handle_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.function_handle_at(idx)),
        }
    }

    /// The type index of a function, for cross-instance identity checks.
    pub fn function_type_index_at(&self, idx: usize) -> Option<u32> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_type_index_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst
                .with_instance(|inst| inst.module().functions().get(idx).map(|f| f.type_index())),
        }
    }

    /// Page count of an exported memory.
    pub fn memory_pages(&self, name: &str) -> Option<usize> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.memory_pages(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| {
                // By EXPORT NAME, not memory 0: a module with several
                // exported memories would otherwise report the first one's
                // size for all of them, and a caller sizing an import from
                // that gets the wrong limits.
                let idx = inst
                    .module()
                    .memories()
                    .iter()
                    .position(|m| m.export_names().iter().any(|export| export == name))?;
                inst.shared_memory_at(idx)
                    .map(|m| m.memory_len() / crate::constants::WASM_PAGE_SIZE)
            }),
        }
    }

    /// Element count of an exported table.
    pub fn table_size(&self, name: &str) -> Option<usize> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.table_size(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| {
                // By export name and from the LIVE table, as memory_pages
                // does: a table grown after instantiation must report its
                // current size, or an import sized from it gets stale limits.
                let idx = inst
                    .module()
                    .tables()
                    .iter()
                    .position(|t| t.export_names().iter().any(|e| e == name))?;
                inst.table_len(idx)
            }),
        }
    }

    /// A tag handle for exception handling.
    pub fn tag_identity(&self, name: &str) -> Option<TagIdentity> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.tag_identity(name),
            // Tag IDENTITY is a linking concern, so the interpreter mints
            // handles even though it cannot yet throw or catch.
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| {
                let idx = inst
                    .module()
                    .tags()
                    .iter()
                    .position(|t| t.export_names().iter().any(|e| e == name))?;
                inst.tag_identity_at(idx)
            }),
        }
    }

    /// Shared entity state, for linking one instance's exports into
    /// another's imports. The interpreter does not link yet, so it offers
    /// nothing to share.
    pub fn shared_memory_at(&self, idx: usize) -> Option<MemInst> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.shared_memory_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.shared_memory_at(idx)),
        }
    }

    pub fn shared_table_state_at(&self, idx: usize) -> Option<ImportedTableState> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.shared_table_state_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| {
                inst.table_state_at(idx).map(|table| ImportedTableState {
                    table,
                    type_ctx: None,
                })
            }),
        }
    }

    pub fn shared_global_state_at(&self, idx: usize) -> Option<ImportedGlobalState> {
        match &self.inner {
            #[cfg(sf_interp)]
            // With the exporter's type context: a reference type's concrete
            // heap type names an index in THIS module's type space, and the
            // importer cannot resolve it otherwise.
            Inner::Interp(inst) => inst.with_instance(|inst| {
                inst.global_state_at(idx).map(|global| ImportedGlobalState {
                    global,
                    type_ctx: Some(inst.module().types().clone()),
                })
            }),
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.shared_global_state_at(idx),
        }
    }

    /// Append a host function after instantiation, for the spec runner's
    /// late-bound imports.
    pub fn append_host_function<F>(&mut self, func_type: FunctionType, callback: F) -> usize
    where
        F: for<'a, 'b, 'c, 'd> Fn(
                &'a mut Caller<'b>,
                &'c [Value],
                &'d mut [Value],
            ) -> Result<(), WasmError>
            + 'static,
    {
        match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.append_host_function(func_type, callback),
            #[cfg(sf_interp)]
            Inner::Interp(_) => {
                let _ = (func_type, callback);
                usize::MAX
            }
        }
    }

    /// The JIT's instance, for the entity-model surface the interpreter has
    /// no counterpart for. `None` when this instance is on another engine.
    #[cfg(sf_jit)]
    #[inline]
    pub fn as_jit(&self) -> Option<&JitInstanceLease> {
        match &self.inner {
            Inner::Jit(inst) => Some(inst),
            #[cfg(sf_interp)]
            Inner::Interp(_) => None,
        }
    }

    #[cfg(sf_jit)]
    #[inline]
    pub fn as_jit_mut(&mut self) -> Option<&mut JitInstanceLease> {
        match &mut self.inner {
            Inner::Jit(inst) => Some(inst),
            #[cfg(sf_interp)]
            Inner::Interp(_) => None,
        }
    }

    /// Scoped access to the interpreter's instance. `None` when this
    /// instance is on another engine.
    ///
    /// The higher-ranked closure prevents a reference derived from the
    /// instance-table token from escaping this call.
    #[cfg(sf_interp)]
    #[inline]
    pub fn with_interp<R>(
        &self,
        use_instance: impl for<'instance> FnOnce(&'instance InterpInstance) -> R,
    ) -> Option<R> {
        match &self.inner {
            Inner::Interp(inst) => Some(inst.with_instance(use_instance)),
            #[cfg(sf_jit)]
            Inner::Jit(_) => None,
        }
    }

    #[cfg(sf_interp)]
    #[inline]
    pub fn with_interp_mut<R>(
        &mut self,
        use_instance: impl for<'instance> FnOnce(&'instance mut InterpInstance) -> R,
    ) -> Option<R> {
        match &mut self.inner {
            Inner::Interp(inst) => Some(inst.with_instance_mut(use_instance)),
            #[cfg(sf_jit)]
            Inner::Jit(_) => None,
        }
    }

    /// Test-only checked-out interpreter access for exercising an alternate
    /// driver through the same re-entrant instance-table boundary as `invoke`.
    #[cfg(all(test, sf_interp))]
    pub(crate) fn checkout_interp_for_test(
        &self,
    ) -> Result<crate::vm::link::InstanceToken, WasmError> {
        match &self.inner {
            Inner::Interp(instance) => instance.checkout_for_invocation(),
            #[cfg(sf_jit)]
            Inner::Jit(_) => Err(WasmError::invalid("instance is not an interpreter")),
        }
    }
}

impl RuntimeWorld {
    /// An empty world. The first instantiation fixes its tier.
    #[inline]
    pub fn new() -> Self {
        Self {
            registry: LinkRegistry::new(),
            instances: collections::Vec::new(),
            tier: None,
        }
    }

    fn from_registry(registry: LinkRegistry) -> Self {
        Self {
            registry,
            instances: collections::Vec::new(),
            tier: None,
        }
    }

    fn take(&mut self, id: InstanceId) -> Option<Instance> {
        let index = self
            .instances
            .iter()
            .position(|(candidate, _)| *candidate == id)?;
        self.instances.remove(index).1
    }

    /// A cheap, clonable capability for indexed calls into this world.
    ///
    /// The handle does not borrow or keep the world alive. This is the
    /// callback shape that has to work: guest code calls a host function while
    /// the embedder still holds `&mut RuntimeWorld`, and that host function
    /// calls a runtime-chosen peer. Every call through the handle performs a
    /// fresh generation-checked checkout.
    pub fn handle(&self) -> WorldAccess {
        self.registry.instance_table().world_access()
    }

    /// Instantiate `module` into this world, returning its id.
    ///
    /// Every instance in a world runs on the same engine: they resolve each
    /// other's function identities out of one address space, and the engines
    /// do not share a call path for those. Mixing tiers is rejected rather
    /// than half-supported.
    #[inline]
    pub fn instantiate(
        &mut self,
        engine: &Engine,
        module: Module,
        imports: &[Import],
    ) -> Result<InstanceId, InstanceInstantiationError> {
        match self.tier {
            Some(tier) if tier != engine.tier() => {
                return Err(InstanceInstantiationError::Complete(WasmError::invalid(
                    "a runtime world runs one engine; this world is already instantiated on the other tier",
                )));
            }
            _ => {}
        }
        match Instance::from_module_in_registry(engine, module, imports, &self.registry) {
            Ok(instance) => {
                let id = instance.instance_id();
                self.instances.push((id, Some(instance)));
                self.tier = Some(engine.tier());
                Ok(id)
            }
            Err(InstanceInstantiationError::Partial { id, error }) => {
                // The slot is occupied, so the tier is fixed even though no
                // usable facade came back.
                self.instances.push((id, None));
                self.tier = Some(engine.tier());
                Err(InstanceInstantiationError::Partial { id, error })
            }
            Err(error) => Err(error),
        }
    }

    /// The instance behind `id`, for reaching its exports when linking a
    /// later module against it.
    ///
    /// `None` once the id stops resolving, and also for a slot left occupied
    /// by a partial instantiation, which has no usable facade.
    #[inline]
    pub fn instance(&self, id: InstanceId) -> Option<&Instance> {
        self.instances
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .and_then(|(_, instance)| instance.as_ref())
    }

    /// Mutable access to the instance facade behind `id`.
    pub fn instance_mut(&mut self, id: InstanceId) -> Option<&mut Instance> {
        self.instances
            .iter_mut()
            .find(|(candidate, _)| *candidate == id)
            .and_then(|(_, instance)| instance.as_mut())
    }

    /// Drop an instance and retire its slot. Its id stops resolving.
    ///
    /// Fails while the instance is checked out — that is, while a call into
    /// it is on the stack.
    #[inline]
    pub fn free(&mut self, id: InstanceId) -> Result<(), WasmError> {
        let index = self
            .instances
            .iter()
            .position(|(candidate, _)| *candidate == id)
            .ok_or_else(|| WasmError::invalid("unknown runtime-world instance"))?;
        if let Some(instance) = self.instances[index].1.as_ref() {
            if !instance.has_exclusive_lease() {
                return Err(WasmError::invalid(
                    "cannot free a checked-out runtime-world instance",
                ));
            }
            let (_, instance) = self.instances.remove(index);
            drop(instance);
            return Ok(());
        }

        match self.registry.instance_table().free(id) {
            Ok(()) => {}
            Err(InstanceFreeError::InUse) => {
                return Err(WasmError::invalid(
                    "cannot free a checked-out runtime-world instance",
                ));
            }
            Err(InstanceFreeError::InvalidId) => {
                return Err(WasmError::invalid("unknown runtime-world instance"));
            }
        }
        self.instances.remove(index);
        Ok(())
    }

    /// Call an exported function on one instance of this world.
    ///
    /// The callee is checked out of the instance table and this borrow of the
    /// world ends before it runs, so a cross-instance call made from inside
    /// the callee re-enters through a fresh checkout rather than nesting.
    pub fn invoke(
        &mut self,
        id: InstanceId,
        name: &str,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        let token = self
            .registry
            .instance_table()
            .checkout(id)
            .ok_or_else(|| WasmError::invalid("unknown runtime-world instance"))?;
        #[cfg(sf_jit)]
        if token.jit().is_some() {
            return JitInstanceLease::invoke_token(token, name, args);
        }
        #[cfg(sf_interp)]
        if token.interp().is_some() {
            return interp_imports::invoke_by_name(token, name, args);
        }
        Err(WasmError::invalid(
            "runtime-world instance has no enabled engine",
        ))
    }
}

impl WorldAccess {
    /// Invoke a function by its instance identity and local function index.
    ///
    /// This takes only `&self`: a host callback can therefore call back into
    /// the world without aliasing the embedder's live mutable borrow of the
    /// [`RuntimeWorld`] facade. The weak table reference is upgraded only for
    /// this call, and `checkout` rejects an expired world, a freed instance,
    /// and a stale generation with an ordinary error.
    pub fn invoke(
        &self,
        id: InstanceId,
        function_index: usize,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        let token = self
            .checkout(id)
            .ok_or_else(|| WasmError::invalid("unknown runtime-world instance"))?;
        #[cfg(sf_jit)]
        if token.jit().is_some() {
            return JitInstanceLease::invoke_function_index_token(token, function_index, args);
        }
        #[cfg(sf_interp)]
        if token.interp().is_some() {
            return interp_imports::invoke_by_index(token, function_index, args);
        }
        Err(WasmError::invalid(
            "runtime-world instance has no enabled engine",
        ))
    }
}

impl Drop for RuntimeWorld {
    fn drop(&mut self) {
        while let Some((id, _)) = self.instances.last() {
            let id = *id;
            if self.free(id).is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::vm::engine::Tier;
    use core::cell::Cell;
    use std::rc::Rc;

    const ADD_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f,
        0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    ];
    #[cfg(feature = "memprof")]
    const EMPTY_WASM: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn interp_validation_retains_one_plan_and_still_predecodes_every_function() {
        let _guard = crate::vm::interpreter::baseline_artifact_guard_for_test();
        let wasm = wat::parse_str(
            r#"(module
                (func (export "countdown") (param $n i32) (result i32)
                    block $exit
                        loop $again
                            local.get $n
                            i32.eqz
                            br_if $exit
                            local.get $n
                            i32.const 1
                            i32.sub
                            local.set $n
                            br $again
                        end
                    end
                    local.get $n)
                (func (export "cold") (result i32) i32.const 7))"#,
        )
        .expect("encode baseline plan module");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let before = crate::vm::interpreter::baseline_artifact_build_count_for_test();
        let mut instance = Instance::from_module(
            &engine,
            Module::new("retained-baseline-plan", &wasm).expect("parse module"),
            &[],
        )
        .expect("instantiate interpreter");
        assert_eq!(
            crate::vm::interpreter::baseline_artifact_build_count_for_test(),
            before + 1,
            "one interpreter instance must publish exactly one composite artifact"
        );

        let interp = match &instance.inner {
            Inner::Interp(interp) => interp,
            #[cfg(sf_jit)]
            Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
        };
        interp.with_instance(|interp| {
            assert_eq!(interp.baseline_plan_label_for_test(0), Some("hybrid"));
            assert_eq!(interp.baseline_plan_label_for_test(1), Some("raw"));
            assert_eq!(
                interp.baseline_execution_plan_label_for_test(0),
                Some("full-fold")
            );
            assert_eq!(
                interp.baseline_execution_plan_label_for_test(1),
                Some("raw")
            );
            assert_eq!(
                interp.baseline_artifact_has_function_for_test(0),
                Some(true)
            );
            assert_eq!(
                interp.baseline_artifact_has_function_for_test(1),
                Some(true)
            );
            assert!(
                interp.predecode_matches_runtime_plan_for_test(),
                "published runtime entries must match the effective plan"
            );
        });
        assert_eq!(
            instance
                .invoke("countdown", &[Value::I32(4)])
                .expect("effective mixed execution remains active"),
            collections::vec![Value::I32(0)]
        );
    }

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn production_selector_drives_whole_function_mixed_execution() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (data (i32.const 0) "\07\00\00\00")
                (global $state (mut i32) (i32.const 0))
                (func $raw_add (export "raw_root") (param i32) (result i32)
                    local.get 0 i32.const 2 i32.add)
                (func $fold_add (param i32) (result i32)
                    ref.null func drop
                    local.get 0 i32.const 3 i32.add)
                (func (export "fold_to_raw") (param i32) (result i32)
                    ref.null func drop
                    local.get 0 call $raw_add i32.const 5 i32.add)
                (func (export "raw_to_fold") (param i32) (result i32)
                    local.get 0 call $fold_add i32.const 7 i32.add)
                (func (export "memory") (result i32)
                    i32.const 0 i32.load)
                (func (export "global") (param i32) (result i32)
                    local.get 0 global.set $state
                    global.get $state)
                (func (export "multi") (param i32) (result i32 i32)
                    local.get 0
                    local.get 0 i32.const 1 i32.add))"#,
        )
        .expect("encode production mixed module");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let mut instance = Instance::from_module(
            &engine,
            Module::new("production-whole-mixed", &wasm).expect("parse module"),
            &[],
        )
        .expect("instantiate production mixed module");

        let interp = match &instance.inner {
            Inner::Interp(interp) => interp,
            #[cfg(sf_jit)]
            Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
        };
        interp.with_instance(|interp| {
            assert_eq!(
                (0..7)
                    .map(|index| interp.baseline_execution_plan_label_for_test(index))
                    .collect::<collections::Vec<_>>(),
                [
                    Some("raw"),
                    Some("full-fold"),
                    Some("full-fold"),
                    Some("raw"),
                    Some("raw"),
                    Some("raw"),
                    Some("raw"),
                ]
            );
            assert!(
                interp.whole_direct_call_is_slow_for_test(2, 0),
                "a folded caller must enter a Raw callee through the slow checkpoint"
            );
            assert!(
                interp.predecode_matches_runtime_plan_for_test(),
                "selective predecode must match the effective runtime plan"
            );
        });

        for (export, args, expected) in [
            ("raw_root", &[Value::I32(40)][..], &[Value::I32(42)][..]),
            ("fold_to_raw", &[Value::I32(11)][..], &[Value::I32(18)][..]),
            ("raw_to_fold", &[Value::I32(13)][..], &[Value::I32(23)][..]),
            ("memory", &[][..], &[Value::I32(7)][..]),
            ("global", &[Value::I32(17)][..], &[Value::I32(17)][..]),
            (
                "multi",
                &[Value::I32(19)][..],
                &[Value::I32(19), Value::I32(20)][..],
            ),
        ] {
            assert_eq!(
                instance.invoke(export, args).expect("mixed invocation"),
                expected,
                "{export}"
            );
        }

        #[cfg(not(feature = "memprof"))]
        {
            let lease = match &instance.inner {
                Inner::Interp(lease) => lease,
                #[cfg(sf_jit)]
                Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
            };
            let raw_to_fold = lease
                .with_instance(|interp| interp.find_export("raw_to_fold"))
                .expect("mixed export");
            let invoke = |input, result: &mut [u64; 1]| {
                let mut access = crate::vm::interpreter::InterpInstanceAccess::checked_out(
                    lease.checkout_for_invocation()?,
                );
                InterpInstance::invoke_access(&mut access, raw_to_fold, &[input], result)
            };
            let mut result = [0];
            invoke(23, &mut result).expect("warm mixed path");
            let (outcome, census) = crate::test_alloc::measure(|| invoke(29, &mut result));
            outcome.expect("warm mixed invocation");
            assert_eq!(census, crate::test_alloc::Census::default());
            assert_eq!(result, [39]);

            // The public typed adapter currently materializes one raw argument
            // and one raw result Vec for both folded and mixed calls. Keep that
            // pre-existing boundary cost out of the driver's zero-allocation
            // assertion, while proving that mixed routing adds nothing to it.
            let mixed_func = instance.get_func("raw_to_fold").expect("mixed export");
            let mut mixed_result = [Value::I32(0)];
            instance
                .call(&mixed_func, &[Value::I32(23)], &mut mixed_result)
                .expect("warm public mixed path");
            let (mixed_outcome, mixed_census) = crate::test_alloc::measure(|| {
                instance.call(&mixed_func, &[Value::I32(29)], &mut mixed_result)
            });
            mixed_outcome.expect("public mixed invocation");

            let mut folded = Instance::from_module_full_fold_for_test(
                &engine,
                Module::new("production-whole-mixed-folded", &wasm).expect("parse oracle"),
                &[],
            )
            .expect("instantiate folded oracle");
            let folded_func = folded.get_func("raw_to_fold").expect("folded export");
            let mut folded_result = [Value::I32(0)];
            folded
                .call(&folded_func, &[Value::I32(23)], &mut folded_result)
                .expect("warm public folded path");
            let (folded_outcome, folded_census) = crate::test_alloc::measure(|| {
                folded.call(&folded_func, &[Value::I32(29)], &mut folded_result)
            });
            folded_outcome.expect("public folded invocation");
            assert_eq!(mixed_result, folded_result);
            assert!(mixed_census.allocations <= folded_census.allocations);
            assert!(mixed_census.reallocations <= folded_census.reallocations);
            assert!(mixed_census.allocated_bytes <= folded_census.allocated_bytes);
            assert!(mixed_census.reallocated_bytes <= folded_census.reallocated_bytes);
        }
    }

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn production_mixed_plan_keeps_unsupported_call_boundaries_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (import "host" "noop" (func $host))
                (table 1 funcref)
                (func $recursive (export "recursive") (param i32) (result i32)
                    local.get 0
                    i32.eqz
                    if (result i32)
                        i32.const 0
                    else
                        local.get 0 i32.const 1 i32.sub call $recursive
                    end)
                (func $tail_target (type $unary)
                    local.get 0 i32.const 1 i32.add)
                (func (export "tail") (type $unary)
                    local.get 0 return_call $tail_target)
                (func $indirect_target (type $unary)
                    local.get 0 i32.const 2 i32.add)
                (elem (i32.const 0) func $indirect_target)
                (func (export "indirect") (type $unary)
                    local.get 0 i32.const 0 call_indirect (type $unary))
                (func (export "import_call") (result i32)
                    call $host i32.const 9)
                (func (export "raw") (type $unary)
                    local.get 0 i32.const 3 i32.add))"#,
        )
        .expect("encode mixed boundary module");
        let import = Import::func("host", "noop", |_caller, _args, _results| Ok(()));
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let mut instance = Instance::from_module(
            &engine,
            Module::new("production-mixed-boundaries", &wasm).expect("parse module"),
            &[import],
        )
        .expect("instantiate mixed boundary module");

        let interp = match &instance.inner {
            Inner::Interp(interp) => interp,
            #[cfg(sf_jit)]
            Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
        };
        interp.with_instance(|interp| {
            assert_eq!(
                (0..8)
                    .map(|index| interp.baseline_execution_plan_label_for_test(index))
                    .collect::<collections::Vec<_>>(),
                [
                    Some("import"),
                    Some("full-fold"),
                    Some("full-fold"),
                    Some("full-fold"),
                    Some("raw"),
                    Some("full-fold"),
                    Some("full-fold"),
                    Some("raw"),
                ]
            );
        });

        for (export, input, expected) in [
            ("recursive", 4, 0),
            ("tail", 11, 12),
            ("indirect", 13, 15),
            ("raw", 17, 20),
        ] {
            assert_eq!(
                instance
                    .invoke(export, &[Value::I32(input)])
                    .expect("folded boundary invocation"),
                collections::vec![Value::I32(expected)],
                "{export}"
            );
        }
        assert_eq!(
            instance
                .invoke("import_call", &[])
                .expect("folded import invocation"),
            collections::vec![Value::I32(9)]
        );
    }

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn selective_predecode_publishes_explicit_entries_and_no_raw_cells() {
        let wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (import "host" "noop" (func $host))
                (func $raw (export "raw") (type $unary)
                    local.get 0 i32.const 1 i32.add)
                (func $folded_leaf (type $unary)
                    ref.null func drop
                    local.get 0 i32.const 2 i32.add)
                (func (export "folded") (type $unary)
                    ref.null func drop
                    local.get 0 call $folded_leaf i32.const 3 i32.add))"#,
        )
        .expect("encode selective module");
        let import = Import::func("host", "noop", |_caller, _args, _results| Ok(()));
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let mut instance = Instance::from_module(
            &engine,
            Module::new("selective-predecode", &wasm).expect("parse module"),
            &[import],
        )
        .expect("instantiate selective module");

        let interp = match &instance.inner {
            Inner::Interp(interp) => interp,
            #[cfg(sf_jit)]
            Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
        };
        interp.with_instance(|interp| {
            assert_eq!(
                (0..4)
                    .map(|index| interp.function_runtime_label_for_test(index))
                    .collect::<collections::Vec<_>>(),
                [Some("import"), Some("raw"), Some("folded"), Some("folded"),]
            );
            assert_eq!(
                interp.function_instr_and_cell_count_for_test(0),
                Some((0, 0))
            );
            assert_eq!(
                interp.function_instr_and_cell_count_for_test(1),
                Some((0, 0)),
                "Raw must allocate neither Instr nor DCell storage"
            );
            for function in [2, 3] {
                let (instructions, cells) = interp
                    .function_instr_and_cell_count_for_test(function)
                    .expect("Folded census");
                assert!(instructions > 0, "Folded function has no instructions");
                assert_eq!(cells, instructions + 1, "Folded prefetch pad is missing");
            }
            assert!(!interp.function_has_native_range_for_test(0));
            assert!(!interp.function_has_native_range_for_test(1));
            assert!(interp.function_has_native_range_for_test(2));
            assert!(interp.function_has_native_range_for_test(3));
            assert!(
                interp.whole_direct_call_is_native_for_test(3, 2),
                "Folded-to-Folded direct calls must retain native fixup"
            );
        });

        assert_eq!(
            instance
                .invoke("raw", &[Value::I32(40)])
                .expect("Raw invocation"),
            collections::vec![Value::I32(41)]
        );
        assert_eq!(
            instance
                .invoke("folded", &[Value::I32(7)])
                .expect("Folded invocation"),
            collections::vec![Value::I32(12)]
        );
    }

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn selective_predecode_executes_a_raw_start_function() {
        let wasm = wat::parse_str(
            r#"(module
                (global $state (mut i32) (i32.const 0))
                (func $start
                    i32.const 9 global.set $state)
                (start $start)
                (func (export "read") (result i32)
                    global.get $state))"#,
        )
        .expect("encode Raw start module");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let mut instance = Instance::from_module(
            &engine,
            Module::new("selective-raw-start", &wasm).expect("parse module"),
            &[],
        )
        .expect("run Raw start");
        let interp = match &instance.inner {
            Inner::Interp(interp) => interp,
            #[cfg(sf_jit)]
            Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
        };
        interp.with_instance(|interp| {
            assert_eq!(interp.function_runtime_label_for_test(0), Some("raw"));
            assert_eq!(
                interp.function_instr_and_cell_count_for_test(0),
                Some((0, 0))
            );
        });
        assert_eq!(
            instance
                .invoke("read", &[])
                .expect("read initialized global"),
            collections::vec![Value::I32(9)]
        );
    }

    #[cfg(all(sf_interp, sf_module_validator, not(feature = "memprof")))]
    #[test]
    fn selective_predecode_reduces_instance_allocation_census() {
        let wasm = wat::parse_str(
            r#"(module
                (func (export "run") (param $x i32) (result i32)
                    local.get $x i32.const 1 i32.add
                    i32.const 2 i32.mul
                    i32.const 3 i32.add
                    i32.const 4 i32.mul
                    i32.const 5 i32.add
                    i32.const 6 i32.mul
                    i32.const 7 i32.add
                    i32.const 8 i32.mul
                    i32.const 9 i32.add
                    i32.const 10 i32.mul
                    i32.const 11 i32.add
                    i32.const 12 i32.mul))"#,
        )
        .expect("encode allocation census module");
        let raw_module = Module::new("selective-allocation-raw", &wasm).expect("parse Raw module");
        let folded_module =
            Module::new("selective-allocation-folded", &wasm).expect("parse Folded module");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");

        let (raw, raw_census) =
            crate::test_alloc::measure(|| Instance::from_module(&engine, raw_module, &[]));
        let raw = raw.expect("instantiate selective Raw module");
        let (folded, folded_census) = crate::test_alloc::measure(|| {
            Instance::from_module_full_fold_for_test(&engine, folded_module, &[])
        });
        let folded = folded.expect("instantiate forced Folded module");

        assert!(
            raw_census.allocations < folded_census.allocations,
            "Raw should remove the Instr/DCell arena allocations: Raw={raw_census:?}, Folded={folded_census:?}"
        );
        assert!(
            raw_census.allocated_bytes < folded_census.allocated_bytes,
            "Raw should retain fewer startup bytes: Raw={raw_census:?}, Folded={folded_census:?}"
        );
        let assert_storage = |instance: &Instance, expected_raw: bool| {
            let interp = match &instance.inner {
                Inner::Interp(interp) => interp,
                #[cfg(sf_jit)]
                Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
            };
            interp.with_instance(|interp| {
                let (instructions, cells) = interp
                    .function_instr_and_cell_count_for_test(0)
                    .expect("function census");
                if expected_raw {
                    assert_eq!((instructions, cells), (0, 0));
                } else {
                    assert!(instructions > 0);
                    assert_eq!(cells, instructions + 1);
                }
            });
        };
        assert_storage(&raw, true);
        assert_storage(&folded, false);
    }

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn dynamic_tail_sites_keep_every_local_runtime_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (table 1 funcref)
                (func $target (type $unary) local.get 0)
                (elem (i32.const 0) func $target)
                (func (export "tail_indirect") (type $unary)
                    local.get 0 i32.const 0 return_call_indirect (type $unary))
                (func (export "tail_ref") (type $unary)
                    local.get 0 ref.func $target return_call_ref $unary)
                (func (export "otherwise_raw") (type $unary)
                    local.get 0 i32.const 1 i32.add))"#,
        )
        .expect("encode dynamic tail module");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let instance = Instance::from_module(
            &engine,
            Module::new("dynamic-tail-folded", &wasm).expect("parse module"),
            &[],
        )
        .expect("instantiate dynamic tail module");
        let interp = match &instance.inner {
            Inner::Interp(interp) => interp,
            #[cfg(sf_jit)]
            Inner::Jit(_) => panic!("explicit interpreter engine selected another tier"),
        };
        interp.with_instance(|interp| {
            for function in 0..4 {
                assert_eq!(
                    interp.function_runtime_label_for_test(function),
                    Some("folded"),
                    "dynamic tail target set is not closed at function {function}"
                );
            }
        });
    }

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn invalid_interp_module_does_not_publish_a_baseline_artifact() {
        let _guard = crate::vm::interpreter::baseline_artifact_guard_for_test();
        let wasm = wat::parse_str("(module (global i32 (f64.const 0)) (func nop))")
            .expect("encode invalid suffix module");
        let module = Module::new("invalid-baseline-suffix", &wasm).expect("parse module");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let before = crate::vm::interpreter::baseline_artifact_build_count_for_test();
        assert!(
            Instance::from_module(&engine, module, &[]).is_err(),
            "validation must reject module"
        );
        assert_eq!(
            crate::vm::interpreter::baseline_artifact_build_count_for_test(),
            before,
            "failed validation must not publish staged artifact state"
        );
    }

    #[cfg(all(sf_interp, sf_module_validator))]
    #[test]
    fn interp_validation_plans_artifact_ineligible_functions_for_full_fold() {
        fn assert_plans(wat: &str, expected: &[&str]) {
            let wasm = wat::parse_str(wat).expect("encode plan fixture");
            let module = Module::new("full-fold-plan", &wasm).expect("parse plan fixture");
            let baseline = crate::vm::interpreter::ValidatedBaselinePlan::validate(&module)
                .expect("validate and plan fixture");
            for (index, &expected) in expected.iter().enumerate() {
                assert_eq!(
                    baseline.plan_label_for_test(index),
                    Some(expected),
                    "function {index}"
                );
            }
        }

        assert_plans(
            r#"(module
                (tag $e)
                (func nop)
                (func throw $e)
                (func (param exnref) local.get 0 throw_ref))"#,
            &["raw", "full-fold", "full-fold"],
        );
        assert_plans(
            r#"(module
                (type $s (struct))
                (func nop)
                (func (result (ref $s)) struct.new $s))"#,
            &["raw", "full-fold"],
        );
        #[cfg(sf_has_simd)]
        assert_plans(
            r#"(module
                (func nop)
                (func (result v128) v128.const i32x4 0 0 0 0))"#,
            &["raw", "full-fold"],
        );
    }

    #[test]
    fn runtime_world_invokes_and_frees_by_generation_checked_id() {
        let mut world = RuntimeWorld::new();
        let module = Module::new("runtime-world-add", ADD_WASM).expect("parse add module");
        let id = world
            .instantiate(&Engine::with_defaults(), module, &[])
            .expect("instantiate in world");

        assert_eq!(
            world
                .invoke(id, "add", &[Value::I32(3), Value::I32(4)])
                .expect("invoke through world"),
            collections::vec![Value::I32(7)]
        );

        let checkout = world
            .registry
            .instance_table()
            .checkout(id)
            .expect("instance remains occupied");
        assert!(world.free(id).is_err());
        drop(checkout);
        world.free(id).expect("free after checkout ends");
        assert!(world.invoke(id, "add", &[]).is_err());
    }

    /// The embedder's `&mut RuntimeWorld` receiver stays live for the outer
    /// invocation while `a` enters a host callback. The callback owns only a
    /// weak world handle, so it can check out a peer or re-enter `a` itself
    /// without reconstructing or aliasing that mutable borrow.
    #[test]
    fn world_handle_reenters_from_a_host_callback() {
        let provider_wasm = wat::parse_str(
            r#"
            (module
              (func (export "add_forty") (param i32) (result i32)
                local.get 0
                i32.const 40
                i32.add))
            "#,
        )
        .expect("encode handle provider");
        let caller_wasm = wat::parse_str(
            r#"
            (module
              (func $call_b (import "host" "call_b") (param i32) (result i32))
              (func (export "run_a") (param i32) (result i32)
                local.get 0
                call $call_b
                i32.const 1
                i32.add))
            "#,
        )
        .expect("encode handle caller");
        let self_caller_wasm = wat::parse_str(
            r#"
            (module
              (func $call_self (import "host" "call_self") (param i32) (result i32))
              (func (export "nested") (param i32) (result i32)
                local.get 0
                i32.const 40
                i32.add)
              (func (export "run_self") (param i32) (result i32)
                local.get 0
                call $call_self
                i32.const 1
                i32.add))
            "#,
        )
        .expect("encode same-instance handle caller");
        let callback_type = FunctionType::new(
            collections::vec![crate::value_type::ValueType::I32],
            collections::vec![crate::value_type::ValueType::I32],
        );

        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).expect("engine config");
            let mut world = RuntimeWorld::new();
            let handle = world.handle();
            let b = world
                .instantiate(
                    &engine,
                    Module::new("handle-provider-b", &provider_wasm).expect("parse provider"),
                    &[],
                )
                .expect("instantiate provider b");
            let callback_handle = handle.clone();
            let host = Import::func_typed(
                "host",
                "call_b",
                move |_caller, args, results| {
                    let returned = callback_handle.invoke(b, 0, args)?;
                    if returned.len() != results.len() {
                        return Err(WasmError::invalid("argument/result arity mismatch"));
                    }
                    results.copy_from_slice(&returned);
                    Ok(())
                },
                callback_type.clone(),
            );
            let a = world
                .instantiate(
                    &engine,
                    Module::new("handle-caller-a", &caller_wasm).expect("parse caller"),
                    &[host],
                )
                .expect("instantiate caller a");

            assert_eq!(
                world
                    .invoke(a, "run_a", &[Value::I32(2)])
                    .expect("a host callback invokes b"),
                collections::vec![Value::I32(43)],
                "{tier:?}: host callback did not reach b through the handle"
            );

            let self_id = Rc::new(Cell::new(None));
            let callback_id = Rc::clone(&self_id);
            let callback_handle = handle.clone();
            let self_host = Import::func_typed(
                "host",
                "call_self",
                move |_caller, args, results| {
                    let id = callback_id.get().ok_or_else(|| {
                        WasmError::internal("same-instance callback id is not initialized")
                    })?;
                    let returned = callback_handle.invoke(id, 1, args)?;
                    if returned.len() != results.len() {
                        return Err(WasmError::invalid("argument/result arity mismatch"));
                    }
                    results.copy_from_slice(&returned);
                    Ok(())
                },
                callback_type.clone(),
            );
            let self_caller = world
                .instantiate(
                    &engine,
                    Module::new("handle-self-caller", &self_caller_wasm)
                        .expect("parse same-instance caller"),
                    &[self_host],
                )
                .expect("instantiate same-instance caller");
            self_id.set(Some(self_caller));

            assert_eq!(
                world
                    .invoke(self_caller, "run_self", &[Value::I32(2)])
                    .expect("a host callback re-enters the current instance"),
                collections::vec![Value::I32(43)],
                "{tier:?}: host callback did not re-enter the current slot"
            );

            drop(world);
            assert!(
                handle.invoke(b, 0, &[Value::I32(2)]).is_err(),
                "{tier:?}: a handle kept an expired world alive"
            );
        }
    }

    /// Reusing the freed slot makes the stale-id check load-bearing: without
    /// the generation comparison this call would silently reach the
    /// replacement and return its different sentinel.
    #[test]
    fn world_handle_rejects_a_freed_generation_after_slot_reuse() {
        let old_wasm =
            wat::parse_str(r#"(module (func (export "value") (result i32) i32.const 111))"#)
                .expect("encode old handle target");
        let replacement_wasm =
            wat::parse_str(r#"(module (func (export "value") (result i32) i32.const 222))"#)
                .expect("encode replacement handle target");

        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).expect("engine config");
            let mut world = RuntimeWorld::new();
            let handle = world.handle();
            let old = world
                .instantiate(
                    &engine,
                    Module::new("old-handle-target", &old_wasm).expect("parse old target"),
                    &[],
                )
                .expect("instantiate old target");
            world.free(old).expect("free old target");
            let replacement = world
                .instantiate(
                    &engine,
                    Module::new("replacement-handle-target", &replacement_wasm)
                        .expect("parse replacement target"),
                    &[],
                )
                .expect("instantiate replacement target");

            assert_eq!(
                old.index(),
                replacement.index(),
                "{tier:?}: slot not reused"
            );
            assert_ne!(
                old.generation(),
                replacement.generation(),
                "{tier:?}: reused slot kept its generation"
            );
            assert!(
                handle.invoke(old, 0, &[]).is_err(),
                "{tier:?}: stale id misdispatched to the replacement"
            );
            assert_eq!(
                handle
                    .invoke(replacement, 0, &[])
                    .expect("current generation invokes"),
                collections::vec![Value::I32(222)]
            );
        }
    }

    #[test]
    fn runtime_world_registers_only_escapable_functions() {
        let wasm = wat::parse_str(
            r#"
            (module
              (func $hidden)
              (func $code_ref)
              (func $element_declared)
              (func (export "declared_ref") (result funcref)
                ref.func $code_ref)
              ;; An expression-form segment holding only ref.null declares
              ;; nothing: exactness means it must not leak any function
              ;; into the escapable set.
              (elem declare funcref (ref.null func))
              (elem declare func $code_ref)
              (elem declare func $element_declared)
            )
            "#,
        )
        .expect("encode escapable-function module");

        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).expect("engine config");
            let module = Module::new("runtime-world-escapable", &wasm).expect("parse module");
            let mut world = RuntimeWorld::new();
            let id = world
                .instantiate(&engine, module, &[])
                .expect("instantiate in world");

            let instance = &world
                .instances
                .iter()
                .find(|(candidate, _)| *candidate == id)
                .expect("world retains instance")
                .1
                .as_ref()
                .expect("successful instance has a facade");
            assert!(
                instance.function_handle_at(0).is_none(),
                "{tier:?}: hidden function acquired a world address"
            );
            let declared = instance
                .function_handle_at(1)
                .expect("code-section ref.func has a world address");

            let returned = world
                .invoke(id, "declared_ref", &[])
                .expect("execute ref.func");
            assert!(
                matches!(returned.as_slice(), [Value::Ref(handle, _)] if *handle == declared),
                "{tier:?}: ref.func did not return the declared identity"
            );

            let arenas = world.registry.arenas();
            let entries = arenas.functions.borrow();
            assert!(
                entries.iter().all(|entry| entry.owner == id),
                "{tier:?}: registration has the wrong owner"
            );
            assert_eq!(
                entries
                    .iter()
                    .map(|entry| entry.local_index)
                    .collect::<collections::Vec<_>>(),
                collections::vec![1, 2, 3],
                "{tier:?}: escapable function set is not exact"
            );
        }
    }

    /// `ref.func` in a body of a function declared nowhere outside code is
    /// invalid; the escapable set never scans bodies, so the validator must
    /// reject the module rather than the runtime inventing an identity.
    #[cfg(sf_module_validator)]
    #[test]
    fn undeclared_code_ref_func_is_rejected() {
        let wasm = wat::parse_str(
            r#"
            (module
              (func $undeclared)
              (func (export "get") (result funcref)
                ref.func $undeclared)
              ;; Expression-form segments must not over-declare: this one
              ;; names no function, so the ref.func above stays undeclared.
              (elem declare funcref (ref.null func))
            )
            "#,
        )
        .expect("encode undeclared ref.func module");

        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).expect("engine config");
            let module = Module::new("undeclared-ref-func", &wasm).expect("parse module");
            let mut world = RuntimeWorld::new();
            let error = world
                .instantiate(&engine, module, &[])
                .expect_err("undeclared ref.func must fail validation");
            let message = alloc::format!("{:?}", error.error());
            assert!(
                message.contains("undeclared function reference"),
                "{tier:?}: unexpected rejection: {message}"
            );
        }
    }

    #[cfg(all(sf_jit, sf_interp))]
    #[test]
    fn runtime_world_reference_type_answers_match_between_engines() {
        let provider_wasm = wat::parse_str(
            r#"
            (module
              (type $provider (func (param i32) (result i32)))
              (func $target (type $provider) (param i32) (result i32)
                local.get 0)
              (elem declare func $target)
              (func (export "get") (result funcref)
                ref.func $target))
            "#,
        )
        .expect("encode reference provider");
        let consumer_wasm = wat::parse_str(
            r#"
            (module
              ;; The matching type deliberately has a different index from
              ;; the provider's type. Index zero is a same-numbered decoy.
              (type $wrong (func (result i64)))
              (type $right (func (param i32) (result i32)))

              (func (export "test_func") (param funcref) (result i32)
                (ref.test (ref func) (local.get 0)))
              (func (export "test_right") (param funcref) (result i32)
                (ref.test (ref $right) (local.get 0)))
              (func (export "test_wrong") (param funcref) (result i32)
                (ref.test (ref $wrong) (local.get 0)))
              (func (export "cast_right") (param funcref) (result funcref)
                (ref.cast (ref $right) (local.get 0)))
              (func (export "cast_wrong") (param funcref) (result funcref)
                (ref.cast (ref $wrong) (local.get 0)))

              (func (export "test_any") (param (ref null any)) (result i32)
                (ref.test (ref any) (local.get 0)))
              (func (export "cast_any") (param (ref null any))
                (result (ref null any))
                (ref.cast (ref any) (local.get 0)))

              (func (export "test_nullable") (param funcref) (result i32)
                (ref.test (ref null func) (local.get 0)))
              (func (export "test_nonnull") (param funcref) (result i32)
                (ref.test (ref func) (local.get 0)))
              (func (export "cast_nullable") (param funcref) (result funcref)
                (ref.cast (ref null func) (local.get 0)))
              (func (export "cast_nonnull") (param funcref) (result funcref)
                (ref.cast (ref func) (local.get 0))))
            "#,
        )
        .expect("encode reference consumer");

        fn answers_for(
            tier: Tier,
            provider_wasm: &[u8],
            consumer_wasm: &[u8],
        ) -> collections::Vec<i32> {
            fn invoke_i32(
                world: &mut RuntimeWorld,
                consumer: InstanceId,
                tier: Tier,
                name: &str,
                argument: Value,
            ) -> i32 {
                let values = world
                    .invoke(consumer, name, &[argument])
                    .unwrap_or_else(|error| panic!("{tier:?} {name}: {error}"));
                match values.as_slice() {
                    [Value::I32(value)] => *value,
                    other => panic!("{tier:?} {name}: unexpected results {other:?}"),
                }
            }

            fn invoke_ref(
                world: &mut RuntimeWorld,
                consumer: InstanceId,
                tier: Tier,
                name: &str,
                argument: Value,
            ) -> RefValue {
                let values = world
                    .invoke(consumer, name, &[argument])
                    .unwrap_or_else(|error| panic!("{tier:?} {name}: {error}"));
                match values.as_slice() {
                    [Value::Ref(handle, _)] => *handle,
                    other => panic!("{tier:?} {name}: unexpected results {other:?}"),
                }
            }

            fn cast_traps(
                world: &mut RuntimeWorld,
                consumer: InstanceId,
                tier: Tier,
                name: &str,
                argument: Value,
            ) -> i32 {
                let error = world
                    .invoke(consumer, name, &[argument])
                    .expect_err("cast should trap");
                assert_eq!(
                    error,
                    WasmError::Trap("cast failure"),
                    "{tier:?} {name}: wrong failure"
                );
                1
            }

            let engine = Engine::new(Config::new().tier(tier)).expect("engine config");
            let mut world = RuntimeWorld::new();
            let provider = world
                .instantiate(
                    &engine,
                    Module::new("reference-provider", provider_wasm).expect("parse provider"),
                    &[],
                )
                .expect("instantiate provider");
            let consumer = world
                .instantiate(
                    &engine,
                    Module::new("reference-consumer", consumer_wasm).expect("parse consumer"),
                    &[],
                )
                .expect("instantiate consumer");

            let function = world
                .invoke(provider, "get", &[])
                .expect("get provider function")
                .into_iter()
                .next()
                .expect("one provider result");
            let Value::Ref(function_handle, _) = function else {
                panic!("{tier:?}: provider returned a non-reference")
            };
            let host_handle = RefValue::hostref(7);
            let host = Value::Ref(host_handle, crate::value_type::RefType::anyref());
            let null_handle = RefValue::null();
            let null = Value::Ref(null_handle, crate::value_type::RefType::funcref());

            let mut answers = collections::vec![
                invoke_i32(&mut world, consumer, tier, "test_func", function),
                invoke_i32(&mut world, consumer, tier, "test_right", function),
                invoke_i32(&mut world, consumer, tier, "test_wrong", function),
                i32::from(
                    invoke_ref(&mut world, consumer, tier, "cast_right", function)
                        == function_handle,
                ),
                cast_traps(&mut world, consumer, tier, "cast_wrong", function),
                // Host references historically took different paths. Both
                // engines now answer the aggregate `any` type uniformly.
                invoke_i32(&mut world, consumer, tier, "test_any", host),
                i32::from(invoke_ref(&mut world, consumer, tier, "cast_any", host) == host_handle),
                // Nullability remains at the opcode call sites, not in the
                // shared non-null matcher.
                invoke_i32(&mut world, consumer, tier, "test_nullable", null),
                invoke_i32(&mut world, consumer, tier, "test_nonnull", null),
                i32::from(
                    invoke_ref(&mut world, consumer, tier, "cast_nullable", null) == null_handle,
                ),
                cast_traps(&mut world, consumer, tier, "cast_nonnull", null),
            ];

            // Absolute function handles may outlive their owners. Abstract
            // function matching needs only the arena provenance; concrete
            // matching is the arm that checks out the owner's type context.
            world.free(provider).expect("free reference provider");
            answers.push(invoke_i32(
                &mut world,
                consumer,
                tier,
                "test_func",
                function,
            ));
            answers
        }

        let jit = answers_for(Tier::Jit, &provider_wasm, &consumer_wasm);
        let interp = answers_for(Tier::Interp, &provider_wasm, &consumer_wasm);
        assert_eq!(jit, interp, "reference-type answers diverged by engine");
        assert_eq!(jit, collections::vec![1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]);
    }

    /// The design's Embedder API example: `b` calls a function owned by `a`
    /// while `b` is mid-execution. Both instances are checked out at once, and
    /// neither call needs the `&mut RuntimeWorld` the embedder is holding --
    /// `invoke` lets its own borrow end before the callee runs.
    #[test]
    fn runtime_world_invokes_across_instances_mid_execution() {
        let provider = wat::parse_str(
            r#"
            (module
              (func (export "answer") (result i32)
                i32.const 305419896))
            "#,
        )
        .expect("encode provider");
        let consumer = wat::parse_str(
            r#"
            (module
              (func $answer (import "a" "answer") (result i32))
              (func (export "run_b") (result i32)
                call $answer
                i32.const 1
                i32.add))
            "#,
        )
        .expect("encode consumer");

        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).expect("engine config");
            let mut world = RuntimeWorld::new();

            let a = world
                .instantiate(
                    &engine,
                    Module::new("provider", &provider).expect("parse provider"),
                    &[],
                )
                .expect("instantiate provider");

            let (handle, func_type) = {
                let instance = world.instance(a).expect("world retains the provider");
                (
                    instance
                        .function_handle_at(0)
                        .expect("exported function has a world address"),
                    instance
                        .function_type_at(0)
                        .expect("exported function has a type"),
                )
            };

            let b = world
                .instantiate(
                    &engine,
                    Module::new("consumer", &consumer).expect("parse consumer"),
                    &[Import::linked_func_typed("a", "answer", handle, func_type)],
                )
                .expect("instantiate consumer");

            assert_ne!(a, b);
            assert_eq!(
                world.invoke(b, "run_b", &[]).expect("nested invoke"),
                collections::vec![Value::I32(305_419_897)],
                "{tier:?}: cross-instance call did not reach the provider"
            );

            // The flat case still works afterwards, so the call above left no
            // checkout behind on either engine.
            assert_eq!(
                world.invoke(a, "answer", &[]).expect("flat invoke"),
                collections::vec![Value::I32(305_419_896)]
            );
            world.free(b).expect("free consumer");
            world.free(a).expect("free provider");
        }
    }

    /// A funcref-carrying imported global must reach a peer as an ABSOLUTE
    /// handle. Linking synthesizes a host function past the parsed module's
    /// function count for this shape, and that synthetic index used to be
    /// skipped by world registration -- leaving a LOCAL index in a container
    /// the reachability fact marks shared, which a peer would resolve against
    /// its own function space. This path had no in-tree coverage.
    #[cfg(sf_jit)]
    #[test]
    fn linked_function_in_an_imported_global_carries_the_absolute_form() {
        fn answer(
            _caller: &mut Caller<'_>,
            _args: &[Value],
            results: &mut [Value],
        ) -> Result<(), WasmError> {
            results[0] = Value::I32(0x5eed);
            Ok(())
        }

        let wasm = wat::parse_str(
            r#"
            (module
              (type $ft (func (result i32)))
              (global $g (import "host" "fn_global") funcref)
              (table 1 funcref)
              (func (export "call_it") (result i32)
                (table.set (i32.const 0) (global.get $g))
                (call_indirect (type $ft) (i32.const 0))))
            "#,
        )
        .expect("encode module");

        let engine = Engine::new(Config::new().tier(Tier::Jit)).expect("engine");
        let func_type = FunctionType::new(
            collections::Vec::new(),
            collections::vec![crate::value_type::ValueType::I32],
        );
        let import = Import::global_with_linked_function(
            "host",
            "fn_global",
            Value::Ref(RefValue::null(), crate::value_type::RefType::funcref()),
            false,
            Some((answer as crate::vm::entities::HostFn, func_type)),
        );

        let mut instance = Instance::from_module(
            &engine,
            Module::new("linked-global", &wasm).expect("parse"),
            &[import],
        )
        .expect("instantiate with a linked-function global");

        // The global must not hand out a local index.
        let handle = match instance.global_at(0).expect("global") {
            Some(Value::Ref(handle, _)) => handle,
            other => panic!("expected a funcref global, got {other:?}"),
        };
        assert!(
            !handle.is_null(),
            "the linked function must have an address"
        );

        assert_eq!(
            instance
                .invoke("call_it", &[])
                .expect("call through the global"),
            collections::vec![Value::I32(0x5eed)]
        );
    }

    /// An uncaught exception hands the embedder a `RefValue`, and until
    /// `exception_fields` existed there was no way to resolve it -- the
    /// resolver was `pub(crate)`. This pins the capability the method claims,
    /// on both engines, since it is otherwise the only reader of
    /// `ExnInstance::fields` in a JIT-only build.
    #[test]
    fn uncaught_exception_payload_is_readable_by_the_embedder() {
        let wasm = wat::parse_str(
            r#"
            (module
              (tag $t (param i32 i64))
              (func (export "boom")
                i32.const 1234
                i64.const 5678
                throw $t))
            "#,
        )
        .expect("encode throwing module");

        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).expect("engine");
            let mut instance =
                Instance::from_module(&engine, Module::new("thrower", &wasm).expect("parse"), &[])
                    .expect("instantiate");

            let error = instance.invoke("boom", &[]).expect_err("must throw");
            let WasmError::Exception { exn, .. } = error else {
                panic!("{tier:?}: expected an uncaught exception, got {error:?}");
            };
            assert_eq!(
                instance.exception_fields(exn),
                Some(collections::vec![Value::I32(1234), Value::I64(5678)]),
                "{tier:?}: the embedder must be able to read the payload"
            );
        }
    }

    /// A world runs one engine; the second tier is refused by name.
    #[test]
    fn runtime_world_refuses_a_second_tier() {
        if Tier::ALL.len() < 2 {
            return;
        }
        let mut world = RuntimeWorld::new();
        let first = Engine::new(Config::new().tier(Tier::ALL[0])).expect("first engine");
        let second = Engine::new(Config::new().tier(Tier::ALL[1])).expect("second engine");

        world
            .instantiate(
                &first,
                Module::new("first", ADD_WASM).expect("parse add module"),
                &[],
            )
            .expect("first instantiation fixes the tier");

        let error = world
            .instantiate(
                &second,
                Module::new("second", ADD_WASM).expect("parse add module"),
                &[],
            )
            .expect_err("a world must refuse a second tier");
        let message = alloc::format!("{:?}", error.error());
        assert!(
            message.contains("runs one engine"),
            "unexpected rejection: {message}"
        );
    }

    #[test]
    fn runtime_world_keeps_failed_instantiation_occupied() {
        let wasm = wat::parse_str(
            r#"
            (module
              (func $start unreachable)
              (start $start))
            "#,
        )
        .expect("encode failing instantiation");

        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).expect("engine config");
            let mut world = RuntimeWorld::new();
            let error = world
                .instantiate(
                    &engine,
                    Module::new("runtime-world-partial", &wasm).expect("parse partial module"),
                    &[],
                )
                .expect_err("start function must trap");
            let (id, error) = match error {
                InstanceInstantiationError::Partial { id, error } => (id, error),
                InstanceInstantiationError::Complete(error) => {
                    panic!("{tier:?}: failure did not retain an occupied slot: {error}")
                }
            };
            assert!(
                matches!(error, WasmError::Trap(_)),
                "{tier:?}: wrong instantiation failure: {error}"
            );
            assert!(
                world
                    .instances
                    .iter()
                    .any(|(candidate, instance)| *candidate == id && instance.is_none()),
                "{tier:?}: failed slot is not world-owned"
            );

            let checkout = world
                .registry
                .instance_table()
                .checkout(id)
                .unwrap_or_else(|| panic!("{tier:?}: failed slot was freed or regenerated"));
            #[cfg(all(sf_interp, sf_module_validator))]
            if tier == Tier::Interp {
                let interp = checkout
                    .interp()
                    .expect("interpreter partial slot lost its runtime");
                assert_eq!(interp.function_runtime_label_for_test(0), Some("raw"));
                assert_eq!(
                    interp.function_instr_and_cell_count_for_test(0),
                    Some((0, 0)),
                    "trapping Raw start must still skip folded storage"
                );
            }
            assert!(world.free(id).is_err());
            drop(checkout);
            world.free(id).expect("free occupied failed slot");
            assert!(
                world.registry.instance_table().checkout(id).is_none(),
                "{tier:?}: freed generation still resolves"
            );
        }
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn empty_world_after_free_has_no_live_tracked_bytes() {
        static TRACKING: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let tracking_guard = TRACKING.lock().unwrap_or_else(|poison| poison.into_inner());
        let failed_wasm = wat::parse_str(
            r#"
            (module
              (memory 0)
              (data (i32.const 0) "x"))
            "#,
        )
        .expect("encode failing instantiation");
        let engines = Tier::ALL
            .iter()
            .map(|&tier| Engine::new(Config::new().tier(tier)).expect("engine config"))
            .collect::<std::vec::Vec<_>>();

        for engine in &engines {
            tracked_alloc::set_tracking_enabled(true);
            tracked_alloc::reset_tracking();

            {
                let mut world = RuntimeWorld::new();
                let module = Module::new("runtime-world-live-bytes", EMPTY_WASM)
                    .expect("parse empty module");
                let id = world
                    .instantiate(engine, module, &[])
                    .expect("instantiate in world");
                world.free(id).expect("free world instance");
                assert!(world.instances.is_empty());

                let error = world
                    .instantiate(
                        engine,
                        Module::new("runtime-world-failed-live-bytes", &failed_wasm)
                            .expect("parse failing module"),
                        &[],
                    )
                    .expect_err("active data segment must be out of bounds");
                let id = match error {
                    InstanceInstantiationError::Partial { id, .. } => id,
                    InstanceInstantiationError::Complete(error) => {
                        panic!("failure did not retain an occupied slot: {error}")
                    }
                };
                let checkout = world
                    .registry
                    .instance_table()
                    .checkout(id)
                    .expect("failed slot remains occupied at its original generation");
                drop(checkout);
            }

            let snapshot = tracked_alloc::snapshot();
            assert_eq!(
                snapshot.total_bytes,
                0,
                "{:?} live records: {:#?}",
                engine.tier(),
                snapshot.records
            );
            tracked_alloc::set_tracking_enabled(false);
            tracked_alloc::reset_tracking();
        }
        drop(tracking_guard);
    }
}
