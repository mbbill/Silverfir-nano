//! An instantiated module, on whichever engine this build selected.
//!
//! Which engine is underneath is not the embedder's problem. The same
//! bytes, the same `&[Import]`, and the same `invoke(name, &[Value])` work
//! on the JIT and on the interpreter, and the code that calls them needs no
//! `cfg`. What differs -- native code versus a threaded dispatch chain,
//! a `Store` full of entities versus a flat predecoded frame -- differs
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
use crate::vm::link::{InstanceId, LinkRegistry};
use crate::vm::tag::TagHandle;
use crate::vm::value::{RefHandle, Value};

#[cfg(sf_interp)]
use crate::vm::interpreter::{InterpInstance, InterpInstanceLease};
#[cfg(sf_jit)]
use crate::vm::jit::instance::{InstanceInstantiationError, JitInstance};

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
    Jit(JitInstance),
    #[cfg(sf_interp)]
    Interp(InterpInstanceLease),
}

/// Reject a module the spec says is invalid.
///
/// Ahead of the tier split, because validation is not "how code is run":
/// a module either conforms or it does not, and which engine is about to
/// run it cannot change that. It used to live inside the JIT's
/// instantiation, so an interpreter instance accepted modules the spec
/// requires be rejected.
#[inline]
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

/// An engine-owned collection of boxed instances addressed by generational
/// ids. The existing `Instance` API uses a private one-slot world.
pub(crate) struct RuntimeWorld {
    registry: LinkRegistry,
    instances: collections::Vec<(InstanceId, Instance)>,
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
    ) -> Result<Self, (Option<Self>, WasmError)> {
        validate(&module).map_err(|error| (None, error))?;
        match engine.tier() {
            #[cfg(sf_jit)]
            Tier::Jit => {
                match JitInstance::from_module_with_registry(engine, module, imports, registry) {
                    Ok(inst) => Ok(Self {
                        inner: Inner::Jit(inst),
                        registry: registry.clone(),
                    }),
                    Err(error) => {
                        let (partial, error) = error.into_parts();
                        Err((
                            partial.map(|inst| Self {
                                inner: Inner::Jit(inst),
                                registry: registry.clone(),
                            }),
                            error,
                        ))
                    }
                }
            }
            #[cfg(sf_interp)]
            Tier::Interp => {
                let dispatch =
                    interp_imports::bind(&module, imports).map_err(|error| (None, error))?;
                InterpInstance::new_partial_with_registry(
                    engine,
                    module,
                    Some(InterpInstance::boxed_caller_host(dispatch)),
                    imports,
                    None,
                    registry,
                )
                .map(|inst| Self {
                    inner: Inner::Interp(inst),
                    registry: registry.clone(),
                })
                .map_err(|(partial, error)| {
                    (
                        partial.map(|inst| Self {
                            inner: Inner::Interp(inst),
                            registry: registry.clone(),
                        }),
                        error,
                    )
                })
            }
        }
    }

    /// Instantiate in `engine`, which decides the tier and the budgets.
    pub fn new(engine: &Engine, wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        Self::from_module(engine, Module::new("main", wasm_bytes)?, imports)
    }

    /// Instantiate against a shared link registry, so instances can
    /// resolve each other's exports.
    ///
    /// The JIT uses the registry for linked functions and references. The
    /// interpreter still links functions through its import host, but shares
    /// the registry's reference identities and payloads with other instances.
    #[cfg(sf_jit)]
    pub fn from_module_with_registry(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        registry: &crate::vm::link::LinkRegistry,
    ) -> Result<Self, InstanceInstantiationError> {
        match Self::from_module_in_registry(engine, module, imports, registry) {
            Ok(instance) => Ok(instance),
            Err((partial, error)) => match partial.map(|instance| instance.inner) {
                Some(Inner::Jit(instance)) => {
                    Err(InstanceInstantiationError::Partial { instance, error })
                }
                #[cfg(sf_interp)]
                Some(Inner::Interp(_)) => Err(InstanceInstantiationError::Complete(error)),
                None => Err(InstanceInstantiationError::Complete(error)),
            },
        }
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
    /// The interpreter's function forwarding remains host-driven, while the
    /// registry preserves reference identity and payloads across instances.
    #[cfg(sf_interp)]
    pub fn from_module_with_registry_and_funcref_host(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        registry: &crate::vm::link::LinkRegistry,
        funcref_host: crate::vm::interpreter::FuncRefHost,
    ) -> Result<Self, (Option<Self>, WasmError)> {
        validate(&module).map_err(|e| (None, e))?;
        let dispatch = interp_imports::bind(&module, imports).map_err(|e| (None, e))?;
        match InterpInstance::new_partial_with_registry(
            engine,
            module,
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
        let id = world.instantiate(engine, module, imports)?;
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
            Inner::Interp(inst) => {
                let index = inst.find_export(name)?;
                let (params, results) = inst.func_arity(index)?;
                (index, params, results)
            }
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
            Inner::Interp(inst) => interp_imports::call_by_index(inst, func.index, args, results),
        }
    }

    pub fn has_function_export(&self, name: &str) -> bool {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.has_function_export(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.find_export(name).is_some(),
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
            Inner::Interp(inst) => interp_imports::invoke_by_index(inst, idx, args),
        }
    }

    /// A global's value by index.
    pub fn global_at(&self, idx: usize) -> Result<Option<Value>, WasmError> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.global_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => interp_imports::global_at(inst, idx),
        }
    }

    /// Overwrite a global by index.
    pub fn replace_global_at(&mut self, idx: usize, value: Value) -> Result<(), WasmError> {
        match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.replace_global_at(idx, value),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => interp_imports::replace_global_at(inst, idx, value),
        }
    }

    /// An exported global's value by name.
    pub fn get_global(&self, name: &str) -> Result<Option<Value>, WasmError> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.get_global(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => match inst.find_export_global(name) {
                Some(idx) => interp_imports::global_at(inst, idx),
                None => Ok(None),
            },
        }
    }

    /// The declared type of a function by index.
    pub fn function_type_at(&self, idx: usize) -> Option<FunctionType> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_type_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst
                .module()
                .functions()
                .get(idx)
                .map(|f| f.func_type().clone()),
        }
    }

    /// An absolute reference handle for a function, suitable for crossing
    /// instance boundaries.
    pub fn function_handle_at(&self, idx: usize) -> Option<RefHandle> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_handle_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(_) => {
                let _ = idx;
                None
            }
        }
    }

    /// The type index of a function, for cross-instance identity checks.
    pub fn function_type_index_at(&self, idx: usize) -> Option<u32> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_type_index_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.module().functions().get(idx).map(|f| f.type_index()),
        }
    }

    /// Page count of an exported memory.
    pub fn memory_pages(&self, name: &str) -> Option<usize> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.memory_pages(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => {
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
            }
        }
    }

    /// Element count of an exported table.
    pub fn table_size(&self, name: &str) -> Option<usize> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.table_size(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => {
                // By export name and from the LIVE table, as memory_pages
                // does: a table grown after instantiation must report its
                // current size, or an import sized from it gets stale limits.
                let idx = inst
                    .module()
                    .tables()
                    .iter()
                    .position(|t| t.export_names().iter().any(|e| e == name))?;
                inst.table_len(idx)
            }
        }
    }

    /// A tag handle for exception handling.
    pub fn tag_handle(&self, name: &str) -> Option<TagHandle> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.tag_handle(name),
            // Tag IDENTITY is a linking concern, so the interpreter mints
            // handles even though it cannot yet throw or catch.
            #[cfg(sf_interp)]
            Inner::Interp(inst) => {
                let idx = inst
                    .module()
                    .tags()
                    .iter()
                    .position(|t| t.export_names().iter().any(|e| e == name))?;
                inst.tag_handle_at(idx)
            }
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
            Inner::Interp(inst) => inst.shared_memory_at(idx),
        }
    }

    pub fn shared_table_state_at(&self, idx: usize) -> Option<ImportedTableState> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.shared_table_state_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.table_state_at(idx).map(|table| ImportedTableState {
                table,
                type_ctx: None,
            }),
        }
    }

    pub fn shared_global_state_at(&self, idx: usize) -> Option<ImportedGlobalState> {
        match &self.inner {
            #[cfg(sf_interp)]
            // With the exporter's type context: a reference type's concrete
            // heap type names an index in THIS module's type space, and the
            // importer cannot resolve it otherwise.
            Inner::Interp(inst) => inst.global_state_at(idx).map(|global| ImportedGlobalState {
                global,
                type_ctx: Some(inst.module().types().clone()),
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
    pub fn as_jit(&self) -> Option<&JitInstance> {
        match &self.inner {
            Inner::Jit(inst) => Some(inst),
            #[cfg(sf_interp)]
            Inner::Interp(_) => None,
        }
    }

    #[cfg(sf_jit)]
    #[inline]
    pub fn as_jit_mut(&mut self) -> Option<&mut JitInstance> {
        match &mut self.inner {
            Inner::Jit(inst) => Some(inst),
            #[cfg(sf_interp)]
            Inner::Interp(_) => None,
        }
    }

    /// The interpreter's instance, for its dispatch statistics and its
    /// index-based call path. `None` when this instance is on another
    /// engine.
    #[cfg(sf_interp)]
    #[inline]
    pub fn as_interp(&self) -> Option<&InterpInstance> {
        match &self.inner {
            Inner::Interp(inst) => Some(inst),
            #[cfg(sf_jit)]
            Inner::Jit(_) => None,
        }
    }

    #[cfg(sf_interp)]
    #[inline]
    pub fn as_interp_mut(&mut self) -> Option<&mut InterpInstance> {
        match &mut self.inner {
            Inner::Interp(inst) => Some(inst),
            #[cfg(sf_jit)]
            Inner::Jit(_) => None,
        }
    }
}

impl RuntimeWorld {
    pub(crate) fn new() -> Self {
        Self {
            registry: LinkRegistry::new(),
            instances: collections::Vec::new(),
        }
    }

    fn from_registry(registry: LinkRegistry) -> Self {
        Self {
            registry,
            instances: collections::Vec::new(),
        }
    }

    fn take(&mut self, id: InstanceId) -> Option<Instance> {
        let index = self
            .instances
            .iter()
            .position(|(candidate, _)| *candidate == id)?;
        Some(self.instances.remove(index).1)
    }

    fn instantiate(
        &mut self,
        engine: &Engine,
        module: Module,
        imports: &[Import],
    ) -> Result<InstanceId, WasmError> {
        let instance = Instance::from_module_in_registry(engine, module, imports, &self.registry)
            .map_err(|(_, error)| error)?;
        let id = instance.instance_id();
        self.instances.push((id, instance));
        Ok(id)
    }

    fn free(&mut self, id: InstanceId) -> Result<(), WasmError> {
        let index = self
            .instances
            .iter()
            .position(|(candidate, _)| *candidate == id)
            .ok_or_else(|| WasmError::invalid("unknown runtime-world instance"))?;
        if !self.instances[index].1.has_exclusive_lease() {
            return Err(WasmError::invalid(
                "cannot free a checked-out runtime-world instance",
            ));
        }
        let (_, instance) = self.instances.remove(index);
        drop(instance);
        Ok(())
    }

    fn invoke(
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
            return JitInstance::invoke_token(token, name, args);
        }
        #[cfg(sf_interp)]
        {
            let mut token = token;
            if let Some(instance) = token.interp_mut() {
                return interp_imports::invoke_by_name(instance, name, args);
            }
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

    const ADD_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f,
        0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    ];
    #[cfg(feature = "memprof")]
    const EMPTY_WASM: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

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

    #[cfg(feature = "memprof")]
    #[test]
    fn empty_world_after_free_has_no_live_tracked_bytes() {
        static TRACKING: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let tracking_guard = TRACKING.lock().unwrap_or_else(|poison| poison.into_inner());
        tracked_alloc::set_tracking_enabled(true);
        tracked_alloc::reset_tracking();

        {
            let mut world = RuntimeWorld::new();
            let module =
                Module::new("runtime-world-live-bytes", EMPTY_WASM).expect("parse empty module");
            let id = world
                .instantiate(&Engine::with_defaults(), module, &[])
                .expect("instantiate in world");
            world.free(id).expect("free world instance");
            assert!(world.instances.is_empty());
        }

        let snapshot = tracked_alloc::snapshot();
        assert_eq!(
            snapshot.total_bytes, 0,
            "live records: {:#?}",
            snapshot.records
        );
        tracked_alloc::set_tracking_enabled(false);
        tracked_alloc::reset_tracking();
        drop(tracking_guard);
    }
}
