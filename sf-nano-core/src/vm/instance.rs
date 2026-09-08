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
//! Engine diagnostics expose owned snapshots and scalar answers; instance
//! bodies, leases and instruction representations remain private.

use crate::collections;
use crate::error::WasmError;
use crate::module::Module;
use crate::vm::engine::{Engine, Tier};
use crate::vm::link::RefTypeOwner;
use crate::vm::link::{InstanceFreeError, InstanceId, LinkRegistry, WorldAccess};
use crate::vm::memory::{MemoryView, MemoryViewMut};
use crate::vm::tag::TagIdentity;
use crate::vm::value::Value as RawValue;
use crate::{RefValue, Value};

#[cfg(sf_interp)]
use crate::vm::interpreter::{InterpInstance, InterpInstanceLease};
#[cfg(sf_jit)]
use crate::vm::jit::instantiate::JitInstanceLease;

// One import model for both engines. The interpreter's raw host-dispatch
// boundary is an implementation detail that `interp_imports` drives from
// these same declarations.
use crate::vm::imports::{Extern, Import};

#[cfg(sf_interp)]
mod interp_imports;

enum Inner {
    #[cfg(sf_jit)]
    Jit(JitInstanceLease),
    #[cfg(sf_interp)]
    Interp(InterpInstanceLease),
}

fn validate_import_world(
    module: &Module,
    imports: &[Import],
    registry: &LinkRegistry,
) -> Result<(), WasmError> {
    for tag in module.tags() {
        let crate::module::entities::TagDef::Import {
            module: import_module,
            name,
            ..
        } = tag.def()
        else {
            continue;
        };
        let Some(Import {
            value: crate::vm::imports::ImportValue::Tag(state),
            ..
        }) = imports
            .iter()
            .find(|import| import.module == *import_module && import.name == *name)
        else {
            continue;
        };
        if state.type_ctx.is_none()
            && state.func_type.params().iter().any(|value| {
                matches!(
                    value,
                    crate::value_type::ValueType::Ref(crate::value_type::RefType {
                        heap_type: crate::value_type::HeapType::Concrete(_),
                        ..
                    })
                )
            })
        {
            return Err(WasmError::unlinkable(
                "host tags with concrete types require a typed module export",
            ));
        }
    }
    for global in module.globals() {
        if let crate::module::entities::GlobalDef::Import {
            module: import_module,
            name,
            value_type,
            ..
        } = global.def()
        {
            if let Some(Import {
                value:
                    crate::vm::imports::ImportValue::Global(
                        crate::vm::imports::ImportedGlobal::Value(value),
                        _,
                    ),
                ..
            }) = imports
                .iter()
                .find(|import| import.module == *import_module && import.name == *name)
            {
                registry.validate_host_global(*value, *value_type, module.types())?;
            }
        }
    }
    if imports.iter().any(|import| {
        import
            .source
            .is_some_and(|source| !registry.instance_table().owns(source))
    }) {
        return Err(WasmError::unlinkable(
            "export belongs to a different runtime world",
        ));
    }
    if imports.iter().any(|import| {
        matches!(&import.value, crate::vm::imports::ImportValue::Memory(_, Some(memory))
            if memory.host_callback_borrowed())
    }) {
        return Err(WasmError::trap(
            "imported linear memory is borrowed by a host callback",
        ));
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
/// Instantiate a validated module, then call an export by name:
///
/// ```rust
/// use sf_nano_core::{Config, Engine, Module, RuntimeWorld, Value};
///
/// // (module (func (export "answer") (result i32) i32.const 42))
/// let wasm = b"\0asm\x01\0\0\0\x01\x05\x01\x60\0\x01\x7f\x03\x02\x01\0\x07\x0a\x01\x06answer\0\0\x0a\x06\x01\x04\0\x41\x2a\x0b";
/// let engine = Engine::new(Config::new()).expect("configuration");
/// let module = Module::new("example", wasm).expect("valid module");
/// let mut world = RuntimeWorld::new();
/// let id = world.instantiate(&engine, module, &[]).expect("instance");
/// assert_eq!(world.invoke(id, "answer", &[]).expect("call"), vec![Value::I32(42)]);
/// world.free(id).expect("free instance");
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
    owner: InstanceId,
    index: usize,
    params: usize,
    results: usize,
    reference: RefValue,
}

impl Func {
    /// A world-bound function reference suitable for a Wasm reference argument.
    pub fn to_value(&self) -> Value {
        Value::Ref(
            self.reference,
            crate::value_type::RefType::funcref().to_non_nullable(),
        )
    }

    /// How many arguments this function takes, and how many it returns.
    #[inline]
    pub const fn arity(&self) -> (usize, usize) {
        (self.params, self.results)
    }
}

impl Instance {
    fn lower_args(&self, args: &[Value]) -> Result<crate::value::ValueBuffer<RawValue>, WasmError> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => RefTypeOwner::Jit(inst.store()).import_values(args),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => {
                inst.with_instance(|inst| RefTypeOwner::Interp(inst).import_values(args))
            }
        }
    }

    fn function_reference(&self, index: usize) -> Option<RefValue> {
        let raw = match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_handle_at(index),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.function_handle_at(index)),
        }?;
        Some(RefValue::from_vm(raw, self.instance_id().world()))
    }

    /// Look up an export with its live identity and private linking metadata.
    /// Exported objects can be bound into another instance in this RuntimeWorld.
    pub fn get_export(&self, name: &str) -> Result<Option<Extern>, WasmError> {
        let value = match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.export_value(name)?,
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.export_value(name))?,
        };
        Ok(value.map(|value| Extern {
            value,
            source: self.instance_id(),
        }))
    }

    /// Collect named exports for linking, preserving aliases and current sizes.
    /// Functions fail safely if their source instance is later freed. Shared
    /// memory/global/table storage remains owned by its exported objects.
    pub fn exports(&self) -> Result<collections::Vec<(alloc::string::String, Extern)>, WasmError> {
        let names = match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.export_names(),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.export_names()),
        };
        names
            .into_iter()
            .map(|name| {
                let value = self
                    .get_export(&name)?
                    .ok_or_else(|| WasmError::internal("export disappeared"))?;
                Ok((name, value))
            })
            .collect()
    }

    fn from_module_in_registry(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        registry: &LinkRegistry,
    ) -> Result<Self, InstanceInstantiationError> {
        let execution = registry.memory_access().enter()?;
        validate_import_world(&module, imports, registry)?;
        let result = match engine.tier() {
            #[cfg(sf_jit)]
            Tier::Jit => JitInstanceLease::from_module_with_registry(
                engine, module, imports, registry,
            )
            .map(|inst| Self {
                inner: Inner::Jit(inst),
                registry: registry.clone(),
            }),
            #[cfg(sf_interp)]
            Tier::Interp => {
                let dispatch = interp_imports::bind(&module, imports)?;
                match InterpInstance::new_partial_with_registry(
                    engine,
                    module,
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
        };
        drop(execution);
        result
    }

    /// Instantiate in `engine`, which decides the tier and the budgets.
    pub fn new(engine: &Engine, wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        Self::from_module(engine, Module::new("main", wasm_bytes)?, imports)
    }

    /// Instantiate a validated module in `engine` without validating it again.
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
        let index = match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.function_index_of_export(name)?,
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.find_export(name))?,
        };
        self.get_func_by_index(index)
    }

    /// Resolve a referenceable module function index to an instance-owned handle.
    /// Returns `None` for an invalid index or a function not declared eligible
    /// to escape the module (through an export, element or `ref.func`).
    pub fn get_func_by_index(&self, index: usize) -> Option<Func> {
        let (params, results) = match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => {
                let ft = inst.function_type_at(index)?;
                (ft.params().len(), ft.results().len())
            }
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.func_arity(index))?,
        };
        Some(Func {
            owner: self.instance_id(),
            reference: self.function_reference(index)?,
            index,
            params,
            results,
        })
    }

    /// Call a resolved export, writing results into `results`.
    ///
    /// Avoids name lookup and writes into the caller's result storage.
    /// Engines may still allocate temporary invocation storage internally.
    /// A function resolved from a different instance is rejected before dispatch.
    pub fn call(
        &mut self,
        func: &Func,
        args: &[Value],
        results: &mut [Value],
    ) -> Result<(), WasmError> {
        if func.owner != self.instance_id() {
            return Err(WasmError::invalid(
                "function belongs to a different instance",
            ));
        }
        if args.len() != func.params || results.len() != func.results {
            return Err(WasmError::invalid("argument/result arity mismatch"));
        }
        let execution = self.registry.memory_access().enter()?;
        let args = self.lower_args(args)?;
        let mut raw_results = crate::value::ValueBuffer::new(results.len(), RawValue::Unknown);
        let result = match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.call_function_index(func.index, &args, &mut raw_results),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => interp_imports::call_by_index(
                inst.checkout_for_invocation()?,
                func.index,
                &args,
                &mut raw_results,
            ),
        };
        drop(execution);
        result?;
        for (result, raw) in results.iter_mut().zip(raw_results.iter().copied()) {
            *result = Value::from_vm(raw, self.instance_id().world());
        }
        Ok(())
    }

    pub fn has_function_export(&self, name: &str) -> bool {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.has_function_export(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.find_export(name).is_some()),
        }
    }

    /// Borrow the first linear memory for reading.
    ///
    /// Calls and instantiation in this world fail while the returned view lives.
    /// Returns an error if no memory exists, the world is executing, or a
    /// mutable view is already held. Host callbacks should use Caller::memory.
    pub fn memory(&self) -> Result<MemoryView, WasmError> {
        self.registry.memory_access().read(|| match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.shared_memory_at(0),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.shared_memory_at(0)),
        })
    }

    /// Exclusively borrow the first linear memory for host mutation.
    ///
    /// Drop the returned view before executing or instantiating in this world.
    /// Other views and views requested during execution return an error.
    pub fn memory_mut(&mut self) -> Result<MemoryViewMut, WasmError> {
        self.registry.memory_access().write(|| match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.shared_memory_at(0),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| inst.shared_memory_at(0)),
        })
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
        let execution = self.registry.memory_access().enter()?;
        let args = self.lower_args(args)?;
        let result = match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.invoke_function_index(idx, &args),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => {
                interp_imports::invoke_by_index(inst.checkout_for_invocation()?, idx, &args)
            }
        };
        drop(execution);
        result.map(|values| {
            values
                .into_iter()
                .map(|value| Value::from_vm(value, self.instance_id().world()))
                .collect()
        })
    }

    /// An exported global's value by name.
    pub fn get_global(&self, name: &str) -> Result<Option<Value>, WasmError> {
        let value = match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.get_global(name),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => inst.with_instance(|inst| match inst.find_export_global(name) {
                Some(idx) => interp_imports::global_at(inst, idx),
                None => Ok(None),
            }),
        }?;
        Ok(value.map(|value| Value::from_vm(value, self.instance_id().world())))
    }

    /// The payload of an exception, using the handle from [`WasmError::exception`].
    ///
    /// An uncaught exception surfaces with a reference handle, its tag, and
    /// the module's name for that tag. The handle alone is opaque to an
    /// embedder, so this resolves it to the values the throw carried.
    /// Exception objects are registry-owned, so this keeps working after the
    /// instance that threw has been dropped.
    pub fn exception_fields(&self, exn: RefValue) -> Option<collections::Vec<Value>> {
        if exn.world != self.instance_id().world() {
            return None;
        }
        self.registry.arenas().resolve_exn(exn.raw).map(|instance| {
            instance
                .fields
                .iter()
                .copied()
                .map(|value| Value::from_vm(value, self.instance_id().world()))
                .collect()
        })
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

    /// Whether an exported local JIT function has native code. Returns None
    /// for interpreter instances, host/linked functions or unknown names.
    #[cfg(sf_jit)]
    #[inline]
    pub fn function_has_native_code(&self, name: &str) -> Option<bool> {
        match &self.inner {
            Inner::Jit(inst) => inst.function_has_native_code(name),
            #[cfg(sf_interp)]
            Inner::Interp(_) => None,
        }
    }

    /// Collect an owned interpreter diagnostic snapshot. Returns None when
    /// this instance uses another engine. Collection may allocate.
    #[cfg(sf_interp)]
    #[inline]
    pub fn interpreter_stats(&self) -> Option<crate::InterpreterStats> {
        match &self.inner {
            Inner::Interp(inst) => Some(inst.with_instance(|inst| inst.statistics())),
            #[cfg(sf_jit)]
            Inner::Jit(_) => None,
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
        let execution = self.registry.memory_access().enter()?;
        let result = self.invoke_inner(id, name, args);
        drop(execution);
        result
    }

    fn invoke_inner(
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
        let owner = RefTypeOwner::from_token(&token)
            .ok_or_else(|| WasmError::invalid("instance has no engine"))?;
        let args = owner.import_values(args)?;
        let expose = |values: collections::Vec<RawValue>| {
            values
                .into_iter()
                .map(|value| Value::from_vm(value, id.world()))
                .collect()
        };
        #[cfg(sf_jit)]
        if token.jit().is_some() {
            return JitInstanceLease::invoke_token(token, name, &args).map(expose);
        }
        #[cfg(sf_interp)]
        if token.interp().is_some() {
            return interp_imports::invoke_by_name(token, name, &args).map(expose);
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
        let execution = self.begin_execution()?;
        let result = self.invoke_inner(id, function_index, args);
        drop(execution);
        result
    }

    fn invoke_inner(
        &self,
        id: InstanceId,
        function_index: usize,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        let token = self
            .checkout(id)
            .ok_or_else(|| WasmError::invalid("unknown runtime-world instance"))?;
        let owner = RefTypeOwner::from_token(&token)
            .ok_or_else(|| WasmError::invalid("instance has no engine"))?;
        let args = owner.import_values(args)?;
        let expose = |values: collections::Vec<RawValue>| {
            values
                .into_iter()
                .map(|value| Value::from_vm(value, id.world()))
                .collect()
        };
        #[cfg(sf_jit)]
        if token.jit().is_some() {
            return JitInstanceLease::invoke_function_index_token(token, function_index, &args)
                .map(expose);
        }
        #[cfg(sf_interp)]
        if token.interp().is_some() {
            return interp_imports::invoke_by_index(token, function_index, &args).map(expose);
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
        let callback_type = crate::FunctionType::new(
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
            let error = Instance::new(&engine, &wasm, &[])
                .err()
                .expect("undeclared ref.func must fail validation");
            let message = alloc::format!("{:?}", error);
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
                    WasmError {
                        repr: crate::error::ErrorRepr::Trap("cast failure")
                    },
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

            let answers = collections::vec![
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

            // A retained public handle is inert after its owner is freed:
            // reject it before even an abstract ref.test enters Wasm.
            world.free(provider).expect("free reference provider");
            let error = world
                .invoke(consumer, "test_func", &[function])
                .expect_err("stale public reference must not enter Wasm");
            assert_eq!(error.class(), "invalid");
            assert_eq!(error.message(), "reference owner is no longer available");
            answers
        }

        let jit = answers_for(Tier::Jit, &provider_wasm, &consumer_wasm);
        let interp = answers_for(Tier::Interp, &provider_wasm, &consumer_wasm);
        assert_eq!(jit, interp, "reference-type answers diverged by engine");
        assert_eq!(jit, collections::vec![1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]);
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

            let exported = world
                .instance(a)
                .expect("provider")
                .get_export("answer")
                .expect("export metadata")
                .expect("answer export");

            let b = world
                .instantiate(
                    &engine,
                    Module::new("consumer", &consumer).expect("parse consumer"),
                    &[Import::new("a", "answer", exported)],
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

    /// A funcref exported in a shared global preserves its absolute identity,
    /// including when the referenced function re-exports a host callback.
    #[test]
    fn linked_function_in_an_imported_global_carries_the_absolute_form() {
        let source = wat::parse_str(
            r#"(module
            (import "host" "answer" (func $answer (result i32)))
            (global (export "fn_global") funcref (ref.func $answer)))"#,
        )
        .unwrap();
        let target = wat::parse_str(
            r#"(module
            (type $ft (func (result i32)))
            (global $g (import "source" "fn_global") funcref)
            (table 1 funcref)
            (func (export "call_it") (result i32)
                (table.set (i32.const 0) (global.get $g))
                (call_indirect (type $ft) (i32.const 0))))"#,
        )
        .unwrap();
        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).unwrap();
            let mut world = RuntimeWorld::new();
            let host = Import::func("host", "answer", |_, _, out| {
                out[0] = Value::I32(0x5eed);
                Ok(())
            });
            let source = world
                .instantiate(&engine, Module::new("source", &source).unwrap(), &[host])
                .unwrap();
            let value = world
                .instance(source)
                .unwrap()
                .get_export("fn_global")
                .unwrap()
                .unwrap();
            let target = world
                .instantiate(
                    &engine,
                    Module::new("target", &target).unwrap(),
                    &[Import::new("source", "fn_global", value)],
                )
                .unwrap();
            assert_eq!(
                world.invoke(target, "call_it", &[]).unwrap(),
                [Value::I32(0x5eed)]
            );
        }
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
            let WasmError {
                repr: crate::error::ErrorRepr::Exception { exn, .. },
            } = error
            else {
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
                matches!(
                    error,
                    WasmError {
                        repr: crate::error::ErrorRepr::Trap(_)
                    }
                ),
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
        // Tracking is process-wide. Other tests do not take a shared lock,
        // so run this zero-live-bytes assertion in its own test process.
        const CHILD: &str = "SF_NANO_EMPTY_WORLD_TRACKING_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args([
                    "--exact",
                    "vm::instance::tests::empty_world_after_free_has_no_live_tracked_bytes",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .expect("run isolated allocation test");
            assert!(
                output.status.success(),
                "isolated allocation test failed:\n{}\n{}",
                std::string::String::from_utf8_lossy(&output.stdout),
                std::string::String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
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
    }
}

#[cfg(test)]
impl Instance {
    /// An absolute reference handle for a function, suitable for crossing
    /// instance boundaries.
    fn function_handle_at(&self, idx: usize) -> Option<RefValue> {
        self.function_reference(idx)
    }
}
