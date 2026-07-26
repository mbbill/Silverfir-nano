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
use crate::vm::tag::TagHandle;
use crate::vm::value::{RefHandle, Value};

#[cfg(sf_interp)]
use crate::vm::interpreter::InterpInstance;
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
    Interp(InterpInstance),
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
    /// Instantiate in `engine`, which decides the tier and the budgets.
    pub fn new(engine: &Engine, wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        Self::from_module(engine, Module::new("main", wasm_bytes)?, imports)
    }

    /// Instantiate against a shared link registry, so instances can
    /// resolve each other's exports.
    ///
    /// The registry is the JIT's linking machinery; an interpreter
    /// instance ignores it and links only what its import list carries.
    #[cfg(sf_jit)]
    pub fn from_module_with_registry(
        engine: &Engine,
        module: Module,
        imports: &[Import],
        registry: &crate::vm::store::LinkRegistry,
    ) -> Result<Self, InstanceInstantiationError> {
        validate(&module).map_err(InstanceInstantiationError::Complete)?;
        match engine.tier() {
            Tier::Jit => JitInstance::from_module_with_registry(engine, module, imports, registry)
                .map(|inst| Self {
                    inner: Inner::Jit(inst),
                }),
            #[cfg(sf_interp)]
            Tier::Interp => Self::from_module(engine, module, imports)
                .map_err(InstanceInstantiationError::Complete),
        }
    }

    /// Instantiate an already-parsed module in `engine`.
    pub fn from_module(
        engine: &Engine,
        module: Module,
        imports: &[Import],
    ) -> Result<Self, WasmError> {
        validate(&module)?;
        let inner = match engine.tier() {
            #[cfg(sf_jit)]
            Tier::Jit => Inner::Jit(JitInstance::from_module(engine, module, imports)?),
            #[cfg(sf_interp)]
            Tier::Interp => {
                // The host goes in with the module: a start function may
                // call an import, and it runs during instantiation.
                let dispatch = interp_imports::bind(&module, imports)?;
                Inner::Interp(InterpInstance::new(
                    engine,
                    module,
                    Some(InterpInstance::boxed_host(dispatch)),
                    imports,
                )?)
            }
        };
        Ok(Self { inner })
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

    /// Call an exported function by name.
    pub fn invoke(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Result<collections::Vec<Value>, WasmError> {
        match &mut self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.invoke(name, args),
            #[cfg(sf_interp)]
            Inner::Interp(inst) => interp_imports::invoke_by_name(inst, name, args),
        }
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

    /// A reference handle for a function, for `ref.func` across instances.
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
                let _ = name;
                inst.memory()
                    .map(|m| m.len() / crate::constants::WASM_PAGE_SIZE)
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
                let _ = (inst, name);
                None
            }
        }
    }

    /// A tag handle for exception handling.
    pub fn tag_handle(&self, name: &str) -> Option<TagHandle> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.tag_handle(name),
            // The interpreter has no exception handling, so it mints no tags.
            #[cfg(sf_interp)]
            Inner::Interp(_) => {
                let _ = name;
                None
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
            Inner::Interp(_) => {
                let _ = idx;
                None
            }
        }
    }

    pub fn shared_global_state_at(&self, idx: usize) -> Option<ImportedGlobalState> {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(inst) => inst.shared_global_state_at(idx),
            #[cfg(sf_interp)]
            Inner::Interp(_) => {
                let _ = idx;
                None
            }
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
