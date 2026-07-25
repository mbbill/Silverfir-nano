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
use crate::module::Module;
use crate::vm::engine::{engine as active_engine, Engine};
use crate::vm::value::Value;

#[cfg(sf_interp)]
use crate::vm::interpreter::InterpInstance;
#[cfg(sf_jit)]
use crate::vm::jit::instance::JitInstance;

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
    /// Instantiate on the engine currently selected by
    /// [`crate::set_engine`].
    pub fn new(wasm_bytes: &[u8], imports: &[Import]) -> Result<Self, WasmError> {
        Self::with_engine(active_engine(), Module::new("main", wasm_bytes)?, imports)
    }

    /// Instantiate an already-parsed module on the selected engine.
    pub fn from_module(module: Module, imports: &[Import]) -> Result<Self, WasmError> {
        Self::with_engine(active_engine(), module, imports)
    }

    /// Instantiate on a named engine, ignoring the process-wide selection.
    pub fn with_engine(
        engine: Engine,
        module: Module,
        imports: &[Import],
    ) -> Result<Self, WasmError> {
        let inner = match engine {
            #[cfg(sf_jit)]
            Engine::Jit => Inner::Jit(JitInstance::from_module(module, imports)?),
            #[cfg(sf_interp)]
            Engine::Interp => {
                // Bind before instantiating: the module's start function
                // runs inside `InterpInstance::new`, and it may already
                // call an import.
                let dispatch = interp_imports::bind(&module, imports)?;
                let mut inst = InterpInstance::new(module)?;
                inst.set_host(dispatch);
                Inner::Interp(inst)
            }
        };
        Ok(Self { inner })
    }

    /// Which engine is running this instance.
    #[inline]
    pub fn engine(&self) -> Engine {
        match &self.inner {
            #[cfg(sf_jit)]
            Inner::Jit(_) => Engine::Jit,
            #[cfg(sf_interp)]
            Inner::Interp(_) => Engine::Interp,
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
