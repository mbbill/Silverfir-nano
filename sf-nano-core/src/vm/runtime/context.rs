//! Native runtime context and ABI-visible cached storage views.
//!
//! This is the runtime object that native lowering targets. It stays local to
//! the native runtime boundary rather than becoming part of the generic VM API.
//!
//! NOTE: This Rust struct follows the host compiler ABI. Shared lowering and
//! the emulator must use the explicit target-side layout helpers in
//! [`super::layout`] rather than assuming these host offsets/strides directly.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        entities::{FunctionInst, GlobalInst, ModuleInst},
        runtime::{code::CompiledNativeModule, dispatch_view::NativeDispatchMetadata},
        store::Store,
        value::RefHandle,
    },
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeMemoryView {
    pub(crate) base: *mut u8,
    pub(crate) len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeTableView {
    pub(crate) elements_base: *mut RefHandle,
    pub(crate) elements_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeGlobalsView {
    pub(crate) base: *mut GlobalInst,
    pub(crate) len: usize,
}

pub(crate) use super::dispatch_view::function_kind;
pub(crate) use super::dispatch_view::CallDispatchView as NativeFunctionView;

#[repr(C)]
#[derive(Debug)]
pub(crate) struct NativeContext {
    pub(crate) stack_end: *mut u64,
    pub(crate) mem0_base: *mut u8,
    pub(crate) mem0_size: u64,
    pub(crate) globals_view: NativeGlobalsView,
    pub(crate) memory_views_base: *const NativeMemoryView,
    pub(crate) memory_views_len: usize,
    pub(crate) table_views_base: *const NativeTableView,
    pub(crate) table_views_len: usize,
    pub(crate) function_views_base: *const NativeFunctionView,
    pub(crate) function_views_len: usize,
    pub(crate) local_call_infos_base: *const u8,
    pub(crate) local_call_infos_len: usize,
    pub(crate) type_canon_base: *const u32,
    pub(crate) type_canon_len: usize,
    pub(crate) store: *mut Store,
    pub(crate) current_module: *const ModuleInst,
    pub(crate) error: Option<WasmError>,
    /// Trap kind set by the guard-page signal handler (no allocation needed).
    /// 0 = no trap, 1 = memory out of bounds.
    #[cfg(sf_has_guard_pages)]
    pub(crate) trap_kind: u32,
    memory_views: collections::Vec<NativeMemoryView>,
    table_views: collections::Vec<NativeTableView>,
    function_views: collections::Vec<NativeFunctionView>,
    type_canon: collections::Vec<u32>,
    #[cfg(sf_call_trace)]
    pub(crate) trace_stack: collections::Vec<u32>,
}

impl NativeContext {
    #[inline]
    pub(crate) fn new(store: *mut Store, stack_end: *mut u64) -> Self {
        let mut ctx = Self {
            stack_end,
            mem0_base: core::ptr::null_mut(),
            mem0_size: 0,
            globals_view: NativeGlobalsView::default(),
            memory_views_base: core::ptr::null(),
            memory_views_len: 0,
            table_views_base: core::ptr::null(),
            table_views_len: 0,
            function_views_base: core::ptr::null(),
            function_views_len: 0,
            local_call_infos_base: core::ptr::null(),
            local_call_infos_len: 0,
            type_canon_base: core::ptr::null(),
            type_canon_len: 0,
            store,
            current_module: core::ptr::null(),
            error: None,
            #[cfg(sf_has_guard_pages)]
            trap_kind: 0,
            memory_views: collections::Vec::new(),
            table_views: collections::Vec::new(),
            function_views: collections::Vec::new(),
            type_canon: collections::Vec::new(),
            #[cfg(sf_call_trace)]
            trace_stack: collections::Vec::new(),
        };
        ctx.refresh_cached_views();
        ctx
    }

    #[inline]
    pub(crate) fn refresh_cached_views(&mut self) {
        if let Some(store) = self.store() {
            self.current_module = store.module() as *const ModuleInst;
        } else {
            self.current_module = core::ptr::null();
        }
        self.refresh_globals_view();
        self.refresh_memory_views();
        self.refresh_table_views();
        self.refresh_type_canon();
        self.refresh_function_views();
    }

    #[inline]
    pub(crate) fn seed_local_call_infos(&mut self, compiled: &CompiledNativeModule) {
        self.seed_dispatch_metadata(compiled.dispatch_metadata());
    }

    #[inline]
    pub(crate) fn seed_dispatch_metadata(&mut self, metadata: &NativeDispatchMetadata) {
        self.local_call_infos_base = metadata.local_call_infos().base();
        self.local_call_infos_len = metadata.local_call_infos().len();
    }

    #[inline]
    pub(crate) fn refresh_memory_views(&mut self) {
        let Some(store) = self.store() else {
            self.mem0_base = core::ptr::null_mut();
            self.mem0_size = 0;
            self.memory_views.clear();
            self.memory_views_base = core::ptr::null();
            self.memory_views_len = 0;
            return;
        };

        let memory_views: collections::Vec<_> = store
            .module()
            .memories
            .iter()
            .map(|memory| {
                let ptr = memory.memory_ptr();
                let len = memory.memory_len();
                NativeMemoryView {
                    base: if len == 0 { core::ptr::null_mut() } else { ptr },
                    len,
                }
            })
            .collect();

        self.memory_views = memory_views;

        self.memory_views_base = if self.memory_views.is_empty() {
            core::ptr::null()
        } else {
            self.memory_views.as_ptr()
        };
        self.memory_views_len = self.memory_views.len();

        if let Some(view) = self.memory_views.first().copied() {
            self.mem0_base = view.base;
            self.mem0_size = view.len as u64;
        } else {
            self.mem0_base = core::ptr::null_mut();
            self.mem0_size = 0;
        }
    }

    #[inline]
    pub(crate) fn refresh_table_views(&mut self) {
        let Some(store) = self.store() else {
            self.table_views.clear();
            self.table_views_base = core::ptr::null();
            self.table_views_len = 0;
            return;
        };

        let table_views: collections::Vec<_> = store
            .module()
            .tables
            .iter()
            .map(|table| NativeTableView {
                elements_base: if table.elements.is_empty() {
                    core::ptr::null_mut()
                } else {
                    table.elements.as_ptr() as *mut RefHandle
                },
                elements_len: table.elements.len(),
            })
            .collect();

        self.table_views = table_views;

        self.table_views_base = if self.table_views.is_empty() {
            core::ptr::null()
        } else {
            self.table_views.as_ptr()
        };
        self.table_views_len = self.table_views.len();
    }

    #[inline]
    pub(crate) fn refresh_globals_view(&mut self) {
        let Some(store) = self.store() else {
            self.globals_view = NativeGlobalsView::default();
            return;
        };

        let globals = &store.module().globals;
        self.globals_view = NativeGlobalsView {
            base: if globals.is_empty() {
                core::ptr::null_mut()
            } else {
                globals.as_ptr() as *mut GlobalInst
            },
            len: globals.len(),
        };
    }

    #[inline]
    pub(crate) fn refresh_function_views(&mut self) {
        let Some(store) = self.store() else {
            self.function_views.clear();
            self.function_views_base = core::ptr::null();
            self.function_views_len = 0;
            return;
        };

        let module = store.module();
        let type_canon = &self.type_canon;
        let function_views: collections::Vec<_> = module
            .functions
            .iter()
            .enumerate()
            .map(|(func_idx, func)| {
                let (kind, local_target) = match func {
                    FunctionInst::Local { .. } => (function_kind::LOCAL, func_idx as u32),
                    FunctionInst::External { .. } => (function_kind::EXTERNAL, u32::MAX),
                };
                let type_canon = match func {
                    FunctionInst::Local { type_index, .. } => type_canon
                        .get(*type_index as usize)
                        .copied()
                        .unwrap_or(u32::MAX),
                    FunctionInst::External { func_type, .. } => module
                        .types
                        .as_slice()
                        .iter()
                        .position(|candidate| candidate.as_ref() == func_type.as_ref())
                        .and_then(|index| type_canon.get(index).copied())
                        .unwrap_or(u32::MAX),
                };
                NativeFunctionView {
                    kind,
                    type_canon,
                    local_target,
                }
            })
            .collect();

        self.function_views = function_views;
        self.function_views_base = if self.function_views.is_empty() {
            core::ptr::null()
        } else {
            self.function_views.as_ptr()
        };
        self.function_views_len = self.function_views.len();
    }

    #[inline]
    pub(crate) fn refresh_type_canon(&mut self) {
        let Some(store) = self.store() else {
            self.type_canon.clear();
            self.type_canon_base = core::ptr::null();
            self.type_canon_len = 0;
            return;
        };

        let type_ctx = &store.module().types;
        let mut type_canon = collections::Vec::with_capacity(type_ctx.len());
        for idx in 0..type_ctx.len() {
            let idx_u32 = idx as u32;
            let mut canonical = idx_u32;
            for prior in 0..idx {
                let prior_u32 = prior as u32;
                if type_ctx.types_equivalent(prior_u32, idx_u32) {
                    canonical = type_canon[prior];
                    break;
                }
            }
            type_canon.push(canonical);
        }

        self.type_canon = type_canon;
        self.type_canon_base = if self.type_canon.is_empty() {
            core::ptr::null()
        } else {
            self.type_canon.as_ptr()
        };
        self.type_canon_len = self.type_canon.len();
    }

    #[inline]
    pub(crate) fn store(&self) -> Option<&Store> {
        unsafe { self.store.as_ref() }
    }

    #[inline]
    pub(crate) fn store_mut(&mut self) -> Option<&mut Store> {
        unsafe { self.store.as_mut() }
    }
}

pub(crate) mod ctx_offset {
    use super::NativeContext;

    #[cfg(test)]
    pub(crate) const STACK_END: u32 = core::mem::offset_of!(NativeContext, stack_end) as u32;
    pub(crate) const MEM0_BASE: u32 = core::mem::offset_of!(NativeContext, mem0_base) as u32;
    pub(crate) const MEM0_SIZE: u32 = core::mem::offset_of!(NativeContext, mem0_size) as u32;
    #[cfg(sf_has_guard_pages)]
    pub(crate) const TRAP_KIND: u32 = core::mem::offset_of!(NativeContext, trap_kind) as u32;

    #[cfg(test)]
    mod test_only {
        use super::super::NativeContext;
        pub(crate) const GLOBALS_VIEW: u32 =
            core::mem::offset_of!(NativeContext, globals_view) as u32;
        pub(crate) const MEMORY_VIEWS_BASE: u32 =
            core::mem::offset_of!(NativeContext, memory_views_base) as u32;
        pub(crate) const MEMORY_VIEWS_LEN: u32 =
            core::mem::offset_of!(NativeContext, memory_views_len) as u32;
        pub(crate) const TABLE_VIEWS_BASE: u32 =
            core::mem::offset_of!(NativeContext, table_views_base) as u32;
        pub(crate) const TABLE_VIEWS_LEN: u32 =
            core::mem::offset_of!(NativeContext, table_views_len) as u32;
        pub(crate) const FUNCTION_VIEWS_BASE: u32 =
            core::mem::offset_of!(NativeContext, function_views_base) as u32;
        pub(crate) const FUNCTION_VIEWS_LEN: u32 =
            core::mem::offset_of!(NativeContext, function_views_len) as u32;
        pub(crate) const LOCAL_CALL_INFOS_BASE: u32 =
            core::mem::offset_of!(NativeContext, local_call_infos_base) as u32;
        pub(crate) const LOCAL_CALL_INFOS_LEN: u32 =
            core::mem::offset_of!(NativeContext, local_call_infos_len) as u32;
        pub(crate) const TYPE_CANON_BASE: u32 =
            core::mem::offset_of!(NativeContext, type_canon_base) as u32;
        pub(crate) const TYPE_CANON_LEN: u32 =
            core::mem::offset_of!(NativeContext, type_canon_len) as u32;
    }
    #[cfg(test)]
    pub(crate) use test_only::*;
}

#[cfg(test)]
pub(crate) mod memory_view_offset {
    use super::NativeMemoryView;

    pub(crate) const BASE: u32 = core::mem::offset_of!(NativeMemoryView, base) as u32;
    pub(crate) const LEN: u32 = core::mem::offset_of!(NativeMemoryView, len) as u32;
}

#[cfg(test)]
pub(crate) mod table_view_offset {
    use super::NativeTableView;

    pub(crate) const ELEMENTS_BASE: u32 =
        core::mem::offset_of!(NativeTableView, elements_base) as u32;
    pub(crate) const ELEMENTS_LEN: u32 =
        core::mem::offset_of!(NativeTableView, elements_len) as u32;
}

#[cfg(test)]
pub(crate) mod globals_view_offset {
    use super::NativeGlobalsView;

    pub(crate) const BASE: u32 = core::mem::offset_of!(NativeGlobalsView, base) as u32;
    pub(crate) const LEN: u32 = core::mem::offset_of!(NativeGlobalsView, len) as u32;
}

#[cfg(test)]
pub(crate) mod function_view_offset {
    use super::NativeFunctionView;

    pub(crate) const KIND: u32 = core::mem::offset_of!(NativeFunctionView, kind) as u32;
    pub(crate) const TYPE_CANON: u32 = core::mem::offset_of!(NativeFunctionView, type_canon) as u32;
    pub(crate) const LOCAL_TARGET: u32 =
        core::mem::offset_of!(NativeFunctionView, local_target) as u32;
}

#[cfg(test)]
mod tests {
    use tracked_alloc::{boxed::Box, rc::Rc, string::String};

    use super::*;
    use crate::{
        module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType},
        value_type::ValueType,
        vm::{
            entities::{Caller, ModuleInst},
            store::Store,
            value::Value,
        },
    };

    fn external_noop(
        _caller: &mut Caller<'_>,
        _args: &[Value],
        _results: &mut [Value],
    ) -> Result<(), WasmError> {
        Ok(())
    }

    #[test]
    fn refresh_function_views_canonicalizes_equivalent_type_indices() {
        let duplicated_sig = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I64],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&duplicated_sig),
            duplicated_sig
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(
                Rc::new(FunctionType::new(
                    collections::vec![ValueType::I32],
                    collections::vec![ValueType::I64],
                )),
                1,
            ),
            type_index: 1,
        });
        let mut store = Box::new(Store::new(module));
        let ctx = NativeContext::new((&mut *store) as *mut Store, core::ptr::null_mut());

        assert_eq!(ctx.type_canon_len, 2);
        let type_canon =
            unsafe { core::slice::from_raw_parts(ctx.type_canon_base, ctx.type_canon_len) };
        assert_eq!(type_canon, &[0, 0]);
        assert_eq!(ctx.function_views_len, 1);
        let view = unsafe { &*ctx.function_views_base };
        assert_eq!(view.type_canon, 0);
    }

    #[test]
    fn refresh_function_views_canonicalizes_equivalent_external_signatures() {
        let duplicated_sig = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I64],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&duplicated_sig),
            duplicated_sig
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::External {
            func_type: Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I64],
            )),
            callback: external_noop,
        });
        let mut store = Box::new(Store::new(module));
        let ctx = NativeContext::new((&mut *store) as *mut Store, core::ptr::null_mut());

        assert_eq!(ctx.type_canon_len, 2);
        assert_eq!(ctx.function_views_len, 1);
        let view = unsafe { &*ctx.function_views_base };
        assert_eq!(view.type_canon, 0);
    }
}
