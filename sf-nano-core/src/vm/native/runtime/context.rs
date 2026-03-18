//! Native runtime context and ABI-visible cached storage views.
//!
//! This is the runtime object that native lowering targets. It stays local to
//! the native runtime boundary rather than becoming part of the generic VM API.
//!
//! NOTE: The current native ABI and shared MachineIR use a 64-bit machine
//! model. Lowering reads pointer-like fields and cached lengths/counts through
//! `U64` accesses, so 32-bit targets would need an explicit pointer-width
//! abstraction instead of reusing this layout as-is.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        entities::{FunctionInst, GlobalInst, ModuleInst},
        store::Store,
        value::RefHandle,
    },
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeMemoryView {
    pub base: *mut u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeTableView {
    pub elements_base: *mut RefHandle,
    pub elements_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeGlobalsView {
    pub base: *mut GlobalInst,
    pub len: usize,
}

pub mod function_kind {
    pub const LOCAL: u32 = 0;
    pub const EXTERNAL: u32 = 1;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeFunctionView {
    pub kind: u32,
    /// Canonical equivalence class for the callee signature within the current
    /// module type context. `call_indirect` compares this against the cached
    /// canonical id for the expected type index instead of comparing raw type
    /// indices directly.
    pub type_canon: u32,
    /// Native-local target token for `MachineTerminator::CallIndirect`.
    ///
    /// The current lowering path keeps machine function ids aligned with the
    /// module function index space, so local callees use that machine-call
    /// target id directly here.
    pub local_target: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct NativeContext {
    pub stack_end: *mut u64,
    pub mem0_base: *mut u8,
    pub mem0_size: u64,
    pub globals_view: NativeGlobalsView,
    pub memory_views_base: *const NativeMemoryView,
    pub memory_views_len: usize,
    pub table_views_base: *const NativeTableView,
    pub table_views_len: usize,
    pub function_views_base: *const NativeFunctionView,
    pub function_views_len: usize,
    pub type_canon_base: *const u32,
    pub type_canon_len: usize,
    pub store: *mut Store,
    pub current_module: *const ModuleInst,
    pub error: Option<WasmError>,
    /// Trap kind set by the guard-page signal handler (no allocation needed).
    /// 0 = no trap, 1 = memory out of bounds.
    #[cfg(has_guard_pages)]
    pub trap_kind: u32,
    memory_views: Vec<NativeMemoryView>,
    table_views: Vec<NativeTableView>,
    function_views: Vec<NativeFunctionView>,
    type_canon: Vec<u32>,
    #[cfg(feature = "function-trace")]
    pub trace_stack: std::vec::Vec<u32>,
}

impl NativeContext {
    #[inline]
    pub fn new(store: *mut Store, stack_end: *mut u64) -> Self {
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
            type_canon_base: core::ptr::null(),
            type_canon_len: 0,
            store,
            current_module: core::ptr::null(),
            error: None,
            #[cfg(has_guard_pages)]
            trap_kind: 0,
            memory_views: Vec::new(),
            table_views: Vec::new(),
            function_views: Vec::new(),
            type_canon: Vec::new(),
            #[cfg(feature = "function-trace")]
            trace_stack: std::vec::Vec::new(),
        };
        ctx.refresh_cached_views();
        ctx
    }

    #[inline]
    pub fn refresh_cached_views(&mut self) {
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
    pub fn refresh_memory_views(&mut self) {
        let Some(store) = self.store() else {
            self.mem0_base = core::ptr::null_mut();
            self.mem0_size = 0;
            self.memory_views.clear();
            self.memory_views_base = core::ptr::null();
            self.memory_views_len = 0;
            return;
        };

        let memory_views: Vec<_> = store
            .module()
            .memories
            .iter()
            .map(|memory| {
                let ptr = memory.memory_ptr();
                let len = memory.memory_len();
                NativeMemoryView {
                    base: if len == 0 {
                        core::ptr::null_mut()
                    } else {
                        ptr
                    },
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
    pub fn refresh_table_views(&mut self) {
        let Some(store) = self.store() else {
            self.table_views.clear();
            self.table_views_base = core::ptr::null();
            self.table_views_len = 0;
            return;
        };

        let table_views: Vec<_> = store
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
    pub fn refresh_globals_view(&mut self) {
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
    pub fn refresh_function_views(&mut self) {
        let Some(store) = self.store() else {
            self.function_views.clear();
            self.function_views_base = core::ptr::null();
            self.function_views_len = 0;
            return;
        };

        let module = store.module();
        let type_canon = &self.type_canon;
        let function_views: Vec<_> = module
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
    pub fn refresh_type_canon(&mut self) {
        let Some(store) = self.store() else {
            self.type_canon.clear();
            self.type_canon_base = core::ptr::null();
            self.type_canon_len = 0;
            return;
        };

        let type_ctx = &store.module().types;
        let mut type_canon = Vec::with_capacity(type_ctx.len());
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
    pub fn store(&self) -> Option<&Store> {
        unsafe { self.store.as_ref() }
    }

    #[inline]
    pub fn store_mut(&mut self) -> Option<&mut Store> {
        unsafe { self.store.as_mut() }
    }
}

pub mod ctx_offset {
    use super::NativeContext;

    pub const STACK_END: u32 = core::mem::offset_of!(NativeContext, stack_end) as u32;
    pub const MEM0_BASE: u32 = core::mem::offset_of!(NativeContext, mem0_base) as u32;
    pub const MEM0_SIZE: u32 = core::mem::offset_of!(NativeContext, mem0_size) as u32;
    pub const GLOBALS_VIEW: u32 = core::mem::offset_of!(NativeContext, globals_view) as u32;
    pub const MEMORY_VIEWS_BASE: u32 =
        core::mem::offset_of!(NativeContext, memory_views_base) as u32;
    pub const MEMORY_VIEWS_LEN: u32 = core::mem::offset_of!(NativeContext, memory_views_len) as u32;
    pub const TABLE_VIEWS_BASE: u32 = core::mem::offset_of!(NativeContext, table_views_base) as u32;
    pub const TABLE_VIEWS_LEN: u32 = core::mem::offset_of!(NativeContext, table_views_len) as u32;
    pub const FUNCTION_VIEWS_BASE: u32 =
        core::mem::offset_of!(NativeContext, function_views_base) as u32;
    pub const FUNCTION_VIEWS_LEN: u32 =
        core::mem::offset_of!(NativeContext, function_views_len) as u32;
    pub const TYPE_CANON_BASE: u32 = core::mem::offset_of!(NativeContext, type_canon_base) as u32;
    pub const TYPE_CANON_LEN: u32 = core::mem::offset_of!(NativeContext, type_canon_len) as u32;
    pub const STORE: u32 = core::mem::offset_of!(NativeContext, store) as u32;
    pub const CURRENT_MODULE: u32 = core::mem::offset_of!(NativeContext, current_module) as u32;
    #[cfg(has_guard_pages)]
    pub const TRAP_KIND: u32 = core::mem::offset_of!(NativeContext, trap_kind) as u32;
}

pub mod memory_view_offset {
    use super::NativeMemoryView;

    pub const BASE: u32 = core::mem::offset_of!(NativeMemoryView, base) as u32;
    pub const LEN: u32 = core::mem::offset_of!(NativeMemoryView, len) as u32;
}

pub mod table_view_offset {
    use super::NativeTableView;

    pub const ELEMENTS_BASE: u32 = core::mem::offset_of!(NativeTableView, elements_base) as u32;
    pub const ELEMENTS_LEN: u32 = core::mem::offset_of!(NativeTableView, elements_len) as u32;
}

pub mod globals_view_offset {
    use super::NativeGlobalsView;

    pub const BASE: u32 = core::mem::offset_of!(NativeGlobalsView, base) as u32;
    pub const LEN: u32 = core::mem::offset_of!(NativeGlobalsView, len) as u32;
}

pub mod function_view_offset {
    use super::NativeFunctionView;

    pub const KIND: u32 = core::mem::offset_of!(NativeFunctionView, kind) as u32;
    pub const TYPE_CANON: u32 = core::mem::offset_of!(NativeFunctionView, type_canon) as u32;
    pub const LOCAL_TARGET: u32 = core::mem::offset_of!(NativeFunctionView, local_target) as u32;
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::*;
    use crate::{
        module::{type_context::TypeContext, type_defs::FunctionType},
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
            vec![ValueType::I32],
            vec![ValueType::I64],
        ));
        let types = TypeContext::new(vec![Rc::clone(&duplicated_sig), duplicated_sig]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::Local {
            spec: crate::module::entities::FunctionSpec::new(
                Rc::new(FunctionType::new(
                    vec![ValueType::I32],
                    vec![ValueType::I64],
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
            vec![ValueType::I32],
            vec![ValueType::I64],
        ));
        let types = TypeContext::new(vec![Rc::clone(&duplicated_sig), duplicated_sig]);
        let mut module = ModuleInst::new(String::from("m"), types);
        module.functions.push(FunctionInst::External {
            func_type: Rc::new(FunctionType::new(
                vec![ValueType::I32],
                vec![ValueType::I64],
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
