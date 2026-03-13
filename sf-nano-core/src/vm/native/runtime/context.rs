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
        entities::{GlobalInst, ModuleInst},
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

#[repr(C)]
#[derive(Debug)]
pub struct NativeContext {
    pub stack_end: *mut u64,
    pub call_depth: u64,
    pub mem0_base: *mut u8,
    pub mem0_size: u64,
    pub globals_view: NativeGlobalsView,
    pub memory_views_base: *const NativeMemoryView,
    pub memory_views_len: usize,
    pub table_views_base: *const NativeTableView,
    pub table_views_len: usize,
    pub store: *mut Store,
    pub current_module: *const ModuleInst,
    pub error: Option<WasmError>,
    memory_views: Vec<NativeMemoryView>,
    table_views: Vec<NativeTableView>,
    #[cfg(feature = "function-trace")]
    pub trace_stack: std::vec::Vec<u32>,
}

impl NativeContext {
    #[inline]
    pub fn new(store: *mut Store, stack_end: *mut u64) -> Self {
        let mut ctx = Self {
            stack_end,
            call_depth: 0,
            mem0_base: core::ptr::null_mut(),
            mem0_size: 0,
            globals_view: NativeGlobalsView::default(),
            memory_views_base: core::ptr::null(),
            memory_views_len: 0,
            table_views_base: core::ptr::null(),
            table_views_len: 0,
            store,
            current_module: core::ptr::null(),
            error: None,
            memory_views: Vec::new(),
            table_views: Vec::new(),
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
            .map(|memory| NativeMemoryView {
                base: if memory.data.is_empty() {
                    core::ptr::null_mut()
                } else {
                    memory.data.as_ptr() as *mut u8
                },
                len: memory.data.len(),
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
    pub const CALL_DEPTH: u32 = core::mem::offset_of!(NativeContext, call_depth) as u32;
    pub const MEM0_BASE: u32 = core::mem::offset_of!(NativeContext, mem0_base) as u32;
    pub const MEM0_SIZE: u32 = core::mem::offset_of!(NativeContext, mem0_size) as u32;
    pub const GLOBALS_VIEW: u32 = core::mem::offset_of!(NativeContext, globals_view) as u32;
    pub const MEMORY_VIEWS_BASE: u32 =
        core::mem::offset_of!(NativeContext, memory_views_base) as u32;
    pub const MEMORY_VIEWS_LEN: u32 = core::mem::offset_of!(NativeContext, memory_views_len) as u32;
    pub const TABLE_VIEWS_BASE: u32 = core::mem::offset_of!(NativeContext, table_views_base) as u32;
    pub const TABLE_VIEWS_LEN: u32 = core::mem::offset_of!(NativeContext, table_views_len) as u32;
    pub const STORE: u32 = core::mem::offset_of!(NativeContext, store) as u32;
    pub const CURRENT_MODULE: u32 = core::mem::offset_of!(NativeContext, current_module) as u32;
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
