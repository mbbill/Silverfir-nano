//! Heap-backed memory fixtures for native runtime tests, including aliasing
//! paths that deliberately do not use guard-page memory.
use crate::vm::entities::{check_memory_quota, MemBacking, MemInst};
use crate::{collections, config::Config, error::WasmError, utils::limits::Limits};
use core::cell::{Cell, RefCell};
use tracked_alloc::rc::Rc;

pub(crate) fn heap_memory(config: &Config, limits: Limits) -> Result<MemInst, WasmError> {
    check_memory_quota(config, &limits)?;
    let initial_bytes = limits.min() * crate::constants::WASM_PAGE_SIZE;
    Ok(MemInst {
        backing: Rc::new(RefCell::new(MemBacking {
            data: collections::vec![0u8; initial_bytes],
            host_callback_borrowed: Cell::new(false),
            #[cfg(sf_has_guard_pages)]
            guard: None,
        })),
        limits,
    })
}
