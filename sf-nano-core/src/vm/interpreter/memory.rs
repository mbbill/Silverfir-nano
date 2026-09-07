//! The interpreter allocates linear memory on the heap at instantiation.
use crate::vm::entities::{check_memory_quota, MemBacking, MemInst};
use crate::{collections, config::Config, error::WasmError, utils::limits::Limits};
use core::cell::{Cell, RefCell};
use tracked_alloc::rc::Rc;

impl MemInst {
    pub(crate) fn new_heap(config: &Config, limits: Limits) -> Result<Self, WasmError> {
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
}
