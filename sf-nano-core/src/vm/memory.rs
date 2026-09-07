//! Host memory views and the world access discipline at embedding boundaries.

use alloc::rc::Rc;
use core::{
    cell::Cell,
    fmt,
    ops::{Deref, DerefMut},
};

use crate::{vm::entities::MemInst, WasmError};

/// Memory views borrow the world, like a slice borrowed from a Wasmtime store.
/// Guest-to-guest calls do not touch these counters. Reentrant embedding calls
/// are allowed, but cannot begin while an external memory view exists.
#[derive(Default)]
pub(crate) struct WorldBorrowState {
    executions: Cell<usize>,
    readers: Cell<usize>,
    writer: Cell<bool>,
}

impl WorldBorrowState {
    pub(crate) fn enter(self: &Rc<Self>) -> Result<ExecutionGuard, WasmError> {
        if self.readers.get() != 0 || self.writer.get() {
            return Err(WasmError::trap(
                "runtime world has an outstanding memory view",
            ));
        }
        let count = self
            .executions
            .get()
            .checked_add(1)
            .ok_or_else(|| WasmError::trap("runtime world execution depth exhausted"))?;
        self.executions.set(count);
        Ok(ExecutionGuard(Rc::clone(self)))
    }

    pub(crate) fn read(
        self: &Rc<Self>,
        memory: impl FnOnce() -> Option<MemInst>,
    ) -> Result<MemoryView, WasmError> {
        if self.executions.get() != 0 || self.writer.get() {
            return Err(WasmError::trap(
                "runtime world is executing or memory is mutably borrowed",
            ));
        }
        let count = self
            .readers
            .get()
            .checked_add(1)
            .ok_or_else(|| WasmError::trap("runtime world memory borrow count exhausted"))?;
        let memory = memory().ok_or_else(|| WasmError::invalid("instance has no linear memory"))?;
        self.readers.set(count);
        Ok(MemoryView {
            memory,
            state: Rc::clone(self),
        })
    }

    pub(crate) fn write(
        self: &Rc<Self>,
        memory: impl FnOnce() -> Option<MemInst>,
    ) -> Result<MemoryViewMut, WasmError> {
        if self.executions.get() != 0 || self.readers.get() != 0 || self.writer.get() {
            return Err(WasmError::trap(
                "runtime world is executing or memory is already borrowed",
            ));
        }
        let memory = memory().ok_or_else(|| WasmError::invalid("instance has no linear memory"))?;
        self.writer.set(true);
        Ok(MemoryViewMut {
            memory,
            state: Rc::clone(self),
        })
    }
}

pub(crate) struct ExecutionGuard(Rc<WorldBorrowState>);

impl Drop for ExecutionGuard {
    fn drop(&mut self) {
        self.0.executions.set(self.0.executions.get() - 1);
    }
}

/// A read-only view of linear memory.
///
/// Dereferences to `[u8]`. Multiple readers may coexist. While any view lives,
/// calls and instantiation in its runtime world return an error. Drop the view
/// before resuming execution. The backing remains alive if its instance is
/// released, and views cannot cross threads.
pub struct MemoryView {
    memory: MemInst,
    state: Rc<WorldBorrowState>,
}

impl Deref for MemoryView {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // The view owns the backing; the world borrow blocks execution,
        // initialization and mutable host views until the last reader drops.
        unsafe { core::slice::from_raw_parts(self.memory.memory_ptr(), self.memory.memory_len()) }
    }
}

impl Drop for MemoryView {
    fn drop(&mut self) {
        self.state.readers.set(self.state.readers.get() - 1);
    }
}

impl fmt::Debug for MemoryView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}

/// An exclusive mutable view of linear memory.
///
/// Dereferences to `[u8]` and supports slice mutation. Other memory views,
/// calls and instantiation in this runtime world fail until it is dropped.
/// The backing remains alive if its instance is released.
pub struct MemoryViewMut {
    memory: MemInst,
    state: Rc<WorldBorrowState>,
}

impl Deref for MemoryViewMut {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // The world admits exactly this view and no running guest.
        unsafe { core::slice::from_raw_parts(self.memory.memory_ptr(), self.memory.memory_len()) }
    }
}

impl DerefMut for MemoryViewMut {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // The exclusive world borrow prevents any other slice or guest access;
        // &mut self also excludes aliases through this guard.
        unsafe {
            core::slice::from_raw_parts_mut(self.memory.memory_ptr(), self.memory.memory_len())
        }
    }
}

impl Drop for MemoryViewMut {
    fn drop(&mut self) {
        self.state.writer.set(false);
    }
}

impl fmt::Debug for MemoryViewMut {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::WorldBorrowState;
    use crate::{
        utils::limits::Limits,
        vm::entities::{MemBacking, MemInst},
    };
    use alloc::{rc::Rc, vec};
    use core::cell::{Cell, RefCell};

    #[test]
    fn memory_view_aliasing_and_backing_lifetime() {
        // Heap-only backing keeps this test executable in Miri on either
        // aliasing model without entering generated code or OS mappings.
        let memory = MemInst {
            backing: Rc::new(RefCell::new(MemBacking {
                data: vec![1, 2, 3],
                host_callback_borrowed: Cell::new(false),
                #[cfg(sf_has_guard_pages)]
                guard: None,
            })),
            limits: Limits::new(0, None).unwrap(),
        };
        let state = Rc::new(WorldBorrowState::default());
        assert!(state.read(|| None).is_err());
        let execution = state.enter().unwrap();
        assert!(state.read(|| Some(memory.clone())).is_err());
        drop(execution);
        let first = state.read(|| Some(memory.clone())).unwrap();
        let second = state.read(|| Some(memory.clone())).unwrap();
        assert_eq!(&*first, &[1, 2, 3]);
        assert_eq!(&*second, &[1, 2, 3]);
        assert!(state.write(|| Some(memory.clone())).is_err());
        assert!(state.enter().is_err());
        drop(first);
        assert!(state.enter().is_err());
        drop(second);
        let mut writer = state.write(|| Some(memory.clone())).unwrap();
        drop(memory);
        writer[1] = 9;
        assert_eq!(&*writer, &[1, 9, 3]);
        assert!(state.read(|| None).is_err());
        drop(writer);
        let execution = state.enter().unwrap();
        drop(execution);
    }
}
