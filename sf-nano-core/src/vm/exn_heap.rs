//! Per-store exception-instance arena.
//!
//! `ExnRef` is deliberately a `Copy` index. The heap and the shared reference
//! registry retain the same `Rc<ExnInstance>`, so an exception object stays
//! alive after its originating instance is dropped.

use crate::collections;
use crate::vm::tag::TagHandle;
use crate::vm::value::Value;
use tracked_alloc::rc::Rc;

/// Opaque index into a `Store`'s `ExnHeap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExnRef(usize);

impl ExnRef {
    #[inline]
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExnInstance {
    pub(crate) tag: TagHandle,
    // JIT-only builds retain payloads for exception identity/lifetime even
    // though only the interpreter currently reads them back at a catch.
    #[cfg_attr(not(feature = "interp"), allow(dead_code))]
    pub(crate) fields: collections::Vec<Value>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExnHeap {
    entries: collections::Vec<Rc<ExnInstance>>,
}

impl ExnHeap {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            entries: collections::Vec::new(),
        }
    }

    pub(crate) fn alloc(&mut self, tag: TagHandle, fields: collections::Vec<Value>) -> ExnRef {
        let index = self.entries.len();
        self.entries.push(Rc::new(ExnInstance { tag, fields }));
        ExnRef::new(index)
    }

    #[inline]
    pub(crate) fn get_shared(&self, exn_ref: ExnRef) -> Option<Rc<ExnInstance>> {
        self.entries.get(exn_ref.index()).cloned()
    }
}
