//! Per-store exception-instance arena.
//!
//! `ExnRef` is deliberately a `Copy` index so `RefRegistryEntry::Exn` stays
//! `Copy`; the payload `Vec` lives out-of-line in the heap.

use crate::collections;
use crate::vm::tag::TagHandle;
use crate::vm::value::Value;

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
    pub(crate) fields: collections::Vec<Value>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExnHeap {
    entries: collections::Vec<ExnInstance>,
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
        self.entries.push(ExnInstance { tag, fields });
        ExnRef::new(index)
    }

    #[inline]
    pub(crate) fn get(&self, exn_ref: ExnRef) -> Option<&ExnInstance> {
        self.entries.get(exn_ref.index())
    }
}
