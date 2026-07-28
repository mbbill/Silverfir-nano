//! No-std GC object arena for Wasm GC references.

use crate::{collections, vm::value::Value, WasmError};

/// Reference to one arena-allocated GC object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GcRef(usize);

impl GcRef {
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
enum GcObject {
    Struct(GcStruct),
    Array(GcArray),
}

#[derive(Debug, Clone)]
pub(crate) struct GcStruct {
    pub(crate) type_idx: u32,
    pub(crate) fields: collections::Vec<Value>,
}

#[derive(Debug, Clone)]
pub(crate) struct GcArray {
    pub(crate) type_idx: u32,
    pub(crate) elements: collections::Vec<Value>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GcHeap {
    objects: collections::Vec<GcObject>,
}

impl GcHeap {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            objects: collections::Vec::new(),
        }
    }

    pub(crate) fn alloc_struct(&mut self, type_idx: u32, fields: collections::Vec<Value>) -> GcRef {
        let index = self.objects.len();
        self.objects
            .push(GcObject::Struct(GcStruct { type_idx, fields }));
        GcRef::new(index)
    }

    pub(crate) fn alloc_array(
        &mut self,
        type_idx: u32,
        elements: collections::Vec<Value>,
    ) -> GcRef {
        let index = self.objects.len();
        self.objects
            .push(GcObject::Array(GcArray { type_idx, elements }));
        GcRef::new(index)
    }

    fn get(&self, gc_ref: GcRef) -> Result<&GcObject, WasmError> {
        self.objects
            .get(gc_ref.index())
            .ok_or_else(|| WasmError::invalid("invalid GC reference"))
    }

    fn get_mut(&mut self, gc_ref: GcRef) -> Result<&mut GcObject, WasmError> {
        self.objects
            .get_mut(gc_ref.index())
            .ok_or_else(|| WasmError::invalid("invalid GC reference"))
    }

    #[inline]
    pub(crate) fn get_struct(&self, gc_ref: GcRef) -> Result<&GcStruct, WasmError> {
        match self.get(gc_ref)? {
            GcObject::Struct(obj) => Ok(obj),
            GcObject::Array(_) => Err(WasmError::invalid("GC reference is not a struct")),
        }
    }

    #[inline]
    pub(crate) fn get_struct_mut(&mut self, gc_ref: GcRef) -> Result<&mut GcStruct, WasmError> {
        match self.get_mut(gc_ref)? {
            GcObject::Struct(obj) => Ok(obj),
            GcObject::Array(_) => Err(WasmError::invalid("GC reference is not a struct")),
        }
    }

    #[inline]
    pub(crate) fn get_array(&self, gc_ref: GcRef) -> Result<&GcArray, WasmError> {
        match self.get(gc_ref)? {
            GcObject::Array(obj) => Ok(obj),
            GcObject::Struct(_) => Err(WasmError::invalid("GC reference is not an array")),
        }
    }

    #[inline]
    pub(crate) fn get_array_mut(&mut self, gc_ref: GcRef) -> Result<&mut GcArray, WasmError> {
        match self.get_mut(gc_ref)? {
            GcObject::Array(obj) => Ok(obj),
            GcObject::Struct(_) => Err(WasmError::invalid("GC reference is not an array")),
        }
    }

    #[inline]
    pub(crate) fn type_idx(&self, gc_ref: GcRef) -> Result<u32, WasmError> {
        match self.get(gc_ref)? {
            GcObject::Struct(obj) => Ok(obj.type_idx),
            GcObject::Array(obj) => Ok(obj.type_idx),
        }
    }

    #[inline]
    pub(crate) fn is_struct(&self, gc_ref: GcRef) -> bool {
        matches!(self.objects.get(gc_ref.index()), Some(GcObject::Struct(_)))
    }

    #[inline]
    pub(crate) fn is_array(&self, gc_ref: GcRef) -> bool {
        matches!(self.objects.get(gc_ref.index()), Some(GcObject::Array(_)))
    }
}
