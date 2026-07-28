//! The JIT's resolution of constant expressions: references come from the
//! `Store`'s registries, globals from its entity model, GC objects from its
//! heap. The grammar itself lives in [`crate::vm::const_eval`].

use crate::collections;
use crate::error::WasmError;
use crate::module::entities::ConstExpr;
use crate::value_type::{AbstractHeapType, HeapType, RefType};
use crate::vm::const_eval::{self, ConstResolver};
use crate::vm::entities::FunctionInst;
use crate::vm::jit::store::Store;
use crate::vm::jit::value_encoding::try_raw_to_value_in_store;
use crate::vm::value::Value;

struct StoreResolver<'a> {
    store: &'a mut Store,
}

impl ConstResolver for StoreResolver<'_> {
    fn func_ref(&mut self, func_idx: u32) -> Result<Value, WasmError> {
        let func_idx = func_idx as usize;
        let module = self.store.module();
        if func_idx >= module.functions.len() {
            return Err(WasmError::invalid("ref.func: function index out of range"));
        }
        let handle = module
            .function_handle(func_idx)
            .ok_or_else(|| WasmError::invalid("ref.func: function handle missing"))?;
        let type_idx = match &module.functions[func_idx] {
            FunctionInst::Local { type_index, .. } => *type_index,
            FunctionInst::Host { .. } | FunctionInst::Linked { .. } => 0,
        };
        Ok(Value::Ref(
            handle,
            RefType::new(false, HeapType::Concrete(type_idx)),
        ))
    }

    fn global_get(&mut self, global_idx: u32) -> Result<Value, WasmError> {
        let module = self.store.module();
        let global = module
            .globals
            .get(global_idx as usize)
            .ok_or_else(|| WasmError::invalid("global.get: index out of range"))?;
        try_raw_to_value_in_store(global.raw(), global.value_type, self.store)
    }

    fn ref_i31(&mut self, value: i32) -> Result<Value, WasmError> {
        let handle = self.store.register_i31(value);
        Ok(Value::Ref(
            handle,
            RefType::new(false, HeapType::Abstract(AbstractHeapType::I31)),
        ))
    }

    fn alloc_struct(
        &mut self,
        type_idx: u32,
        fields: collections::Vec<Value>,
    ) -> Result<Value, WasmError> {
        let gc_ref = self
            .store
            .gc_heap()
            .borrow_mut()
            .alloc_struct(type_idx, fields);
        let handle = self.store.register_gc_ref(gc_ref);
        Ok(Value::Ref(
            handle,
            RefType::new(false, HeapType::Concrete(type_idx)),
        ))
    }

    fn alloc_array(
        &mut self,
        type_idx: u32,
        elements: collections::Vec<Value>,
    ) -> Result<Value, WasmError> {
        let gc_ref = self
            .store
            .gc_heap()
            .borrow_mut()
            .alloc_array(type_idx, elements);
        let handle = self.store.register_gc_ref(gc_ref);
        Ok(Value::Ref(
            handle,
            RefType::new(false, HeapType::Concrete(type_idx)),
        ))
    }
}

pub(crate) fn eval_const_expr(expr: &ConstExpr, store: &mut Store) -> Result<Value, WasmError> {
    let bytes: &[u8] = expr;
    // The type context is cloned out so the resolver can borrow the store
    // mutably; type contexts are Rc-backed and cheap to clone.
    let types = store.module().types.clone();
    const_eval::eval_const_expr(bytes, &types, &mut StoreResolver { store })
}
