use crate::{
    collections,
    constants::WASM_PAGE_SIZE,
    error::WasmError,
    module::type_defs::{CompositeType, FieldType, PackedType, StorageType},
    value_type::ValueType,
    vm::{
        entities::{MemInst, TableInst},
        jit::arch::backend_config,
        jit::gc_heap::GcRef,
        jit::runtime::{
            self,
            common::{internal_error, trap_error},
            context::{NativeContext, PendingEscape},
            StoreAccess,
        },
        jit::store::Store,
        jit::value_encoding::{
            absolutize, retag_for_container, try_machine_raw_to_value_in_store,
            value_to_machine_raw_in_store,
        },
        link::{ref_type_matches, InstanceId, RefRegistryEntry, RefTypeOwner},
        tag::TagIdentity,
        value::{machine_raw_to_ref, ref_to_machine_raw, RefValue, Value},
    },
};

#[inline]
fn current_store(ctx: &NativeContext) -> Result<&Store, WasmError> {
    let store = ctx
        .store()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    Ok(store)
}

#[inline]
fn current_store_mut(ctx: &mut NativeContext) -> Result<&mut Store, WasmError> {
    ctx.store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))
}

#[inline]
pub(super) fn active_gp_unit_bytes() -> u8 {
    backend_config().gp_unit_bytes
}

#[inline]
fn ref_from_machine(raw_ref: u64) -> RefValue {
    machine_raw_to_ref(raw_ref, active_gp_unit_bytes())
}

#[inline]
fn ref_to_machine(handle: RefValue) -> u64 {
    ref_to_machine_raw(handle, active_gp_unit_bytes())
}

pub(super) fn do_ref_absolutize(ctx: &NativeContext, raw_ref: u64) -> Result<u64, WasmError> {
    Ok(ref_to_machine(absolutize(
        current_store(ctx)?,
        ref_from_machine(raw_ref),
    )))
}

#[inline]
fn value_from_machine(
    ctx: &NativeContext,
    raw: u64,
    ty: crate::value_type::ValueType,
) -> Result<Value, WasmError> {
    try_machine_raw_to_value_in_store(
        raw,
        ty,
        active_gp_unit_bytes(),
        current_store(ctx).expect("store"),
    )
}

#[inline]
fn value_to_machine(store: &mut Store, value: Value) -> u64 {
    value_to_machine_raw_in_store(value, active_gp_unit_bytes(), store)
}

#[inline]
unsafe fn frame_read(frame: *mut u64, slot: u16) -> u64 {
    unsafe { *frame.add(slot as usize) }
}

#[inline]
fn require_arg_slots(actual: u16, expected: u16, label: &'static str) -> Result<(), WasmError> {
    if actual != expected {
        return Err(internal_error(label));
    }
    Ok(())
}

#[inline]
fn frame_slot(start_slot: u16, index: u16) -> Result<u16, WasmError> {
    start_slot
        .checked_add(index)
        .ok_or_else(|| internal_error("EH helper frame slot overflow"))
}

fn read_exn_payload_fields(
    frame: *mut u64,
    start_slot: u16,
    slot_count: u16,
    params: &[ValueType],
    store: &Store,
    label: &'static str,
) -> Result<collections::Vec<Value>, WasmError> {
    require_arg_slots(slot_count, params.len() as u16, label)?;
    params
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            let slot = frame_slot(start_slot, index as u16)?;
            let raw = unsafe { frame_read(frame, slot) };
            try_machine_raw_to_value_in_store(raw, *ty, active_gp_unit_bytes(), store)
        })
        .collect::<Result<_, _>>()
}

pub(super) fn do_eh_alloc_exn_ref(
    ctx: &mut NativeContext,
    frame: *mut u64,
    tag_idx: u32,
    start_slot: u16,
    slot_count: u16,
) -> Result<u64, WasmError> {
    let store = current_store_mut(ctx)?;
    let tag_inst = store
        .module()
        .tags
        .get(tag_idx as usize)
        .copied()
        .ok_or_else(|| internal_error("eh_alloc_exn_ref: tag index out of range"))?;
    let tag_ty = store
        .module()
        .types
        .get_function_type(tag_inst.type_index)
        .cloned()
        .ok_or_else(|| internal_error("eh_alloc_exn_ref: tag has no function type"))?;
    let fields = read_exn_payload_fields(
        frame,
        start_slot,
        slot_count,
        tag_ty.params(),
        store,
        "eh_alloc_exn_ref args span does not match tag arity",
    )?;

    let handle = store.alloc_exn(tag_inst.handle, fields);
    Ok(ref_to_machine(handle))
}

pub(super) fn do_eh_throw(
    ctx: &mut NativeContext,
    frame: *mut u64,
    tag_idx: u32,
    start_slot: u16,
    slot_count: u16,
) -> Result<(), WasmError> {
    let (tag_identity, exn_handle) = {
        let store = current_store_mut(ctx)?;
        let tag_inst = store
            .module()
            .tags
            .get(tag_idx as usize)
            .copied()
            .ok_or_else(|| internal_error("eh_throw: tag index out of range"))?;
        let tag_ty = store
            .module()
            .types
            .get_function_type(tag_inst.type_index)
            .cloned()
            .ok_or_else(|| internal_error("eh_throw: tag has no function type"))?;
        let fields = read_exn_payload_fields(
            frame,
            start_slot,
            slot_count,
            tag_ty.params(),
            store,
            "eh_throw args span does not match tag arity",
        )?;
        let exn_handle = store.alloc_exn(tag_inst.handle, fields);
        (tag_inst.handle, exn_handle)
    };

    raise_exception(ctx, tag_identity, exn_handle)
}

pub(super) fn do_eh_throw_ref(
    ctx: &mut NativeContext,
    frame: *mut u64,
    exnref_slot: u16,
) -> Result<(), WasmError> {
    let raw = unsafe { frame_read(frame, exnref_slot) };
    let exn_handle = ref_from_machine(raw);
    if exn_handle.is_null() {
        return Err(trap_error("null reference"));
    }

    let tag_identity = {
        let store = current_store(ctx)?;
        let exn_idx = exn_handle
            .pooled_index()
            .ok_or_else(|| trap_error("invalid exception reference"))?;
        let ref_registry = store.clone_ref_registry();
        let registry = ref_registry.borrow();
        let entry = registry
            .get(exn_idx)
            .cloned()
            .ok_or_else(|| trap_error("invalid exception reference"))?;
        match entry {
            // Exception objects are immutable and can enter the registry only
            // through the typed Wasm throw path or validated HostThrow path.
            RefRegistryEntry::Exn(exn) => exn.tag,
            _ => return Err(trap_error("throw_ref operand is not an exception")),
        }
    };

    raise_exception(ctx, tag_identity, exn_handle)
}

fn raise_exception(
    ctx: &mut NativeContext,
    tag_identity: TagIdentity,
    exn_handle: RefValue,
) -> Result<(), WasmError> {
    ctx.pending_escape = PendingEscape::Throw {
        exn: exn_handle,
        tag: tag_identity,
    };

    Err(WasmError::Exception {
        exn: exn_handle,
        tag: tag_identity,
        module_tag_name: None,
    })
}

#[inline]
pub(super) fn memory_mut(ctx: &mut NativeContext, mem_idx: u32) -> Result<&mut MemInst, WasmError> {
    current_store_mut(ctx)?
        .module_mut()
        .memories
        .get_mut(mem_idx as usize)
        .ok_or_else(|| internal_error("preserved helper referenced invalid memory index"))
}

#[inline]
pub(super) fn table_mut(
    ctx: &mut NativeContext,
    table_idx: u32,
) -> Result<&mut TableInst, WasmError> {
    current_store_mut(ctx)?
        .module_mut()
        .tables
        .get_mut(table_idx as usize)
        .ok_or_else(|| internal_error("preserved helper referenced invalid table index"))
}

pub(super) fn do_ref_func(ctx: &mut NativeContext, func_idx: u32) -> Result<u64, WasmError> {
    let store = current_store(ctx)?;
    let handle = store
        .module()
        .function_handle(func_idx as usize)
        .ok_or_else(|| internal_error("preserved helper referenced invalid function index"))?;
    Ok(ref_to_machine(handle))
}

pub(super) fn do_ref_as_non_null(raw_ref: u64) -> Result<u64, WasmError> {
    let handle = ref_from_machine(raw_ref);
    if handle.is_null() {
        return Err(trap_error("null reference"));
    }
    Ok(ref_to_machine(handle))
}

#[inline]
pub(super) fn do_ref_eq(lhs: u64, rhs: u64) -> u64 {
    u64::from(lhs == rhs)
}

pub(super) fn do_ref_i31(ctx: &mut NativeContext, raw_value: u64) -> Result<u64, WasmError> {
    let handle = current_store_mut(ctx)?.register_i31(raw_value as u32 as i32);
    Ok(ref_to_machine(handle))
}

#[cfg(sf_has_simd)]
pub(super) fn do_v128_to_raw(ctx: &mut NativeContext, bytes: [u8; 16]) -> Result<u64, WasmError> {
    Ok(current_store_mut(ctx)?.intern_v128(bytes))
}

#[cfg(sf_has_simd)]
pub(super) fn do_v128_from_raw(ctx: &NativeContext, raw: u64) -> Result<[u8; 16], WasmError> {
    current_store(ctx)?
        .get_v128(raw)
        .ok_or_else(|| internal_error("invalid v128 raw value"))
}

fn decode_i31(store: &Store, handle: RefValue) -> Result<i32, WasmError> {
    if handle.is_null() {
        return Err(trap_error("null i31 reference"));
    }
    if handle.is_extern() {
        return Err(trap_error("invalid i31 reference"));
    }
    match store.ref_entry_for_handle(handle) {
        Some(RefRegistryEntry::I31(value)) => Ok(value),
        _ => Err(trap_error("invalid i31 reference")),
    }
}

pub(super) fn do_i31_get_s(ctx: &mut NativeContext, raw_ref: u64) -> Result<u64, WasmError> {
    let handle = ref_from_machine(raw_ref);
    let value = decode_i31(current_store(ctx)?, handle)? as u32 & 0x7fff_ffff;
    let signed = if (value & 0x4000_0000) != 0 {
        value | 0x8000_0000
    } else {
        value
    };
    Ok(signed as u64)
}

pub(super) fn do_i31_get_u(ctx: &mut NativeContext, raw_ref: u64) -> Result<u64, WasmError> {
    let handle = ref_from_machine(raw_ref);
    let value = decode_i31(current_store(ctx)?, handle)? as u32 & 0x7fff_ffff;
    Ok(value as u64)
}

pub(super) fn do_any_convert_extern(raw_ref: u64) -> Result<u64, WasmError> {
    let handle = ref_from_machine(raw_ref)
        .to_any()
        .map_err(|_| trap_error("any.convert_extern: value is not in extern hierarchy"))?;
    Ok(ref_to_machine(handle))
}

pub(super) fn do_extern_convert_any(raw_ref: u64) -> Result<u64, WasmError> {
    let handle = ref_from_machine(raw_ref)
        .to_extern()
        .map_err(|_| trap_error("extern.convert_any: value is not in aggregate hierarchy"))?;
    Ok(ref_to_machine(handle))
}

#[inline]
fn decode_reftype(imm0: u32, imm1: u32) -> crate::value_type::RefType {
    let encoded = (u64::from(imm1) << 32) | u64::from(imm0);
    crate::value_type::RefType::decode_from_u64(encoded)
}

pub(super) fn do_ref_test(
    ctx: &NativeContext,
    imm0: u32,
    imm1: u32,
    raw_ref: u64,
) -> Result<u64, WasmError> {
    let target_type = decode_reftype(imm0, imm1);
    let handle = ref_from_machine(raw_ref);
    let is_match = if handle.is_null() {
        target_type.nullable
    } else {
        ref_type_matches(
            handle,
            &target_type.heap_type,
            RefTypeOwner::Jit(current_store(ctx)?),
        )?
    };
    Ok(u64::from(is_match))
}

pub(super) fn do_ref_cast(
    ctx: &NativeContext,
    imm0: u32,
    imm1: u32,
    raw_ref: u64,
) -> Result<u64, WasmError> {
    let target_type = decode_reftype(imm0, imm1);
    let handle = ref_from_machine(raw_ref);

    if handle.is_null() {
        return if target_type.nullable {
            Ok(ref_to_machine(handle))
        } else {
            Err(trap_error("cast failure"))
        };
    }

    if ref_type_matches(
        handle,
        &target_type.heap_type,
        RefTypeOwner::Jit(current_store(ctx)?),
    )? {
        Ok(ref_to_machine(handle))
    } else {
        Err(trap_error("cast failure"))
    }
}

pub(super) fn do_struct_new_default(
    ctx: &mut NativeContext,
    type_idx: u32,
) -> Result<u64, WasmError> {
    let store = current_store_mut(ctx)?;
    let fields = {
        let def_type = store
            .module()
            .types
            .get(type_idx)
            .ok_or_else(|| internal_error("preserved helper referenced invalid type index"))?;
        let struct_type = match &def_type.composite {
            CompositeType::Struct(struct_type) => struct_type,
            _ => return Err(internal_error("preserved helper expected a struct type")),
        };
        struct_type
            .fields
            .iter()
            .map(|field| Value::default_for_type(field.storage.to_valtype()))
            .collect()
    };
    let gc_ref = store.gc_heap().borrow_mut().alloc_struct(type_idx, fields);
    let handle = store.register_gc_ref(gc_ref);
    Ok(ref_to_machine(handle))
}

pub(super) fn do_struct_new(
    ctx: &mut NativeContext,
    type_idx: u32,
    payload_ptr: u64,
    count: u32,
) -> Result<u64, WasmError> {
    let store = current_store_mut(ctx)?;
    let count = count as usize;
    let raw_fields = if count == 0 {
        &[][..]
    } else {
        let ptr = payload_ptr as usize as *const u64;
        if ptr.is_null() {
            return Err(internal_error("struct.new payload pointer was null"));
        }
        unsafe { core::slice::from_raw_parts(ptr, count) }
    };
    let fields = {
        let def_type = store
            .module()
            .types
            .get(type_idx)
            .ok_or_else(|| internal_error("preserved helper referenced invalid type index"))?;
        let struct_type = match &def_type.composite {
            CompositeType::Struct(struct_type) => struct_type,
            _ => return Err(internal_error("preserved helper expected a struct type")),
        };
        if struct_type.fields.len() != count {
            return Err(internal_error(
                "struct.new field count mismatches struct type",
            ));
        }
        struct_type
            .fields
            .iter()
            .zip(raw_fields)
            .map(|(field, raw)| {
                try_machine_raw_to_value_in_store(
                    *raw,
                    field.storage.to_valtype(),
                    active_gp_unit_bytes(),
                    store,
                )
            })
            .collect::<Result<_, _>>()?
    };
    let gc_ref = store.gc_heap().borrow_mut().alloc_struct(type_idx, fields);
    let handle = store.register_gc_ref(gc_ref);
    Ok(ref_to_machine(handle))
}

fn struct_field_storage(
    store: &Store,
    type_idx: u32,
    field_idx: u32,
) -> Result<StorageType, WasmError> {
    let def_type = store
        .module()
        .types
        .get(type_idx)
        .ok_or_else(|| internal_error("preserved helper referenced invalid type index"))?;
    let struct_type = match &def_type.composite {
        CompositeType::Struct(struct_type) => struct_type,
        _ => return Err(internal_error("preserved helper expected a struct type")),
    };
    struct_type
        .fields
        .get(field_idx as usize)
        .map(|field| field.storage)
        .ok_or_else(|| internal_error("preserved helper referenced invalid field index"))
}

struct ResolvedGcRef {
    access: StoreAccess<'static>,
    gc_ref: GcRef,
}

impl ResolvedGcRef {
    #[inline]
    fn owner(&self) -> InstanceId {
        self.access.id()
    }

    #[inline]
    fn with_store<R>(
        &self,
        use_store: impl for<'store> FnOnce(&'store Store) -> Result<R, WasmError>,
    ) -> Result<R, WasmError> {
        self.access.with_store(use_store)?
    }

    #[inline]
    fn with_store_mut<R>(
        &mut self,
        use_store: impl for<'store> FnOnce(&'store mut Store) -> Result<R, WasmError>,
    ) -> Result<R, WasmError> {
        self.access.with_store_mut(use_store)?
    }
}

fn gc_store_access(
    ctx: &NativeContext,
    owner: InstanceId,
    invalid_ref: &'static str,
) -> Result<StoreAccess<'static>, WasmError> {
    let (current_id, instance_backref) = {
        let current = current_store(ctx)?;
        (
            current.instance_backref().self_id(),
            current.instance_backref().clone(),
        )
    };
    if owner == current_id {
        return runtime::current_store_access(ctx);
    }

    let token = instance_backref
        .checkout(owner)
        .ok_or_else(|| trap_error(invalid_ref))?;
    if token.jit_pointer().is_none() {
        return Err(trap_error(invalid_ref));
    }
    Ok(StoreAccess::checked_out(token))
}

fn resolve_struct_ref(ctx: &NativeContext, handle: RefValue) -> Result<ResolvedGcRef, WasmError> {
    if handle.is_null() {
        return Err(trap_error("null structure reference"));
    }
    if handle.is_extern() {
        return Err(trap_error("invalid structure reference"));
    }

    let (owner, gc_ref) = match current_store(ctx)?.ref_entry_for_handle(handle) {
        Some(RefRegistryEntry::Gc { owner, gc_ref }) => (owner, gc_ref),
        _ => return Err(trap_error("invalid structure reference")),
    };
    Ok(ResolvedGcRef {
        access: gc_store_access(ctx, owner, "invalid structure reference")?,
        gc_ref,
    })
}

fn extend_packed_field(value: Value, storage: StorageType, signed: bool) -> Value {
    match (value, storage) {
        (Value::I32(raw), StorageType::Packed(PackedType::I8)) => {
            if signed {
                Value::I32((raw as i8) as i32)
            } else {
                Value::I32(raw & 0xFF)
            }
        }
        (Value::I32(raw), StorageType::Packed(PackedType::I16)) => {
            if signed {
                Value::I32((raw as i16) as i32)
            } else {
                Value::I32(raw & 0xFFFF)
            }
        }
        _ => value,
    }
}

fn array_element_field(store: &Store, type_idx: u32) -> Result<FieldType, WasmError> {
    let def_type = store
        .module()
        .types
        .get(type_idx)
        .ok_or_else(|| internal_error("preserved helper referenced invalid type index"))?;
    let array_type = match &def_type.composite {
        CompositeType::Array(array_type) => array_type,
        _ => return Err(internal_error("preserved helper expected an array type")),
    };
    Ok(array_type.element)
}

fn array_element_storage(store: &Store, type_idx: u32) -> Result<StorageType, WasmError> {
    Ok(array_element_field(store, type_idx)?.storage)
}

fn data_array_element_size(storage: StorageType) -> Result<usize, WasmError> {
    match storage {
        StorageType::Packed(PackedType::I8) => Ok(1),
        StorageType::Packed(PackedType::I16) => Ok(2),
        StorageType::Val(ValueType::I32 | ValueType::F32) => Ok(4),
        StorageType::Val(ValueType::I64 | ValueType::F64) => Ok(8),
        StorageType::Val(ValueType::V128) => Err(internal_error(
            "v128 arrays are not supported by preserved helpers",
        )),
        StorageType::Val(ValueType::Ref(_) | ValueType::Unknown) => Err(internal_error(
            "preserved helper expected a numeric array element type",
        )),
    }
}

fn value_from_data_bytes(storage: StorageType, bytes: &[u8]) -> Result<Value, WasmError> {
    Ok(match storage {
        StorageType::Packed(PackedType::I8) => Value::I32(i32::from(bytes[0])),
        StorageType::Packed(PackedType::I16) => {
            Value::I32(i32::from(u16::from_le_bytes([bytes[0], bytes[1]])))
        }
        StorageType::Val(ValueType::I32) => {
            Value::I32(i32::from_le_bytes(bytes.try_into().unwrap()))
        }
        StorageType::Val(ValueType::I64) => {
            Value::I64(i64::from_le_bytes(bytes.try_into().unwrap()))
        }
        StorageType::Val(ValueType::F32) => Value::F32(f32::from_bits(u32::from_le_bytes(
            bytes.try_into().unwrap(),
        ))),
        StorageType::Val(ValueType::F64) => Value::F64(f64::from_bits(u64::from_le_bytes(
            bytes.try_into().unwrap(),
        ))),
        StorageType::Val(ValueType::V128) => {
            return Err(internal_error(
                "v128 arrays are not supported by preserved helpers",
            ));
        }
        StorageType::Val(ValueType::Ref(_) | ValueType::Unknown) => {
            return Err(internal_error(
                "preserved helper expected a numeric array element type",
            ));
        }
    })
}

fn resolve_array_ref(ctx: &NativeContext, handle: RefValue) -> Result<ResolvedGcRef, WasmError> {
    if handle.is_null() {
        return Err(trap_error("null array reference"));
    }
    if handle.is_extern() {
        return Err(trap_error("invalid array reference"));
    }

    let (owner, gc_ref) = match current_store(ctx)?.ref_entry_for_handle(handle) {
        Some(RefRegistryEntry::Gc { owner, gc_ref }) => (owner, gc_ref),
        _ => return Err(trap_error("invalid array reference")),
    };
    Ok(ResolvedGcRef {
        access: gc_store_access(ctx, owner, "invalid array reference")?,
        gc_ref,
    })
}

pub(super) fn do_struct_get(
    ctx: &mut NativeContext,
    type_idx: u32,
    field_idx: u32,
    raw_ref: u64,
    signed: Option<bool>,
) -> Result<u64, WasmError> {
    let storage = struct_field_storage(current_store(ctx)?, type_idx, field_idx)?;
    let origin = resolve_struct_ref(ctx, ref_from_machine(raw_ref))?;
    let gc_ref = origin.gc_ref;
    let value = origin.with_store(|origin_store| {
        let gc_heap = origin_store.gc_heap().borrow();
        let value = gc_heap
            .get_struct(gc_ref)
            .map_err(|_| trap_error("invalid structure reference"))?
            .fields
            .get(field_idx as usize)
            .copied()
            .ok_or_else(|| internal_error("preserved helper referenced invalid struct field"))?;
        Ok(value)
    })?;
    let value = match signed {
        Some(is_signed) => extend_packed_field(value, storage, is_signed),
        None => value,
    };
    // GC fields are absolute Values. The origin scope is closed before the
    // value is localized into the current instance's frame representation.
    Ok(value_to_machine(current_store_mut(ctx)?, value))
}

pub(super) fn do_struct_set(
    ctx: &mut NativeContext,
    type_idx: u32,
    field_idx: u32,
    raw_ref: u64,
    raw_value: u64,
) -> Result<(), WasmError> {
    let handle = ref_from_machine(raw_ref);
    let storage = struct_field_storage(current_store(ctx)?, type_idx, field_idx)?;
    let value = value_from_machine(ctx, raw_value, storage.to_valtype())?;
    let mut origin = resolve_struct_ref(ctx, handle)?;
    let gc_ref = origin.gc_ref;
    origin.with_store_mut(|origin_store| {
        let mut gc_heap = origin_store.gc_heap().borrow_mut();
        let gc_struct = gc_heap
            .get_struct_mut(gc_ref)
            .map_err(|_| trap_error("invalid structure reference"))?;
        let Some(field) = gc_struct.fields.get_mut(field_idx as usize) else {
            return Err(internal_error(
                "preserved helper referenced invalid struct field",
            ));
        };
        *field = value;
        Ok(())
    })
}

pub(super) fn do_array_new_default(
    ctx: &mut NativeContext,
    type_idx: u32,
    length_raw: u64,
) -> Result<u64, WasmError> {
    let (elem_type, length) = {
        let store = current_store(ctx)?;
        let def_type = store
            .module()
            .types
            .get(type_idx)
            .ok_or_else(|| internal_error("preserved helper referenced invalid type index"))?;
        let array_type = match &def_type.composite {
            CompositeType::Array(array_type) => array_type,
            _ => return Err(internal_error("preserved helper expected an array type")),
        };
        (
            array_type.element.storage.to_valtype(),
            length_raw as u32 as usize,
        )
    };
    let init = Value::default_for_type(elem_type);
    let mut elements = crate::collections::Vec::new();
    elements
        .try_reserve(length)
        .map_err(|_| trap_error("out of memory"))?;
    elements.resize(length, init);
    let handle = {
        let store = current_store_mut(ctx)?;
        let gc_ref = store.gc_heap().borrow_mut().alloc_array(type_idx, elements);
        store.register_gc_ref(gc_ref)
    };
    Ok(ref_to_machine(handle))
}

pub(super) fn do_array_new(
    ctx: &mut NativeContext,
    type_idx: u32,
    init_raw: u64,
    length_raw: u64,
) -> Result<u64, WasmError> {
    let (init, length) = {
        let store = current_store_mut(ctx)?;
        let storage = array_element_storage(store, type_idx)?;
        let elem_type = storage.to_valtype();
        (
            try_machine_raw_to_value_in_store(init_raw, elem_type, active_gp_unit_bytes(), store)?,
            length_raw as u32 as usize,
        )
    };
    let mut elements = crate::collections::Vec::new();
    elements
        .try_reserve(length)
        .map_err(|_| trap_error("out of memory"))?;
    elements.resize(length, init);
    let handle = {
        let store = current_store_mut(ctx)?;
        let gc_ref = store.gc_heap().borrow_mut().alloc_array(type_idx, elements);
        store.register_gc_ref(gc_ref)
    };
    Ok(ref_to_machine(handle))
}

pub(super) fn do_array_get(
    ctx: &mut NativeContext,
    type_idx: u32,
    raw_ref: u64,
    raw_index: u64,
    signed: Option<bool>,
) -> Result<u64, WasmError> {
    let storage = array_element_storage(current_store(ctx)?, type_idx)?;
    let origin = resolve_array_ref(ctx, ref_from_machine(raw_ref))?;
    let gc_ref = origin.gc_ref;
    let index = raw_index as u32 as usize;
    let value = origin.with_store(|origin_store| {
        let gc_heap = origin_store.gc_heap().borrow();
        let value = gc_heap
            .get_array(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?
            .elements
            .get(index)
            .copied()
            .ok_or_else(|| trap_error("array element index out of bounds"))?;
        Ok(value)
    })?;
    let value = match signed {
        Some(is_signed) => extend_packed_field(value, storage, is_signed),
        None => value,
    };
    // As for struct.get, localize only after the absolute GC value has left
    // the origin materialization scope.
    Ok(value_to_machine(current_store_mut(ctx)?, value))
}

pub(super) fn do_array_len(ctx: &NativeContext, raw_ref: u64) -> Result<u64, WasmError> {
    let origin = resolve_array_ref(ctx, ref_from_machine(raw_ref))?;
    let gc_ref = origin.gc_ref;
    let len = origin.with_store(|origin_store| {
        let gc_heap = origin_store.gc_heap().borrow();
        Ok(gc_heap
            .get_array(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?
            .elements
            .len())
    })?;
    Ok(len as u32 as u64)
}

pub(super) fn do_array_set(
    ctx: &mut NativeContext,
    type_idx: u32,
    raw_ref: u64,
    raw_index: u64,
    raw_value: u64,
) -> Result<(), WasmError> {
    let handle = ref_from_machine(raw_ref);
    let index = raw_index as u32 as usize;
    let storage = array_element_storage(current_store(ctx)?, type_idx)?;
    let value = value_from_machine(ctx, raw_value, storage.to_valtype())?;
    let mut origin = resolve_array_ref(ctx, handle)?;
    let gc_ref = origin.gc_ref;
    origin.with_store_mut(|origin_store| {
        let mut gc_heap = origin_store.gc_heap().borrow_mut();
        let gc_array = gc_heap
            .get_array_mut(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?;
        let Some(element) = gc_array.elements.get_mut(index) else {
            return Err(trap_error("array element index out of bounds"));
        };
        *element = value;
        Ok(())
    })
}

pub(super) fn do_array_new_fixed(
    ctx: &mut NativeContext,
    type_idx: u32,
    payload_ptr: u64,
    count: u32,
) -> Result<u64, WasmError> {
    let store = current_store_mut(ctx)?;
    let storage = array_element_storage(store, type_idx)?;
    let count = count as usize;
    let raw_values = if count == 0 {
        &[][..]
    } else {
        let ptr = payload_ptr as usize as *const u64;
        if ptr.is_null() {
            return Err(internal_error("array.new_fixed payload pointer was null"));
        }
        unsafe { core::slice::from_raw_parts(ptr, count) }
    };
    let mut elements = crate::collections::Vec::new();
    elements
        .try_reserve(count)
        .map_err(|_| trap_error("out of memory"))?;
    for raw in raw_values {
        elements.push(try_machine_raw_to_value_in_store(
            *raw,
            storage.to_valtype(),
            active_gp_unit_bytes(),
            store,
        )?);
    }
    let gc_ref = store.gc_heap().borrow_mut().alloc_array(type_idx, elements);
    let handle = store.register_gc_ref(gc_ref);
    Ok(ref_to_machine(handle))
}

pub(super) fn do_array_fill(
    ctx: &mut NativeContext,
    type_idx: u32,
    raw_ref: u64,
    raw_index: u64,
    raw_value: u64,
    raw_len: u64,
) -> Result<(), WasmError> {
    let handle = ref_from_machine(raw_ref);
    let index = raw_index as u32 as usize;
    let len = raw_len as u32 as usize;
    let storage = array_element_storage(current_store(ctx)?, type_idx)?;
    let value = value_from_machine(ctx, raw_value, storage.to_valtype())?;
    let mut origin = resolve_array_ref(ctx, handle)?;
    let gc_ref = origin.gc_ref;
    origin.with_store_mut(|origin_store| {
        let mut gc_heap = origin_store.gc_heap().borrow_mut();
        let gc_array = gc_heap
            .get_array_mut(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?;
        if index.saturating_add(len) > gc_array.elements.len() {
            return Err(trap_error("array element index out of bounds"));
        }
        gc_array.elements[index..index + len].fill(value);
        Ok(())
    })
}

pub(super) fn do_array_copy(
    ctx: &mut NativeContext,
    _dst_type_idx: u32,
    _src_type_idx: u32,
    raw_dst_ref: u64,
    raw_dst_index: u64,
    raw_src_ref: u64,
    raw_src_index: u64,
    raw_len: u64,
) -> Result<(), WasmError> {
    let dst_handle = ref_from_machine(raw_dst_ref);
    let src_handle = ref_from_machine(raw_src_ref);
    let dst_index = raw_dst_index as u32 as usize;
    let src_index = raw_src_index as u32 as usize;
    let len = raw_len as u32 as usize;
    let mut dst = resolve_array_ref(ctx, dst_handle)?;
    let src = resolve_array_ref(ctx, src_handle)?;
    let dst_gc_ref = dst.gc_ref;
    let src_gc_ref = src.gc_ref;

    if dst.owner() == src.owner() && dst_gc_ref == src_gc_ref {
        return dst.with_store_mut(|origin_store| {
            let mut gc_heap = origin_store.gc_heap().borrow_mut();
            let gc_array = gc_heap
                .get_array_mut(dst_gc_ref)
                .map_err(|_| trap_error("invalid array reference"))?;
            if src_index.saturating_add(len) > gc_array.elements.len()
                || dst_index.saturating_add(len) > gc_array.elements.len()
            {
                return Err(trap_error("array element index out of bounds"));
            }
            gc_array
                .elements
                .copy_within(src_index..src_index + len, dst_index);
            Ok(())
        });
    }

    let copied = src.with_store(|src_store| {
        let gc_heap = src_store.gc_heap().borrow();
        let src_array = gc_heap
            .get_array(src_gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?;
        if src_index.saturating_add(len) > src_array.elements.len() {
            return Err(trap_error("array element index out of bounds"));
        }
        Ok(src_array.elements[src_index..src_index + len].to_vec())
    })?;

    dst.with_store_mut(|dst_store| {
        let mut gc_heap = dst_store.gc_heap().borrow_mut();
        let dst_array = gc_heap
            .get_array_mut(dst_gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?;
        if dst_index.saturating_add(len) > dst_array.elements.len() {
            return Err(trap_error("array element index out of bounds"));
        }
        dst_array.elements[dst_index..dst_index + len].copy_from_slice(&copied);
        Ok(())
    })
}

pub(super) fn do_array_init_data(
    ctx: &mut NativeContext,
    type_idx: u32,
    data_idx: u32,
    raw_ref: u64,
    raw_dst_index: u64,
    raw_src_index: u64,
    raw_len: u64,
) -> Result<(), WasmError> {
    let handle = ref_from_machine(raw_ref);
    let dst_index = raw_dst_index as u32 as usize;
    let src_index = raw_src_index as u32 as usize;
    let len = raw_len as u32 as usize;
    let storage = array_element_storage(current_store(ctx)?, type_idx)?;
    let elem_size = data_array_element_size(storage)?;
    let mut origin = resolve_array_ref(ctx, handle)?;
    let gc_ref = origin.gc_ref;
    let array_len = origin.with_store(|origin_store| {
        let gc_heap = origin_store.gc_heap().borrow();
        Ok(gc_heap
            .get_array(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?
            .elements
            .len())
    })?;
    let values = {
        let current = current_store(ctx)?;
        let module = current.module();
        let data = module
            .data
            .get(data_idx as usize)
            .ok_or_else(|| trap_error("out of bounds memory access"))?;
        let src_byte = src_index;
        if len == 0 {
            if src_byte > data.bytes.len() || dst_index > array_len {
                return Err(trap_error("out of bounds memory access"));
            }
            return Ok(());
        }
        if data.is_dropped() {
            return Err(trap_error("out of bounds memory access"));
        }
        let byte_len = len
            .checked_mul(elem_size)
            .ok_or_else(|| trap_error("out of bounds memory access"))?;
        if src_byte.saturating_add(byte_len) > data.bytes.len()
            || dst_index.saturating_add(len) > array_len
        {
            return Err(trap_error("out of bounds memory access"));
        }
        let mut values = crate::collections::Vec::new();
        values
            .try_reserve(len)
            .map_err(|_| trap_error("out of memory"))?;
        for offset in 0..len {
            let start = src_byte + offset * elem_size;
            values.push(value_from_data_bytes(
                storage,
                &data.bytes[start..start + elem_size],
            )?);
        }
        values
    };
    origin.with_store_mut(|origin_store| {
        let mut gc_heap = origin_store.gc_heap().borrow_mut();
        let dst_array = gc_heap
            .get_array_mut(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?;
        dst_array.elements[dst_index..dst_index + len].copy_from_slice(&values);
        Ok(())
    })
}

pub(super) fn do_array_new_data(
    ctx: &mut NativeContext,
    type_idx: u32,
    data_idx: u32,
    raw_src_index: u64,
    raw_len: u64,
) -> Result<u64, WasmError> {
    let store = current_store_mut(ctx)?;
    let storage = array_element_storage(store, type_idx)?;
    let elem_size = data_array_element_size(storage)?;
    let src_index = raw_src_index as u32 as usize;
    let len = raw_len as u32 as usize;
    let data = store
        .module()
        .data
        .get(data_idx as usize)
        .ok_or_else(|| trap_error("out of bounds memory access"))?;
    let src_byte = src_index;
    if len == 0 {
        if src_byte > data.bytes.len() {
            return Err(trap_error("out of bounds memory access"));
        }
    } else {
        if data.is_dropped() {
            return Err(trap_error("out of bounds memory access"));
        }
        let byte_len = len
            .checked_mul(elem_size)
            .ok_or_else(|| trap_error("out of bounds memory access"))?;
        if src_byte.saturating_add(byte_len) > data.bytes.len() {
            return Err(trap_error("out of bounds memory access"));
        }
    }
    let mut elements = crate::collections::Vec::new();
    elements
        .try_reserve(len)
        .map_err(|_| trap_error("out of memory"))?;
    for offset in 0..len {
        let start = src_byte + offset * elem_size;
        elements.push(value_from_data_bytes(
            storage,
            &data.bytes[start..start + elem_size],
        )?);
    }
    let gc_ref = store.gc_heap().borrow_mut().alloc_array(type_idx, elements);
    let handle = store.register_gc_ref(gc_ref);
    Ok(ref_to_machine(handle))
}

pub(super) fn do_array_init_elem(
    ctx: &mut NativeContext,
    type_idx: u32,
    elem_idx: u32,
    raw_ref: u64,
    raw_dst_index: u64,
    raw_src_index: u64,
    raw_len: u64,
) -> Result<(), WasmError> {
    let handle = ref_from_machine(raw_ref);
    let dst_index = raw_dst_index as u32 as usize;
    let src_index = raw_src_index as u32 as usize;
    let len = raw_len as u32 as usize;
    let element_ref_type = match array_element_field(current_store(ctx)?, type_idx)?.storage {
        StorageType::Val(ValueType::Ref(ref_type)) => ref_type,
        _ => {
            return Err(internal_error(
                "preserved helper expected a reference array element type",
            ));
        }
    };
    let mut origin = resolve_array_ref(ctx, handle)?;
    let gc_ref = origin.gc_ref;
    let array_len = origin.with_store(|origin_store| {
        let gc_heap = origin_store.gc_heap().borrow();
        Ok(gc_heap
            .get_array(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?
            .elements
            .len())
    })?;
    let values = {
        let current = current_store(ctx)?;
        let module = current.module();
        let elem = module
            .elements
            .get(elem_idx as usize)
            .ok_or_else(|| trap_error("out of bounds table access"))?;
        if len == 0 {
            if src_index > elem.refs.len() || dst_index > array_len {
                return Err(trap_error("out of bounds table access"));
            }
            return Ok(());
        }
        if elem.is_dropped() {
            return Err(trap_error("out of bounds table access"));
        }
        if src_index.saturating_add(len) > elem.refs.len()
            || dst_index.saturating_add(len) > array_len
        {
            return Err(trap_error("out of bounds table access"));
        }
        let mut values = crate::collections::Vec::new();
        values
            .try_reserve(len)
            .map_err(|_| trap_error("out of memory"))?;
        for handle in &elem.refs[src_index..src_index + len] {
            values.push(Value::Ref(*handle, element_ref_type));
        }
        values
    };
    origin.with_store_mut(|origin_store| {
        let mut gc_heap = origin_store.gc_heap().borrow_mut();
        let dst_array = gc_heap
            .get_array_mut(gc_ref)
            .map_err(|_| trap_error("invalid array reference"))?;
        dst_array.elements[dst_index..dst_index + len].copy_from_slice(&values);
        Ok(())
    })
}

pub(super) fn do_array_new_elem(
    ctx: &mut NativeContext,
    type_idx: u32,
    elem_idx: u32,
    raw_src_index: u64,
    raw_len: u64,
) -> Result<u64, WasmError> {
    let store = current_store_mut(ctx)?;
    let element_ref_type = match array_element_field(store, type_idx)?.storage {
        StorageType::Val(ValueType::Ref(ref_type)) => ref_type,
        _ => {
            return Err(internal_error(
                "preserved helper expected a reference array element type",
            ));
        }
    };
    let src_index = raw_src_index as u32 as usize;
    let len = raw_len as u32 as usize;
    let elem = store
        .module()
        .elements
        .get(elem_idx as usize)
        .ok_or_else(|| trap_error("out of bounds table access"))?;
    if len == 0 {
        if src_index > elem.refs.len() {
            return Err(trap_error("out of bounds table access"));
        }
    } else {
        if elem.is_dropped() {
            return Err(trap_error("out of bounds table access"));
        }
        if src_index.saturating_add(len) > elem.refs.len() {
            return Err(trap_error("out of bounds table access"));
        }
    }
    let mut elements = crate::collections::Vec::new();
    elements
        .try_reserve(len)
        .map_err(|_| trap_error("out of memory"))?;
    for handle in &elem.refs[src_index..src_index + len] {
        elements.push(Value::Ref(*handle, element_ref_type));
    }
    let gc_ref = store.gc_heap().borrow_mut().alloc_array(type_idx, elements);
    let handle = store.register_gc_ref(gc_ref);
    Ok(ref_to_machine(handle))
}

/// Core memory-grow logic owned by the preserved-helper system. Returns the
/// Wasm result value (old page count on success, or the appropriate sentinel).
pub(super) fn do_memory_grow(
    ctx: &mut NativeContext,
    mem_idx: u32,
    delta_raw: u64,
) -> Result<u64, WasmError> {
    // Read the budget before borrowing the memory out of the context.
    let runtime_cap = ctx
        .store()
        .map(|store| store.config().get_wasm_memory_max_pages() as usize)
        .unwrap_or(0);
    let mem = memory_mut(ctx, mem_idx)?;
    let is_64 = mem.limits.is64;
    let error_value = memory_grow_error_value(is_64);
    let delta_pages = decode_memory_grow_delta(delta_raw, is_64);

    let old_pages = mem.current_pages();
    let effective_max = mem.limits.get_max().min(runtime_cap);
    let result = match old_pages.checked_add(delta_pages) {
        None => error_value,
        Some(new_pages) if new_pages > effective_max => error_value,
        Some(new_pages) => {
            #[cfg(sf_has_guard_pages)]
            {
                let mut backing = mem.backing_mut();
                if let Some(guard) = backing.guard.as_mut() {
                    match guard.grow(delta_pages) {
                        Ok(_) => old_pages as u64,
                        Err(_) => error_value,
                    }
                } else {
                    drop(backing);
                    grow_heap(mem, new_pages, old_pages, error_value)
                }
            }
            #[cfg(not(sf_has_guard_pages))]
            {
                grow_heap(mem, new_pages, old_pages, error_value)
            }
        }
    };

    if memory_grow_succeeded(result, error_value) {
        ctx.refresh_memory_views();
    }
    Ok(result)
}

fn grow_heap(mem: &mut MemInst, new_pages: usize, old_pages: usize, error_value: u64) -> u64 {
    if let Some(new_len) = new_pages.checked_mul(WASM_PAGE_SIZE) {
        let mut backing = mem.backing_mut();
        let additional = new_len.saturating_sub(backing.data.len());
        if backing.data.try_reserve(additional).is_err() {
            error_value
        } else {
            backing.data.resize(new_len, 0);
            old_pages as u64
        }
    } else {
        error_value
    }
}

#[inline]
fn memory_grow_error_value(is_64: bool) -> u64 {
    if is_64 {
        u64::MAX
    } else {
        u32::MAX as u64
    }
}

#[inline]
fn decode_memory_grow_delta(delta_pages_raw: u64, is_64: bool) -> usize {
    if is_64 {
        delta_pages_raw as usize
    } else {
        (delta_pages_raw as u32) as usize
    }
}

#[inline]
fn memory_grow_succeeded(result: u64, error_value: u64) -> bool {
    result != error_value
}

pub(super) fn do_memory_copy(
    ctx: &mut NativeContext,
    dst_mem_idx: u32,
    src_mem_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    let di = dst_mem_idx as usize;
    let si = src_mem_idx as usize;
    if di >= store.module().memories.len() || si >= store.module().memories.len() {
        return Err(internal_error(
            "preserved helper referenced invalid memory index",
        ));
    }
    if di == si {
        let mem = store.memory_mut(di);
        let mem_len = mem.memory_len();
        if src.saturating_add(len) > mem_len || dest.saturating_add(len) > mem_len {
            return Err(trap_error("out of bounds memory access"));
        }
        let data = unsafe { core::slice::from_raw_parts_mut(mem.memory_ptr(), mem_len) };
        data.copy_within(src..src + len, dest);
    } else {
        let module = store.module_mut();
        let (src_mem, dst_mem) = if si < di {
            let (left, right) = module.memories.split_at_mut(di);
            (&left[si], &mut right[0])
        } else {
            let (left, right) = module.memories.split_at_mut(si);
            (&right[0] as &MemInst, &mut left[di])
        };
        let sl = src_mem.memory_len();
        let dl = dst_mem.memory_len();
        if src.saturating_add(len) > sl || dest.saturating_add(len) > dl {
            return Err(trap_error("out of bounds memory access"));
        }
        let ss = unsafe { core::slice::from_raw_parts(src_mem.memory_ptr(), sl) };
        let ds = unsafe { core::slice::from_raw_parts_mut(dst_mem.memory_ptr(), dl) };
        ds[dest..dest + len].copy_from_slice(&ss[src..src + len]);
    }
    Ok(())
}

pub(super) fn do_memory_init(
    ctx: &mut NativeContext,
    mem_idx: u32,
    data_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    let module = store.module_mut();
    let mi = mem_idx as usize;
    let di = data_idx as usize;
    if mi >= module.memories.len() {
        return Err(internal_error(
            "preserved helper referenced invalid memory index",
        ));
    }
    if di >= module.data.len() {
        return Err(trap_error("out of bounds memory access"));
    }
    let data_bytes = &module.data[di].bytes;
    let data_dropped = module.data[di].is_dropped();
    let mem_len = module.memories[mi].memory_len();
    if len == 0 {
        if src > data_bytes.len() || dest > mem_len {
            return Err(trap_error("out of bounds memory access"));
        }
        return Ok(());
    }
    if data_dropped {
        return Err(trap_error("out of bounds memory access"));
    }
    if src.saturating_add(len) > data_bytes.len() {
        return Err(trap_error("out of bounds memory access"));
    }
    if dest.saturating_add(len) > mem_len {
        return Err(trap_error("out of bounds memory access"));
    }
    unsafe {
        core::ptr::copy_nonoverlapping(
            data_bytes[src..].as_ptr(),
            module.memories[mi].memory_ptr().add(dest),
            len,
        );
    }
    Ok(())
}

pub(super) fn do_table_grow(
    ctx: &mut NativeContext,
    table_idx: u32,
    init_val_raw: u64,
    delta_raw: u64,
) -> Result<u64, WasmError> {
    let fill = {
        let store = current_store(ctx)?;
        let reachable = store.module().table_is_reachable(table_idx as usize);
        retag_for_container(store, ref_from_machine(init_val_raw), reachable)
    };
    let table = table_mut(ctx, table_idx)?;
    let is_64 = table.limits.is64;
    let error_value = if is_64 { u64::MAX } else { u32::MAX as u64 };
    let delta = if is_64 {
        delta_raw as usize
    } else {
        delta_raw as u32 as usize
    };
    let mut elements = table.elements_mut();
    let old_len = elements.len();
    let result = match old_len.checked_add(delta) {
        None => error_value,
        Some(new_len) if new_len > table.limits.get_max() => error_value,
        Some(_) if elements.try_reserve(delta).is_err() => error_value,
        Some(new_len) => {
            elements.resize_with(new_len, || fill);
            old_len as u64
        }
    };
    drop(elements);
    if result != error_value {
        ctx.refresh_table_views();
    }
    Ok(result)
}

pub(super) fn do_table_fill(
    ctx: &mut NativeContext,
    table_idx: u32,
    start: usize,
    val_raw: u64,
    len: usize,
) -> Result<(), WasmError> {
    let fill = {
        let store = current_store(ctx)?;
        let reachable = store.module().table_is_reachable(table_idx as usize);
        retag_for_container(store, ref_from_machine(val_raw), reachable)
    };
    let table = table_mut(ctx, table_idx)?;
    let mut elements = table.elements_mut();
    if start.saturating_add(len) > elements.len() {
        return Err(trap_error("out of bounds table access"));
    }
    elements[start..start + len].fill(fill);
    Ok(())
}

pub(super) fn do_table_copy(
    ctx: &mut NativeContext,
    dst_tbl: u32,
    src_tbl: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let di = dst_tbl as usize;
    let si = src_tbl as usize;
    if di == si {
        let store = current_store_mut(ctx)?;
        if di >= store.module().tables.len() {
            return Err(internal_error(
                "preserved helper referenced invalid table index",
            ));
        }
        let table = store.table_mut(di);
        let mut elements = table.elements_mut();
        if src.saturating_add(len) > elements.len() || dest.saturating_add(len) > elements.len() {
            return Err(trap_error("out of bounds table access"));
        }
        elements.copy_within(src..src + len, dest);
    } else {
        let copied = {
            let store = current_store(ctx)?;
            let module = store.module();
            if di >= module.tables.len() || si >= module.tables.len() {
                return Err(internal_error(
                    "preserved helper referenced invalid table index",
                ));
            }
            let src_elements = module.tables[si].elements();
            let dst_elements = module.tables[di].elements();
            if src.saturating_add(len) > src_elements.len()
                || dest.saturating_add(len) > dst_elements.len()
            {
                return Err(trap_error("out of bounds table access"));
            }
            let source_reachable = module.table_is_reachable(si);
            let destination_reachable = module.table_is_reachable(di);
            let mut copied = collections::Vec::new();
            copied
                .try_reserve(len)
                .map_err(|_| trap_error("out of memory"))?;
            if source_reachable == destination_reachable {
                copied.extend_from_slice(&src_elements[src..src + len]);
            } else {
                copied.extend(
                    src_elements[src..src + len]
                        .iter()
                        .map(|handle| retag_for_container(store, *handle, destination_reachable)),
                );
            }
            copied
        };
        let store = current_store_mut(ctx)?;
        let mut dst_elements = store.module_mut().tables[di].elements_mut();
        dst_elements[dest..dest + len].copy_from_slice(&copied);
    }
    Ok(())
}

pub(super) fn do_table_init(
    ctx: &mut NativeContext,
    table_idx: u32,
    elem_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let ti = table_idx as usize;
    let ei = elem_idx as usize;
    let copied = {
        let store = current_store(ctx)?;
        let module = store.module();
        if ti >= module.tables.len() {
            return Err(internal_error(
                "preserved helper referenced invalid table index",
            ));
        }
        if ei >= module.elements.len() {
            return Err(trap_error("out of bounds table access"));
        }
        let elem = &module.elements[ei];
        let table_len = module.tables[ti].size();
        if len == 0 {
            if src > elem.refs.len() || dest > table_len {
                return Err(trap_error("out of bounds table access"));
            }
            return Ok(());
        }
        if src.saturating_add(len) > elem.refs.len() || dest.saturating_add(len) > table_len {
            return Err(trap_error("out of bounds table access"));
        }
        if elem.is_dropped() {
            return Err(trap_error("out of bounds table access"));
        }
        let reachable = module.table_is_reachable(ti);
        let mut copied = collections::Vec::new();
        copied
            .try_reserve(len)
            .map_err(|_| trap_error("out of memory"))?;
        if reachable {
            copied.extend_from_slice(&elem.refs[src..src + len]);
        } else {
            copied.extend(
                elem.refs[src..src + len]
                    .iter()
                    .map(|handle| retag_for_container(store, *handle, false)),
            );
        }
        copied
    };
    let store = current_store_mut(ctx)?;
    let mut table_elements = store.module_mut().tables[ti].elements_mut();
    table_elements[dest..dest + len].copy_from_slice(&copied);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode_memory_grow_delta, memory_grow_succeeded};

    #[test]
    fn memory_grow_uses_unsigned_delta_for_32_bit_memories() {
        assert_eq!(
            decode_memory_grow_delta(u32::MAX as u64, false),
            u32::MAX as usize
        );
        assert_eq!(
            decode_memory_grow_delta(0x8000_0000u64, false),
            0x8000_0000usize
        );
    }

    #[test]
    fn memory_grow_success_check_uses_selected_error_sentinel() {
        assert!(memory_grow_succeeded(u32::MAX as u64, u64::MAX));
        assert!(!memory_grow_succeeded(u64::MAX, u64::MAX));
        assert!(!memory_grow_succeeded(u32::MAX as u64, u32::MAX as u64));
    }
}
