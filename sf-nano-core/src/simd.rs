use crate::{
    error::WasmError,
    module::{
        entities::{ElementInit, FunctionType},
        type_defs::{CompositeType, StorageType},
        Module,
    },
    value_type::ValueType,
};

#[cfg(sf_simd)]
#[inline]
pub(crate) fn simd_unsupported_error() -> WasmError {
    WasmError::invalid("SIMD support is not yet implemented")
}

#[cfg(not(sf_simd))]
#[inline]
pub(crate) fn simd_unsupported_error() -> WasmError {
    WasmError::invalid("SIMD is disabled in this build")
}

#[inline]
pub(crate) fn ensure_simd_value_type_supported(value_type: ValueType) -> Result<(), WasmError> {
    if matches!(value_type, ValueType::V128) {
        return Err(simd_unsupported_error());
    }
    Ok(())
}

#[inline]
fn ensure_simd_value_types_supported(value_types: &[ValueType]) -> Result<(), WasmError> {
    for &value_type in value_types {
        ensure_simd_value_type_supported(value_type)?;
    }
    Ok(())
}

#[inline]
fn ensure_simd_storage_type_supported(storage: &StorageType) -> Result<(), WasmError> {
    if let StorageType::Val(value_type) = storage {
        ensure_simd_value_type_supported(*value_type)?;
    }
    Ok(())
}

#[inline]
fn ensure_simd_function_type_supported(func_type: &FunctionType) -> Result<(), WasmError> {
    ensure_simd_value_types_supported(func_type.params())?;
    ensure_simd_value_types_supported(func_type.results())
}

pub(crate) fn validate_simd_module(module: &Module) -> Result<(), WasmError> {
    for def_type in module.types().as_slice() {
        match &def_type.composite {
            CompositeType::Func(func_type) => ensure_simd_function_type_supported(func_type)?,
            CompositeType::Struct(struct_type) => {
                for field in &struct_type.fields {
                    ensure_simd_storage_type_supported(&field.storage)?;
                }
            }
            CompositeType::Array(array_type) => {
                ensure_simd_storage_type_supported(&array_type.element.storage)?;
            }
        }
    }

    for func in module.functions() {
        ensure_simd_function_type_supported(func.func_type())?;
        if let Some(spec) = func.spec() {
            ensure_simd_value_types_supported(spec.locals())?;
        }
    }

    for global in module.globals() {
        ensure_simd_value_type_supported(global.value_type())?;
    }

    for tag in module.tags() {
        ensure_simd_function_type_supported(tag.func_type())?;
    }

    for element in module.elements() {
        if let ElementInit::InitExprs { value_type, .. } = element.get_init() {
            ensure_simd_value_type_supported(*value_type)?;
        }
    }

    Ok(())
}
