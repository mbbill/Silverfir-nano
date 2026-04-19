//! Shared dynamic-budget accounting helpers.

use crate::value_type::ValueType;

pub(crate) fn count_live_bank_budget_units(
    types: &[ValueType],
    gp_unit_bytes: u8,
) -> (usize, usize) {
    let mut gp = 0usize;
    let mut fp = 0usize;
    for ty in types {
        match ty {
            ValueType::F32 | ValueType::F64 | ValueType::V128 => fp += 1,
            _ => gp += gp_value_budget_units(*ty, gp_unit_bytes),
        }
    }
    (gp, fp)
}

pub(crate) fn gp_value_budget_units(ty: ValueType, gp_unit_bytes: u8) -> usize {
    match ty {
        ValueType::I64 if gp_unit_bytes < 8 => 2,
        _ => 1,
    }
}
