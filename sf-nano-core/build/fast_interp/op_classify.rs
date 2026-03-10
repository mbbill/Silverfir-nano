// Op classification — single source of truth for fusible op categories.
// Shared by gen_fusion_c, gen_fusion_ir_match.
//
// Categories are defined in handlers.toml via the `category` field.
// This module provides lookup-based classification using the category map
// built from HandlersFile::category_map().

use std::collections::HashMap;

use super::types::{FusedHandler, OpCategory};

/// Map from expanded handler name to its OpCategory.
pub type CategoryMap = HashMap<String, OpCategory>;

/// Parse a variant-qualified pattern element into (base_op, variant_num).
/// "i32_const_d1" → ("i32_const", 1)
/// "i32_add_d3" → ("i32_add", 3)
/// Panics if no variant suffix found.
pub fn parse_variant_qualified(name: &str) -> (&str, u8) {
    for variant in (1u8..=4).rev() {
        let suffix = format!("_d{}", variant);
        if name.ends_with(&suffix) {
            return (&name[..name.len() - suffix.len()], variant);
        }
    }
    panic!(
        "Pattern element '{}' has no variant suffix (_d1.._d4)",
        name
    );
}

/// Strip variant suffix if present, returning the base op name.
/// "i32_add_d1" → "i32_add", "i32_add" → "i32_add"
pub fn strip_variant_suffix(name: &str) -> &str {
    for variant in (1u8..=4).rev() {
        let suffix = format!("_d{}", variant);
        if name.ends_with(&suffix) {
            return &name[..name.len() - suffix.len()];
        }
    }
    name
}

/// Compute the TOS register name for a given variant and stack position.
/// variant: 1-4 (D1..D4), pos: 1-indexed from top of stack
/// Formula: t[(variant + 4 - pos) % 4]
pub fn tos_reg(variant: u8, pos: usize) -> String {
    let n = variant as usize;
    let idx = (n + 4 - pos) % 4;
    format!("t{}", idx)
}

/// Pure expression binops: pop 2, push 1, no side effects.
pub fn is_pure_binop(categories: &CategoryMap, op: &str) -> bool {
    categories.get(op) == Some(&OpCategory::PureBinop)
}

/// Trapping binops: pop 2, push 1, needs ctx for trap path.
pub fn is_trapping_binop(categories: &CategoryMap, op: &str) -> bool {
    categories.get(op) == Some(&OpCategory::TrappingBinop)
}

/// Pure expression unary ops: pop 1, push 1, no side effects.
pub fn is_pure_unary(categories: &CategoryMap, op: &str) -> bool {
    categories.get(op) == Some(&OpCategory::PureUnary)
}

/// Load ops: pop 1 (addr), push 1 (value), needs ctx, has MemArg immediate.
pub fn is_load_op(categories: &CategoryMap, op: &str) -> bool {
    categories.get(op) == Some(&OpCategory::Load)
}

/// Store ops: pop 2 (addr, value), push 0, needs ctx, has MemArg immediate.
pub fn is_store_op(categories: &CategoryMap, op: &str) -> bool {
    categories.get(op) == Some(&OpCategory::Store)
}

/// Check if an op is a comparison/test op (produces a boolean result).
/// Handles variant-qualified names by stripping _dN suffix.
pub fn is_comparison_op(op: &str) -> bool {
    let base = strip_variant_suffix(op);
    matches!(
        base,
        "i32_eq"
            | "i32_ne"
            | "i32_eqz"
            | "i32_lt_s"
            | "i32_lt_u"
            | "i32_gt_s"
            | "i32_gt_u"
            | "i32_le_s"
            | "i32_le_u"
            | "i32_ge_s"
            | "i32_ge_u"
            | "i64_eq"
            | "i64_ne"
            | "i64_eqz"
            | "i64_lt_s"
            | "i64_lt_u"
            | "i64_gt_s"
            | "i64_gt_u"
            | "i64_le_s"
            | "i64_le_u"
            | "i64_ge_s"
            | "i64_ge_u"
            | "f32_eq"
            | "f32_ne"
            | "f32_lt"
            | "f32_gt"
            | "f32_le"
            | "f32_ge"
            | "f64_eq"
            | "f64_ne"
            | "f64_lt"
            | "f64_gt"
            | "f64_le"
            | "f64_ge"
    )
}

/// Check if pattern has a br_if branch target (handles variant-qualified names)
pub fn has_branch(fused: &FusedHandler) -> bool {
    fused
        .pattern
        .iter()
        .any(|op| strip_variant_suffix(op) == "br_if")
}

/// Check if pattern contains if_ (handles variant-qualified names)
pub fn has_if(fused: &FusedHandler) -> bool {
    fused
        .pattern
        .iter()
        .any(|op| strip_variant_suffix(op) == "if_")
}

/// Determine whether ctx is used by the handler (memory ops and trapping ops need it).
/// Handles variant-qualified pattern names by stripping _dN suffix before lookup.
pub fn needs_ctx(categories: &CategoryMap, fused: &FusedHandler) -> bool {
    fused.pattern.iter().any(|op| {
        let base = strip_variant_suffix(op);
        matches!(
            categories.get(base),
            Some(OpCategory::Load) | Some(OpCategory::Store) | Some(OpCategory::TrappingBinop)
        )
    })
}
