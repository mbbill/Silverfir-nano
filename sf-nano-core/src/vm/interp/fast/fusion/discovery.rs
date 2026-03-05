//! Automatic fusion candidate discovery for variant-qualified patterns.
//!
//! Analyzes a [`PatternTrie`] to find the best instruction sequences to fuse,
//! accounting for prefix overlaps and encoding budget constraints.
//! Outputs ready-to-paste `[[fused]]` TOML entries for `handlers_fused.toml`.
//!
//! All pattern elements are variant-qualified: "i32_const_d1", "i32_add_d2", etc.

extern crate std;

use std::collections::{HashMap, HashSet};
use std::string::String;
use std::vec::Vec;

use super::pattern_trie::PatternTrie;

// =============================================================================
// Variant suffix handling
// =============================================================================

/// Strip _dN variant suffix from a handler name.
/// "i32_add_d1" → "i32_add", "i32_add" → "i32_add"
fn strip_variant(name: &str) -> &str {
    if name.len() >= 3 {
        let suffix = &name[name.len() - 3..];
        if matches!(suffix, "_d1" | "_d2" | "_d3" | "_d4") {
            return &name[..name.len() - 3];
        }
    }
    name
}

// =============================================================================
// Op classification (operates on base op names after stripping variant)
// =============================================================================

fn is_pure_binop(op: &str) -> bool {
    matches!(
        op,
        "i32_add" | "i32_sub" | "i32_mul"
        | "i32_and" | "i32_or" | "i32_xor"
        | "i32_shl" | "i32_shr_s" | "i32_shr_u" | "i32_rotl" | "i32_rotr"
        | "i32_eq" | "i32_ne"
        | "i32_lt_s" | "i32_lt_u" | "i32_gt_s" | "i32_gt_u"
        | "i32_le_s" | "i32_le_u" | "i32_ge_s" | "i32_ge_u"
        | "i64_add" | "i64_sub" | "i64_mul"
        | "i64_and" | "i64_or" | "i64_xor"
        | "i64_shl" | "i64_shr_s" | "i64_shr_u" | "i64_rotl" | "i64_rotr"
        | "i64_eq" | "i64_ne"
        | "i64_lt_s" | "i64_lt_u" | "i64_gt_s" | "i64_gt_u"
        | "i64_le_s" | "i64_le_u" | "i64_ge_s" | "i64_ge_u"
        | "f32_add" | "f32_sub" | "f32_mul" | "f32_div"
        | "f32_min" | "f32_max" | "f32_copysign"
        | "f32_eq" | "f32_ne" | "f32_lt" | "f32_gt" | "f32_le" | "f32_ge"
        | "f64_add" | "f64_sub" | "f64_mul" | "f64_div"
        | "f64_min" | "f64_max" | "f64_copysign"
        | "f64_eq" | "f64_ne" | "f64_lt" | "f64_gt" | "f64_le" | "f64_ge"
    )
}

fn is_trapping_binop(op: &str) -> bool {
    matches!(
        op,
        "i32_div_s" | "i32_div_u" | "i32_rem_s" | "i32_rem_u"
        | "i64_div_s" | "i64_div_u" | "i64_rem_s" | "i64_rem_u"
    )
}

fn is_pure_unary(op: &str) -> bool {
    matches!(
        op,
        "i32_eqz" | "i32_clz" | "i32_ctz" | "i32_popcnt"
        | "i64_eqz" | "i64_clz" | "i64_ctz" | "i64_popcnt"
        | "f32_abs" | "f32_neg" | "f32_ceil" | "f32_floor"
        | "f32_trunc" | "f32_nearest" | "f32_sqrt"
        | "f64_abs" | "f64_neg" | "f64_ceil" | "f64_floor"
        | "f64_trunc" | "f64_nearest" | "f64_sqrt"
        | "i32_wrap_i64" | "i64_extend_i32_s" | "i64_extend_i32_u"
        | "i32_extend8_s" | "i32_extend16_s"
        | "i64_extend8_s" | "i64_extend16_s" | "i64_extend32_s"
        | "i32_reinterpret_f32" | "i64_reinterpret_f64"
        | "f32_reinterpret_i32" | "f64_reinterpret_i64"
        | "f32_convert_i32_s" | "f32_convert_i32_u"
        | "f32_convert_i64_s" | "f32_convert_i64_u"
        | "f64_convert_i32_s" | "f64_convert_i32_u"
        | "f64_convert_i64_s" | "f64_convert_i64_u"
        | "f32_demote_f64" | "f64_promote_f32"
    )
}

fn is_load_op(op: &str) -> bool {
    matches!(
        op,
        "i32_load" | "i32_load8_s" | "i32_load8_u" | "i32_load16_s" | "i32_load16_u"
        | "i64_load" | "i64_load8_s" | "i64_load8_u"
        | "i64_load16_s" | "i64_load16_u" | "i64_load32_s" | "i64_load32_u"
        | "f32_load" | "f64_load"
    )
}

fn is_store_op(op: &str) -> bool {
    matches!(
        op,
        "i32_store" | "i32_store8" | "i32_store16"
        | "i64_store" | "i64_store8" | "i64_store16" | "i64_store32"
        | "f32_store" | "f64_store"
    )
}

/// Check if a variant-qualified op is fusible.
pub fn is_fusible_op(op: &str) -> bool {
    let base = strip_variant(op);
    matches!(base, "local_get" | "local_set" | "local_tee"
        | "local_get_l0" | "local_set_l0" | "local_tee_l0"
        | "local_get_l1" | "local_set_l1" | "local_tee_l1"
        | "local_get_l2" | "local_set_l2" | "local_tee_l2"
        | "i32_const" | "i64_const" | "br_if" | "if_")
        || is_pure_binop(base)
        || is_trapping_binop(base)
        || is_pure_unary(base)
        || is_load_op(base)
        || is_store_op(base)
}

// =============================================================================
// Encoding budget validation
// =============================================================================

fn op_encoding_bits(op: &str) -> u32 {
    let base = strip_variant(op);
    match base {
        "local_get" | "local_set" | "local_tee" => 8,
        "local_get_l0" | "local_set_l0" | "local_tee_l0"
        | "local_get_l1" | "local_set_l1" | "local_tee_l1"
        | "local_get_l2" | "local_set_l2" | "local_tee_l2" => 0,
        "i32_const" => 32,
        "i64_const" | "br_if" | "if_" => 64,
        _ if is_load_op(base) || is_store_op(base) => 32,
        _ => 0,
    }
}

pub fn encoding_fits(pattern: &[&str]) -> bool {
    let mut current_slot = 0u32;
    let mut current_bit = 0u32;
    for op in pattern {
        let bits = op_encoding_bits(op);
        if bits == 0 {
            continue;
        }
        if current_bit + bits > 64 {
            current_slot += 1;
            current_bit = 0;
        }
        if current_slot > 2 {
            return false;
        }
        current_bit += bits;
    }
    true
}

// =============================================================================
// Encoding field auto-generation
// =============================================================================

#[derive(Debug, Clone)]
pub struct EncodingField {
    pub name: String,
    pub bits: u32,
    pub kind: Option<String>,
    pub from: usize,
}

pub fn auto_encoding_fields(pattern: &[&str]) -> Vec<EncodingField> {
    let mut fields = Vec::new();

    // Count field types (using base op names)
    let mut local_count = 0u32;
    let mut const32_count = 0u32;
    let mut const64_count = 0u32;
    let mut load_store_count = 0u32;
    let mut brif_count = 0u32;
    let mut if_count = 0u32;

    for op in pattern {
        let base = strip_variant(op);
        match base {
            "local_get" | "local_set" | "local_tee" => local_count += 1,
            "local_get_l0" | "local_set_l0" | "local_tee_l0"
            | "local_get_l1" | "local_set_l1" | "local_tee_l1"
            | "local_get_l2" | "local_set_l2" | "local_tee_l2" => {}
            "i32_const" => const32_count += 1,
            "i64_const" => const64_count += 1,
            "br_if" => brif_count += 1,
            "if_" => if_count += 1,
            _ if is_load_op(base) || is_store_op(base) => load_store_count += 1,
            _ => {}
        }
    }

    let total_const_count = const32_count + const64_count;
    let total_branch_count = brif_count + if_count;

    for (i, op) in pattern.iter().enumerate() {
        let base = strip_variant(op);
        match base {
            "local_get" | "local_set" | "local_tee" => {
                let name = if local_count == 1 {
                    String::from("local_idx")
                } else {
                    std::format!("local_idx_{}", i)
                };
                fields.push(EncodingField { name, bits: 8, kind: None, from: i });
            }
            "i32_const" => {
                let name = if total_const_count == 1 {
                    String::from("const_val")
                } else {
                    std::format!("const_val_{}", i)
                };
                fields.push(EncodingField { name, bits: 32, kind: None, from: i });
            }
            "i64_const" => {
                let name = if total_const_count == 1 {
                    String::from("const_val")
                } else {
                    std::format!("const_val_{}", i)
                };
                fields.push(EncodingField { name, bits: 64, kind: None, from: i });
            }
            "br_if" => {
                let name = if total_branch_count == 1 {
                    String::from("target")
                } else {
                    std::format!("target_{}", i)
                };
                fields.push(EncodingField { name, bits: 64, kind: Some(String::from("target")), from: i });
            }
            "if_" => {
                let name = if total_branch_count == 1 {
                    String::from("target")
                } else {
                    std::format!("target_{}", i)
                };
                fields.push(EncodingField { name, bits: 64, kind: Some(String::from("target")), from: i });
            }
            _ if is_load_op(base) || is_store_op(base) => {
                let name = if load_store_count == 1 {
                    String::from("offset")
                } else {
                    std::format!("offset_{}", i)
                };
                fields.push(EncodingField { name, bits: 32, kind: None, from: i });
            }
            _ => {}
        }
    }

    fields
}

// =============================================================================
// Greedy selection algorithm
// =============================================================================

#[derive(Debug, Clone)]
pub struct FusionCandidate {
    pub pattern: Vec<String>,
    pub name: String,
    pub raw_count: u64,
    pub effective_count: u64,
    pub savings: u64,
    pub encoding_fields: Vec<EncodingField>,
}

pub struct DiscoveryConfig {
    pub max_candidates: usize,
    /// Minimum savings as a percentage of total instructions (e.g., 0.01 = 0.01%).
    pub min_savings_pct: f64,
    pub reserved_names: HashSet<String>,
}

fn is_pattern_fusible(pattern: &[&str]) -> bool {
    if pattern.len() < 2 {
        return false;
    }
    if !pattern.iter().all(|op| is_fusible_op(op)) {
        return false;
    }
    // br_if and if_ can only appear at the END of a pattern
    for op in &pattern[..pattern.len() - 1] {
        let base = strip_variant(op);
        if base == "br_if" || base == "if_" {
            return false;
        }
    }
    let mem_count = pattern
        .iter()
        .filter(|op| {
            let base = strip_variant(op);
            is_load_op(base) || is_store_op(base)
        })
        .count();
    if mem_count > 1 {
        return false;
    }
    true
}

/// Run the greedy fusion discovery algorithm.
pub fn discover(trie: &PatternTrie, config: &DiscoveryConfig) -> Vec<FusionCandidate> {
    let raw_candidates = trie.collect_candidates(2, 1);

    let min_savings = (config.min_savings_pct / 100.0 * trie.total_instructions as f64) as u64;

    let mut scored: Vec<(Vec<String>, u64, u64)> = Vec::new();
    for c in &raw_candidates {
        let pattern_refs: Vec<&str> = c.pattern.iter().map(|s| s.as_str()).collect();

        if !is_pattern_fusible(&pattern_refs) {
            continue;
        }
        if !encoding_fits(&pattern_refs) {
            continue;
        }

        let savings = c.count * (c.pattern.len() as u64 - 1);
        if savings < min_savings {
            continue;
        }

        scored.push((c.pattern.clone(), c.count, savings));
    }

    scored.sort_by(|a, b| b.2.cmp(&a.2));

    let mut selected: Vec<FusionCandidate> = Vec::new();
    let mut consumed: HashMap<Vec<String>, u64> = HashMap::new();
    let mut used_names: HashSet<String> = config.reserved_names.clone();

    for (pattern, raw_count, _) in &scored {
        if selected.len() >= config.max_candidates {
            break;
        }

        let already_consumed = consumed.get(pattern).copied().unwrap_or(0);
        let effective_count = raw_count.saturating_sub(already_consumed);
        let effective_savings = effective_count * (pattern.len() as u64 - 1);

        if effective_savings < min_savings {
            continue;
        }

        let pattern_refs: Vec<&str> = pattern.iter().map(|s| s.as_str()).collect();
        let fields = auto_encoding_fields(&pattern_refs);

        // Name = concatenation of pattern elements
        let name = pattern_refs.join("_");
        if used_names.contains(&name) {
            continue; // skip duplicates
        }
        used_names.insert(name.clone());

        for prefix_len in 2..pattern.len() {
            let prefix = pattern[..prefix_len].to_vec();
            *consumed.entry(prefix).or_insert(0) += effective_count;
        }

        selected.push(FusionCandidate {
            pattern: pattern.clone(),
            name,
            raw_count: *raw_count,
            effective_count,
            savings: effective_savings,
            encoding_fields: fields,
        });
    }

    selected
}

// =============================================================================
// TOML output
// =============================================================================

pub fn format_toml_entry(candidate: &FusionCandidate) -> String {
    let mut out = String::new();

    out.push_str("[[fused]]\n");
    out.push_str(&std::format!("op = \"{}\"\n", candidate.name));

    let pattern_strs: Vec<String> = candidate
        .pattern
        .iter()
        .map(|s| std::format!("\"{}\"", s))
        .collect();
    out.push_str(&std::format!("pattern = [{}]\n", pattern_strs.join(", ")));

    out.push_str("c_impl = true\n");

    if !candidate.encoding_fields.is_empty() {
        out.push_str("encoding.fields = [\n");
        for field in &candidate.encoding_fields {
            let mut parts = std::vec![
                std::format!("name = \"{}\"", field.name),
                std::format!("bits = {}", field.bits),
                std::format!("from = {}", field.from),
            ];
            if let Some(ref kind) = field.kind {
                parts.push(std::format!("kind = \"{}\"", kind));
            }
            out.push_str(&std::format!("    {{ {} }},\n", parts.join(", ")));
        }
        out.push_str("]\n");
    }

    out
}

pub fn format_all_toml(candidates: &[FusionCandidate]) -> String {
    let mut out = String::new();
    out.push_str("# =============================================================================\n");
    out.push_str("# AUTO-GENERATED — do not edit manually\n");
    out.push_str("# =============================================================================\n");
    out.push_str("# Regenerate: sf-nano-cli discover-fusion <wasm-file>\n\n");

    for (i, c) in candidates.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format_toml_entry(c));
    }
    out
}
