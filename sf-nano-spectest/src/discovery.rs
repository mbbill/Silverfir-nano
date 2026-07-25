//! Test discovery and filtering

use std::{fs, path::Path};

pub fn find_wast_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut wast_files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            // Prefer the directory entry type over Path metadata probes here.
            // qemu-riscv32 has crashed in optimized Rust std statx metadata
            // paths before any WAST execution; read_dir already gives us the
            // file type we need without changing discovery semantics.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };

            if file_type.is_file() && path.extension().is_some_and(|ext| ext == "wast") {
                wast_files.push(path);
            } else if file_type.is_dir() {
                wast_files.extend(find_wast_files(&path));
            }
        }
    }

    wast_files.sort();
    wast_files
}

/// Determines if a test should be skipped based on feature support.
///
fn is_enabled_simd_test(test_name: &str) -> bool {
    let name = test_name.replace('\\', "/");
    let stem = Path::new(&name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&name);
    matches!(
        stem,
        "simd_const"
            | "simd_boolean"
            | "simd_select"
            | "simd_bitwise"
            | "simd_splat"
            | "simd_bit_shift"
            | "simd_i8x16_arith"
            | "simd_i16x8_arith"
            | "simd_i32x4_arith"
            | "simd_i64x2_arith"
            | "simd_store"
            | "simd_address"
            | "simd_load"
            | "simd_f32x4_arith"
            | "simd_f64x2_arith"
            | "simd_i8x16_cmp"
            | "simd_i16x8_cmp"
            | "simd_i32x4_cmp"
            | "simd_i64x2_cmp"
            | "simd_f32x4_cmp"
            | "simd_f64x2_cmp"
            | "simd_i8x16_sat_arith"
            | "simd_i16x8_sat_arith"
            | "simd_i8x16_arith2"
            | "simd_i16x8_arith2"
            | "simd_i32x4_arith2"
            | "simd_i64x2_arith2"
            | "simd_f32x4"
            | "simd_f64x2"
            | "simd_f32x4_rounding"
            | "simd_f64x2_rounding"
            | "simd_i32x4_trunc_sat_f32x4"
            | "simd_i32x4_trunc_sat_f64x2"
            | "simd_linking"
            | "simd_f32x4_pmin_pmax"
            | "simd_f64x2_pmin_pmax"
            | "simd_align"
            | "simd_load_splat"
            | "simd_load_zero"
            | "simd_load_extend"
            | "simd_conversions"
            | "simd_i16x8_extadd_pairwise_i8x16"
            | "simd_i16x8_extmul_i8x16"
            | "simd_i16x8_q15mulr_sat_s"
            | "simd_i32x4_dot_i16x8"
            | "simd_i32x4_extadd_pairwise_i16x8"
            | "simd_i32x4_extmul_i16x8"
            | "simd_i64x2_extmul_i32x4"
            | "simd_int_to_int_extend"
            | "simd_lane"
            | "simd_load8_lane"
            | "simd_load16_lane"
            | "simd_load32_lane"
            | "simd_load64_lane"
            | "simd_store8_lane"
            | "simd_store16_lane"
            | "simd_store32_lane"
            | "simd_store64_lane"
            | "simd_memory-multi"
            | "relaxed_dot_product"
            | "relaxed_laneselect"
            | "relaxed_madd_nmadd"
            | "relaxed_min_max"
            | "i8x16_relaxed_swizzle"
            | "i32x4_relaxed_trunc"
            | "i16x8_relaxed_q15mulr_s"
    )
}

fn is_lane_named_simd_test(test_name: &str) -> bool {
    let name = test_name.replace('\\', "/");
    let stem = Path::new(&name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&name);
    stem.starts_with("i8x16_")
        || stem.starts_with("i16x8_")
        || stem.starts_with("i32x4_")
        || stem.starts_with("i64x2_")
        || stem.starts_with("f32x4_")
        || stem.starts_with("f64x2_")
}

/// This interpreter currently implements a partial WebAssembly 3.0 surface.
/// The upstream testsuite includes features that still need parser, validator,
/// lowering, and runtime work, so we skip them here until the implementation is
/// complete.
pub fn should_skip_test(test_name: &str, simd_enabled: bool) -> bool {
    // Skip advanced features that may not be implemented yet

    // The upstream testsuite's `custom/` directory uses parser directives such
    // as `assert_malformed_custom` that the current `wast` crate does not
    // understand, so keep those files out of the runner for now.
    if test_name.starts_with("custom/")
        || test_name.starts_with("custom\\")
        || test_name.contains("/custom/")
        || test_name.contains("\\custom\\")
    {
        return true;
    }

    // Skip/enable SIMD tests across the testsuite naming patterns:
    // `simd_*`, `relaxed_*`, and lane-named files like `i32x4_*`.
    if test_name.starts_with("relaxed_")
        || test_name.starts_with("simd_")
        || is_lane_named_simd_test(test_name)
    {
        if simd_enabled && is_enabled_simd_test(test_name) {
            return false;
        }
        return true;
    }

    // Skip WebAssembly 3.0 features not yet implemented

    // Exception Handling proposal — `rethrow` + the `legacy/` directory
    // are legacy-proposal-shaped (pre-merge `try`/`catch`/`rethrow` /
    // `delegate` form) and stay skipped. The other EH tests
    // (tag/throw/throw_ref/try_table) are enabled; the runner surfaces wasm
    // exceptions via `WasmError::Exception` and `assert_exception`.
    if test_name == "rethrow"
        || test_name.starts_with("legacy/")
        || test_name.starts_with("legacy\\")
    {
        return true;
    }

    // What is OUT OF SCOPE, as opposed to unsupported. These never enter
    // the totals for either engine, because a suite total that mixes "we
    // do not do this" with "nobody targets this" cannot answer the only
    // question worth asking: does this engine pass Wasm 3.0.
    //
    // Above: the superseded exception-handling spelling (`try_table` is
    // the 3.0 one, and it runs), and custom-section annotations. Below:
    // post-3.0 proposals -- GC descriptors, exact references, custom page
    // sizes, threads, wide arithmetic.
    test_name.starts_with("proposals/") ||  // Unix paths
    test_name.starts_with("proposals\\") || // Windows paths
    test_name.contains("/proposals/") ||    // Unix paths (anywhere in path)
    test_name.contains("\\proposals\\") // Windows paths (anywhere in path)
}

#[cfg(test)]
mod tests {
    use super::should_skip_test;

    #[test]
    fn call_ref_is_not_skipped() {
        assert!(!should_skip_test("call_ref", false));
    }

    #[test]
    fn gc_array_and_recursive_type_tests_are_not_skipped() {
        assert!(!should_skip_test("array", false));
        assert!(!should_skip_test("type-rec", false));
        assert!(!should_skip_test("type-subtyping", false));
    }

    #[test]
    fn tail_call_tests_are_not_skipped() {
        assert!(!should_skip_test("return_call_ref", false));
    }

    #[test]
    fn instance_is_enabled() {
        assert!(!should_skip_test("instance", false));
        assert!(!should_skip_test("instance.wast", false));
    }

    #[test]
    fn memory64_tests_are_not_skipped() {
        assert!(!should_skip_test("memory64", false));
        assert!(!should_skip_test("memory64.wast", false));
        assert!(!should_skip_test("memory64-imports", false));
        assert!(!should_skip_test("memory_grow64", false));
    }

    #[test]
    fn simd_const_is_enabled_when_simd_is_available() {
        assert!(!should_skip_test("simd_const", true));
        assert!(!should_skip_test("simd_const.wast", true));
    }

    #[test]
    fn enabled_simd_files_are_not_skipped() {
        assert!(!should_skip_test("simd_boolean", true));
        assert!(!should_skip_test("simd_select", true));
        assert!(!should_skip_test("simd_bitwise", true));
        assert!(!should_skip_test("simd_splat", true));
        assert!(!should_skip_test("simd_bit_shift", true));
        assert!(!should_skip_test("simd_i8x16_arith", true));
        assert!(!should_skip_test("simd_i16x8_arith", true));
        assert!(!should_skip_test("simd_i32x4_arith", true));
        assert!(!should_skip_test("simd_i64x2_arith", true));
        assert!(!should_skip_test("simd_store", true));
        assert!(!should_skip_test("simd_address", true));
        assert!(!should_skip_test("simd_load", true));
        assert!(!should_skip_test("simd_f32x4_arith", true));
        assert!(!should_skip_test("simd_f64x2_arith", true));
        assert!(!should_skip_test("simd_i8x16_cmp", true));
        assert!(!should_skip_test("simd_i16x8_cmp", true));
        assert!(!should_skip_test("simd_i32x4_cmp", true));
        assert!(!should_skip_test("simd_i64x2_cmp", true));
        assert!(!should_skip_test("simd_f32x4_cmp", true));
        assert!(!should_skip_test("simd_f64x2_cmp", true));
        assert!(!should_skip_test("simd_i8x16_sat_arith", true));
        assert!(!should_skip_test("simd_i16x8_sat_arith", true));
        assert!(!should_skip_test("simd_i8x16_arith2", true));
        assert!(!should_skip_test("simd_i16x8_arith2", true));
        assert!(!should_skip_test("simd_i32x4_arith2", true));
        assert!(!should_skip_test("simd_i64x2_arith2", true));
        assert!(!should_skip_test("simd_f32x4", true));
        assert!(!should_skip_test("simd_f64x2", true));
        assert!(!should_skip_test("simd_f32x4_rounding", true));
        assert!(!should_skip_test("simd_f64x2_rounding", true));
        assert!(!should_skip_test("simd_i32x4_trunc_sat_f32x4", true));
        assert!(!should_skip_test("simd_i32x4_trunc_sat_f64x2", true));
        assert!(!should_skip_test("simd_linking", true));
        assert!(!should_skip_test("simd_f32x4_pmin_pmax", true));
        assert!(!should_skip_test("simd_f64x2_pmin_pmax", true));
        assert!(!should_skip_test("simd_align", true));
        assert!(!should_skip_test("simd_load_splat", true));
        assert!(!should_skip_test("simd_load_zero", true));
        assert!(!should_skip_test("simd_load_extend", true));
        assert!(!should_skip_test("simd_conversions", true));
        assert!(!should_skip_test("simd_i16x8_extadd_pairwise_i8x16", true));
        assert!(!should_skip_test("simd_i16x8_extmul_i8x16", true));
        assert!(!should_skip_test("simd_i16x8_q15mulr_sat_s", true));
        assert!(!should_skip_test("simd_i32x4_dot_i16x8", true));
        assert!(!should_skip_test("simd_i32x4_extadd_pairwise_i16x8", true));
        assert!(!should_skip_test("simd_i32x4_extmul_i16x8", true));
        assert!(!should_skip_test("simd_i64x2_extmul_i32x4", true));
        assert!(!should_skip_test("simd_int_to_int_extend", true));
        assert!(!should_skip_test("simd_lane", true));
        assert!(!should_skip_test("simd_load8_lane", true));
        assert!(!should_skip_test("simd_load16_lane", true));
        assert!(!should_skip_test("simd_load32_lane", true));
        assert!(!should_skip_test("simd_load64_lane", true));
        assert!(!should_skip_test("simd_store8_lane", true));
        assert!(!should_skip_test("simd_store16_lane", true));
        assert!(!should_skip_test("simd_store32_lane", true));
        assert!(!should_skip_test("simd_store64_lane", true));
        assert!(!should_skip_test("simd_memory-multi", true));
    }

    #[test]
    fn unsupported_simd_files_stay_skipped() {
        assert!(should_skip_test("simd_relaxed_unknown", true));
    }

    #[test]
    fn relaxed_simd_files_are_enabled_when_simd_is_available() {
        assert!(!should_skip_test("relaxed_dot_product", true));
        assert!(!should_skip_test("relaxed_laneselect", true));
        assert!(!should_skip_test("relaxed_madd_nmadd", true));
        assert!(!should_skip_test("relaxed_min_max", true));
        assert!(!should_skip_test("i8x16_relaxed_swizzle", true));
        assert!(!should_skip_test("i32x4_relaxed_trunc", true));
        assert!(!should_skip_test("i16x8_relaxed_q15mulr_s", true));
        assert!(should_skip_test("relaxed_dot_product", false));
    }

    #[test]
    fn unrelated_x_suffix_names_are_not_treated_as_simd() {
        assert!(!should_skip_test("x64-addressing", false));
        assert!(!should_skip_test("matrix16", false));
    }

    #[test]
    fn custom_annotation_tests_are_skipped() {
        assert!(should_skip_test("custom/branch_hint.wast", true));
        assert!(should_skip_test("custom\\name_annot.wast", true));
    }
}
