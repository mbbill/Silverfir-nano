//! Test discovery and filtering

use std::{fs, path::Path};

pub fn find_wast_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut wast_files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|ext| ext == "wast") {
                wast_files.push(path);
            } else if path.is_dir() {
                wast_files.extend(find_wast_files(&path));
            }
        }
    }

    wast_files.sort();
    wast_files
}

/// Determines if a test should be skipped based on feature support.
///
/// This interpreter currently implements a partial WebAssembly 3.0 surface.
/// The upstream testsuite includes features that still need parser, validator,
/// lowering, and runtime work, so we skip them here until the implementation is
/// complete.
pub fn should_skip_test(test_name: &str) -> bool {
    // Skip advanced features that may not be implemented yet

    // Skip SIMD/vector tests (various naming patterns)
    if test_name.starts_with("simd_") || test_name.starts_with("relaxed_") {
        return true;
    }

    // Skip SIMD lane-based tests (e.g., i32x4, f64x2, i16x8, etc.)
    // These contain vector opcodes that are not yet implemented
    let simd_patterns = ["x2", "x4", "x8", "x16"];
    for pattern in &simd_patterns {
        if test_name.contains(pattern) {
            return true;
        }
    }

    // Skip WebAssembly 3.0 features not yet implemented

    // Array tests still need runtime and lowering work. Some files also trip
    // known wast encoding issues, so keep the whole group disabled until the
    // implementation and test ingestion story are both in better shape.
    if test_name.starts_with("array") {
        return true;
    }

    // Type recursion tests: DISABLED due to wast crate encoder bug
    // The wast crate incorrectly encodes WAT with recursion groups, creating extra
    // types outside the rec group when handling implicit function types.
    //
    // Issue:
    // - WAT: (rec (type $ft (func)) (type (func))) (func $f)
    // - Expected: Rec group with 2 types (indices 0-1), function $f uses type 1
    // - Actual: Rec group with 2 types (indices 0-1), PLUS type 2 outside rec group, function $f
    //   uses type 2
    // - Result: Our validator correctly applies structural equivalence between type 2 (not in rec
    //   group) and type 0 (in rec group), accepting the module
    // - Expected: Test expects rejection because function should use type 1 (which IS in the same
    //   rec group as type 0), requiring nominal typing within the rec group
    //
    // Our type equivalence implementation is correct per WebAssembly 3.0 spec:
    // - Types in SAME rec group: nominal typing (must have same index)
    // - Types in DIFFERENT rec groups or one/both not in any rec group: structural (isorecursive)
    //   equivalence
    // - Function types: structural equivalence across rec groups
    // - Struct/Array types: not equivalent across different rec groups
    //
    // Affected tests: type-rec (directive #5), type-subtyping (directive #10)
    // Status: Our implementation is correct; the wast crate's encoding creates
    //         binaries that don't match the intended WAT semantics
    //
    // TODO: Re-enable when wast crate encoding is fixed
    // Tested with wast 239.0 (latest as of Oct 2025) - issue persists
    if test_name == "type-rec" || test_name == "type-subtyping" {
        return true;
    }

    // `br_on_cast*.wast` still trips the current `wast`/`wat` encoder on GC
    // array ops (`array.get_u` in folded forms and direct `array.set` were
    // observed to encode incorrectly on April 16, 2026), so keep the upstream
    // files disabled until the runner or dependency stack can ingest those
    // modules faithfully.
    if test_name.starts_with("br_on_cast") {
        return true;
    }

    // Exception Handling proposal
    if test_name.starts_with("tag")
        || test_name.starts_with("throw")
        || test_name == "rethrow"
        || test_name.starts_with("try_")
        || test_name.starts_with("instance")
    {
        return true;
    }

    // Skip Tail Call proposal
    if test_name.starts_with("return_call") {
        return true;
    }

    // Skip proposal tests (advanced WebAssembly features)
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
        assert!(!should_skip_test("call_ref"));
    }

    #[test]
    fn tail_call_tests_remain_skipped() {
        assert!(should_skip_test("return_call_ref"));
    }
}
