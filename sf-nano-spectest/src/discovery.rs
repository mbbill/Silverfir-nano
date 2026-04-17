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

    // `br_on_cast*.wast` is no longer skipped. The earlier failure mode was
    // our own GC array opcode-table bug, not a persistent upstream `wast`/`wat`
    // limitation, and the targeted spectests now pass again.

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
    fn gc_array_and_recursive_type_tests_are_not_skipped() {
        assert!(!should_skip_test("array"));
        assert!(!should_skip_test("type-rec"));
        assert!(!should_skip_test("type-subtyping"));
    }

    #[test]
    fn tail_call_tests_remain_skipped() {
        assert!(should_skip_test("return_call_ref"));
    }
}
