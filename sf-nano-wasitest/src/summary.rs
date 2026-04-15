//! Test result summary and reporting.

use crate::types::TestSuite;

pub fn print_test_summary(suites: &[TestSuite], total_duration: std::time::Duration) {
    let total_tests: usize = suites.iter().map(|suite| suite.total_count()).sum();
    let total_passed: usize = suites.iter().map(|suite| suite.passed_count()).sum();
    let total_failed: usize = suites.iter().map(|suite| suite.failed_count()).sum();
    let total_skipped: usize = suites.iter().map(|suite| suite.skipped_count()).sum();

    println!();
    println!("=== WASI Test Summary ===");
    println!("Test Suites: {}", suites.len());
    println!("Total Tests: {}", total_tests);
    println!(
        "Passed:  {} ({:.1}%)",
        total_passed,
        if total_tests > 0 {
            (total_passed as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "Failed:  {} ({:.1}%)",
        total_failed,
        if total_tests > 0 {
            (total_failed as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "Skipped: {} ({:.1}%)",
        total_skipped,
        if total_tests > 0 {
            (total_skipped as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("Duration: {:.2?}", total_duration);
}
