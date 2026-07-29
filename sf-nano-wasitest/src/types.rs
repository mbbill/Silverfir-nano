//! Test data structures.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Default)]
pub struct TestConfig {
    pub args: Option<Vec<String>>,
    pub env: Option<HashMap<String, String>>,
    pub exit_code: Option<i32>,
    /// Directory, relative to the test's own directory, to preopen as the
    /// guest's root filesystem (`/`). This is how the suite declares a test's
    /// scratch area; there is no other preopen key in its schema.
    pub root: Option<String>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
}

#[derive(Debug)]
pub struct TestOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug)]
pub struct TestFailure {
    pub failure_type: String,
    pub message: String,
}

#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub is_executed: bool,
    pub skip_reason: Option<String>,
    pub failures: Vec<TestFailure>,
    pub duration: std::time::Duration,
}

impl TestResult {
    pub fn failed(&self) -> bool {
        !self.failures.is_empty()
    }

    pub fn status(&self) -> &'static str {
        if !self.is_executed {
            "SKIPPED"
        } else if self.failed() {
            "FAILED"
        } else {
            "PASSED"
        }
    }
}

#[derive(Debug)]
pub struct TestSuite {
    pub name: String,
    pub test_results: Vec<TestResult>,
}

impl TestSuite {
    pub fn total_count(&self) -> usize {
        self.test_results.len()
    }

    pub fn passed_count(&self) -> usize {
        self.test_results
            .iter()
            .filter(|result| result.is_executed && !result.failed())
            .count()
    }

    pub fn failed_count(&self) -> usize {
        self.test_results
            .iter()
            .filter(|result| result.failed())
            .count()
    }

    pub fn skipped_count(&self) -> usize {
        self.test_results
            .iter()
            .filter(|result| !result.is_executed)
            .count()
    }
}
