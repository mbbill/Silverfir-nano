//! Test data structures.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
pub struct TestConfig {
    pub args: Option<Vec<String>>,
    pub env: Option<HashMap<String, String>>,
    pub exit_code: Option<i32>,
    pub dirs: Option<Vec<String>>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub wasi_functions: Option<Vec<String>>,
}

#[derive(Debug)]
#[allow(dead_code)]
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
#[allow(dead_code)]
pub struct TestResult {
    pub name: String,
    pub config: TestConfig,
    pub output: Option<TestOutput>,
    pub is_executed: bool,
    pub failures: Vec<TestFailure>,
    pub duration: std::time::Duration,
    pub command_line: Option<String>,
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
#[allow(dead_code)]
pub struct TestSuite {
    pub name: String,
    pub test_results: Vec<TestResult>,
    pub duration: std::time::Duration,
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
