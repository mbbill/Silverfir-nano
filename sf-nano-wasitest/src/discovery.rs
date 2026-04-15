//! Test discovery functions.

use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn find_test_suite_directories(testsuite_path: &Path) -> Vec<PathBuf> {
    let mut test_dirs = Vec::new();
    find_wasm_directories(&testsuite_path.join("tests"), &mut test_dirs);
    test_dirs.sort();
    test_dirs
}

fn find_wasm_directories(dir: &Path, result: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }

    if let Some(dir_name) = dir.file_name().and_then(|name| name.to_str()) {
        if dir_name == "wasm32-wasip3" {
            return;
        }
    }

    if has_wasm_files(dir) {
        result.push(dir.to_path_buf());
        return;
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_wasm_directories(&path, result);
            }
        }
    }
}

fn has_wasm_files(dir: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "wasm") {
                return true;
            }
        }
    }
    false
}

pub fn discover_suite_tests(suite_dir: &Path) -> (String, Vec<(PathBuf, String)>) {
    let suite_name = read_manifest(suite_dir).unwrap_or_else(|| {
        suite_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string()
    });
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(suite_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "wasm") {
                let name = path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                out.push((path, name));
            }
        }
    }
    (suite_name, out)
}

pub fn match_filters(patterns: &[String], name: &str) -> bool {
    if patterns.is_empty() {
        return true;
    }
    let lname = name.to_lowercase();
    patterns
        .iter()
        .any(|pattern| lname.contains(&pattern.to_lowercase()))
}

pub fn read_manifest(suite_dir: &Path) -> Option<String> {
    let manifest_path = suite_dir.join("manifest.json");
    if !manifest_path.exists() {
        return None;
    }

    match fs::read_to_string(&manifest_path) {
        Ok(content) => {
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
                manifest
                    .get("name")
                    .and_then(|name| name.as_str())
                    .map(|name| name.to_string())
            } else {
                None
            }
        }
        Err(_) => None,
    }
}
