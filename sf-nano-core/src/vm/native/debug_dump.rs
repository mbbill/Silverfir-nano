//! Optional native artifact dump.
//!
//! Enabled by `feature = "native-dump"` and `SF_DUMP_DIR=/path/to/root`.
//! Dumps per-function native artifacts alongside shared IR dumps.

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec::Vec,
};

use crate::vm::{
    debug_dump,
    lowered::{IrOp, IrOpKind, OpIndex},
    native::group_meta::{sanitize_token as sanitize_summary, summarize_group},
};

use std::sync::{Mutex, OnceLock};

#[derive(Clone)]
struct ArtifactRecord {
    code_start: usize,
    code_len: usize,
    kind: &'static str,
    ir_start: usize,
    ir_end: usize,
    terminator: String,
    branch_target: Option<usize>,
    summary: String,
}

type ArtifactMap = BTreeMap<(String, u32), Vec<ArtifactRecord>>;

static RECORDS: OnceLock<Mutex<ArtifactMap>> = OnceLock::new();

fn records() -> &'static Mutex<ArtifactMap> {
    RECORDS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn enabled() -> bool {
    debug_dump::enabled()
}

pub fn record_group(
    code_start: usize,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    ir_start: usize,
    group: &[IrOp],
    terminator: Option<&'static str>,
    branch_target: Option<OpIndex>,
) {
    if !enabled() {
        return;
    }
    if !debug_dump::should_dump_func(func_idx) {
        return;
    }
    let mut guard = records().lock().expect("native debug dump lock poisoned");
    guard
        .entry((module_name.to_string(), func_idx))
        .or_default()
        .push(ArtifactRecord {
            code_start,
            code_len,
            kind: "group",
            ir_start,
            ir_end: ir_start + group.len(),
            terminator: terminator.unwrap_or("-").to_string(),
            branch_target: branch_target.map(|idx| idx.as_usize()),
            summary: summarize_group(group),
        });
}

pub fn record_wrapper(
    code_start: usize,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    ir_index: usize,
    helper_name: &str,
    op_kind: &IrOpKind,
) {
    if !enabled() {
        return;
    }
    if !debug_dump::should_dump_func(func_idx) {
        return;
    }
    let mut guard = records().lock().expect("native debug dump lock poisoned");
    guard
        .entry((module_name.to_string(), func_idx))
        .or_default()
        .push(ArtifactRecord {
            code_start,
            code_len,
            kind: "wrapper",
            ir_start: ir_index,
            ir_end: ir_index + 1,
            terminator: sanitize_summary(helper_name),
            branch_target: None,
            summary: sanitize_summary(&format!("{:?}", op_kind)),
        });
}

pub fn flush_function(
    code_base: *const u8,
    module_name: &str,
    func_idx: u32,
) {
    if !enabled() {
        return;
    }
    if !debug_dump::should_dump_func(func_idx) {
        return;
    }

    let maybe_records = {
        let mut guard = records().lock().expect("native debug dump lock poisoned");
        guard.remove(&(module_name.to_string(), func_idx))
    };
    let Some(mut entries) = maybe_records else {
        return;
    };
    entries.sort_by_key(|entry| entry.code_start);

    let Some(dir) = debug_dump::subdir(module_name, func_idx, "native") else {
        return;
    };

    let mut index = String::from("# idx\tkind\tir\tterm\ttarget\tstart\tend\tblob\tsummary\n");
    for (idx, entry) in entries.iter().enumerate() {
        let blob_name = blob_name(idx, entry);
        let blob_path = dir.join(&blob_name);
        let bytes = unsafe {
            core::slice::from_raw_parts(code_base.add(entry.code_start), entry.code_len)
        };
        let _ = debug_dump::write_bytes(&blob_path, bytes);
        index.push_str(&format!(
            "{}\t{}\t{}..{}\t{}\t{}\t0x{:x}\t0x{:x}\t{}\t{}\n",
            idx,
            entry.kind,
            entry.ir_start,
            entry.ir_end,
            entry.terminator,
            entry
                .branch_target
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            code_base as usize + entry.code_start,
            code_base as usize + entry.code_start + entry.code_len,
            blob_name,
            entry.summary,
        ));
    }
    let _ = debug_dump::write_text(&dir.join("index.tsv"), &index);
}

fn blob_name(idx: usize, entry: &ArtifactRecord) -> String {
    format!(
        "{idx:04}_{}_ir{}_{}_{}.bin",
        entry.kind,
        entry.ir_start,
        entry.ir_end,
        debug_dump::sanitize_token(&entry.terminator)
    )
}
