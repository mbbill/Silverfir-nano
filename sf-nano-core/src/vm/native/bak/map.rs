//! Optional native address-to-function map dumping.
//!
//! Enable by setting `SF_JIT_MAP=/path/to/map.txt` before running.

use alloc::format;

use crate::vm::native::symbols::{sanitize_token, NativeSymbolStats};

#[cfg(any(feature = "wasi", feature = "std", test))]
use std::{
    env,
    fs::{File, OpenOptions},
    io::Write,
    sync::{Mutex, OnceLock},
};

#[cfg(any(feature = "wasi", feature = "std", test))]
static MAP_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

#[cfg(any(feature = "wasi", feature = "std", test))]
fn writer() -> Option<&'static Mutex<File>> {
    MAP_FILE
        .get_or_init(|| {
            let path = env::var("SF_JIT_MAP").ok()?;
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)
                .ok()?;
            if writeln!(
                file,
                "# start\tend\tmodule\tfunc\tsegment\tmode\tblocks\tops\tcall_local\tcall_indirect\tcall_external\thelpers_t\thelpers_ct\tbr_table\tloads\tstores"
            )
            .is_err()
            {
                return None;
            }
            Some(Mutex::new(file))
        })
        .as_ref()
}

pub fn record_function(
    code_start: *const u8,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    segment: &str,
    direct: bool,
    stats: NativeSymbolStats,
) {
    #[cfg(any(feature = "wasi", feature = "std", test))]
    {
        let Some(writer) = writer() else {
            return;
        };
        let line = render_line(
            code_start,
            code_len,
            module_name,
            func_idx,
            segment,
            direct,
            stats,
        );
        if let Ok(mut file) = writer.lock() {
            let _ = file.write_all(line.as_bytes());
        }
    }

    #[cfg(not(any(feature = "wasi", feature = "std", test)))]
    let _ = (code_start, code_len, module_name, func_idx, segment, direct, stats);
}

fn render_line(
    code_start: *const u8,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    segment: &str,
    direct: bool,
    stats: NativeSymbolStats,
) -> alloc::string::String {
    let start = code_start as usize;
    let end = start + code_len;
    format!(
        "0x{start:016x}\t0x{end:016x}\tmodule={}\tfunc={}\tsegment={}\tmode={}\tblocks={}\tops={}\tcall_local={}\tcall_indirect={}\tcall_external={}\thelpers_t={}\thelpers_ct={}\tbr_table={}\tloads={}\tstores={}\n",
        sanitize_token(module_name),
        func_idx,
        sanitize_token(segment),
        if direct { "direct" } else { "shared" },
        stats.blocks,
        stats.ops,
        stats.call_local,
        stats.call_indirect,
        stats.call_external,
        stats.transparent_helpers,
        stats.control_transfer_helpers,
        stats.br_table,
        stats.loads,
        stats.stores,
    )
}

#[cfg(test)]
mod tests {
    use super::render_line;
    use crate::vm::native::symbols::NativeSymbolStats;

    #[test]
    fn render_line_includes_function_metadata() {
        let line = render_line(
            0x1000 as *const u8,
            0x30,
            "main module",
            6,
            "body",
            true,
            NativeSymbolStats {
                blocks: 4,
                ops: 32,
                call_local: 8,
                br_table: 2,
                loads: 9,
                stores: 5,
                ..NativeSymbolStats::default()
            },
        );

        assert!(line.contains("0x0000000000001000"));
        assert!(line.contains("0x0000000000001030"));
        assert!(line.contains("module=main_module"));
        assert!(line.contains("func=6"));
        assert!(line.contains("segment=body"));
        assert!(line.contains("mode=direct"));
        assert!(line.contains("call_local=8"));
        assert!(line.contains("br_table=2"));
        assert!(line.contains("loads=9"));
        assert!(line.contains("stores=5"));
    }
}
