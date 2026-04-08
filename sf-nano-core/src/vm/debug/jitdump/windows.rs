//! Windows host primitives for `jitdump`: file open via `File::create`,
//! monotonic time via `QueryPerformanceCounter`/`QueryPerformanceFrequency`,
//! and ELF machine arch tag set to `EM_NONE` (Windows does not use the
//! Linux perf `jitdump` format; the exporter still compiles so the feature
//! is available on dev builds targeting Windows).

use std::fs::File;
use std::io;
use std::path::Path;

use super::EM_NONE;

unsafe extern "system" {
    fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    fn QueryPerformanceCounter(count: *mut i64) -> i32;
}

pub(super) fn open_tracking_file(path: &Path) -> io::Result<File> {
    File::create(path)
}

pub(super) fn monotonic_timestamp_nanos() -> u64 {
    unsafe {
        let mut freq: i64 = 0;
        let mut count: i64 = 0;
        QueryPerformanceFrequency(&mut freq);
        QueryPerformanceCounter(&mut count);
        if freq == 0 {
            return 0;
        }
        let secs = count / freq;
        let rem = count % freq;
        (secs as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add((rem as u64).saturating_mul(1_000_000_000) / freq as u64)
    }
}

pub(super) fn elf_machine_arch() -> u32 {
    EM_NONE
}
