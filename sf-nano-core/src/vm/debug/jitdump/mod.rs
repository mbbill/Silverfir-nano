//! Emit the Linux-perf `jitdump` symbol/code format so that external
//! profilers (samply, perf) can resolve JIT-compiled code regions to
//! symbols.
//!
//! This module is the generic half. All host-specific pieces — file open,
//! monotonic clock, ELF machine arch tag — live in the per-OS submodules
//! below and are selected by the `sf_os_*` cfgs:
//!
//! - `linux`   (glibc/musl fopen + clock_gettime)
//! - `macos`   (Darwin fopen + mach_absolute_time)
//! - `windows` (std::fs + QueryPerformance*)
//!
//! `jitdump` is gated on `sf_jitdump`, which itself requires `sf_has_std`,
//! so none of the `target_os = "none"` path applies here.

use std::{
    env,
    fs::{self, File},
    io::{self, Write},
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

// ── Host-specific primitives ────────────────────────────────────────────────
//
// Exactly one submodule is active; it exports the three host-coupled free
// functions the generic code uses. Adding another host is a matter of
// dropping a new file beside these and extending the cfg selection here.

#[cfg(sf_os_linux)]
mod linux;
#[cfg(sf_os_linux)]
use linux::{elf_machine_arch, monotonic_timestamp_nanos, open_tracking_file};

#[cfg(sf_os_macos)]
mod macos;
#[cfg(sf_os_macos)]
use macos::{elf_machine_arch, monotonic_timestamp_nanos, open_tracking_file};

#[cfg(sf_os_windows)]
mod windows;
#[cfg(sf_os_windows)]
use windows::{elf_machine_arch, monotonic_timestamp_nanos, open_tracking_file};

// ── Shared ELF machine tag constants ────────────────────────────────────────
//
// Kept here rather than in each host module so that the three hosts agree
// on the integer values without duplication.

#[cfg(any(sf_os_linux, sf_os_windows))]
pub(super) const EM_NONE: u32 = 0;
#[cfg(any(sf_os_macos, all(sf_os_linux, sf_arch_arm64)))]
pub(super) const EM_AARCH64: u32 = 183;

// ── Public entry point ──────────────────────────────────────────────────────

static JITDUMP_FILE: OnceLock<Option<Mutex<JitDumpWriter>>> = OnceLock::new();

pub(crate) fn record_function(code_start: *const u8, code_bytes: &[u8], symbol_name: &str) {
    let Some(writer) = writer() else {
        return;
    };
    if let Ok(mut writer) = writer.lock() {
        let _ = writer.write_code_load(symbol_name, code_start as u64, code_bytes);
    }
}

fn writer() -> Option<&'static Mutex<JitDumpWriter>> {
    JITDUMP_FILE.get_or_init(create_writer).as_ref()
}

fn create_writer() -> Option<Mutex<JitDumpWriter>> {
    if !jitdump_enabled() {
        return None;
    }

    let pid = std::process::id();
    let path = jitdump_path(pid);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok()?;
    }

    let file = open_tracking_file(&path).ok()?;
    let writer = JitDumpWriter::new(file, pid).ok()?;
    Some(Mutex::new(writer))
}

fn jitdump_enabled() -> bool {
    env::var_os("SF_JITDUMP").is_some() || env::var_os("SAMPLY_BOOTSTRAP_SERVER_NAME").is_some()
}

fn jitdump_path(pid: u32) -> PathBuf {
    let dir = env::var_os("SF_JITDUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    dir.join(tracked_alloc::format!("jit-{pid}.dump"))
}

// ── Writer ──────────────────────────────────────────────────────────────────

struct JitDumpWriter {
    file: File,
    pid: u32,
    tid: u32,
    next_code_index: u64,
}

impl JitDumpWriter {
    fn new(mut file: File, pid: u32) -> io::Result<Self> {
        write_file_header(
            &mut file,
            pid,
            monotonic_timestamp_nanos(),
            elf_machine_arch(),
        )?;
        file.flush()?;
        Ok(Self {
            file,
            pid,
            tid: pid,
            next_code_index: 1,
        })
    }

    fn write_code_load(
        &mut self,
        function_name: &str,
        code_addr: u64,
        code_bytes: &[u8],
    ) -> io::Result<()> {
        let code_index = self.next_code_index;
        self.next_code_index += 1;
        write_code_load_record(
            &mut self.file,
            monotonic_timestamp_nanos(),
            self.pid,
            self.tid,
            code_addr,
            code_index,
            function_name,
            code_bytes,
        )?;
        self.file.flush()
    }
}

impl Drop for JitDumpWriter {
    fn drop(&mut self) {
        let _ = write_record_header(
            &mut self.file,
            JIT_CODE_CLOSE,
            RECORD_HEADER_SIZE as u32,
            monotonic_timestamp_nanos(),
        );
        let _ = self.file.flush();
    }
}

// ── Format ──────────────────────────────────────────────────────────────────

const FILE_HEADER_SIZE: u32 = 40;
const RECORD_HEADER_SIZE: usize = 16;
const JIT_CODE_LOAD: u32 = 0;
const JIT_CODE_CLOSE: u32 = 3;

fn write_file_header<W: Write>(
    out: &mut W,
    pid: u32,
    timestamp: u64,
    elf_machine_arch: u32,
) -> io::Result<()> {
    #[cfg(target_endian = "little")]
    out.write_all(b"DTiJ")?;
    #[cfg(target_endian = "big")]
    out.write_all(b"JiTD")?;

    out.write_all(&1u32.to_le_bytes())?;
    out.write_all(&FILE_HEADER_SIZE.to_le_bytes())?;
    out.write_all(&elf_machine_arch.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?;
    out.write_all(&pid.to_le_bytes())?;
    out.write_all(&timestamp.to_le_bytes())?;
    out.write_all(&0u64.to_le_bytes())?;
    Ok(())
}

fn write_code_load_record<W: Write>(
    out: &mut W,
    timestamp: u64,
    pid: u32,
    tid: u32,
    code_addr: u64,
    code_index: u64,
    function_name: &str,
    code_bytes: &[u8],
) -> io::Result<()> {
    let name_bytes = function_name.as_bytes();
    let total_size =
        RECORD_HEADER_SIZE + 4 + 4 + 8 + 8 + 8 + 8 + name_bytes.len() + 1 + code_bytes.len();
    write_record_header(out, JIT_CODE_LOAD, total_size as u32, timestamp)?;
    out.write_all(&pid.to_le_bytes())?;
    out.write_all(&tid.to_le_bytes())?;
    out.write_all(&code_addr.to_le_bytes())?;
    out.write_all(&code_addr.to_le_bytes())?;
    out.write_all(&(code_bytes.len() as u64).to_le_bytes())?;
    out.write_all(&code_index.to_le_bytes())?;
    out.write_all(name_bytes)?;
    out.write_all(&[0])?;
    out.write_all(code_bytes)?;
    Ok(())
}

fn write_record_header<W: Write>(
    out: &mut W,
    record_type: u32,
    total_size: u32,
    timestamp: u64,
) -> io::Result<()> {
    out.write_all(&record_type.to_le_bytes())?;
    out.write_all(&total_size.to_le_bytes())?;
    out.write_all(&timestamp.to_le_bytes())?;
    Ok(())
}
