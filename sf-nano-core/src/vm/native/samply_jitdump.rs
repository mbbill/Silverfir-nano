use alloc::format;

use crate::vm::lowered::{IrOp, OpIndex};
use crate::vm::native::group_meta::render_symbol_name;

#[cfg(any(feature = "wasi", feature = "std", test))]
use std::{
    env,
    ffi::CString,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

#[cfg(any(feature = "wasi", feature = "std", test))]
static JITDUMP_FILE: OnceLock<Option<Mutex<JitDumpWriter>>> = OnceLock::new();

#[cfg(any(feature = "wasi", feature = "std", test))]
pub fn record_group(
    code_base: *const u8,
    code_start: usize,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    ir_start: usize,
    group: &[IrOp],
    terminator: Option<&'static str>,
    branch_target: Option<OpIndex>,
) {
    let Some(writer) = writer() else {
        return;
    };

    let name = render_symbol_name(
        module_name,
        func_idx,
        ir_start,
        group,
        terminator,
        branch_target,
    );
    let code_addr = (code_base as usize + code_start) as u64;
    let code_ptr = unsafe { code_base.add(code_start) };
    let code_bytes = unsafe { std::slice::from_raw_parts(code_ptr, code_len) };

    if let Ok(mut writer) = writer.lock() {
        let _ = writer.write_code_load(&name, code_addr, code_bytes);
    }
}

#[cfg(not(any(feature = "wasi", feature = "std", test)))]
pub fn record_group(
    code_base: *const u8,
    code_start: usize,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    ir_start: usize,
    group: &[IrOp],
    terminator: Option<&'static str>,
    branch_target: Option<OpIndex>,
) {
    let _ = (
        code_base,
        code_start,
        code_len,
        module_name,
        func_idx,
        ir_start,
        group,
        terminator,
        branch_target,
    );
}

#[cfg(any(feature = "wasi", feature = "std", test))]
fn writer() -> Option<&'static Mutex<JitDumpWriter>> {
    JITDUMP_FILE.get_or_init(create_writer).as_ref()
}

#[cfg(any(feature = "wasi", feature = "std", test))]
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

#[cfg(any(feature = "wasi", feature = "std", test))]
fn jitdump_enabled() -> bool {
    env::var_os("SF_JITDUMP").is_some() || env::var_os("SAMPLY_BOOTSTRAP_SERVER_NAME").is_some()
}

#[cfg(any(feature = "wasi", feature = "std", test))]
fn jitdump_path(pid: u32) -> PathBuf {
    let dir = env::var_os("SF_JITDUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    dir.join(format!("jit-{pid}.dump"))
}

#[cfg(any(feature = "wasi", feature = "std", test))]
struct JitDumpWriter {
    file: File,
    pid: u32,
    tid: u32,
    next_code_index: u64,
}

#[cfg(any(feature = "wasi", feature = "std", test))]
impl JitDumpWriter {
    fn new(mut file: File, pid: u32) -> io::Result<Self> {
        write_file_header(&mut file, pid, monotonic_timestamp_nanos(), elf_machine_arch())?;
        file.flush()?;
        Ok(Self {
            file,
            pid,
            tid: pid,
            next_code_index: 1,
        })
    }

    fn write_code_load(&mut self, function_name: &str, code_addr: u64, code_bytes: &[u8]) -> io::Result<()> {
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

#[cfg(any(feature = "wasi", feature = "std", test))]
impl Drop for JitDumpWriter {
    fn drop(&mut self) {
        let _ = write_record_header(&mut self.file, JIT_CODE_CLOSE, RECORD_HEADER_SIZE as u32, monotonic_timestamp_nanos());
        let _ = self.file.flush();
    }
}

#[cfg(any(feature = "wasi", feature = "std", test))]
const FILE_HEADER_SIZE: u32 = 40;
#[cfg(any(feature = "wasi", feature = "std", test))]
const RECORD_HEADER_SIZE: usize = 16;
#[cfg(any(feature = "wasi", feature = "std", test))]
const JIT_CODE_LOAD: u32 = 0;
#[cfg(any(feature = "wasi", feature = "std", test))]
const JIT_CODE_CLOSE: u32 = 3;

#[cfg(any(feature = "wasi", feature = "std", test))]
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

#[cfg(any(feature = "wasi", feature = "std", test))]
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

#[cfg(any(feature = "wasi", feature = "std", test))]
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

#[cfg(any(feature = "wasi", feature = "std", test))]
#[cfg(target_family = "unix")]
fn open_tracking_file(path: &Path) -> io::Result<File> {
    use std::os::fd::FromRawFd;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "jitdump path contains NUL"))?;
    let file_ptr = unsafe { fopen(c_path.as_ptr(), b"wb\0".as_ptr() as *const core::ffi::c_char) };
    if file_ptr.is_null() {
        return Err(io::Error::last_os_error());
    }
    // We create the file with `fopen` so samply's preload library sees the path.
    // After that, all writes go through Rust's `File`; the leaked FILE* is tiny and
    // process-global, so this is acceptable for an opt-in profiling path.
    let fd = unsafe { fileno(file_ptr) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(any(feature = "wasi", feature = "std", test))]
#[cfg(not(target_family = "unix"))]
fn open_tracking_file(path: &Path) -> io::Result<File> {
    File::create(path)
}

#[cfg(any(feature = "wasi", feature = "std", test))]
#[cfg(target_family = "unix")]
extern "C" {
    fn fopen(path: *const core::ffi::c_char, mode: *const core::ffi::c_char) -> *mut core::ffi::c_void;
    fn fileno(stream: *mut core::ffi::c_void) -> i32;
}

#[cfg(any(feature = "wasi", feature = "std", test))]
fn elf_machine_arch() -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        return 183;
    }
    #[cfg(target_arch = "x86_64")]
    {
        return 62;
    }
    #[cfg(target_arch = "riscv64")]
    {
        return 243;
    }
    #[allow(unreachable_code)]
    0
}

#[cfg(any(feature = "wasi", feature = "std", test))]
#[cfg(target_os = "macos")]
fn monotonic_timestamp_nanos() -> u64 {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MachTimebaseInfo {
        numer: u32,
        denom: u32,
    }

    extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    }

    static TIMEBASE: OnceLock<MachTimebaseInfo> = OnceLock::new();

    let info = TIMEBASE.get_or_init(|| unsafe {
        let mut info = MachTimebaseInfo { numer: 1, denom: 1 };
        if mach_timebase_info(&mut info as *mut _) != 0 || info.denom == 0 {
            info = MachTimebaseInfo { numer: 1, denom: 1 };
        }
        info
    });

    let ticks = unsafe { mach_absolute_time() };
    ticks.saturating_mul(info.numer as u64) / info.denom as u64
}

#[cfg(any(feature = "wasi", feature = "std", test))]
#[cfg(target_os = "linux")]
fn monotonic_timestamp_nanos() -> u64 {
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }

    extern "C" {
        fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
    }

    const CLOCK_MONOTONIC: i32 = 1;

    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts as *mut _) };
    if rc != 0 {
        return 0;
    }
    (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64)
}

#[cfg(any(feature = "wasi", feature = "std", test))]
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn monotonic_timestamp_nanos() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_write_code_load_record_includes_name_and_bytes() {
        let mut bytes = Vec::new();
        write_code_load_record(
            &mut bytes,
            123,
            7,
            7,
            0x1000,
            1,
            "jit::test",
            &[0xaa, 0xbb, 0xcc, 0xdd],
        )
        .unwrap();

        assert_eq!(&bytes[0..4], &JIT_CODE_LOAD.to_le_bytes());
        assert_eq!(&bytes[4..8], &(bytes.len() as u32).to_le_bytes());
        assert_eq!(&bytes[8..16], &123u64.to_le_bytes());
        assert!(bytes.windows(b"jit::test\0".len()).any(|w| w == b"jit::test\0"));
        assert_eq!(&bytes[bytes.len() - 4..], &[0xaa, 0xbb, 0xcc, 0xdd]);
    }
}
