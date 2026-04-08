use std::{
    env,
    ffi::CString,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

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
    dir.join(alloc::format!("jit-{pid}.dump"))
}

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

#[cfg(target_family = "unix")]
fn open_tracking_file(path: &Path) -> io::Result<File> {
    use std::os::fd::FromRawFd;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "jitdump path contains NUL"))?;
    let file_ptr = unsafe { fopen(c_path.as_ptr(), c"wb".as_ptr()) };
    if file_ptr.is_null() {
        return Err(io::Error::last_os_error());
    }
    let fd = unsafe { fileno(file_ptr) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(target_family = "unix"))]
fn open_tracking_file(path: &Path) -> io::Result<File> {
    File::create(path)
}

#[cfg(target_family = "unix")]
unsafe extern "C" {
    fn fopen(
        path: *const core::ffi::c_char,
        mode: *const core::ffi::c_char,
    ) -> *mut core::ffi::c_void;
    fn fileno(stream: *mut core::ffi::c_void) -> i32;
}

fn monotonic_timestamp_nanos() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut timebase = mach_timebase_info { numer: 0, denom: 0 };
        mach_timebase_info(&mut timebase);
        let ticks = mach_absolute_time();
        ticks.saturating_mul(timebase.numer as u64) / timebase.denom.max(1) as u64
    }

    #[cfg(target_os = "linux")]
    unsafe {
        let mut ts = timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        clock_gettime(CLOCK_MONOTONIC, &mut ts);
        (ts.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(ts.tv_nsec as u64)
    }
    #[cfg(target_os = "windows")]
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

#[cfg(target_os = "macos")]
fn elf_machine_arch() -> u32 {
    EM_AARCH64
}

#[cfg(target_os = "linux")]
fn elf_machine_arch() -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        EM_AARCH64
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        EM_NONE
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn elf_machine_arch() -> u32 {
    EM_NONE
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
const EM_NONE: u32 = 0;
const EM_AARCH64: u32 = 183;

#[cfg(target_os = "linux")]
const CLOCK_MONOTONIC: i32 = 1;

#[cfg(target_os = "linux")]
#[repr(C)]
struct timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut timespec) -> i32;
}

#[cfg(target_os = "windows")]
unsafe extern "system" {
    fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    fn QueryPerformanceCounter(count: *mut i64) -> i32;
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct mach_timebase_info {
    numer: u32,
    denom: u32,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut mach_timebase_info) -> i32;
}
